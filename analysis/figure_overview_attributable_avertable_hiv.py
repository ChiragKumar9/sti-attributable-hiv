import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
from matplotlib import rc

from averted_burden.delta_method import ci_via_log, ci_via_logit

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

SEXES = ["Male", "Female", "MSM"]
STIS = ["gc", "chlamydia", "syphilis", "trichomoniasis"]


def _sum_field(rows, field):
    """Sum a UFloat-valued field across a list of raw (not-yet-extracted)
    rows. Summing the UFloat objects themselves -- rather than summing
    already-extracted lower/upper CI bounds -- is what actually propagates
    variance/covariance correctly: the `uncertainties` library's linear
    error propagation picks up the correlations coming from any shared RR,
    resistance, or GBD/UNAIDS-cluster leaves automatically. Only extract an
    interval (via ci_via_log) from the final summed UFloat."""
    total = None
    for r in rows:
        val = r[field]
        total = val if total is None else total + val
    return total


def _sum_attributable_across_stis(rows, prefix):
    """Sum UFloat attributable-case counts across all four STIs AND across
    the given rows (e.g. every country in a region) by summing the UFloat
    objects themselves, for the same reason as `_sum_field` above."""
    total = None
    for r in rows:
        for sti in STIS:
            val = r[f"{prefix}{sti}"]
            total = val if total is None else total + val
    return total


def _fmt_count(triple):
    """Format a (mean, lower, upper) count triple as comma-separated integers:
    '12,345 (95% CI 9,876-15,432)'. Cases are whole numbers, so no decimals."""
    mean, lower, upper = triple
    if mean is None:
        return "n/a"
    return f"{round(mean):,} (95% CI {round(lower):,}-{round(upper):,})"


def _fmt_pct(triple):
    """Format a (mean, lower, upper) proportion triple as a percentage with one
    decimal: '12.4% (95% CI 9.8-15.4%)'."""
    mean, lower, upper = triple
    if mean is None:
        return "n/a"
    return f"{100 * mean:.1f}% (95% CI {100 * lower:.1f}-{100 * upper:.1f}%)"


_QUAD_Z95 = 1.959963984540054  # 97.5th percentile of the standard normal


def _block_combine(means, lowers, uppers, blocks):
    """Combine per-row CSV intervals with a two-level correlation structure:
    rows sharing a `block` id are treated as PERFECTLY correlated (their spreads
    add linearly -- appropriate for the multiple years of one (country, sex)
    series, driven by shared RR / curve / autocorrelated-incidence leaves);
    distinct blocks are treated as INDEPENDENT (quadrature). Intervals are
    read as LOG-symmetric (as ci_via_log produced them): each row's linear sd is
    recovered from the log CV, sd_i = mean_i * (log upper_i - log lower_i) /
    (2*Z95). The summed sd is re-expressed as a log-symmetric band, so the lower
    edge stays strictly positive.

    blocks: iterable of hashable ids, one per row (e.g. (country_code, sex)).
    Returns (mean_total, lower, upper).
    """
    means = np.asarray(means, dtype=float)
    lowers = np.asarray(lowers, dtype=float)
    uppers = np.asarray(uppers, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (np.log(uppers) - np.log(lowers)) / (2 * _QUAD_Z95)
    sd = means * cv
    ok = np.isfinite(sd) & (means > 0) & (lowers > 0) & (uppers > 0)
    sd = np.where(ok, sd, 0.0)
    m = np.where(np.isfinite(means), means, 0.0)

    block_mean = defaultdict(float)
    block_sd = defaultdict(float)
    for i, b in enumerate(blocks):
        block_mean[b] += m[i]
        block_sd[b] += sd[i]  # linear sd addition => perfect corr within block

    mean_total = float(sum(block_mean.values()))
    if not np.isfinite(mean_total) or mean_total <= 0:
        return mean_total, mean_total, mean_total
    sd_total = float(
        np.sqrt(sum(s * s for s in block_sd.values()))
    )  # quadrature across blocks
    cv_total = sd_total / mean_total
    lower = mean_total * np.exp(-_QUAD_Z95 * cv_total)
    upper = mean_total * np.exp(_QUAD_Z95 * cv_total)
    return mean_total, float(lower), float(upper)


def _quadrature_sum(means, lowers, uppers):
    """Combine per-row CSV intervals into one interval by treating the rows as
    INDEPENDENT and adding variances (quadrature). Each row's interval is
    treated as LOG-symmetric -- which is how ci_via_log produced it -- rather
    than linear-symmetric: the row's linear-space sd is recovered from the
    log-space coefficient of variation, sd_i = mean_i * (log upper_i -
    log lower_i) / (2*Z95), not from the raw linear half-width. Variances add
    under independence, then the summed sd is re-expressed as a log-symmetric
    band around the total mean via the same delta-method transform ci_via_log
    uses. This keeps the lower edge strictly positive (no floor needed) and, for
    a single row, reproduces that row's own interval.

    Returns (mean_total, lower, upper).
    """
    means = np.asarray(means, dtype=float)
    lowers = np.asarray(lowers, dtype=float)
    uppers = np.asarray(uppers, dtype=float)

    mean_total = float(np.nansum(means))
    if not np.isfinite(mean_total) or mean_total <= 0:
        return mean_total, mean_total, mean_total

    # per-row linear sd recovered from the LOG interval (log-symmetric):
    #   cv_i = (log upper_i - log lower_i) / (2 * Z95);  sd_i = mean_i * cv_i
    # rows with a non-positive mean/lower/upper contribute zero variance.
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = (np.log(uppers) - np.log(lowers)) / (2 * _QUAD_Z95)
    sd_rows = means * cv
    sd_rows = np.where(
        np.isfinite(sd_rows) & (means > 0) & (lowers > 0) & (uppers > 0),
        sd_rows,
        0.0,
    )
    sd_total = float(np.sqrt(np.nansum(sd_rows**2)))

    cv_total = sd_total / mean_total
    lower = mean_total * np.exp(-_QUAD_Z95 * cv_total)
    upper = mean_total * np.exp(_QUAD_Z95 * cv_total)
    return mean_total, float(lower), float(upper)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 27.5))
    if nrows * ncols == 1:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        ax.set_axisbelow(True)
        return fig, ax
    for temp in ax.flatten():  # type: ignore
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        temp.set_axisbelow(True)
    return fig, ax


def plot_hiv_burden_by_sex(hiv, ax, fig):
    # Aggregate UNAIDS incidence; combine Male + MSM into a single Male total
    hiv_agg = (
        hiv.group_by(["year", "sex"])
        .agg(
            pl.sum("unaids_incidence_number").alias("incidence"),
            pl.sum("unaids_incidence_number_upper").alias("incidence_upper"),
            pl.sum("unaids_incidence_number_lower").alias("incidence_lower"),
        )
        .sort("year")
    )

    male_hist = (
        hiv_agg.filter(pl.col("sex").is_in(["Male", "MSM"]))
        .group_by("year")
        .agg(
            pl.sum("incidence"),
            pl.sum("incidence_upper"),
            pl.sum("incidence_lower"),
        )
        .sort("year")
    )
    female_hist = hiv_agg.filter(pl.col("sex") == "Female").sort("year")

    ax.plot(
        male_hist["year"],
        male_hist["incidence"],
        linewidth=5,
        label="Men",
        color="midnightblue",
    )
    ax.fill_between(
        male_hist["year"],
        male_hist["incidence_lower"],
        male_hist["incidence_upper"],
        alpha=0.3,
        color="midnightblue",
    )

    ax.plot(
        female_hist["year"],
        female_hist["incidence"],
        linewidth=5,
        label="Women",
        color="magenta",
    )
    ax.fill_between(
        female_hist["year"],
        female_hist["incidence_lower"],
        female_hist["incidence_upper"],
        alpha=0.3,
        color="magenta",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence in\nsub-Saharan Africa (N)")
    ax.legend(loc=(0.5, 0.73), edgecolor="black")
    ax.set_xticks([2000, 2010, 2020])
    ax.set_yticks([0, 300000, 600000, 900000, 1200000])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    # y axis minor ticks every 100,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100000))
    # x axis ticks every 2 years
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_hiv_burden_by_region(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum("unaids_incidence_number"),
            pl.sum("unaids_incidence_number_upper"),
            pl.sum("unaids_incidence_number_lower"),
        )
        .rename(
            {
                "unaids_incidence_number": "incidence",
                "unaids_incidence_number_upper": "incidence_upper",
                "unaids_incidence_number_lower": "incidence_lower",
            }
        )
    )
    hiv = hiv.sort(by="year")

    region_colors = {
        "Western": "darkorange",
        "Eastern": "olivedrab",
        "Central": "teal",
        "Southern": "maroon",
    }

    for region, color in region_colors.items():
        hist = hiv.filter(pl.col("region") == region)
        ax.plot(
            hist["year"],
            hist["incidence"],
            linewidth=5,
            label=region,
            color=color,
        )
        ax.fill_between(
            hist["year"],
            hist["incidence_lower"],
            hist["incidence_upper"],
            alpha=0.3,
            color=color,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence (N)")
    ax.legend(loc=(0.5, 0.7), edgecolor="black")
    ax.set_xticks([2000, 2010, 2020])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    ax.set_ylim(0)
    # minor y ticks every 50,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50000))
    # x axis ticks every 2 years
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_rr_estimates(aggregated_rrs, ax, fig):
    # error bars showing the RR estimates from each study by sex
    # combine the two dataframes into one
    aggregated_rrs = aggregated_rrs.rename(
        {"rr_mu": "val", "rr_lower": "lower", "rr_upper": "upper"}
    )

    female_data = aggregated_rrs.filter(pl.col("sex") == "Heterosexual Women")
    heterosexual_male_data = aggregated_rrs.filter(
        pl.col("sex") == "Heterosexual Men"
    )
    msm_data = aggregated_rrs.filter(pl.col("sex") == "MSM")
    # we have to get all the RRs in the same order
    female_data = female_data.sort("bacteria")
    heterosexual_male_data = heterosexual_male_data.sort("bacteria")
    msm_data = msm_data.sort("bacteria")
    # for heterosexual male and msm, filter out trichomoniasis since the RRs
    # are inferred/1.0
    heterosexual_male_data = heterosexual_male_data.filter(
        pl.col("bacteria") != "Trichomoniasis"
    )
    msm_data = msm_data.filter(pl.col("bacteria") != "Trichomoniasis")

    male_xs = np.arange(0, heterosexual_male_data.shape[0])
    female_xs = np.arange(0, female_data.shape[0])

    offset = 0.2
    ax.errorbar(
        male_xs - offset,
        heterosexual_male_data["val"],
        yerr=[
            heterosexual_male_data["val"].to_numpy()
            - heterosexual_male_data["lower"].to_numpy(),
            heterosexual_male_data["upper"].to_numpy()
            - heterosexual_male_data["val"].to_numpy(),
        ],
        fmt="o",
        label="Heterosexual men",
        color="midnightblue",
        elinewidth=1,
        capsize=10,
        capthick=2,
        markersize=8,
    )

    ax.errorbar(
        female_xs,
        female_data["val"],
        yerr=[
            female_data["val"].to_numpy() - female_data["lower"].to_numpy(),
            female_data["upper"].to_numpy() - female_data["val"].to_numpy(),
        ],
        fmt="o",
        label="Heterosexual women",
        color="magenta",
        elinewidth=1,
        capsize=10,
        capthick=2,
        markersize=8,
    )

    ax.errorbar(
        male_xs + offset,
        msm_data["val"],
        yerr=[
            msm_data["val"].to_numpy() - msm_data["lower"].to_numpy(),
            msm_data["upper"].to_numpy() - msm_data["val"].to_numpy(),
        ],
        fmt="o",
        label="MSM",
        color="slategrey",
        elinewidth=1,
        capsize=10,
        capthick=2,
        markersize=8,
    )

    ax.set_ylim(0.1)
    ax.legend(loc=(0.3, 0.05), edgecolor="black")
    ax.set_xlabel("STI")
    ax.set_xticks(
        female_xs,
        female_data["bacteria"].unique(maintain_order=True),
        rotation=-30,
        ha="center",
    )

    ax.set_yscale("log")
    ax.set_yticks([1, 10])
    ax.set_yticklabels(["1", "10"])
    ax.axhline(1, color="grey", linestyle="--")
    ax.set_xlabel("STI")
    ax.set_ylabel("Causal RR(HIV acquisition | untreated STI)")


def plot_attributable_hiv_burden_sex(ufloat_rows, ax, fig, number):
    """New panel d: trajectory of STBI-attributable HIV incidence by sex.
    Sums the raw UFloat objects across STIs and across countries within
    each year-sex group before extracting any interval."""

    # merge MSM into Male, same grouping as before, but sum the underlying
    # UFloat objects (not pre-extracted bounds) across STIs and across
    # countries within each year-sex group before extracting any interval.
    def merged_sex(sex):
        return "Male" if sex in ("Male", "MSM") else sex

    groups = defaultdict(list)
    for r in ufloat_rows:
        groups[(r["year"], merged_sex(r["sex"]))].append(r)

    series = {
        "Male": {"years": [], "mean": [], "lower": [], "upper": []},
        "Female": {"years": [], "mean": [], "lower": [], "upper": []},
    }

    for (year, sex), rows in groups.items():
        attributable = _sum_attributable_across_stis(
            rows, "unaids_hiv_incidence_number_attributable_to_"
        )
        if number:
            mean, lower, upper = ci_via_log(attributable)
        else:
            un_pop = _sum_field(rows, "un_pop")
            rate = 100000 * attributable / un_pop  # type: ignore
            mean, lower, upper = ci_via_log(rate)
        series[sex]["years"].append(year)
        series[sex]["mean"].append(mean)
        series[sex]["lower"].append(lower)
        series[sex]["upper"].append(upper)

    colors = {"Male": "midnightblue", "Female": "magenta"}
    for sex in ["Male", "Female"]:
        order = np.argsort(series[sex]["years"])
        years = np.array(series[sex]["years"])[order]
        mean = np.array(series[sex]["mean"])[order]
        lower = np.array(series[sex]["lower"])[order]
        upper = np.array(series[sex]["upper"])[order]

        ax.plot(
            years,
            mean,
            linewidth=5,
            label={"Male": "Men", "Female": "Women"}[sex],
            color=colors[sex],
        )
        ax.fill_between(years, lower, upper, alpha=0.3, color=colors[sex])

    ax.set_xlabel("Year")
    if number:
        ax.set_ylabel("STBI-attributable HIV incidence (N)")
    else:
        ax.set_ylabel(
            "STBI-attributable HIV incidence rate (per 100,000 population)"
        )
    ax.legend(loc="upper right", edgecolor="black")
    ax.set_ylim(0)
    if number:
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
        )
        # ticks every 25,000
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(25000))
    else:
        # ticks every 25
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.set_xticks([2000, 2010, 2020])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

    # cumulative STBI-attributable cases (raw UFloats summed across all years
    # within each merged-sex group, then extracted once)
    male_rows = [
        r for (yr, sx), rs in groups.items() if sx == "Male" for r in rs
    ]
    female_rows = [
        r for (yr, sx), rs in groups.items() if sx == "Female" for r in rs
    ]
    cum_male = _sum_attributable_across_stis(
        male_rows, "unaids_hiv_incidence_number_attributable_to_"
    )
    cum_female = _sum_attributable_across_stis(
        female_rows, "unaids_hiv_incidence_number_attributable_to_"
    )
    cum_total = cum_male + cum_female  # type: ignore
    print(
        "Cumulative STBI-attributable HIV cases, total:",
        _fmt_count(ci_via_log(cum_total)),
    )
    print(
        "Cumulative STBI-attributable HIV cases, Male (incl. MSM):",
        _fmt_count(ci_via_log(cum_male)),
    )
    print(
        "Cumulative STBI-attributable HIV cases, Female:",
        _fmt_count(ci_via_log(cum_female)),
    )

    # cumulative STBI-attributable cases as a percentage of cumulative total HIV
    # incidence over 2000-2023, per stratum. attributable = incidence * PAF, so
    # the ratio shares the incidence leaves and their uncertainty cancels; form
    # it as a UFloat ratio and extract on the (0,1) proportion via ci_via_logit.
    inc_male = _sum_field(male_rows, "unaids_incidence_number")
    inc_female = _sum_field(female_rows, "unaids_incidence_number")
    inc_total = inc_male + inc_female  # type: ignore
    print(
        "Cumulative STBI-attributable HIV, % of total incidence, total:",
        _fmt_pct(ci_via_logit(cum_total / inc_total)),  # type: ignore
    )
    print(
        "Cumulative STBI-attributable HIV, % of total incidence, Male (incl. MSM):",
        _fmt_pct(ci_via_logit(cum_male / inc_male)),  # type: ignore
    )
    print(
        "Cumulative STBI-attributable HIV, % of total incidence, Female:",
        _fmt_pct(ci_via_logit(cum_female / inc_female)),  # type: ignore
    )


def plot_attributable_hiv_burden_region(ufloat_rows, ax, fig):
    """New panel e: trajectory of STBI-attributable HIV incidence by
    region. Sums the raw UFloat objects across STIs and across countries
    within each year-region group before extracting any interval."""
    region_colors = {
        "Western": "darkorange",
        "Eastern": "olivedrab",
        "Central": "teal",
        "Southern": "maroon",
    }

    groups = defaultdict(list)
    for r in ufloat_rows:
        groups[(r["year"], r["region"])].append(r)

    series = {
        region: {"years": [], "mean": [], "lower": [], "upper": []}
        for region in region_colors
    }
    for (year, region), rows in groups.items():
        attributable = _sum_attributable_across_stis(
            rows, "unaids_hiv_incidence_number_attributable_to_"
        )
        mean, lower, upper = ci_via_log(attributable)
        series[region]["years"].append(year)
        series[region]["mean"].append(mean)
        series[region]["lower"].append(lower)
        series[region]["upper"].append(upper)

    for region, color in region_colors.items():
        order = np.argsort(series[region]["years"])
        years = np.array(series[region]["years"])[order]
        mean = np.array(series[region]["mean"])[order]
        lower = np.array(series[region]["lower"])[order]
        upper = np.array(series[region]["upper"])[order]

        ax.plot(years, mean, linewidth=5, label=region, color=color)
        ax.fill_between(years, lower, upper, alpha=0.3, color=color)

    ax.set_xlabel("Year")
    ax.set_ylabel("STBI-attributable HIV incidence (N)")
    ax.legend(loc=(0.5, 0.85), edgecolor="black")
    ax.set_xticks([2000, 2010, 2020])
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    # ticks every 10,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

    # cumulative STBI-attributable cases by region (raw UFloats summed across
    # all years within each region, then extracted once)
    for region in region_colors:
        region_rows = [
            r for (yr, rg), rs in groups.items() if rg == region for r in rs
        ]
        cum = _sum_attributable_across_stis(
            region_rows, "unaids_hiv_incidence_number_attributable_to_"
        )
        print(
            f"Cumulative STBI-attributable HIV cases, {region}:",
            _fmt_count(ci_via_log(cum)),
        )
        inc = _sum_field(region_rows, "unaids_incidence_number")
        print(
            f"Cumulative STBI-attributable HIV, % of total incidence, {region}:",
            _fmt_pct(ci_via_logit(cum / inc)),  # type: ignore
        )


def plot_attributable_hiv_burden_sti(ufloat_rows, ax, fig):
    groups = defaultdict(list)
    for r in ufloat_rows:
        groups[r["year"]].append(r)

    years = sorted(groups)
    series = {sti: {"mean": [], "lower": [], "upper": []} for sti in STIS}
    for year in years:
        rows = groups[year]
        for sti in STIS:
            total = _sum_field(
                rows, f"unaids_hiv_incidence_number_attributable_to_{sti}"
            )
            mean, lower, upper = ci_via_log(total)
            series[sti]["mean"].append(mean)
            series[sti]["lower"].append(lower)
            series[sti]["upper"].append(upper)

    years_arr = np.array(years)

    ax.plot(
        years_arr,
        series["chlamydia"]["mean"],
        linewidth=5,
        label="Chlamydia",
        color="forestgreen",
    )

    ax.fill_between(
        years_arr,
        series["chlamydia"]["lower"],
        series["chlamydia"]["upper"],
        alpha=0.3,
        color="forestgreen",
    )

    ax.plot(
        years_arr,
        series["gc"]["mean"],
        linewidth=5,
        label="Gonorrhea",
        color="firebrick",
    )

    ax.fill_between(
        years_arr,
        series["gc"]["lower"],
        series["gc"]["upper"],
        alpha=0.3,
        color="firebrick",
    )

    ax.plot(
        years_arr,
        series["syphilis"]["mean"],
        linewidth=5,
        label="Syphilis",
        color="purple",
    )

    ax.fill_between(
        years_arr,
        series["syphilis"]["lower"],
        series["syphilis"]["upper"],
        alpha=0.3,
        color="purple",
    )

    ax.plot(
        years_arr,
        series["trichomoniasis"]["mean"],
        linewidth=5,
        label="Trichomoniasis",
        color="gold",
    )

    ax.fill_between(
        years_arr,
        series["trichomoniasis"]["lower"],
        series["trichomoniasis"]["upper"],
        alpha=0.3,
        color="gold",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("STBI-attributable HIV incidence (N)")
    ax.legend(loc=(0.4, 0.95), edgecolor="black")
    ax.set_ylim(0)
    ax.set_xticks([2000, 2010, 2020])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    # ticks every 10,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

    # cumulative STBI-attributable cases by pathogen (raw UFloats summed across
    # all years and all rows, then extracted once), plus each as a percentage
    # of cumulative total HIV incidence over the same window. attributable =
    # incidence * PAF, so the ratio shares the incidence leaves; form as a
    # UFloat ratio and extract the (0,1) proportion via ci_via_logit.
    all_rows = [r for rs in groups.values() for r in rs]
    inc_all = _sum_field(all_rows, "unaids_incidence_number")
    for sti in STIS:
        cum = _sum_field(
            all_rows, f"unaids_hiv_incidence_number_attributable_to_{sti}"
        )
        print(
            f"Cumulative STBI-attributable HIV cases, {sti}:",
            _fmt_count(ci_via_log(cum)),
        )
        print(
            f"Cumulative STBI-attributable HIV, % of total incidence, {sti}:",
            _fmt_pct(ci_via_logit(cum / inc_all)),  # type: ignore
        )


def ax_formatting(ax):
    ax.set_xticks([2000, 2010, 2020])
    # minor x ticks every two years
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_best_case(ufloat_rows, ax, year, fig):
    avert_field = "hiv_averted_upper_bound"
    rows = [r for r in ufloat_rows if r["year"] <= 2024]
    rows = [r for r in rows if r["year"] >= year]

    # Per-row averted UFloats (the "change in cases"): total effect from
    # `avert_field` and direct-only from `direct_hiv_averted_upper_bound`.
    # Each row's log interval is extracted from the positive averted UFloat and
    # tagged with its (country_code, sex) independence block; _block_combine
    # does quadrature across blocks and sums across years within a block.
    years_present = sorted({r["year"] for r in rows})

    def bucket():
        return {"m": [], "lo": [], "hi": [], "blk": []}

    per_year = {
        nm: {y: bucket() for y in years_present} for nm in ("total", "direct")
    }
    across = {nm: bucket() for nm in ("ref", "averted", "direct_averted")}

    for r in rows:
        y = r["year"]
        blk = (r["country_code"], r["sex"])
        ref_uf = r["unaids_incidence_number"]
        tot_av = r[avert_field]
        dir_av = r["direct_hiv_averted_upper_bound"]

        # per-year averted quantities (change in cases). ci_via_log needs
        # positive counts, so extract from the positive averted UFloats.
        for nm, uf in (("total", tot_av), ("direct", dir_av)):
            m, lo, hi = ci_via_log(uf)
            b = per_year[nm][y]
            b["m"].append(m)
            b["lo"].append(lo)
            b["hi"].append(hi)
            b["blk"].append(blk)

        # averted quantities themselves, for the across-year printouts
        for nm, uf in (("averted", tot_av), ("direct_averted", dir_av)):
            m, lo, hi = ci_via_log(uf)
            a = across[nm]
            a["m"].append(m)
            a["lo"].append(lo)
            a["hi"].append(hi)
            a["blk"].append(blk)
        a = across["ref"]
        m, lo, hi = ci_via_log(ref_uf)
        a["m"].append(m)
        a["lo"].append(lo)
        a["hi"].append(hi)
        a["blk"].append(blk)

    def series(nm):
        ys, ms, los, his = [], [], [], []
        for y in years_present:
            b = per_year[nm][y]
            if not b["m"]:
                continue
            mm, ll, hh = _block_combine(b["m"], b["lo"], b["hi"], b["blk"])
            ys.append(y)
            ms.append(mm)
            los.append(ll)
            his.append(hh)
        return np.array(ys), np.array(ms), np.array(los), np.array(his)

    tot_s = series("total")
    dir_s = series("direct")

    # across-year totals (sum within (country,sex), quadrature across)
    tot_av_ci = _block_combine(
        across["averted"]["m"],
        across["averted"]["lo"],
        across["averted"]["hi"],
        across["averted"]["blk"],
    )
    dir_av_ci = _block_combine(
        across["direct_averted"]["m"],
        across["direct_averted"]["lo"],
        across["direct_averted"]["hi"],
        across["direct_averted"]["blk"],
    )
    ref_ci = _block_combine(
        across["ref"]["m"],
        across["ref"]["lo"],
        across["ref"]["hi"],
        across["ref"]["blk"],
    )
    print("Total HIV averted:", _fmt_count(tot_av_ci))
    print("Direct only HIV averted:", _fmt_count(dir_av_ci))
    print("Reference incidence total:", _fmt_count(ref_ci))
    if ref_ci[0] and ref_ci[0] > 0:
        print(
            "Total HIV averted as % of reference:",
            _fmt_pct(
                (
                    tot_av_ci[0] / ref_ci[0],
                    tot_av_ci[1] / ref_ci[0],
                    tot_av_ci[2] / ref_ci[0],
                )
            ),
        )

    yrs, m, lo, hi = dir_s
    ax.plot(yrs, m, linewidth=5, label="Direct", color="grey")
    ax.fill_between(yrs, lo, hi, alpha=0.3, color="grey")
    yrs, m, lo, hi = tot_s
    ax.plot(yrs, m, linewidth=5, label="Total", color="black")
    ax.fill_between(yrs, lo, hi, alpha=0.3, color="black")

    ax.set_xlabel("Year")
    ax.set_ylabel("Change in HIV incidence (N)")
    ax.legend(loc=(0.6, 0.7), edgecolor="black")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax_formatting(ax)
    ax.set_ylim(0)


def plot_averted_sti(hiv, ax, year, fig):
    stub = "upper_bound"
    hiv = hiv.filter(pl.col("year") <= 2023)

    hiv = hiv.filter(pl.col("year") >= year)
    raw = hiv  # keep the un-aggregated rows for the per-pathogen prints

    # Combine per-row intervals across locations/sexes within each year by
    # QUADRATURE (independent rows), per STI, rather than summing the lower/
    # upper columns (which assumes perfect correlation). The lower band carries
    # a positive floor from _quadrature_sum.
    sti_cols = {
        "gc": "gc",
        "chlamydia": "chlamydia",
        "syphilis": "syphilis",
        "trichomoniasis": "trichomoniasis",
    }
    years_sorted = sorted(hiv["year"].unique().to_list())
    agg = {"year": []}
    for c in sti_cols.values():
        agg[c] = []
        agg[f"{c}_lower"] = []
        agg[f"{c}_upper"] = []
    for y in years_sorted:
        rows = hiv.filter(pl.col("year") == y)
        agg["year"].append(y)
        for sti, c in sti_cols.items():
            m, lo, hi = _quadrature_sum(
                rows[f"hiv_averted_{sti}_{stub}"].to_numpy(),
                rows[f"hiv_averted_{sti}_{stub}_lower"].to_numpy(),
                rows[f"hiv_averted_{sti}_{stub}_upper"].to_numpy(),
            )
            agg[c].append(m)
            agg[f"{c}_lower"].append(lo)
            agg[f"{c}_upper"].append(hi)
    hiv = pl.DataFrame(agg).sort(by="year")

    # ---- diagnostic prints: per-pathogen averted totals, quadrature-combined
    # across ALL that pathogen's rows (independent), matching how the per-sex
    # prints in plot_averted_sex combine. Cases only (no per-pathogen
    # proportion, since there is no pathogen-specific HIV incidence
    # denominator). ----
    blocks = list(zip(raw["country_code"].to_list(), raw["sex"].to_list()))
    for sti in sti_cols:
        m, lo, hi = _block_combine(
            raw[f"hiv_averted_{sti}_{stub}"].to_numpy(),
            raw[f"hiv_averted_{sti}_{stub}_lower"].to_numpy(),
            raw[f"hiv_averted_{sti}_{stub}_upper"].to_numpy(),
            blocks,
        )
        print(f"HIV numbers averted {sti}:", _fmt_count((m, lo, hi)))

    ax.plot(
        hiv["year"],
        hiv["chlamydia"],
        linewidth=5,
        label="Chlamydia",
        color="forestgreen",
    )

    ax.fill_between(
        hiv["year"],
        hiv["chlamydia_lower"],
        hiv["chlamydia_upper"],
        alpha=0.3,
        color="forestgreen",
    )

    ax.plot(
        hiv["year"],
        hiv["gc"],
        linewidth=5,
        label="Gonorrhea",
        color="firebrick",
    )

    ax.fill_between(
        hiv["year"],
        hiv["gc_lower"],
        hiv["gc_upper"],
        alpha=0.3,
        color="firebrick",
    )

    ax.plot(
        hiv["year"],
        hiv["syphilis"],
        linewidth=5,
        label="Syphilis",
        color="purple",
    )

    ax.fill_between(
        hiv["year"],
        hiv["syphilis_lower"],
        hiv["syphilis_upper"],
        alpha=0.3,
        color="purple",
    )

    ax.plot(
        hiv["year"],
        hiv["trichomoniasis"],
        linewidth=5,
        label="Trichomoniasis",
        color="gold",
    )

    ax.fill_between(
        hiv["year"],
        hiv["trichomoniasis_lower"],
        hiv["trichomoniasis_upper"],
        alpha=0.3,
        color="gold",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Change in HIV incidence (N)")
    ax.legend(loc=(0.4, 0.95), edgecolor="black")
    # make the numbers appear in non scientific notation
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax_formatting(ax)
    ax.set_ylim(0)


def plot_averted_sex(hiv, ax, year, fig):
    stub = "upper_bound"
    hiv = hiv.filter(pl.col("year") <= 2023)

    hiv = hiv.filter(pl.col("year") >= year)
    raw = hiv  # keep the un-aggregated rows for the across-years prints

    # Per (year, sex) quadrature combine across locations (independent rows),
    # rather than summing lower/upper columns. Produces total/total_lower/
    # total_upper; lower band positive-floored.
    plot_sexes = ["Male", "Female", "MSM"]
    years_sorted = sorted(hiv["year"].unique().to_list())
    agg = {
        "year": [],
        "sex": [],
        "total": [],
        "total_lower": [],
        "total_upper": [],
    }
    for s in plot_sexes:
        for y in years_sorted:
            rows = hiv.filter((pl.col("year") == y) & (pl.col("sex") == s))
            if rows.height == 0:
                continue
            m, lo, hi = _quadrature_sum(
                rows[f"hiv_averted_{stub}"].to_numpy(),
                rows[f"hiv_averted_{stub}_lower"].to_numpy(),
                rows[f"hiv_averted_{stub}_upper"].to_numpy(),
            )
            agg["year"].append(y)
            agg["sex"].append(s)
            agg["total"].append(m)
            agg["total_lower"].append(lo)
            agg["total_upper"].append(hi)
    hiv = pl.DataFrame(agg)

    # ---- diagnostic prints: per-sex across-years totals and proportions,
    # quadrature-combined across ALL that sex's rows (independent), not summed
    # over pre-extracted bounds ----
    for s in plot_sexes:
        srows = raw.filter(pl.col("sex") == s)
        if srows.height == 0:
            print(f"HIV numbers averted {s}: (no rows)")
            continue
        m, lo, hi = _block_combine(
            srows[f"hiv_averted_{stub}"].to_numpy(),
            srows[f"hiv_averted_{stub}_lower"].to_numpy(),
            srows[f"hiv_averted_{stub}_upper"].to_numpy(),
            srows[
                "country_code"
            ].to_list(),  # block = country (sex fixed here)
        )
        print(f"HIV numbers averted {s}:", _fmt_count((m, lo, hi)))
        inc = float(srows["unaids_incidence_number"].sum())
        if inc > 0:
            print(
                f"HIV proportion averted {s}:",
                _fmt_pct((m / inc, lo / inc, hi / inc)),
            )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv.filter(pl.col("sex") == "Male")["year"],
        hiv.filter(pl.col("sex") == "Male")["total"],
        linewidth=5,
        label="Heterosexual men",
        color="midnightblue",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Male")["year"],
        hiv.filter(pl.col("sex") == "Male")["total_lower"],
        hiv.filter(pl.col("sex") == "Male")["total_upper"],
        alpha=0.3,
        color="midnightblue",
    )

    ax.plot(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["total"],
        linewidth=5,
        label="Heterosexual women",
        color="magenta",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["total_lower"],
        hiv.filter(pl.col("sex") == "Female")["total_upper"],
        alpha=0.3,
        color="magenta",
    )

    ax.plot(
        hiv.filter(pl.col("sex") == "MSM")["year"],
        hiv.filter(pl.col("sex") == "MSM")["total"],
        linewidth=5,
        label="MSM",
        color="slategrey",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "MSM")["year"],
        hiv.filter(pl.col("sex") == "MSM")["total_lower"],
        hiv.filter(pl.col("sex") == "MSM")["total_upper"],
        alpha=0.3,
        color="slategrey",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Change in HIV incidence (N)")
    ax.legend(loc=(0.35, 0.95), edgecolor="black")
    # make the numbers appear in non scientific notation
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax_formatting(ax)
    ax.set_ylim(0)


if __name__ == "__main__":
    fig, ax = setup_plot(3, 3)  # type: ignore

    # Panels a-c: unchanged, straight from make_figure.py
    hiv = pl.read_csv(os.path.join(output_dir, "hiv_sti.csv"))
    hiv = hiv.filter(pl.col("year") >= 2000)

    ax[0, 0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_hiv_burden_by_sex(hiv, ax[0, 0], fig)  # type: ignore

    ax[0, 1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_hiv_burden_by_region(hiv, ax[0, 1], fig)  # type: ignore

    aggregated_rrs = pl.read_csv(
        os.path.join(output_dir, "meta_estimated_RRs_causal_STI_given_HIV.csv")
    )
    ax[0, 2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[0, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_rr_estimates(aggregated_rrs, ax[0, 2], fig)  # type: ignore

    # Panels d-f: trajectory of STBI-attributable HIV incidence over time,
    # from the attributable-HIV UFloat pickle.
    with open(
        os.path.join(output_dir, "hiv_attributable_to_stis.ufloat.pkl"), "rb"
    ) as f:
        hiv_attributable_ufloat = pickle.load(f)
    hiv_attributable_ufloat = [
        r for r in hiv_attributable_ufloat if 2000 <= r["year"] <= 2023
    ]

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.1,
        "d",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_sex(
        hiv_attributable_ufloat,
        ax[1, 0],  # type: ignore
        fig,
        number=True,
    )

    ax[1, 1].text(  # type: ignore
        -0.25,
        1.1,
        "e",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_region(
        hiv_attributable_ufloat,
        ax[1, 1],  # type: ignore
        fig,
    )

    ax[1, 2].text(  # type: ignore
        -0.25,
        1.1,
        "f",
        transform=ax[1, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_sti(
        hiv_attributable_ufloat,
        ax[1, 2],  # type: ignore
        fig,
    )

    # Panels g-i (previously d-f): no-antibiotic-access counterfactual
    # trajectory, read straight from hiv_averted.csv as in the original
    # figure script.
    hiv_averted = pl.read_csv(os.path.join(output_dir, "hiv_averted.csv"))

    first_year_upper_bound = 2000

    with open(os.path.join(output_dir, "hiv_averted.ufloat.pkl"), "rb") as f:
        hiv_averted_ufloat = pickle.load(f)

    ax[2, 0].text(  # type: ignore
        -0.25,
        1.08,
        "g",
        transform=ax[2, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_best_case(
        hiv_averted_ufloat,
        ax[2, 0],  # type: ignore
        first_year_upper_bound,
        fig,
    )

    ax[2, 1].text(  # type: ignore
        -0.25,
        1.08,
        "h",
        transform=ax[2, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_sex(
        hiv_averted,
        ax[2, 1],  # type: ignore
        first_year_upper_bound,
        fig,
    )

    ax[2, 2].text(  # type: ignore
        -0.25,
        1.08,
        "i",
        transform=ax[2, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_sti(
        hiv_averted,
        ax[2, 2],  # type: ignore
        first_year_upper_bound,
        fig,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(fig_dir, "figure_historic_attributable_averted_hiv.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_historic_attributable_averted_hiv.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
