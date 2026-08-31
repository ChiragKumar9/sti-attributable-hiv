import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
from matplotlib import rc

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

STIS = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
SEXES = ["Male", "Female", "MSM"]

# analysis window shared by panels b, c and d. The untreated/resistance
# scenarios now run with a 2007 warm-up year, so the indirect sim emits from
# 2008 and the decomposition (direct + indirect) is fully populated from 2008
# onward.
FIRST_YEAR = 2008
LAST_YEAR = 2023

_Z95 = 1.959963984540054  # 97.5th percentile of the standard normal
_QUAD_LOWER_FLOOR = 1.0


def _row_sd(lowers, uppers):
    """Per-row linear-space sd recovered from a 95% CI: (upper - lower)/(2*Z)."""
    return (np.asarray(uppers, float) - np.asarray(lowers, float)) / (2 * _Z95)


def _share_of_total(U, R, var_U, var_R, cov_UR):
    """Share of the summed access+resistance decomposition attributable to one
    lever: U / (U + R), with a delta-method CI. var_U / var_R are the linear-
    space variances of the two summed pathways; cov_UR uses the within-row rho=1
    assumption (sum of sd_u_i * sd_r_i), so the shared burden mostly cancels in
    the ratio. Returns (mean, lower, upper) clipped to [0, 1], or None if the
    total is non-positive. Bounded <=1 by construction."""
    tot = U + R
    if tot <= 0:
        return None
    f = U / tot
    df_dU = R / tot**2
    df_dR = -U / tot**2
    var_f = df_dU**2 * var_U + df_dR**2 * var_R + 2 * df_dU * df_dR * cov_UR
    sd_f = np.sqrt(max(var_f, 0.0))
    return f, max(f - _Z95 * sd_f, 0.0), min(f + _Z95 * sd_f, 1.0)


def _quadrature_sum(means, lowers, uppers):
    """Combine per-row CSV intervals by treating rows as INDEPENDENT and adding
    in quadrature. Fast closed-form arithmetic on the pre-extracted bounds --
    no `uncertainties` graph. Returns (mean_total, lower, upper); lower floored
    positive for symlog."""
    means = np.asarray(means, dtype=float)
    mean_total = float(np.nansum(means))
    sd_total = float(np.sqrt(np.nansum(_row_sd(lowers, uppers) ** 2)))
    lower = max(mean_total - _Z95 * sd_total, _QUAD_LOWER_FLOOR)
    upper = mean_total + _Z95 * sd_total
    return mean_total, lower, upper


def _style_axis(ax):
    """Apply the shared spine/tick styling to a single axis: hide the top and
    right spines, keep ticks on the bottom/left pointing outward, offset the
    spines, and put the grid behind. Applied to every panel (the 2x2 block and
    the full-width panel e) by the GridSpec layout below."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    ax.tick_params(which="major", length=5.25, width=1.2)
    ax.tick_params(which="minor", length=3.0, width=0.9)
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
# Panel a: estimated gonorrhoea resistance RATES across all of Africa, by
# antibiotic. Reads the Beta-derived intervals straight from
# estimated_resistance_rates.csv and averages across locations per year.
# ----------------------------------------------------------------------
def plot_drug_resistance(estimated_resistance_rates, ax, fig):
    estimated_resistance_rates = estimated_resistance_rates.filter(
        pl.col("year") <= LAST_YEAR
    )
    estimated_resistance_rates = estimated_resistance_rates.group_by(
        ["year"]
    ).agg(
        pl.mean("Ciprofloxacin"),
        pl.mean("Cefixime"),
        pl.mean("Azithromycin"),
        pl.mean("Ceftriaxone"),
        pl.mean("Ciprofloxacin_lower"),
        pl.mean("Cefixime_lower"),
        pl.mean("Azithromycin_lower"),
        pl.mean("Ceftriaxone_lower"),
        pl.mean("Ciprofloxacin_upper"),
        pl.mean("Cefixime_upper"),
        pl.mean("Azithromycin_upper"),
        pl.mean("Ceftriaxone_upper"),
    )
    estimated_resistance_rates = estimated_resistance_rates.sort("year")

    drug_colors = {
        "Ciprofloxacin": "red",
        "Cefixime": "blue",
        "Azithromycin": "green",
        "Ceftriaxone": "purple",
    }
    for drug, color in drug_colors.items():
        ax.plot(
            estimated_resistance_rates["year"],
            estimated_resistance_rates[drug] * 100,
            label=drug,
            linewidth=3,
            color=color,
        )
        ax.fill_between(
            estimated_resistance_rates["year"],
            estimated_resistance_rates[f"{drug}_lower"] * 100,
            estimated_resistance_rates[f"{drug}_upper"] * 100,
            color=color,
            alpha=0.3,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Gonorrhea isolates resistant (%)")
    ax.legend(loc=(0.55, 0.23), edgecolor="black")
    ax.set_xticks([2010, 2015, 2020])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(0)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))


# ----------------------------------------------------------------------
# Supplemental: same resistance rates as panel a, faceted by African
# region instead of averaged across all locations. `estimated_resistance_
# rates.csv`'s `location` column holds region rows (Western/Eastern/
# Central/Southern) alongside country rows -- filter to one region
# instead of grouping/averaging. Note: Central has no GASP country data
# and is synthesized by tethering to Southern (see
# 03_assemble_resistance_data.py), so its curve is extrapolated.
# ----------------------------------------------------------------------
def plot_drug_resistance_by_region(
    estimated_resistance_rates, ax, region, fig, legend=False
):
    df = estimated_resistance_rates.filter(
        (pl.col("location") == region) & (pl.col("year") <= LAST_YEAR)
    ).sort("year")

    drug_colors = {
        "Ciprofloxacin": "red",
        "Cefixime": "blue",
        "Azithromycin": "green",
        "Ceftriaxone": "purple",
    }
    for drug, color in drug_colors.items():
        ax.plot(
            df["year"],
            df[drug] * 100,
            label=drug,
            linewidth=3,
            color=color,
        )
        ax.fill_between(
            df["year"],
            df[f"{drug}_lower"] * 100,
            df[f"{drug}_upper"] * 100,
            color=color,
            alpha=0.3,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Gonorrhea isolates resistant\nin {region} Africa (%)")
    if legend:
        ax.legend(loc=(0.55, 0.23), edgecolor="black")
    ax.set_xticks([2010, 2015, 2020])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))


# ----------------------------------------------------------------------
# Panel b: access-vs-resistance decomposition, as COUNTS, aggregated globally
# per year. Reads the flat hiv_averted.csv and combines rows within each year
# by QUADRATURE (independent rows). Plotted as reductions (negatives) on a
# symlog axis.
# ----------------------------------------------------------------------
def plot_access_vs_resistance(hiv, ax, fig):
    hiv = hiv.filter(
        (pl.col("year") >= FIRST_YEAR) & (pl.col("year") <= LAST_YEAR)
    )
    years = sorted(hiv["year"].unique().to_list())

    series = {
        "untreated": {"mean": [], "lower": [], "upper": []},
        "resistance": {"mean": [], "lower": [], "upper": []},
    }
    for y in years:
        rows = hiv.filter(pl.col("year") == y)
        for key, col in [
            ("untreated", "hiv_averted_gc_untreated"),
            ("resistance", "hiv_averted_gc_resistance"),
        ]:
            m, lo, hi = _quadrature_sum(
                rows[col].to_numpy(),
                rows[f"{col}_lower"].to_numpy(),
                rows[f"{col}_upper"].to_numpy(),
            )
            series[key]["mean"].append(-m)
            series[key]["lower"].append(-lo)
            series[key]["upper"].append(-hi)

    years_arr = np.array(years)
    ax.plot(
        years_arr,
        series["untreated"]["mean"],
        linewidth=3,
        label="Universal access",
        color="darkgoldenrod",
    )
    ax.fill_between(
        years_arr,
        series["untreated"]["lower"],
        series["untreated"]["upper"],
        alpha=0.3,
        color="darkgoldenrod",
    )
    ax.plot(
        years_arr,
        series["resistance"]["mean"],
        linewidth=3,
        label="Susceptible to first treatment",
        color="brown",
    )
    ax.fill_between(
        years_arr,
        series["resistance"]["lower"],
        series["resistance"]["upper"],
        alpha=0.3,
        color="brown",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Change in HIV incidence (N)")
    ax.legend(loc=(0.03, 0.95), edgecolor="black")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5000))

    # ---- cumulative shares over the plotted window (2009-2023), pooled across
    # all locations and sexes, AS A FRACTION OF THE ACCESS+RESISTANCE AVERTIBLE
    # TOTAL. denominator is untreated + resistance, so the two shares are
    # complementary and sum to 1. Point estimates exact; CIs use within-row
    # rho=1 between untreated and resistance, independent rows. ----
    u = hiv["hiv_averted_gc_untreated"].to_numpy()
    r = hiv["hiv_averted_gc_resistance"].to_numpy()
    sd_u = _row_sd(
        hiv["hiv_averted_gc_untreated_lower"].to_numpy(),
        hiv["hiv_averted_gc_untreated_upper"].to_numpy(),
    )
    sd_r = _row_sd(
        hiv["hiv_averted_gc_resistance_lower"].to_numpy(),
        hiv["hiv_averted_gc_resistance_upper"].to_numpy(),
    )
    U = float(np.nansum(u))
    R = float(np.nansum(r))
    var_U = float(np.nansum(sd_u**2))
    var_R = float(np.nansum(sd_r**2))
    cov_UR = float(np.nansum(sd_u * sd_r))  # within-row rho=1

    acc_m, acc_lo, acc_hi = _share_of_total(U, R, var_U, var_R, cov_UR)  # type: ignore
    res_m, res_lo, res_hi = _share_of_total(R, U, var_R, var_U, cov_UR)  # type: ignore
    print(
        f"Cumulative {FIRST_YEAR}-{LAST_YEAR} gonorrhea-attributable HIV, "
        f"share avertible by universal access: "
        f"{100 * acc_m:.1f}% (95% CI {100 * acc_lo:.1f}-{100 * acc_hi:.1f}%)"
    )
    print(
        f"Cumulative {FIRST_YEAR}-{LAST_YEAR} gonorrhea-attributable HIV, "
        f"share avertible by susceptibility to first-line treatment: "
        f"{100 * res_m:.1f}% (95% CI {100 * res_lo:.1f}-{100 * res_hi:.1f}%)"
    )

    # cumulative averted CASES under each pathway (absolute counts). Independent
    # -rows quadrature on the summed counts, same assumption as the panel curves.
    U_sd = np.sqrt(var_U)
    R_sd = np.sqrt(var_R)
    print(
        f"Cumulative {FIRST_YEAR}-{LAST_YEAR} gonorrhea-attributable HIV averted by "
        f"universal access (cases): "
        f"{round(U):,} (95% CI {round(max(U - _Z95 * U_sd, 0)):,}-"
        f"{round(U + _Z95 * U_sd):,})"
    )
    print(
        f"Cumulative {FIRST_YEAR}-{LAST_YEAR} gonorrhea-attributable HIV averted by "
        f"susceptibility to first-line treatment (cases): "
        f"{round(R):,} (95% CI {round(max(R - _Z95 * R_sd, 0)):,}-"
        f"{round(R + _Z95 * R_sd):,})"
    )


# ----------------------------------------------------------------------
# Panel c: access SHARE by sex, over time. Reads the flat hiv_averted.csv --
# no pickle, no uncertainties graph, so it is instant.
#
# THE ONE SIMPLIFICATION: within each row, untreated and resistance are taken
# to be perfectly positively correlated (rho = 1). Both are slices of the same
# gc-attributable burden, so the burden's uncertainty is shared and mostly
# cancels in the ratio U/(U+R). Rows are treated as independent of each other,
# exactly as panel b does. Per (year, sex):
#     Var(U)   = sum sd_u_i^2
#     Var(R)   = sum sd_r_i^2
#     Cov(U,R) = sum sd_u_i * sd_r_i        (the rho=1 within-row term)
# and the share band follows from a one-line delta method on U/(U+R). No graph
# is ever built or traced. Bounded <=100% by construction.
# ----------------------------------------------------------------------
def plot_access_vs_resistance_sex(hiv, ax, fig):
    hiv = hiv.filter(
        (pl.col("year") >= FIRST_YEAR) & (pl.col("year") <= LAST_YEAR)
    )

    sex_colors = {
        "Female": "magenta",
        "Male": "midnightblue",
        "MSM": "slategrey",
    }
    labels = {
        "Female": "Heterosexual women",
        "Male": "Heterosexual men",
        "MSM": "MSM",
    }
    for sex, color in sex_colors.items():
        sub = hiv.filter(pl.col("sex") == sex).sort("year")
        years = sorted(sub["year"].unique().to_list())
        xs, mean, lower, upper = [], [], [], []
        for y in years:
            rows = sub.filter(pl.col("year") == y)
            u = rows["hiv_averted_gc_untreated"].to_numpy()
            r = rows["hiv_averted_gc_resistance"].to_numpy()
            sd_u = _row_sd(
                rows["hiv_averted_gc_untreated_lower"].to_numpy(),
                rows["hiv_averted_gc_untreated_upper"].to_numpy(),
            )
            sd_r = _row_sd(
                rows["hiv_averted_gc_resistance_lower"].to_numpy(),
                rows["hiv_averted_gc_resistance_upper"].to_numpy(),
            )

            U = float(np.nansum(u))
            R = float(np.nansum(r))
            var_U = float(np.nansum(sd_u**2))
            var_R = float(np.nansum(sd_r**2))
            cov_UR = float(np.nansum(sd_u * sd_r))  # within-row rho = 1

            res = _share_of_total(U, R, var_U, var_R, cov_UR)
            if res is None:
                continue
            f, lo, hi = res

            xs.append(y)
            mean.append(100 * f)
            lower.append(100 * lo)
            upper.append(100 * hi)

        ax.plot(xs, mean, linewidth=3, label=labels[sex], color=color)
        ax.fill_between(xs, lower, upper, alpha=0.3, color=color)

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Gonorrhea-attributable HIV\nincidence from lack of antibiotic access (%)"
    )
    ax.legend(edgecolor="black")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(30, 100)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))


# ----------------------------------------------------------------------
# Panel d: access SHARE by region, over time. Identical math to panel c
# (within-row rho = 1, independent rows, one-line delta method on U/(U+R)),
# just grouped by region instead of sex.
# ----------------------------------------------------------------------
def plot_access_vs_resistance_region(hiv, ax, fig):
    hiv = hiv.filter(
        (pl.col("year") >= FIRST_YEAR) & (pl.col("year") <= LAST_YEAR)
    )

    region_colors = {
        "Western": "darkorange",
        "Eastern": "olivedrab",
        "Central": "teal",
        "Southern": "maroon",
    }
    for region, color in region_colors.items():
        sub = hiv.filter(pl.col("region") == region).sort("year")
        years = sorted(sub["year"].unique().to_list())
        xs, mean, lower, upper = [], [], [], []
        for y in years:
            rows = sub.filter(pl.col("year") == y)
            u = rows["hiv_averted_gc_untreated"].to_numpy()
            r = rows["hiv_averted_gc_resistance"].to_numpy()
            sd_u = _row_sd(
                rows["hiv_averted_gc_untreated_lower"].to_numpy(),
                rows["hiv_averted_gc_untreated_upper"].to_numpy(),
            )
            sd_r = _row_sd(
                rows["hiv_averted_gc_resistance_lower"].to_numpy(),
                rows["hiv_averted_gc_resistance_upper"].to_numpy(),
            )

            U = float(np.nansum(u))
            R = float(np.nansum(r))
            var_U = float(np.nansum(sd_u**2))
            var_R = float(np.nansum(sd_r**2))
            cov_UR = float(np.nansum(sd_u * sd_r))  # within-row rho = 1

            res = _share_of_total(U, R, var_U, var_R, cov_UR)
            if res is None:
                continue
            f, lo, hi = res

            xs.append(y)
            mean.append(100 * f)
            lower.append(100 * lo)
            upper.append(100 * hi)

        ax.plot(xs, mean, linewidth=3, label=region, color=color)
        ax.fill_between(xs, lower, upper, alpha=0.3, color=color)

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Gonorrhea-attributable HIV\nincidence from lack of antibiotic access (%)"
    )
    ax.legend(edgecolor="black")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(30, 100)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))


# ----------------------------------------------------------------------
# Panel e: % CHANGE in gonorrhoea-attributable HIV under the no-2016-switch
# counterfactual, 2016-2023, by region and sex. Reads the same flat
# hiv_averted.csv as panels b/c/d (no pickle). Per (region, sex):
#   observed = unaids_hiv_incidence_number_attributable_to_gc
#   averted  = hiv_averted_2016_gc_change  (= counterfactual - observed)
#   % change = 100 * averted / observed
# summed across every country/year row in the 2016-2023 window.
#
# Uncertainty follows this file's discipline: rows are INDEPENDENT (quadrature
# across rows) and WITHIN a row the averted amount and the observed burden are
# taken as perfectly correlated (rho=1) -- the same within-row rho=1 assumption
# panels b/c/d use. The ratio CI is a one-line delta method on averted/observed
# (unbounded, unlike the [0,1]-clipped shares in panels c/d).
#
# Two bars per region -- Female and Male (Male includes MSM, merged as in the
# incidence figure's plot_hiv_burden_by_sex) -- coloured magenta / midnightblue
# from panel c; whiskers show the delta-method CI. Merging MSM into Male gives
# the Male stratum a non-negligible observed denominator (heterosexual men have
# ~zero GC-attributable HIV, which alone blows the ratio up).
# ----------------------------------------------------------------------
def plot_gc_2016_counterfactual_region(hiv, ax, fig):
    hiv = hiv.filter((pl.col("year") >= 2016) & (pl.col("year") <= 2023))
    regions = ["Western", "Eastern", "Central", "Southern"]

    OBS = "unaids_hiv_incidence_number_attributable_to_gc"
    AVERT = "hiv_averted_2016_gc_change"

    # Absolute count: how many MORE gonorrhea-attributable HIV cases there would
    # have been over 2016-2023 had the 2016 first-line change not happened.
    # AVERT is the burden averted BY the switch (counterfactual - observed), so
    # its sum over all rows is exactly that additional-cases figure. Sexes and
    # regions pooled; independent-rows quadrature, matching panel b's cumulative
    # case printouts. (This is direct + indirect; the direct-only column is
    # "direct_hiv_averted_2016_gc_change".)
    add_m, add_lo, add_hi = _quadrature_sum(
        hiv[AVERT].to_numpy(),
        hiv[f"{AVERT}_lower"].to_numpy(),
        hiv[f"{AVERT}_upper"].to_numpy(),
    )
    print(
        "Additional gonorrhea-attributable HIV cases 2016-2023 under "
        f"no-2016-switch counterfactual: {round(add_m):,} "
        f"(95% CI {round(add_lo):,}-{round(add_hi):,})"
    )

    # Male includes MSM (merged as in the incidence figure); Female stands
    # alone. Colours/labels from panel c.
    sex_groups = {
        "Female": ["Female"],
        "Male": ["Male", "MSM"],
    }
    group_colors = {
        "Female": "magenta",
        "Male": "midnightblue",
    }
    labels = {
        "Female": "Women",
        "Male": "Men",
    }

    def _pct_change(rows):
        """% change from observed to the no-2016-switch counterfactual =
        100 * averted / observed (counterfactual - observed = averted). Ratio
        delta method with within-row rho=1 (cov = sum sd_av_i * sd_obs_i) and
        independent rows. Unbounded; returns None if the observed total is
        non-positive."""
        A = float(np.nansum(rows[AVERT].to_numpy()))
        obs = float(np.nansum(rows[OBS].to_numpy()))
        if obs <= 0:
            return None
        sd_av = _row_sd(
            rows[f"{AVERT}_lower"].to_numpy(),
            rows[f"{AVERT}_upper"].to_numpy(),
        )
        sd_obs = _row_sd(
            rows[f"{OBS}_lower"].to_numpy(), rows[f"{OBS}_upper"].to_numpy()
        )
        var_A = float(np.nansum(sd_av**2))
        var_O = float(np.nansum(sd_obs**2))
        cov = float(np.nansum(sd_av * sd_obs))  # within-row rho = 1
        g = A / obs
        var_g = var_A / obs**2 + A**2 * var_O / obs**4 - 2 * A * cov / obs**3
        sd_g = np.sqrt(max(var_g, 0.0))
        return 100 * g, 100 * (g - _Z95 * sd_g), 100 * (g + _Z95 * sd_g)

    # region -> group -> (mean, lower, upper) percent
    pct = {region: {} for region in regions}
    for region in regions:
        sub = hiv.filter(pl.col("region") == region)
        for grp, members in sex_groups.items():
            rows = sub.filter(pl.col("sex").is_in(members))
            if rows.height == 0:
                continue
            res = _pct_change(rows)
            if res is not None:
                pct[region][grp] = res

    x = np.arange(len(regions))
    width = 0.35
    offsets = {"Female": -width / 2, "Male": width / 2}

    for grp, color in group_colors.items():
        mean = np.array(
            [
                pct[region].get(grp, (np.nan, np.nan, np.nan))[0]
                for region in regions
            ]
        )
        lower = np.array(
            [
                pct[region].get(grp, (np.nan, np.nan, np.nan))[1]
                for region in regions
            ]
        )
        upper = np.array(
            [
                pct[region].get(grp, (np.nan, np.nan, np.nan))[2]
                for region in regions
            ]
        )
        ax.bar(
            x + offsets[grp],
            mean,
            width=width,
            label=labels[grp],
            color=color,
            alpha=1.0 if color == "magenta" else 0.8,
        )
        ax.errorbar(
            x + offsets[grp],
            mean,
            yerr=[mean - lower, upper - mean],
            fmt="none",
            ecolor="black",
            capsize=5,
        )

    # print the % change by region and group, plus an all-region total per group
    print(
        "% change in gonorrhea-attributable HIV under no-2016-switch counterfactual "
        "(= averted / observed), 2016-2023, by region and sex (Male incl. MSM):"
    )
    for region in regions:
        for grp in sex_groups:
            if grp in pct[region]:
                m, lo, hi = pct[region][grp]
                print(
                    f"  {region} / {labels[grp]}: "
                    f"{m:.1f}% (95% CI {lo:.1f}-{hi:.1f}%)"
                )
    for grp, members in sex_groups.items():
        res = _pct_change(hiv.filter(pl.col("sex").is_in(members)))
        if res is not None:
            m, lo, hi = res
            print(
                f"  ALL REGIONS / {labels[grp]}: "
                f"{m:.1f}% (95% CI {lo:.1f}-{hi:.1f}%)"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=0, ha="center")
    ax.set_xlabel("Region")
    ax.set_ylabel(
        "Change in gonorrhea-attributable\nHIV incidence from 2016 first-line\ntreatment change (%)"
    )
    ax.legend(loc="upper left", edgecolor="black")


if __name__ == "__main__":
    # 3-row layout: top two rows are the original 2x2 -- (a) resistance rates,
    # (b) access vs resistance counts, (c) access share by sex, (d) access
    # share by region. The bottom row is a single full-width panel (e): gc
    # observed vs no-2016-switch cumulative cases by region.
    fig = plt.figure(figsize=(20, 21))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85])

    ax = np.empty((2, 2), dtype=object)
    ax[0, 0] = fig.add_subplot(gs[0, 0])
    ax[0, 1] = fig.add_subplot(gs[0, 1])
    ax[1, 0] = fig.add_subplot(gs[1, 0])
    ax[1, 1] = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])
    for _a in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax_e):
        _style_axis(_a)

    # ---- Panel a: resistance rates ----
    resistance_rates = pl.read_csv(
        os.path.join(output_dir, "estimated_resistance_rates.csv")
    )
    ax[0, 0].text(  # type: ignore
        -0.22,
        1.1,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_drug_resistance(resistance_rates, ax[0, 0], fig)

    # ---- Panels b, c, d all read the same flat CSV; no pickle anywhere ----
    hiv_averted = pl.read_csv(os.path.join(output_dir, "hiv_averted.csv"))

    ax[0, 1].text(  # type: ignore
        -0.22,
        1.1,
        "b",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_access_vs_resistance(
        hiv_averted,
        ax[0, 1],  # type: ignore
        fig,
    )

    ax[1, 0].text(  # type: ignore
        -0.22,
        1.1,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_access_vs_resistance_sex(
        hiv_averted,
        ax[1, 0],  # type: ignore
        fig,
    )

    ax[1, 1].text(  # type: ignore
        -0.22,
        1.1,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_access_vs_resistance_region(
        hiv_averted,
        ax[1, 1],  # type: ignore
        fig,
    )

    # ---- Panel e: gc observed vs no-2016-switch cumulative cases by region ----
    ax_e.text(
        -0.1,
        1.25,
        "e",
        transform=ax_e.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_gc_2016_counterfactual_region(hiv_averted, ax_e, fig)

    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "figure_access_vs_resistance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_access_vs_resistance.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    # ---- Supplemental: resistance rates by region (2x2 small multiples) ----
    fig_supp, ax_supp = plt.subplots(2, 2, figsize=(20, 15))
    for _a in ax_supp.flatten():  # type: ignore
        _style_axis(_a)

    regions = [
        ("Western", (0, 0), "a"),
        ("Eastern", (0, 1), "b"),
        ("Central", (1, 0), "c"),
        ("Southern", (1, 1), "d"),
    ]
    for region, (i, j), letter in regions:
        ax_supp[i, j].text(  # type: ignore
            -0.25,
            1.08,
            letter,
            transform=ax_supp[i, j].transAxes,  # type: ignore
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="right",
        )
        plot_drug_resistance_by_region(
            resistance_rates,
            ax_supp[i, j],  # type: ignore
            region,
            fig_supp,
            legend=(region == "Western"),
        )

    fig_supp.tight_layout()
    fig_supp.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_gc_resistance_by_region.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig_supp.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_gc_resistance_by_region.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )
