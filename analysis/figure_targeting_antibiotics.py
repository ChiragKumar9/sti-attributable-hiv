import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml
from matplotlib import rc
from matplotlib.patches import Patch

from averted_burden.delta_method import ci_via_log, ci_via_logit

output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

STIS = ["gc", "chlamydia", "syphilis", "trichomoniasis"]

# symptomatic proportions (the "antibiotic-access-limited" share of each STI).
# read from params.yml -- the same source the main computation script uses --
# so the figure and the model agree on the numbers.
with open("params.yml", "r") as f:
    _params = yaml.safe_load(f)
sti_symptom_params = _params["sti_symptoms"]

# neutral color used for the symptomatic/asymptomatic legend swatches, and
# the alpha used to draw the asymptomatic portion of each bar. NEUTRAL_COLOR
# is plain black (no hue), chosen to look nothing like the blue-grey
# "slategrey" used elsewhere for the MSM series.
NEUTRAL_COLOR = "darkgrey"
ASYMPTOMATIC_ALPHA = 0.5


def _sex_key(sex):
    """Map the model's sex label to the symptom-param key. MSM uses the male
    symptomatic fractions, matching sex_key in the main computation script."""
    return "female" if sex == "Female" else "male"


def _symptomatic_legend_proxy():
    return Patch(
        facecolor=NEUTRAL_COLOR, edgecolor="black", label="Symptomatic"
    )


def _asymptomatic_legend_proxy():
    return Patch(
        facecolor=NEUTRAL_COLOR,
        edgecolor="black",
        alpha=ASYMPTOMATIC_ALPHA,
        label="Asymptomatic",
    )


def _sum_field(rows, field):
    """Sum a UFloat-valued field across a list of raw (not-yet-extracted)
    rows. Summing the UFloat objects themselves -- rather than summing
    already-extracted lower/upper CI bounds -- is what actually propagates
    variance/covariance correctly: the `uncertainties` library's linear
    error propagation picks up the correlations coming from any shared RR,
    resistance, or GBD/UNAIDS-cluster leaves automatically. Only extract an
    interval (via ci_via_logit) from the final summed UFloat."""
    total = None
    for r in rows:
        val = r[field]
        total = val if total is None else total + val
    return total


def _sum_attributable_across_stis(rows, prefix):
    """Sum UFloat attributable-case counts across all four STIs AND across
    the given rows (e.g. every country in a region, or every age row in an
    age group) by summing the UFloat objects themselves, for the same
    reason as `_sum_field` above."""
    total = None
    for r in rows:
        for sti in STIS:
            val = r[f"{prefix}{sti}"]
            total = val if total is None else total + val
    return total


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 27.5))
    if nrows * ncols == 1:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.tick_params(which="major", length=5.25, width=1.2)
        ax.tick_params(which="minor", length=3.0, width=0.9)
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
        temp.tick_params(which="major", length=5.25, width=1.2)
        temp.tick_params(which="minor", length=3.0, width=0.9)
        # offset the spines
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        temp.set_axisbelow(True)
    return fig, ax


def plot_attributable_hiv_burden_region_sex(ufloat_rows, ax, fig):
    # bar graph of attributable burden in 2023 by region and sex
    rows_2023 = [r for r in ufloat_rows if r["year"] == 2023]

    # print overall fraction of attributable hiv by sex, with error --
    # sum the UFloat attributable counts and the UFloat total incidence
    # across STIs/countries first, then take their ratio and extract the
    # interval on the logit scale (attributable_pct is a [0,1] proportion,
    # so ci_via_logit is the domain-matched transform, matching how PAFs
    # are extracted upstream).
    by_sex_groups = defaultdict(list)
    for r in rows_2023:
        by_sex_groups[r["sex"]].append(r)

    # overall (all sexes, regions, and STIs pooled) fraction of 2023 HIV
    # incidence attributable to STBIs. Same discipline: sum the UFloat
    # attributable counts and total incidence across every row first, then
    # ratio + logit interval.
    attributable_overall = _sum_attributable_across_stis(
        rows_2023, "unaids_hiv_incidence_number_attributable_to_"
    )
    total_incidence_overall = _sum_field(rows_2023, "unaids_incidence_number")
    overall_frac = attributable_overall / total_incidence_overall  # type: ignore
    mean, lower, upper = ci_via_logit(overall_frac)
    print(
        f"Overall fraction of attributable HIV (2023, all sexes/regions): "
        f"{mean * 100:.2f}% (95% CI {lower * 100:.2f}-{upper * 100:.2f}%)"  # type: ignore
    )

    n_mean, n_lower, n_upper = ci_via_log(attributable_overall)
    print(
        f"Overall attributable HIV cases (2023, all sexes/regions): "
        f"{round(n_mean):,} (95% CI {round(n_lower):,}-{round(n_upper):,})"  # type: ignore
    )

    print("Overall fraction of attributable HIV by sex:")
    for sex, rows in by_sex_groups.items():
        attributable = _sum_attributable_across_stis(
            rows, "unaids_hiv_incidence_number_attributable_to_"
        )
        total_incidence = _sum_field(rows, "unaids_incidence_number")
        attributable_frac = attributable / total_incidence  # type: ignore
        mean, lower, upper = ci_via_logit(attributable_frac)
        print(
            f"  sex: {sex}, attributable_pct: {mean * 100:.2f}, "  # type: ignore
            f"attributable_pct_lower: {lower * 100:.2f}, "  # type: ignore
            f"attributable_pct_upper: {upper * 100:.2f}"  # type: ignore
        )

    # enforce ordering of regions and sex groups
    regions = ["Western", "Eastern", "Central", "Southern"]
    sexes = ["Male", "Female", "MSM"]

    region_sex_groups = defaultdict(list)
    for r in rows_2023:
        region_sex_groups[(r["region"], r["sex"])].append(r)

    pct_by_region_sex = {}
    # symptomatic (access-limited) share of each region/sex bar. Bars here pool
    # all four STIs, so this is a burden-weighted blend: for each STI weight its
    # symptomatic fraction by that STI's nominal attributable count, then
    # normalise by the total attributable count. Nominal values only (the split
    # is a visual partition; the whisker stays on the full bar height).
    symp_frac_by_region_sex = {}
    for (region, sex), rows in region_sex_groups.items():
        attributable = _sum_attributable_across_stis(
            rows, "unaids_hiv_incidence_number_attributable_to_"
        )
        total_incidence = _sum_field(rows, "unaids_incidence_number")
        attributable_frac = attributable / total_incidence  # type: ignore
        pct_by_region_sex[(region, sex)] = ci_via_logit(attributable_frac)

        sk = _sex_key(sex)
        symp_num = 0.0
        total_num = 0.0
        for r in rows:
            for sti in STIS:
                a = r[f"unaids_hiv_incidence_number_attributable_to_{sti}"]
                a_nom = getattr(a, "nominal_value", a)
                symp_num += sti_symptom_params[sti][sk] * a_nom
                total_num += a_nom
        symp_frac_by_region_sex[(region, sex)] = (
            symp_num / total_num if total_num else 0.0
        )

    sex_names = {
        "Male": "Heterosexual men",
        "Female": "Heterosexual women",
        "MSM": "MSM",
    }

    colors = {
        "Male": "midnightblue",
        "Female": "magenta",
        "MSM": "slategrey",
    }

    x = np.arange(len(regions))

    width = 0.25
    offsets = {
        "Male": -width,
        "Female": 0,
        "MSM": width,
    }

    for sex in sexes:
        mean = np.array(
            [
                pct_by_region_sex.get((region, sex), (np.nan, np.nan, np.nan))[
                    0
                ]
                * 100
                for region in regions
            ]
        )
        lower = np.array(
            [
                pct_by_region_sex.get((region, sex), (np.nan, np.nan, np.nan))[
                    1
                ]
                * 100
                for region in regions
            ]
        )
        upper = np.array(
            [
                pct_by_region_sex.get((region, sex), (np.nan, np.nan, np.nan))[
                    2
                ]
                * 100
                for region in regions
            ]
        )

        # symptomatic (access-limited) share of this bar, used to split it
        # into a full-alpha bottom segment and a lighter-alpha top segment.
        symp_frac = np.array(
            [
                symp_frac_by_region_sex.get((region, sex), 0.0)
                for region in regions
            ]
        )
        symptomatic_h = mean * symp_frac
        asymptomatic_h = mean * (1 - symp_frac)

        # symptomatic (bottom) segment, drawn at full alpha; carries the
        # legend entry for this sex so the swatch isn't shown translucent
        ax.bar(
            x + offsets[sex],
            symptomatic_h,
            width=width,
            label=sex_names[sex],
            color=colors[sex],
        )
        # asymptomatic (top) segment, drawn at a lighter alpha
        ax.bar(
            x + offsets[sex],
            asymptomatic_h,
            bottom=symptomatic_h,
            width=width,
            color=colors[sex],
            alpha=ASYMPTOMATIC_ALPHA,
        )
        # whisker drawn as its own artist (not part of either bar segment)
        # so it stays fully opaque regardless of the segment alphas above
        ax.errorbar(
            x + offsets[sex],
            mean,
            yerr=[mean - lower, upper - mean],
            fmt="none",
            ecolor="black",
            capsize=5,
        )

    # print the bar and error values by region with nice formatting
    for region in regions:
        for sex in sexes:
            if (region, sex) in pct_by_region_sex:
                mean, lower, upper = pct_by_region_sex[(region, sex)]
                print(
                    f"{sex_names[sex]} in {region}: {mean * 100:.1f}% "
                    f"({lower * 100:.1f}% - {upper * 100:.1f}%)"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, ha="center")
    ax.set_xlabel("Region")
    ax.set_ylabel("CSTI4-attributable HIV incidence (%)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(_symptomatic_legend_proxy())
    labels.append("Symptomatic")
    handles.append(_asymptomatic_legend_proxy())
    labels.append("Asymptomatic")
    ax.legend(handles, labels, loc=(0.01, 0.9), edgecolor="black")
    # add minor y ticks every 5%
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))


def plot_attributable_hiv_burden_region_pathogen(ufloat_rows, ax, fig):
    # bar graph of attributable burden in 2023 by region and pathogen, sexes
    # pooled. Matches the semantics of the equivalent panel in the data-source
    # comparison figure (region on x, one bar per STI), but computes each
    # proportion with correct UFloat propagation: for each region, sum the
    # per-STI attributable UFloat counts and the total-incidence UFloat across
    # every country/sex row first, then take the ratio and extract the interval
    # on the logit scale (a [0,1] proportion) -- rather than dividing already-
    # extracted lower/upper bounds. This is the same discipline as the other
    # panels in this figure.
    rows_2023 = [r for r in ufloat_rows if r["year"] == 2023]

    regions = ["Western", "Eastern", "Central", "Southern"]
    stis = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
    sti_names = {
        "gc": "Gonorrhea",
        "syphilis": "Syphilis",
        "trichomoniasis": "Trichomoniasis",
        "chlamydia": "Chlamydia",
    }
    colors = {
        "gc": "firebrick",
        "syphilis": "purple",
        "trichomoniasis": "gold",
        "chlamydia": "forestgreen",
    }

    region_groups = defaultdict(list)
    for r in rows_2023:
        region_groups[r["region"]].append(r)

    # pct_by_region_sti[(region, sti)] = (mean, lower, upper) proportion of
    # TOTAL (sexes-pooled) incidence attributable to this STI.
    # female_frac_by_region_sti[(region, sti)] = nominal fraction of that STI's
    # attributable burden that is female (point estimate only; printed to the
    # console, not drawn -- see note below).
    # symp_frac_by_region_sti[(region, sti)] = nominal symptomatic (access-
    # limited) share of that STI's attributable burden. Sexes are pooled here,
    # so it's burden-weighted across the rows' sexes (female uses female
    # symptom params, male/MSM use male), normalised by the total attributable.
    pct_by_region_sti = {}
    female_frac_by_region_sti = {}
    symp_frac_by_region_sti = {}
    for region in regions:
        rows = region_groups.get(region, [])
        if not rows:
            continue
        female_rows = [r for r in rows if r["sex"] == "Female"]
        total_incidence = _sum_field(rows, "unaids_incidence_number")
        for sti in stis:
            col = f"unaids_hiv_incidence_number_attributable_to_{sti}"
            attributable = _sum_field(rows, col)
            attributable_frac = attributable / total_incidence  # type: ignore
            pct_by_region_sti[(region, sti)] = ci_via_logit(attributable_frac)
            # nominal-only female share of this STI's attributable burden
            attributable_female = _sum_field(female_rows, col)
            attr_nom = getattr(attributable, "nominal_value", attributable)
            fem_nom = (
                getattr(
                    attributable_female, "nominal_value", attributable_female
                )
                if attributable_female is not None
                else 0.0
            )
            female_frac_by_region_sti[(region, sti)] = (
                fem_nom / attr_nom if attr_nom else 0.0
            )
            # nominal-only symptomatic share (burden-weighted across sexes)
            symp_num = 0.0
            total_num = 0.0
            for r in rows:
                a = r[col]
                a_nom = getattr(a, "nominal_value", a)
                symp_num += sti_symptom_params[sti][_sex_key(r["sex"])] * a_nom
                total_num += a_nom
            symp_frac_by_region_sti[(region, sti)] = (
                symp_num / total_num if total_num else 0.0
            )

    x = np.arange(len(regions))  # type; ignore
    width = 0.2
    for i, pathogen in enumerate(stis):
        offset = width * (i - 1.5)  # center the group of bars
        mean = np.array(
            [
                pct_by_region_sti.get(
                    (region, pathogen), (np.nan, np.nan, np.nan)
                )[0]
                * 100
                for region in regions
            ]
        )
        lower = np.array(
            [
                pct_by_region_sti.get(
                    (region, pathogen), (np.nan, np.nan, np.nan)
                )[1]
                * 100
                for region in regions
            ]
        )
        upper = np.array(
            [
                pct_by_region_sti.get(
                    (region, pathogen), (np.nan, np.nan, np.nan)
                )[2]
                * 100
                for region in regions
            ]
        )

        # symptomatic (access-limited) share of this pooled bar, used to
        # split it into a full-alpha bottom segment and a lighter-alpha top
        # segment. (Note: the female/non-female split that used to be drawn
        # here as two stacked bars is removed -- both segments used the same
        # color/alpha, so it never had a visual effect; female_frac is still
        # tracked above and printed to the console below.)
        symp_frac = np.array(
            [
                symp_frac_by_region_sti.get((region, pathogen), 0.0)
                for region in regions
            ]
        )
        symptomatic_h = mean * symp_frac
        asymptomatic_h = mean * (1 - symp_frac)

        # symptomatic (bottom) segment, drawn at full alpha; carries the
        # legend entry for this pathogen so the swatch isn't shown translucent
        ax.bar(
            x + offset,
            symptomatic_h,
            width=width,
            label=sti_names[pathogen],
            color=colors[pathogen],
        )
        # asymptomatic (top) segment, drawn at a lighter alpha
        ax.bar(
            x + offset,
            asymptomatic_h,
            bottom=symptomatic_h,
            width=width,
            color=colors[pathogen],
            alpha=ASYMPTOMATIC_ALPHA,
        )
        # single whisker on the total bar height (the pooled proportion),
        # drawn as its own artist so it stays fully opaque
        ax.errorbar(
            x + offset,
            mean,
            yerr=[mean - lower, upper - mean],
            fmt="none",
            ecolor="black",
            capsize=5,
        )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(_symptomatic_legend_proxy())
    labels.append("Symptomatic")
    handles.append(_asymptomatic_legend_proxy())
    labels.append("Asymptomatic")

    # print the bar and error values by region and pathogen
    print("CSTI4-attributable HIV by region and pathogen (2023):")
    for region in regions:
        for pathogen in stis:
            if (region, pathogen) in pct_by_region_sti:
                mean, lower, upper = pct_by_region_sti[(region, pathogen)]
                ffrac = female_frac_by_region_sti.get((region, pathogen), 0.0)
                print(
                    f"  {sti_names[pathogen]} in {region}: {mean * 100:.1f}% "
                    f"({lower * 100:.1f}% - {upper * 100:.1f}%), "
                    f"female share {ffrac * 100:.1f}%"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, ha="center")
    ax.set_xlabel("Region")
    ax.set_ylabel("HIV incidence (%)")
    ax.legend(handles, labels, loc=(0.01, 0.7), edgecolor="black")
    # add minor y ticks every 1%
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


def plot_attributable_hiv_burden_age_pathogen(ufloat_rows, ax, sex, fig):
    # we need the age groups in a particular order
    age_groups = ["15-24", "25-49", "50+"]

    sexes = {"Female"} if sex == "Female" else {"Male", "MSM"}
    rows_filtered = [r for r in ufloat_rows if r["sex"] in sexes]

    groups = defaultdict(list)
    for r in rows_filtered:
        groups[r["age_group"]].append(r)

    stis = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
    sti_names = {
        "gc": "Gonorrhea",
        "syphilis": "Syphilis",
        "trichomoniasis": "Trichomoniasis",
        "chlamydia": "Chlamydia",
    }

    colors = {
        "gc": "firebrick",
        "syphilis": "purple",
        "trichomoniasis": "gold",
        "chlamydia": "forestgreen",
    }

    # Determine which STIs to plot for this sex
    if sex == "Female":
        stis = stis
    else:  # Heterosexual male or MSM
        stis = [s for s in stis if s != "trichomoniasis"]

    # this panel is a single sex group (Female, or Male+MSM), so the
    # symptomatic (access-limited) share of a bar is just that STI's
    # symptomatic fraction for the corresponding sex key -- constant across
    # age groups.
    sk = _sex_key(sex)

    pct = {sti: {"mean": [], "lower": [], "upper": []} for sti in stis}
    for age_group in age_groups:
        rows = groups.get(age_group, [])
        # sum the UFloat total incidence and each STI's UFloat attributable
        # count across every country/year row in this age group first, then
        # take the ratio and extract the interval on the logit scale (it's
        # a [0,1] proportion) -- rather than dividing already-extracted
        # lower/upper bounds, which is what this used to do.
        total_incidence = _sum_field(rows, "unaids_incidence_number")
        for sti in stis:
            attributable = _sum_field(
                rows, f"unaids_hiv_incidence_number_attributable_to_{sti}"
            )
            ratio = attributable / total_incidence  # type: ignore
            mean, lower, upper = ci_via_logit(ratio)
            pct[sti]["mean"].append(mean * 100)  # type: ignore
            pct[sti]["lower"].append(lower * 100)  # type: ignore
            pct[sti]["upper"].append(upper * 100)  # type: ignore

    x = np.arange(len(age_groups))

    width = 0.2
    if sex == "Female":
        offsets = [width * (i - 1.5) for i in range(4)]  # 4 bars centered
    else:  # Heterosexual male or MSM
        offsets = [width * (i - 1) for i in range(3)]  # 3 bars centered

    for i, pathogen in enumerate(stis):
        offset = offsets[i]  # center the group of bars
        mean = np.array(pct[pathogen]["mean"])
        lower = np.array(pct[pathogen]["lower"])
        upper = np.array(pct[pathogen]["upper"])

        # symptomatic (access-limited) share of this bar, used to split it
        # into a full-alpha bottom segment and a lighter-alpha top segment.
        symp_frac = sti_symptom_params[pathogen][sk]
        symptomatic_h = mean * symp_frac
        asymptomatic_h = mean * (1 - symp_frac)

        # symptomatic (bottom) segment, drawn at full alpha; carries the
        # legend entry for this pathogen so the swatch isn't shown translucent
        ax.bar(
            x + offset,
            symptomatic_h,
            width=width,
            label=sti_names[pathogen],
            color=colors[pathogen],
        )
        # asymptomatic (top) segment, drawn at a lighter alpha
        ax.bar(
            x + offset,
            asymptomatic_h,
            bottom=symptomatic_h,
            width=width,
            color=colors[pathogen],
            alpha=ASYMPTOMATIC_ALPHA,
        )
        # whisker drawn as its own artist so it stays fully opaque
        ax.errorbar(
            x + offset,
            mean,
            yerr=[mean - lower, upper - mean],
            fmt="none",
            ecolor="black",
            capsize=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(age_groups)
    ax.set_xlabel("Age group")
    ax.set_ylabel(
        f"HIV incidence, {'women' if sex == 'Female' else 'men'} (%)"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(_symptomatic_legend_proxy())
    labels.append("Symptomatic")
    handles.append(_asymptomatic_legend_proxy())
    labels.append("Asymptomatic")
    if sex == "Female":
        ax.legend(handles, labels, loc=(0.02, 0.87), edgecolor="black")
    else:
        ax.legend(handles, labels, loc=(0.5, 0.85), edgecolor="black")
    # add minor y ticks every 1%
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


if __name__ == "__main__":
    # 2x2 grid: (a) region x sex, (b) region x pathogen [both from the
    # attributable pickle, 2023], (c) age x pathogen female, (d) age x
    # pathogen male [from the age-stratified pickle].
    fig, ax = setup_plot(2, 2)  # type: ignore
    fig.set_size_inches(20, 18)

    # Panels a and b read the (country/sex/year) attributable pickle.
    with open(
        os.path.join(output_dir, "hiv_attributable_to_stis.ufloat.pkl"), "rb"
    ) as f:
        hiv_attributable_ufloat = pickle.load(f)
    hiv_attributable_ufloat = [
        r for r in hiv_attributable_ufloat if 2000 <= r["year"] <= 2023
    ]

    # Panel a: attributable HIV burden by region and sex, 2023.
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
    plot_attributable_hiv_burden_region_sex(
        hiv_attributable_ufloat,
        ax[0, 0],  # type: ignore
        fig,
    )

    # Panel b: attributable HIV burden by region and pathogen (sexes pooled),
    # 2023 -- correct UFloat propagation, matching the other panels.
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
    plot_attributable_hiv_burden_region_pathogen(
        hiv_attributable_ufloat,
        ax[0, 1],  # type: ignore
        fig,
    )

    # Panels c, d: attributable HIV burden by age group and pathogen, for
    # females and males respectively (age-stratified pickle).
    with open(
        os.path.join(
            output_dir, "hiv_attributable_to_stis_age_stratified.ufloat.pkl"
        ),
        "rb",
    ) as f:
        hiv_age_stratified_ufloat = pickle.load(f)
    hiv_age_stratified_ufloat = [
        r for r in hiv_age_stratified_ufloat if r["year"] == 2023
    ]

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_age_pathogen(
        hiv_age_stratified_ufloat,
        ax[1, 0],  # type: ignore
        "Female",
        fig,
    )

    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_age_pathogen(
        hiv_age_stratified_ufloat,
        ax[1, 1],  # type: ignore
        "Male",
        fig,
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(fig_dir, "figure_targeting_expanded_antibiotics.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_targeting_expanded_antibiotics.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
