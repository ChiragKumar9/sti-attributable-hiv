import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rc

from averted_burden.delta_method import ci_via_logit

output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

# label, line colour -- same scheme as the template
SEX_STYLE = {
    "Male": ("Male", "midnightblue"),
    "Female": ("Female", "magenta"),
    "MSM": ("MSM", "slategrey"),
}


def _sum_field(rows, field):
    """Sum a UFloat-valued field across raw pickle rows by summing the UFloat
    objects themselves, so the uncertainties library propagates shared-leaf
    correlations (RR / resistance / GBD-UNAIDS clusters) automatically. Only
    extract an interval from the final summed/ratio'd UFloat."""
    total = None
    for r in rows:
        val = r[field]
        total = val if total is None else total + val
    return total


def _fields(sti):
    """(incidence_field, attributable_field) -- UNAIDS estimates."""
    return (
        "unaids_incidence_number",
        f"unaids_hiv_incidence_number_attributable_to_{sti}",
    )


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15))
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


def plot_paf_by_sex(rows, sti, ax, label, legend=False):
    """One panel: PAF (%) trajectory for `sti`, one line per sex. `rows` are the
    raw pickle rows already filtered to a single region. Within each year-sex
    group the single STI's attributable UFloat and the incidence UFloat are
    summed across countries, then PAF = attributable / incidence is formed as a
    UFloat ratio (shared incidence leaves cancel) and its (0,1) interval is
    extracted via ci_via_logit -- never by dividing pre-extracted CI bounds."""
    inc_field, attr_field = _fields(sti)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["year"], r["sex"])].append(r)

    series = {
        s: {"years": [], "mean": [], "lower": [], "upper": []}
        for s in SEX_STYLE
    }

    for (year, sex), grp in groups.items():
        if sex not in series:
            continue
        attributable = _sum_field(grp, attr_field)
        incidence = _sum_field(grp, inc_field)
        paf = attributable / incidence  # type: ignore
        mean, lower, upper = ci_via_logit(paf)
        series[sex]["years"].append(year)
        series[sex]["mean"].append(100 * mean)  # type: ignore
        series[sex]["lower"].append(100 * lower)  # type: ignore
        series[sex]["upper"].append(100 * upper)  # type: ignore

    for sex, (lbl, color) in SEX_STYLE.items():
        if not series[sex]["years"]:
            continue
        order = np.argsort(series[sex]["years"])
        years = np.array(series[sex]["years"])[order]
        mean = np.array(series[sex]["mean"])[order]
        lower = np.array(series[sex]["lower"])[order]
        upper = np.array(series[sex]["upper"])[order]

        ax.plot(years, mean, linewidth=3, label=lbl, color=color)
        ax.fill_between(years, lower, upper, alpha=0.3, color=color)

    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    if legend:
        ax.legend(edgecolor="black")
    ax.set_xticks([2000, 2010, 2020])
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))


if __name__ == "__main__":
    sti_map = {
        "gc": "gonorrhea",
        "chlamydia": "chlamydia",
        "syphilis": "syphilis",
        "trichomoniasis": "trichomoniasis",
    }

    with open(
        os.path.join(output_dir, "hiv_attributable_to_stis.ufloat.pkl"), "rb"
    ) as f:
        all_rows = pickle.load(f)
    all_rows = [r for r in all_rows if 2000 <= r["year"] <= 2023]

    regions = [
        ("Western", (0, 0), "a"),
        ("Eastern", (0, 1), "b"),
        ("Central", (1, 0), "c"),
        ("Southern", (1, 1), "d"),
    ]

    for sti in sti_map:
        fig, ax = setup_plot(2, 2)  # type: ignore

        rows = all_rows
        if sti == "trichomoniasis":
            # RRs for men/MSM are inferred/1.0, so keep women only (as template)
            rows = [r for r in rows if r["sex"] == "Female"]

        base_label = f"PAF of HIV attributable to\n{sti_map[sti]}"

        for region, (i, j), letter in regions:
            ax[i, j].text(  # type: ignore
                -0.25,
                1.08,
                letter,
                transform=ax[i, j].transAxes,  # type: ignore
                fontsize=24,
                fontweight="bold",
                va="top",
                ha="right",
            )
            region_rows = [r for r in rows if r["region"] == region]
            plot_paf_by_sex(
                region_rows,
                sti,
                ax[i, j],  # type: ignore
                base_label + f" in {region} Africa (%)",
                legend=(region == "Western"),
            )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                fig_dir,
                f"figure_supplemental_{sti}_paf_attributable_hiv_region_sex.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(
                fig_dir,
                f"figure_supplemental_{sti}_paf_attributable_hiv_region_sex.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
