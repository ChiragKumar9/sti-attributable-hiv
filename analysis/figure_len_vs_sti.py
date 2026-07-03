import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rc

from averted_burden.delta_method import ci_via_logit

output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

STIS = ["gc", "chlamydia", "syphilis", "trichomoniasis"]

COUNTRIES = ["South Africa", "Zimbabwe"]

# Wu et al., Lancet HIV 2024, MAIN scenario: % of HIV infections averted over
# the 10-year implementation vs an oral-PrEP-only baseline. (mean, lower, upper)
# in percent. NB: this is an AVERTED fraction under a specific modeled rollout
# (~1.6% / ~4.0% population coverage), not an attributable ceiling -- see the
# caption note. The STI bars below are attributable %.
WU_LEN = {
    "South Africa": (12.3, 5.4, 19.5),
    "Zimbabwe": (17.0, 3.3, 28.2),
}

sti_names = {
    "gc": "Gonorrhea",
    "chlamydia": "Chlamydia",
    "syphilis": "Syphilis",
    "trichomoniasis": "Trichomoniasis",
}

# bar order + colours. LEN and the All-STIs summary get their own distinct
# colours; the four individual STIs keep the palette used in the other figures.
LEN_COLOR = "steelblue"
ALL_STI_COLOR = "dimgray"
sti_colors = {
    "gc": "firebrick",
    "chlamydia": "forestgreen",
    "syphilis": "purple",
    "trichomoniasis": "gold",
}


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)


def _sum_field(rows, field):
    """Sum a UFloat-valued field across rows by summing the UFloat objects
    themselves (not pre-extracted bounds), so the shared-leaf correlations
    propagate correctly. Extract an interval only from the final sum."""
    total = None
    for r in rows:
        val = r[field]
        total = val if total is None else total + val
    return total


def _sum_attributable_across_stis(rows, prefix):
    """Sum UFloat attributable-case counts across all four STIs AND across the
    given rows, by summing the UFloat objects themselves (same reason as
    `_sum_field`)."""
    total = None
    for r in rows:
        for sti in STIS:
            val = r[f"{prefix}{sti}"]
            total = val if total is None else total + val
    return total


def country_bar_values(rows_country):
    """Return an ordered list of (label, colour, (mean, lower, upper)) in
    percent for one country's panel: LEN (from Wu), All STIs, then each STI.
    Every STI proportion is TOTAL(sexes-pooled) attributable / total incidence,
    summed as UFloats first then extracted on the logit scale -- the same
    discipline as the attributable-burden figure."""
    total_incidence = _sum_field(rows_country, "unaids_incidence_number")

    # All STIs (coinfection deduction already applied upstream, so the four
    # per-STI parts sum to this without double counting).
    attr_all = _sum_attributable_across_stis(
        rows_country, "unaids_hiv_incidence_number_attributable_to_"
    )
    all_m, all_lo, all_hi = ci_via_logit(attr_all / total_incidence)  # type: ignore

    bars = [
        (
            "All STIs",
            ALL_STI_COLOR,
            (all_m * 100, all_lo * 100, all_hi * 100),  # type: ignore
        )
    ]
    for sti in STIS:
        attr = _sum_field(
            rows_country, f"unaids_hiv_incidence_number_attributable_to_{sti}"
        )
        m, lo, hi = ci_via_logit(attr / total_incidence)  # type: ignore
        bars.append(
            (
                sti_names[sti],
                sti_colors[sti],
                (m * 100, lo * 100, hi * 100),  # type: ignore
            )
        )
    return bars


def plot_country(ax, country, rows_country):
    len_triple = WU_LEN[country]
    bars = [("Lenacapavir", LEN_COLOR, len_triple)] + country_bar_values(
        rows_country
    )

    x = np.arange(len(bars))
    tops = []
    print(f"{country}:")
    for i, (label, color, (m, lo, hi)) in enumerate(bars):
        ax.bar(
            i,
            m,
            yerr=[[m - lo], [hi - m]],
            capsize=5,
            width=0.7,
            color=color,
        )
        tops.append(hi)
        print(f"  {label}: {m:.1f}% ({lo:.1f}% - {hi:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], rotation=30, ha="right")
    ax.set_ylabel(f"Reduction in HIV incidence (%), {country}")
    # ax.set_title(country)
    ax.set_ylim(0, max(tops) * 1.12)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


if __name__ == "__main__":
    with open(
        os.path.join(output_dir, "hiv_attributable_to_stis.ufloat.pkl"), "rb"
    ) as f:
        rows = pickle.load(f)
    rows = [r for r in rows if r["year"] == 2023]

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for a in ax:
        _style_axis(a)
    for a, letter in zip(ax, ["a", "b"]):
        a.text(
            -0.15,
            1.05,
            letter,
            transform=a.transAxes,
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="right",
        )

    for a, country in zip(ax, COUNTRIES):
        rows_country = [r for r in rows if r["location"] == country]
        if not rows_country:
            raise ValueError(f"No 2023 rows found for location == {country!r}")
        plot_country(a, country, rows_country)

    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "figure_len_vs_sti_by_country.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_len_vs_sti_by_country.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
