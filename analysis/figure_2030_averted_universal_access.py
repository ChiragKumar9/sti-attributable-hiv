import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rc

from averted_burden.delta_method import ci_via_log, ci_via_logit

output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

STIS = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
SEXES = ["Male", "Female", "MSM"]
REGIONS = ["Western", "Eastern", "Central", "Southern"]

# ---------------------------------------------------------------- config
ONLY_UNAIDS_ANALYSIS = True
EXCLUDED_COUNTRY_CODES = ["LBR", "GNQ", "STP", "CPV"]  # from data-prep script

TRAJ_START = 2026
CUM_START = 2026

SDG_BASELINE_YEAR = 2010
SDG_TARGET_YEAR = 2030
SDG_FRACTION_OF_BASELINE = 0.10  # SDG level = 10% of the 2010 total
SDG_COLOR = "#009CDE"  # UN blue

DIRECT_COLOR = "grey"  # "#275DAD"
TOTAL_COLOR = "darkred"

sex_colors = {"Male": "midnightblue", "Female": "magenta", "MSM": "slategrey"}
sex_names = {
    "Male": "Heterosexual men",
    "Female": "Heterosexual women",
    "MSM": "MSM",
}
sti_colors = {
    "gc": "firebrick",
    "chlamydia": "forestgreen",
    "syphilis": "purple",
    "trichomoniasis": "gold",
}
sti_names = {
    "gc": "Gonorrhea",
    "chlamydia": "Chlamydia",
    "syphilis": "Syphilis",
    "trichomoniasis": "Trichomoniasis",
}
# ------------------------------------------------------------------------


def _nominal(u):
    return getattr(u, "nominal_value", u)


def _usum(vals):
    """Sum UFloat objects (starting from None, never `0 + UFloat`). Summing the
    objects -- not pre-extracted bounds -- propagates the shared-leaf
    correlations correctly; extract an interval only from the final sum."""
    total = None
    for v in vals:
        total = v if total is None else total + v
    return total


def _style_axis(ax):
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


def load_rows():
    with open(os.path.join(output_dir, "hiv_averted.ufloat.pkl"), "rb") as f:
        rows = pickle.load(f)
    if ONLY_UNAIDS_ANALYSIS:
        rows = [
            r for r in rows if r["country_code"] not in EXCLUDED_COUNTRY_CODES
        ]
    return rows


def _year_total_nominal(rows, year, predicate=None):
    vals = [
        r["unaids_incidence_number"]
        for r in rows
        if r["year"] == year and (predicate is None or predicate(r))
    ]
    return _nominal(_usum(vals)) if vals else None


def sdg_goal_2030(rows, predicate):
    """Minimum single-year change in incidence needed to hit the 2030 SDG
    target: the projected 2030 incidence minus the SDG level (10% of the 2010
    incidence). Per-sex via `predicate`. Returns None if either the 2010 or the
    2030 anchor is missing."""
    inc_2010 = _year_total_nominal(rows, SDG_BASELINE_YEAR, predicate)
    proj_2030 = _year_total_nominal(rows, SDG_TARGET_YEAR, predicate)
    if inc_2010 is None or proj_2030 is None:
        return None
    sdg_level = SDG_FRACTION_OF_BASELINE * inc_2010
    return proj_2030 - sdg_level


def _fmt_count(triple):
    m, lo, hi = triple
    return f"{round(m):,} (95% CI {round(lo):,}-{round(hi):,})"


def _fmt_pct(triple):
    m, lo, hi = triple
    return f"{100 * m:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"


def print_change_in_cases(rows):
    """Forward (2026+) universal-access averted cases, summed as UFloats then
    extracted once. Prints the absolute decrease (total and by sex) and the
    TOTAL cumulative % of cases averted (all sexes/regions/STIs pooled)."""
    fwd = [r for r in rows if r["year"] >= CUM_START]
    print("Total change in HIV cases (decrease), forward universal access:")
    groups = [
        ("Total", lambda r: True),
        ("Male", lambda r: r["sex"] == "Male"),
        ("Female", lambda r: r["sex"] == "Female"),
        ("MSM", lambda r: r["sex"] == "MSM"),
        ("Male + MSM", lambda r: r["sex"] in ("Male", "MSM")),
    ]
    for label, pred in groups:
        total = _usum(
            [r["hiv_averted_upper_bound_future"] for r in fwd if pred(r)]
        )
        print(
            f"  {label}: "
            + (_fmt_count(ci_via_log(total)) if total is not None else "n/a")
        )

    averted_all = _usum([r["hiv_averted_upper_bound_future"] for r in fwd])
    inc_all = _usum([r["unaids_incidence_number"] for r in fwd])
    if averted_all is not None and inc_all is not None:
        print(
            "Total % of cumulative cases averted:",
            _fmt_pct(ci_via_logit(averted_all / inc_all)),  # type: ignore
        )

    print("% of cumulative cases averted by sex (all regions):")
    for label, pred in [
        ("Male", lambda r: r["sex"] == "Male"),
        ("Female", lambda r: r["sex"] == "Female"),
        ("MSM", lambda r: r["sex"] == "MSM"),
        ("Male + MSM", lambda r: r["sex"] in ("Male", "MSM")),
    ]:
        av = _usum(
            [r["hiv_averted_upper_bound_future"] for r in fwd if pred(r)]
        )
        inc = _usum([r["unaids_incidence_number"] for r in fwd if pred(r)])
        if av is not None and inc is not None:
            print(f"  {label}: {_fmt_pct(ci_via_logit(av / inc))}")  # type: ignore


def sex_matches(r, sex):
    return r["sex"] == sex


def print_goal_check(rows):
    """Ground-truth for a/b/c: per-sex cumulative 2026-2030 direct and total
    averted vs the 2030 goal, with a HIT/miss flag (scale-independent)."""
    print("SDG goal check (cumulative 2026-2030 averted vs 2030 goal):")
    for label in ("Male", "MSM", "Female"):
        fwd = [
            r for r in rows if r["year"] >= CUM_START and sex_matches(r, label)
        ]
        direct = _usum([r["direct_hiv_averted_upper_bound"] for r in fwd])
        total = _usum([r["hiv_averted_upper_bound_future"] for r in fwd])
        goal = sdg_goal_2030(rows, lambda r: sex_matches(r, label))
        if goal is None or total is None or direct is None:
            print(f"  {label}: n/a")
            continue
        d, t = _nominal(direct), _nominal(total)
        hit = "HIT" if t >= goal else "miss"
        print(
            f"  {label}: direct {d:,.0f}, total {t:,.0f}, goal {goal:,.0f} -> {hit}"
        )


# ------------------------------------------------------------------------
# Supplemental figure: ABSOLUTE incidence trajectory for one sex, LINEAR y-axis.
# Projected (black), plus residual incidence under the direct-only and
# direct+indirect(total) effects, and the SDG level line. Residual bands are
# summed per-row UFloats (reference - averted) then extracted.
# ------------------------------------------------------------------------
def plot_incidence_trajectory(ax, rows, sexes, ylabel):
    fwd = [r for r in rows if r["year"] >= TRAJ_START and r["sex"] in sexes]
    years = sorted({r["year"] for r in fwd})

    ref, direct, total = {}, {}, {}
    for y in years:
        g = [r for r in fwd if r["year"] == y]
        ref_u = _usum([r["unaids_incidence_number"] for r in g])
        ref_nom = _nominal(ref_u)
        ref[y] = ci_via_log(ref_u)

        # residual = reference - averted. Extract the CI on the strictly-
        # positive averted count (where a log-normal interval is valid), then
        # mirror-subtract from the (zero-variance) reference -- bounding the
        # residual's upper edge at ref_nom and avoiding the overshoot from
        # log-extracting the difference directly (averted's uncertainty can
        # otherwise push the upper bound above the projected line).
        d_m, d_lo, d_hi = ci_via_log(
            _usum([r["direct_hiv_averted_upper_bound"] for r in g])
        )
        t_m, t_lo, t_hi = ci_via_log(
            _usum([r["hiv_averted_upper_bound_future"] for r in g])
        )
        direct[y] = (ref_nom - d_m, max(0.0, ref_nom - d_hi), ref_nom - d_lo)
        total[y] = (ref_nom - t_m, max(0.0, ref_nom - t_hi), ref_nom - t_lo)

    def band(d, color, label):
        m = [d[y][0] for y in years]
        lo = [d[y][1] for y in years]
        hi = [d[y][2] for y in years]
        ax.plot(years, m, lw=3, color=color, label=label)
        ax.fill_between(years, lo, hi, alpha=0.3, color=color)

    band(direct, DIRECT_COLOR, "Direct")
    band(total, TOTAL_COLOR, "Total")
    band(ref, "black", "Projected")

    inc_2010 = _year_total_nominal(
        rows, SDG_BASELINE_YEAR, lambda r: r["sex"] in sexes
    )
    if inc_2010 is not None:
        print(
            f"  {ylabel}: 2010 incidence = {inc_2010:,.0f}, SDG target = {SDG_FRACTION_OF_BASELINE * inc_2010:,.0f}"
        )
        ax.axhline(
            SDG_FRACTION_OF_BASELINE * inc_2010,
            ls="--",
            lw=3,
            color=SDG_COLOR,
            label="SDG target",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0)  # linear, absolute
    ax.set_xticks([TRAJ_START, SDG_TARGET_YEAR])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax.legend(loc=(0.55, 0.75), edgecolor="black")


# ------------------------------------------------------------------------
# Cumulative 2026-2030 averted CASES for one group (sex or region) -- a Direct
# bar and a Total bar (each with a CI), plus a dashed UN-blue 2030-goal line.
# The group is selected via `predicate`; its label lives in the y-axis label.
# ------------------------------------------------------------------------
def plot_group_bars(ax, rows, predicate, ylabel, tick_fontsize=None):
    fwd = [r for r in rows if r["year"] >= CUM_START and predicate(r)]
    direct = _usum([r["direct_hiv_averted_upper_bound"] for r in fwd])
    total = _usum([r["hiv_averted_upper_bound_future"] for r in fwd])

    dm, dlo, dhi = ci_via_log(direct)
    tm, tlo, thi = ci_via_log(total)

    ax.bar(
        0,
        dm,
        yerr=[[dm - dlo], [dhi - dm]],  # type: ignore
        capsize=6,
        width=0.6,
        color=DIRECT_COLOR,
    )
    ax.bar(
        1,
        tm,
        yerr=[[tm - tlo], [thi - tm]],  # type: ignore
        capsize=6,
        width=0.6,
        color=TOTAL_COLOR,
    )

    tops = [dhi, thi]
    goal = sdg_goal_2030(rows, predicate)
    ax.axhline(goal, ls="--", lw=3, color=SDG_COLOR)
    tops = [dhi, thi, goal]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Directly-\nattributable", "Including\ntransmission"],
        fontsize=tick_fontsize,
    )
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(tops) * 1.12)  # type: ignore
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )


# ------------------------------------------------------------------------
# Row 2: cumulative (2026-end) DECREASE in cases, shown POSITIVE. No SDG line.
# decrease % = averted / (that population's projected incidence), formed as a
# UFloat ratio then extracted on the logit scale (a [0,1] proportion).
# ------------------------------------------------------------------------
def _decrease_pct(averted, inc):
    if _nominal(averted) <= 0 or _nominal(inc) <= 0:
        return 0.0, 0.0, 0.0
    m, lo, hi = ci_via_logit(averted / inc)
    return 100.0 * m, 100.0 * (m - lo), 100.0 * (hi - m)  # type: ignore


def plot_pct_by_sex(ax, all_rows):
    fwd = [r for r in all_rows if r["year"] >= CUM_START]
    x = np.arange(len(REGIONS))
    width = 0.25
    offsets = {"Male": -width, "Female": 0.0, "MSM": width}

    tops = []
    print("Cumulative % decrease by region and sex (forward, upper bound):")
    for sex in SEXES:
        heights, elo, ehi = [], [], []
        for region in REGIONS:
            g = [r for r in fwd if r["region"] == region and r["sex"] == sex]
            if not g:
                heights.append(np.nan)
                elo.append(0.0)
                ehi.append(0.0)
                continue
            averted = _usum([r["hiv_averted_upper_bound_future"] for r in g])
            inc = _usum([r["unaids_incidence_number"] for r in g])
            h, lo, hi = _decrease_pct(averted, inc)
            heights.append(h)
            elo.append(lo)
            ehi.append(hi)
            if np.isfinite(h):
                tops.append(h + hi)
                print(
                    f"  {sex_names[sex]} in {region}: {h:.1f}% "
                    f"[{h - lo:.1f}, {h + hi:.1f}]"
                )
        ax.bar(
            x + offsets[sex],
            heights,
            yerr=[elo, ehi],
            capsize=5,
            width=width,
            label=sex_names[sex],
            color=sex_colors[sex],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, ha="center")
    ax.set_xlabel("Region")
    ax.set_ylabel("Cumulative change in HIV incidence (%)")
    # single column, lifted just above the axes
    ax.legend(loc=(0.03, 0.92), edgecolor="black")
    ax.set_ylim(0, (max(tops) * 1.12) if tops else 1)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))


def plot_pct_by_sti(ax, all_rows):
    fwd = [r for r in all_rows if r["year"] >= CUM_START]
    x = np.arange(len(REGIONS))
    width = 0.2

    tops = []
    print("Cumulative % decrease by region and STI (forward, upper bound):")
    for i, sti in enumerate(STIS):
        offset = width * (i - 1.5)
        heights, elo, ehi = [], [], []
        for region in REGIONS:
            g = [r for r in fwd if r["region"] == region]  # sexes pooled
            if not g:
                heights.append(np.nan)
                elo.append(0.0)
                ehi.append(0.0)
                continue
            averted = _usum(
                [r[f"hiv_averted_{sti}_upper_bound_future"] for r in g]
            )
            inc = _usum([r["unaids_incidence_number"] for r in g])
            h, lo, hi = _decrease_pct(averted, inc)
            heights.append(h)
            elo.append(lo)
            ehi.append(hi)
            if np.isfinite(h):
                tops.append(h + hi)
                print(
                    f"  {sti_names[sti]} in {region}: {h:.1f}% "
                    f"[{h - lo:.1f}, {h + hi:.1f}]"
                )
        ax.bar(
            x + offset,
            heights,
            yerr=[elo, ehi],
            capsize=5,
            width=width,
            label=sti_names[sti],
            color=sti_colors[sti],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, ha="center")
    ax.set_xlabel("Region")
    ax.set_ylabel("Cumulative change in HIV incidence (%)")
    ax.legend(loc="upper left", edgecolor="black")
    ax.set_ylim(0, (max(tops) * 1.12) if tops else 1)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))


if __name__ == "__main__":
    rows = load_rows()

    print_change_in_cases(rows)
    print_goal_check(rows)

    # Row 1: two sexes; Row 2: four regions (smaller); Row 3: original panels
    fig = plt.figure(figsize=(22.5, 25))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 0.7, 1])
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[1, 3])
    ax_g = fig.add_subplot(gs[2, 0:2])
    ax_h = fig.add_subplot(gs[2, 2:4])
    panels = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h]

    for a in panels:
        _style_axis(a)
    for a, letter in zip(panels, ["a", "b", "c", "d", "e", "f", "g", "h"]):
        a.text(
            -0.18 if letter in ["a", "b", "g", "h"] else -0.48,
            1.1 if letter in ["a", "b", "g", "h"] else 1.15,
            letter,
            transform=a.transAxes,
            fontsize=30,
            fontweight="bold",
            va="top",
            ha="right",
        )

    _panel_groups = [
        ("a", "Male", lambda r: r["sex"] == "Male"),
        ("b", "Female", lambda r: r["sex"] == "Female"),
        ("c", "Western", lambda r: r["region"] == "Western"),
        ("d", "Eastern", lambda r: r["region"] == "Eastern"),
        ("e", "Central", lambda r: r["region"] == "Central"),
        ("f", "Southern", lambda r: r["region"] == "Southern"),
    ]
    print("Blue dashed line (2030 goal), bar values, and % of goal reached:")
    for letter, name, det in _panel_groups:
        fwd = [r for r in rows if r["year"] >= CUM_START and det(r)]
        direct = _usum([r["direct_hiv_averted_upper_bound"] for r in fwd])
        total = _usum([r["hiv_averted_upper_bound_future"] for r in fwd])
        goal = sdg_goal_2030(rows, det)
        print(f"  {letter} ({name}): goal {goal:,.0f}")
        print(
            f"    Direct: {_fmt_count(ci_via_log(direct))} = "
            + f"{_fmt_pct(ci_via_log(direct / goal))} of goal"  # type: ignore
        )
        print(
            f"    Total:  {_fmt_count(ci_via_log(total))} = "
            + f"{_fmt_pct(ci_via_log(total / goal))} of goal"  # type: ignore
        )

    # Row 1: cumulative averted-case bars (Direct, Total) + goal line, by sex
    plot_group_bars(
        ax_a,
        rows,
        lambda r: r["sex"] == "Male",
        "Cumulative change in HIV incidence,\nmen (N)",
    )
    plot_group_bars(
        ax_b,
        rows,
        lambda r: r["sex"] == "Female",
        "Cumulative change in HIV incidence,\nwomen (N)",
    )

    # Row 2: same plot, one panel per region
    region_axes = [ax_c, ax_d, ax_e, ax_f]
    for a, region in zip(region_axes, REGIONS):
        plot_group_bars(
            a,
            rows,
            lambda r, reg=region: r["region"] == reg,
            f"Cumulative change in HIV\nincidence, {region} Africa (N)",
            tick_fontsize=18,
        )

    # Row 3: cumulative % decrease bars (region x sex, region x STI)
    plot_pct_by_sex(ax_g, rows)
    plot_pct_by_sti(ax_h, rows)

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            fig_dir, "figure_future_un_sdg_averted_universal_access.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir, "figure_future_un_sdg_averted_universal_access.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )

    # ---- Supplemental: absolute incidence trajectory (linear), by sex ----
    fig2, ax2 = plt.subplots(1, 3, figsize=(30, 9))
    for a in ax2:
        _style_axis(a)
    for a, letter in zip(ax2, ["a", "b", "c"]):
        a.text(
            -0.18,
            1.08,
            letter,
            transform=a.transAxes,
            fontsize=30,
            fontweight="bold",
            va="top",
            ha="right",
        )

    plot_incidence_trajectory(
        ax2[0], rows, {"Male"}, "HIV incidence,\nheterosexual male (N)"
    )
    plot_incidence_trajectory(ax2[1], rows, {"MSM"}, "HIV incidence,\nMSM (N)")
    plot_incidence_trajectory(
        ax2[2], rows, {"Female"}, "HIV incidence,\nheterosexual female (N)"
    )

    fig2.tight_layout()
    fig2.savefig(
        os.path.join(fig_dir, "figure_supplemental_future_averted.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig2.savefig(
        os.path.join(fig_dir, "figure_supplemental_future_averted.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
