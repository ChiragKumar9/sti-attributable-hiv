import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml
from matplotlib import rc

from averted_burden.delta_method import (
    ci_via_log,
    ci_via_logit,
    logit_normal_leaf,
    point_mass,
)

output_dir = "outputs"
fig_dir = "figures"
params_path = "params.yml"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)

STIS = ["gc", "chlamydia", "syphilis", "trichomoniasis"]

COUNTRIES = ["South Africa", "Zimbabwe"]

# years available in the pickle to fit the trend on, the year the top row is
# extrapolated to, the single year the bottom row is computed at, and the
# year-pair the (held-constant) transmission rate is anchored on.
FIT_YEARS = list(range(2023, 2031))  # 2023-2030 inclusive
TARGET_YEAR = 2035
SINGLE_YEAR = 2025
ANCHOR_FROM = 2022
ANCHOR_TO = 2023

SEXES = ("Female", "Male", "MSM")

# ART/HIV treatment coverage used in the transmission-rate denominator. Set to
# the 95-95-95 target (0.95**3) as a scenario assumption for the untreated
# (still-transmitting) pool, rather than the actual anchor-year coverage.
TARGET_COVERAGE = 0.95**3

# the indirect (onward-transmission) contribution is accumulated over the number
# of years shown in the window: 2025-2035 inclusive = 11 years.
N_INDIRECT_YEARS = 11

# Test sensitivity for capturing asymptomatic STI infections (e.g. via
# screening rather than symptom-driven care-seeking), per STI.
TEST_SENSITIVITY = {
    "gc": 0.95,
    "chlamydia": 0.95,
    "syphilis": 0.95,
    "trichomoniasis": 0.95,
}

with open(params_path, "r") as _f:
    _params = yaml.safe_load(_f)

# fraction of cases presenting with symptoms, by STI and sex, from params.yml.
# MSM (anatomically male) uses the "male" fraction, since params.yml has no
# MSM-specific entry.
STI_SYMPTOMATIC_FRACTION = _params["sti_symptoms"]

# Wu et al., Lancet HIV 2024, MAIN scenario: % of HIV infections averted over
# the 10-year implementation vs an oral-PrEP-only baseline. (mean, lower, upper)
# in percent. NB: this is an AVERTED fraction under a specific modeled rollout
# (~1.6% / ~4.0% population coverage), not an attributable ceiling. The STI bars
# are direct + indirect attributable %. The LEN bar is a fixed 10-year-rollout
# number with no year dimension, so it is identical in both rows.
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

LEN_COLOR = "steelblue"
ALL_STI_COLOR = "dimgray"
sti_colors = {
    "gc": "firebrick",
    "chlamydia": "forestgreen",
    "syphilis": "purple",
    "trichomoniasis": "gold",
}

ALL_LABEL = "All STBIs"

ATTR_FIELD = "unaids_hiv_incidence_number_attributable_to_{sti}"
INC_FIELD = "unaids_incidence_number"
PREV_YE_FIELD = "unaids_prevalence_year_end_number"


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


def _logit(p):
    return np.log(p / (1.0 - p))


def _inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nom(x):
    return x.nominal_value if hasattr(x, "nominal_value") else x


def safe_ratio(num, den):
    """num / den, returning a zero UFloat when the denominator's nominal value
    is zero or non-finite -- same guard as the indirect-averted script, so the
    transmission rate reproduces that intended behaviour."""
    den_nom = den.nominal_value if hasattr(den, "nominal_value") else den
    if den_nom == 0 or not np.isfinite(den_nom):
        if hasattr(num, "nominal_value"):
            return num - num
        return 0.0
    return num / den


def captured_fraction(sti, sex):
    """Fraction of {sti} cases in {sex} that are captured: symptomatic cases
    (always identified) plus asymptomatic cases identified via testing at
    TEST_SENSITIVITY[sti]. MSM uses the "male" symptomatic fraction."""
    sex_key = "male" if sex in ("Male", "MSM") else "female"
    symptomatic = STI_SYMPTOMATIC_FRACTION[sti][sex_key]
    asymptomatic = 1.0 - symptomatic
    return symptomatic + asymptomatic * TEST_SENSITIVITY[sti]


def guard_ci_via_logit(uf, ctx):
    """ci_via_logit, but assert the nominal value is a valid (0, 1) proportion
    first. The direct + indirect quantity is no longer a pure attributable
    fraction and CAN in principle exceed 1 (onward transmission), where the
    logit transform is undefined -- fail loudly rather than silently."""
    nom = uf.nominal_value if hasattr(uf, "nominal_value") else uf
    if not (0.0 < nom < 1.0):
        raise ValueError(
            f"direct+indirect fraction {nom:.4f} outside (0, 1) for {ctx}; "
            "logit transform invalid (consider log space instead)"
        )
    return ci_via_logit(uf)


def _fit_extrapolate_logit(years, props, target):
    """Linear fit on the logit scale of one (year -> proportion) series,
    evaluated at `target` and mapped back through the inverse logit. Points not
    strictly inside (0, 1) have no finite logit and are dropped."""
    ys, vs = [], []
    for y, p in zip(years, props):
        if p is not None and np.isfinite(p) and 0.0 < p < 1.0:
            ys.append(float(y))
            vs.append(_logit(p))
    if len(ys) < 2:
        raise ValueError(
            f"Need >=2 usable points for logit-linear fit, got {len(ys)}"
        )
    slope, intercept = np.polyfit(ys, vs, 1)
    return float(_inv_logit(slope * target + intercept))


def build_country_index(rows_country):
    """(year, sex) -> row, for one country."""
    idx = {}
    for r in rows_country:
        idx[(r["year"], r["sex"])] = r
    return idx


def _get(idx, year, sex, field):
    row = idx.get((year, sex))
    if row is None:
        raise KeyError(f"missing row for year={year}, sex={sex}")
    return row[field]


def make_wu_factor(country):
    """(1 - Wu averted proportion) as a UFloat leaf (Wu CI folded in), one per
    country, reused across the three directional rates so they stay correlated
    through Wu."""
    m, lo, hi = WU_LEN[country]
    leaf = logit_normal_leaf(
        m / 100.0, lo / 100.0, hi / 100.0, f"wu_len_{country}"
    )
    if leaf is None:
        leaf = point_mass(m / 100.0, f"wu_len_{country}")
    return 1.0 - leaf  # type: ignore


def make_transmission_rates(idx, wu_factor):
    """Directional transmission rates anchored on ANCHOR_FROM -> ANCHOR_TO and
    held constant, scaled by the Wu factor. Definition matches the indirect
    script: recipient incidence (next year) over source untreated year-end
    prevalence (this year). The untreated share uses the fixed TARGET_COVERAGE
    (95-95-95) rather than the anchor-year ART coverage."""
    inc_F = _get(idx, ANCHOR_TO, "Female", INC_FIELD)
    inc_M = _get(idx, ANCHOR_TO, "Male", INC_FIELD)
    inc_S = _get(idx, ANCHOR_TO, "MSM", INC_FIELD)

    untreated_M = _get(idx, ANCHOR_FROM, "Male", PREV_YE_FIELD) * (
        1.0 - TARGET_COVERAGE
    )
    untreated_F = _get(idx, ANCHOR_FROM, "Female", PREV_YE_FIELD) * (
        1.0 - TARGET_COVERAGE
    )
    untreated_S = _get(idx, ANCHOR_FROM, "MSM", PREV_YE_FIELD) * (
        1.0 - TARGET_COVERAGE
    )

    return {
        "M_F": safe_ratio(inc_F, untreated_M) * wu_factor,
        "F_M": safe_ratio(inc_M, untreated_F) * wu_factor,
        "MSM": safe_ratio(inc_S, untreated_S) * wu_factor,
    }


def decompose_for_sti(idx, tr, year, sti):
    """Return (direct_part, indirect_part) UFloats for one STI at `year`, summed
    over sexes. Case counts are scaled by captured_fraction (symptomatic cases
    plus asymptomatic cases identified via testing) before being split into
    direct and indirect. Indirect is routed by the sex WHO HAS the STI to the
    recipient sex: M's STI cases -> F via tr_M_F, F's -> M via tr_F_M, MSM's ->
    MSM."""
    f = ATTR_FIELD.format(sti=sti)
    d_F = _get(idx, year, "Female", f) * captured_fraction(sti, "Female")
    d_M = _get(idx, year, "Male", f) * captured_fraction(sti, "Male")
    d_S = _get(idx, year, "MSM", f) * captured_fraction(sti, "MSM")

    direct = d_F + d_M + d_S
    indirect = (
        d_M * tr["M_F"] + d_F * tr["F_M"] + d_S * tr["MSM"]
    ) * N_INDIRECT_YEARS
    return direct, indirect


def combined_num_for_sti(idx, tr, year, sti):
    direct, indirect = decompose_for_sti(idx, tr, year, sti)
    return direct + indirect


def pooled_incidence(idx, year):
    return sum(_get(idx, year, s, INC_FIELD) for s in SEXES)


def bar_ufloat_for_sti(idx, tr, year, sti):
    return combined_num_for_sti(idx, tr, year, sti) / pooled_incidence(
        idx, year
    )


def bar_ufloat_all(idx, tr, year):
    num = sum(combined_num_for_sti(idx, tr, year, sti) for sti in STIS)
    return num / pooled_incidence(idx, year)


def print_transmission_diagnostics(idx, tr, country):
    """Print the (held-constant) directional transmission rates and, for each
    fit year, the direct vs indirect (transmission-driven) split of every bar,
    as a fraction of pooled incidence."""
    print(
        f"=== {country}: transmission rates "
        f"(anchored {ANCHOR_FROM}->{ANCHOR_TO}, "
        f"treat={TARGET_COVERAGE:.3f}) ==="
    )
    for key in ("M_F", "F_M", "MSM"):
        if _nom(tr[key]) > 0:
            m, lo, hi = ci_via_log(tr[key])
            print(f"  tr_{key}: {m:.4g} ({lo:.4g} - {hi:.4g})")
        else:
            print(f"  tr_{key}: {_nom(tr[key]):.4g} (degenerate)")

    for year in FIT_YEARS:
        if any((year, s) not in idx for s in SEXES):
            continue
        pooled = _nom(pooled_incidence(idx, year))

        dA = iA = None
        for sti in STIS:
            d, i = decompose_for_sti(idx, tr, year, sti)
            dA = d if dA is None else dA + d
            iA = i if iA is None else iA + i
        dAn, iAn = _nom(dA) / pooled, _nom(iA) / pooled  # type: ignore
        print(
            f"  {year} All STIs: direct={100 * dAn:.2f}% "
            f"indirect={100 * iAn:.2f}% total={100 * (dAn + iAn):.2f}%"
        )
        for sti in STIS:
            d, i = decompose_for_sti(idx, tr, year, sti)
            dn, iN = _nom(d) / pooled, _nom(i) / pooled
            share = 100 * iN / (dn + iN) if (dn + iN) != 0 else float("nan")
            print(
                f"    {year} {sti}: direct={100 * dn:.2f}% "
                f"indirect={100 * iN:.2f}% total={100 * (dn + iN):.2f}% "
                f"(indirect share={share:.1f}%)"
            )


def bars_single_year(idx, tr, country, year):
    """Bottom-row bars: propagate the combined direct+indirect UFloats at a
    single year and extract intervals once (logit, guarded)."""
    bars = [("Lenacapavir", LEN_COLOR, WU_LEN[country])]

    m, lo, hi = guard_ci_via_logit(
        bar_ufloat_all(idx, tr, year), f"{country} All {year}"
    )
    bars.append(
        (ALL_LABEL, ALL_STI_COLOR, (m * 100, lo * 100, hi * 100))  # type: ignore
    )
    for sti in STIS:
        m, lo, hi = guard_ci_via_logit(
            bar_ufloat_for_sti(idx, tr, year, sti), f"{country} {sti} {year}"
        )
        bars.append(
            (sti_names[sti], sti_colors[sti], (m * 100, lo * 100, hi * 100))  # type: ignore
        )
    return bars


def bars_extrapolated(idx, tr, country):
    """Top-row bars: extract the combined (mean, lower, upper) at each FIT_YEAR,
    fit logit-linear, extrapolate to TARGET_YEAR. All-STIs mean = sum of the
    four extrapolated STI means; All bounds come from its own combined-series
    fit (so the All mean is generally NOT centred within its own bounds)."""
    series = {ALL_LABEL: {"year": [], "m": [], "lo": [], "hi": []}}
    for sti in STIS:
        series[sti] = {"year": [], "m": [], "lo": [], "hi": []}

    for year in FIT_YEARS:
        if any((year, s) not in idx for s in SEXES):
            continue
        for sti in STIS:
            m, lo, hi = guard_ci_via_logit(
                bar_ufloat_for_sti(idx, tr, year, sti),
                f"{country} {sti} {year}",
            )
            series[sti]["year"].append(year)
            series[sti]["m"].append(m)
            series[sti]["lo"].append(lo)
            series[sti]["hi"].append(hi)
        m, lo, hi = guard_ci_via_logit(
            bar_ufloat_all(idx, tr, year), f"{country} All {year}"
        )
        series[ALL_LABEL]["year"].append(year)
        series[ALL_LABEL]["m"].append(m)
        series[ALL_LABEL]["lo"].append(lo)
        series[ALL_LABEL]["hi"].append(hi)

    extrap = {}
    sti_mean_sum = 0.0
    for sti in STIS:
        s = series[sti]
        m = _fit_extrapolate_logit(s["year"], s["m"], TARGET_YEAR)
        lo = _fit_extrapolate_logit(s["year"], s["lo"], TARGET_YEAR)
        hi = _fit_extrapolate_logit(s["year"], s["hi"], TARGET_YEAR)
        extrap[sti] = (m, lo, hi)
        sti_mean_sum += m

    s_all = series[ALL_LABEL]
    all_lo = _fit_extrapolate_logit(s_all["year"], s_all["lo"], TARGET_YEAR)
    all_hi = _fit_extrapolate_logit(s_all["year"], s_all["hi"], TARGET_YEAR)
    extrap[ALL_LABEL] = (sti_mean_sum, all_lo, all_hi)

    bars = [("Lenacapavir", LEN_COLOR, WU_LEN[country])]
    bars.append(
        (ALL_LABEL, ALL_STI_COLOR, tuple(v * 100 for v in extrap[ALL_LABEL]))
    )
    for sti in STIS:
        bars.append(
            (
                sti_names[sti],
                sti_colors[sti],
                tuple(v * 100 for v in extrap[sti]),
            )
        )
    return bars


def plot_bars(ax, country, bars, header):
    x = np.arange(len(bars))
    tops = []
    print(f"{country} [{header}]:")
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
    ax.set_ylabel(f"Reduction in HIV incidence,\n{country} (%)")
    ax.set_ylim(0, max(tops) * 1.12)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


def make_figure(rows, apply_wu_to_tr, out_stem):
    """Build and save the 2x2 figure. When apply_wu_to_tr is True the indirect
    transmission rate is attenuated by the Wu (1 - LEN reduction) factor (the
    main figure); when False the rate is left un-attenuated (wu_factor = 1.0),
    for the supplemental figure. The Lenacapavir bar (WU_LEN) is unaffected
    either way."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    for a in axes.flat:
        _style_axis(a)

    # per-row scenario labels (a = top row / TARGET_YEAR, b = bottom row / SINGLE_YEAR)
    axes[0][0].text(
        -0.15,
        1.08,
        "a) STBI prevalence trends continue",
        transform=axes[0][0].transAxes,
        fontsize=24,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    axes[1][0].text(
        -0.15,
        1.08,
        "b) STBI prevalence remains at 2025 levels",
        transform=axes[1][0].transAxes,
        fontsize=24,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

    for col, country in enumerate(COUNTRIES):
        rows_country = [r for r in rows if r["location"] == country]
        if not rows_country:
            raise ValueError(f"No rows found for location == {country!r}")
        idx = build_country_index(rows_country)
        wu_factor = make_wu_factor(country) if apply_wu_to_tr else 1.0
        tr = make_transmission_rates(idx, wu_factor)

        print_transmission_diagnostics(idx, tr, country)

        plot_bars(
            axes[0][col],
            country,
            bars_extrapolated(idx, tr, country),
            f"extrapolated to {TARGET_YEAR}",
        )
        plot_bars(
            axes[1][col],
            country,
            bars_single_year(idx, tr, country, SINGLE_YEAR),
            f"{SINGLE_YEAR}",
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, out_stem + ".png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, out_stem + ".pdf"),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    with open(
        os.path.join(output_dir, "hiv_attributable_to_stis.ufloat.pkl"), "rb"
    ) as f:
        rows = pickle.load(f)
    keep_years = set(FIT_YEARS) | {ANCHOR_FROM, ANCHOR_TO, SINGLE_YEAR}
    rows = [r for r in rows if r["year"] in keep_years]

    make_figure(rows, True, "figure_len_vs_sti_by_country")
    make_figure(
        rows,
        False,
        "figure_supplemental_len_vs_sti_by_country_no_indirect_attenuation",
    )
