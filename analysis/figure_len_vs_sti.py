import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib import rc
from matplotlib.patches import Patch

from averted_burden.delta_method import (
    ci_via_log,
    ci_via_logit,
    logit_normal_leaf,
    lognormal_leaf,
    point_mass,
)

output_dir = "outputs"
fig_dir = "figures"
data_dir = "data"
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

# the indirect (onward-transmission) contribution is accumulated over the 10-year
# projection window 2026-2035 (2025 is the "remains at 2025 levels" base year),
# matching the Wu 10-year lenacapavir rollout.
N_INDIRECT_YEARS = 10

# Representative annual cohort year for the DALY panels (c/d): the midpoint of the
# 2026-2035 window. Under "trends continue" the 10-year cumulative is approximated
# as 10 x this representative year's annual flow (see the c/d panel construction).
DALY_REP_YEAR = 2030

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

# GBD 2023 DALY burden (as lognormal leaves, UI propagated) and HIV incidence are
# read by read_daly_weights() for the c/d panels, assigned once it is defined below.

# Wu et al., Lancet HIV 2024, MAIN scenario: % of HIV infections averted over
# the 10-year implementation vs an oral-PrEP-only baseline. (mean, lower, upper)
# in percent. NB: this is an AVERTED fraction under a specific modeled rollout
# concentrated in key populations, not an attributable ceiling. The LEN bar is a
# fixed 10-year-rollout number with no year dimension, so it is identical in the
# "trends continue" and "2025 levels" bars.
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
sti_colors = {
    "gc": "firebrick",
    "chlamydia": "forestgreen",
    "syphilis": "purple",
    "trichomoniasis": "gold",
}

# sexes with sex-split DALY / incidence in the GBD extract (no MSM category there).
STI_DALY_SEXES = ("Female", "Male")

# GBD cause labels in the DALYs extract, keyed by our internal STI code (+ HIV).
DALY_CSV = "IHME-GBD_2023_DATA-DALYs.csv"
DALY_CAUSE_NAME = {
    "gc": "Gonococcal infection",
    "chlamydia": "Chlamydial infection",
    "syphilis": "Syphilis",
    "trichomoniasis": "Trichomoniasis",
}
HIV_CAUSE = "HIV/AIDS"

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
    ax.tick_params(which="major", length=5.25, width=1.2)
    ax.tick_params(which="minor", length=3.0, width=0.9)
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    ax.set_axisbelow(True)


def _logit(p):
    return np.log(p / (1.0 - p))


def _inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


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


def captured_fraction(sti, sex, poc_coverage):
    """Fraction of {sti} cases in {sex} that are captured (and thus treatable):
    symptomatic cases (always identified) plus asymptomatic cases identified via
    POC testing -- reached in a `poc_coverage` fraction of the asymptomatic pool,
    each at TEST_SENSITIVITY[sti]. poc_coverage = 1.0 reproduces universal
    asymptomatic screening. MSM uses the "male" symptomatic fraction."""
    sex_key = "male" if sex in ("Male", "MSM") else "female"
    symptomatic = STI_SYMPTOMATIC_FRACTION[sti][sex_key]
    asymptomatic = 1.0 - symptomatic
    return symptomatic + asymptomatic * poc_coverage * TEST_SENSITIVITY[sti]


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


def len_leaf(country):
    """LEN averted-fraction UFloat leaf (Wu CI folded in), one per country. Used
    both as the LEN stack segment and, via (1 - leaf), as the inclusion-exclusion
    de-overlap factor on the STI segments so the stacked total is the
    (non-double-counted) union of the two reductions."""
    m, lo, hi = WU_LEN[country]
    leaf = logit_normal_leaf(
        m / 100.0, lo / 100.0, hi / 100.0, f"wu_len_{country}"
    )
    if leaf is None:
        leaf = point_mass(m / 100.0, f"wu_len_{country}")
    return leaf


def make_transmission_rates(idx):
    """Directional (raw) transmission rates anchored on ANCHOR_FROM -> ANCHOR_TO
    and held constant. Definition matches the indirect script: recipient incidence
    (next year) over source untreated year-end prevalence (this year). The
    untreated share uses the fixed TARGET_COVERAGE (95-95-95). NB: no Wu
    attenuation -- the LEN/STI overlap correction lives at the bar level (the
    (1 - len_leaf) discount on the STI segments)."""
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
        "M_F": safe_ratio(inc_F, untreated_M),
        "F_M": safe_ratio(inc_M, untreated_F),
        "MSM": safe_ratio(inc_S, untreated_S),
    }


def combined_by_recipient_sex(idx, tr, year, sti, poc_coverage):
    """Attributable-HIV (direct + indirect) for one STI at `year`, split by the sex
    that ACQUIRES the HIV (the recipient), so sex-specific DALY weights can be
    applied. Case counts are scaled by captured_fraction (symptomatic cases plus
    asymptomatic cases reached by POC screening). Direct infections accrue to the
    sex who has the STI; indirect (onward transmission, accumulated over
    N_INDIRECT_YEARS) is routed to the recipient sex: M's STI cases -> F via tr_M_F,
    F's -> M via tr_F_M, MSM's -> MSM. This is the single source of the sex-routing
    logic (combined_num_for_sti and the c/d HIV bar both derive from it), so panels
    a/b and c/d cannot drift apart."""
    f = ATTR_FIELD.format(sti=sti)
    d_F = _get(idx, year, "Female", f) * captured_fraction(
        sti, "Female", poc_coverage
    )
    d_M = _get(idx, year, "Male", f) * captured_fraction(
        sti, "Male", poc_coverage
    )
    d_S = _get(idx, year, "MSM", f) * captured_fraction(
        sti, "MSM", poc_coverage
    )
    return {
        "Female": d_F + d_M * tr["M_F"] * N_INDIRECT_YEARS,
        "Male": d_M + d_F * tr["F_M"] * N_INDIRECT_YEARS,
        "MSM": d_S + d_S * tr["MSM"] * N_INDIRECT_YEARS,
    }


def combined_num_for_sti(idx, tr, year, sti, poc_coverage):
    return sum(
        combined_by_recipient_sex(idx, tr, year, sti, poc_coverage).values()
    )


def pooled_incidence(idx, year):
    return sum(_get(idx, year, s, INC_FIELD) for s in SEXES)


def bar_ufloat_for_sti(idx, tr, year, sti, poc_coverage):
    """STI-attributable HIV as a fraction of pooled incidence at `year` (S_sti,
    before the LEN de-overlap)."""
    return combined_num_for_sti(
        idx, tr, year, sti, poc_coverage
    ) / pooled_incidence(idx, year)


def extrap_sti_leaf(idx, tr, country, sti, poc_coverage):
    """ "Trends continue" S_sti extrapolated to TARGET_YEAR: fit logit-linear to
    the per-year (mean, lower, upper), then rebuild a logit-normal UFloat leaf
    from the extrapolated interval so it composes with len_leaf downstream."""
    ys, ms, los, his = [], [], [], []
    for year in FIT_YEARS:
        if any((year, s) not in idx for s in SEXES):
            continue
        m, lo, hi = guard_ci_via_logit(
            bar_ufloat_for_sti(idx, tr, year, sti, poc_coverage),
            f"{country} {sti} {year}",
        )
        ys.append(year)
        ms.append(m)
        los.append(lo)
        his.append(hi)
    m = _fit_extrapolate_logit(ys, ms, TARGET_YEAR)
    lo = _fit_extrapolate_logit(ys, los, TARGET_YEAR)
    hi = _fit_extrapolate_logit(ys, his, TARGET_YEAR)
    leaf = logit_normal_leaf(m, lo, hi, f"extrap_{country}_{sti}")
    if leaf is None:
        leaf = point_mass(m, f"extrap_{country}_{sti}")
    return leaf


def hiv_bars(idx, tr, country, poc_coverage):
    """Panels a/b: two stacked bars (LEN + 4 STI segments). Each STI segment is
    S_sti * (1 - len_leaf); the LEN segment is len_leaf; the total is their union.
    Returns [(xlabel, [(seglabel, color, ufloat), ...], total_ufloat), ...]."""
    leaf = len_leaf(country)
    scenarios = [
        (
            "CSTI4 prevalence\ntrends continue",
            lambda sti: extrap_sti_leaf(idx, tr, country, sti, poc_coverage),
        ),
        (
            "CSTI4 prevalence\nremains at 2025 levels",
            lambda sti: bar_ufloat_for_sti(
                idx, tr, SINGLE_YEAR, sti, poc_coverage
            ),
        ),
    ]
    bars = []
    for xlabel, s_getter in scenarios:
        segs = [("Lenacapavir", LEN_COLOR, leaf)]
        total = leaf
        for sti in STIS:
            seg = s_getter(sti) * (1 - leaf)  # type: ignore
            segs.append((sti_names[sti], sti_colors[sti], seg))
            total = total + seg
        bars.append((xlabel, segs, total))
    return bars


def read_daly_weights():
    """GBD 2023 per-cause DALY burden and HIV incidence for the c/d panels, by
    country and SEX. Returns (hiv_daly_leaf[country][sex], hiv_inc_val[country][sex],
    sti_daly_leaf[country][sti][sex]).

    DALY Numbers are returned as lognormal UFloat LEAVES (GBD 95% UI folded in) so
    the DALY uncertainty propagates into the c/d error bars. How each bar uses them:
      Bar 1 (STI morbidity) = captured_fraction x total STI DALYs. A per-case weight
        DALY/Inc times an incident case count would put the SAME GBD incidence in
        numerator and denominator, so it cancels -- only the DALY burden and its CI
        remain, and no STI incidence is needed at all.
      Bar 2 (HIV co-benefit) = attributable-HIV x HIV DALY / HIV incidence. The
        attributable-HIV leaf is UNAIDS/PAF-derived (independent of GBD), so there is
        no incidence to cancel: the HIV DALY CI is propagated as its own leaf and the
        GBD HIV incidence denominator is held as a point (its UI is narrower and it
        only normalizes to a per-infection basis).
    Weights are sex-specific: the per-case burden is sharply sex-skewed for the
    female-dominated STI sequelae (trichomoniasis, chlamydia) and materially
    different for HIV (male > female in both countries). CAVEAT: annual DALY Number
    / annual Incidence Number mixes the prevalent burden (YLD+YLL of the whole
    epidemic) with a single year's incident cases, so the per-incident-case HIV
    weight is overstated where prevalence >> incidence (most extreme for Zimbabwe)."""
    df = pl.read_csv(os.path.join(data_dir, DALY_CSV)).filter(
        pl.col("metric") == "Number"
    )
    rows = {
        (r["measure"][:4], r["location"], r["cause"], r["sex"]): r
        for r in df.iter_rows(named=True)
    }

    def daly_leaf(country, cause, sex):
        r = rows[("DALY", country, cause, sex)]
        return lognormal_leaf(
            r["val"], r["lower"], r["upper"], f"daly_{cause}_{country}_{sex}"
        )

    hiv_daly_leaf = {
        c: {sex: daly_leaf(c, HIV_CAUSE, sex) for sex in STI_DALY_SEXES}
        for c in COUNTRIES
    }
    hiv_inc_val = {
        c: {
            sex: rows[("Inci", c, HIV_CAUSE, sex)]["val"]
            for sex in STI_DALY_SEXES
        }
        for c in COUNTRIES
    }
    sti_daly_leaf = {
        c: {
            sti: {
                sex: daly_leaf(c, DALY_CAUSE_NAME[sti], sex)
                for sex in STI_DALY_SEXES
            }
            for sti in STIS
        }
        for c in COUNTRIES
    }
    return hiv_daly_leaf, hiv_inc_val, sti_daly_leaf


HIV_DALY_LEAF, HIV_INC_VAL, STI_DALY_LEAF = read_daly_weights()


def sti_daly_parts(country, poc_coverage):
    """Panel c/d Bar 1: STI-morbidity DALYs from treating ALL captured CSTI4
    infections, per-STI UFloats. captured_fraction x total GBD STI DALYs (a
    lognormal leaf carrying the GBD UI), summed over sexes, x N_INDIRECT_YEARS
    (10-yr horizon). The incidence a per-case weight would carry cancels against the
    case count, so this is simply the captured share of the DALY burden."""
    parts = []
    for sti in STIS:
        s = None
        for sex in STI_DALY_SEXES:
            contrib = (
                captured_fraction(sti, sex, poc_coverage)
                * STI_DALY_LEAF[country][sti][sex]
            )
            s = contrib if s is None else s + contrib
        s = s * N_INDIRECT_YEARS  # type: ignore
        parts.append((sti_names[sti], sti_colors[sti], s))
    return parts


def hiv_daly_parts(idx, tr, country, poc_coverage):
    """Panel c/d Bar 2: HIV DALYs from averted STI-attributable HIV, per-STI
    UFloats. The averted HIV is split by RECIPIENT sex (via
    combined_by_recipient_sex) and each stream is multiplied by its sex-specific HIV
    DALY leaf (GBD UI propagated) over the GBD HIV incidence denominator (held as a
    point); MSM recipients use the male weight (GBD has no MSM category). One
    representative year's full flow (direct + onward-window indirect) is scaled by
    N_INDIRECT_YEARS (the separate count of annual rollout cohorts, 2026-2035) for
    the 10-yr cumulative, and de-overlapped against LEN by the shared (1 - len_leaf)
    factor (matching panels a/b) -- the HIV co-benefit of STI treatment on top of
    the LEN rollout."""
    leaf = len_leaf(country)
    parts = []
    for sti in STIS:
        by_sex = combined_by_recipient_sex(
            idx, tr, DALY_REP_YEAR, sti, poc_coverage
        )
        hiv = None
        for sex, combined in by_sex.items():
            w = "Male" if sex == "MSM" else sex
            term = (
                combined * HIV_DALY_LEAF[country][w] / HIV_INC_VAL[country][w]
            )
            hiv = term if hiv is None else hiv + term
        hiv_dalys = N_INDIRECT_YEARS * (1 - leaf) * hiv  # type: ignore
        parts.append((sti_names[sti], sti_colors[sti], hiv_dalys))
    return parts


def draw_panel(ax, letter, bars, ylabel, ci_fn, scale):
    """Draw one stacked-bar panel and print every segment and total with its own
    CI. `bars` = [(xlabel, [(seglabel, color, ufloat), ...], total_ufloat), ...].
    `ci_fn` extracts (mean, lower, upper) on the appropriate scale (logit for
    fractions, log for counts). A framed legend built from the panel's own
    segments is floated just above the panel's top-left corner."""
    print(f"panel {letter}) [{ylabel.splitlines()[0]} ...]:")
    tops = []
    for i, (xlabel, segs, total) in enumerate(bars):
        bottom = 0.0
        label_flat = xlabel.replace("\n", " ")
        print(f"  bar: {label_flat}")
        for seglabel, color, uf in segs:
            m, lo, hi = ci_fn(uf)
            ax.bar(i, m * scale, bottom=bottom, width=0.7, color=color)
            bottom += m * scale
            print(
                f"    {seglabel}: {m * scale:.3f} "
                f"({lo * scale:.3f} - {hi * scale:.3f})"
            )
        tm, tlo, thi = ci_fn(total)
        ax.errorbar(
            i,
            tm * scale,
            yerr=[[(tm - tlo) * scale], [(thi - tm) * scale]],
            fmt="none",
            ecolor="black",
            capsize=12,
            capthick=2.5,
            elinewidth=2.5,
        )
        tops.append(thi * scale)
        print(
            f"    TOTAL: {tm * scale:.3f} ({tlo * scale:.3f} - {thi * scale:.3f})"
        )

    ax.set_xticks(np.arange(len(bars)))  # type: ignore
    ax.set_xticklabels([b[0] for b in bars], rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    # minimal top padding: the top of the tallest error bar sits just below the
    # frame, so the legend floated just above the frame reads as sitting right on
    # top of the bars rather than in a large empty band.
    ax.set_ylim(0, max(tops) * 1.03)

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=seglabel)
        for seglabel, color, _uf in bars[0][1]
    ]
    # float the legend just above the panel's top-left corner so it clears the
    # error bars without inflating the y-axis
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.97),
        edgecolor="black",
    )

    ax.text(
        -0.15,
        1.05,
        f"{letter})",
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def make_figure(rows, poc_coverage, out_stem):
    """Build and save the 2x2 figure. Top row (a, b): HIV-incidence reduction as
    stacked LEN + CSTI4 bars per country. Bottom row (c, d): DALYs (STI morbidity
    vs averted-HIV) per country. `poc_coverage` sets the asymptomatic POC
    screening reach (1.0 main figure, 0.5 supplemental)."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    for a in axes.flat:
        _style_axis(a)

    letters = [["a", "b"], ["c", "d"]]
    for col, country in enumerate(COUNTRIES):
        rows_country = [r for r in rows if r["location"] == country]
        if not rows_country:
            raise ValueError(f"No rows found for location == {country!r}")
        idx = build_country_index(rows_country)
        tr = make_transmission_rates(idx)

        draw_panel(
            axes[0][col],
            letters[0][col],
            hiv_bars(idx, tr, country, poc_coverage),
            f"Reduction in cumulative HIV incidence\nby 2035, {country} (%)",
            ci_via_logit,
            scale=100.0,
        )

        daly_bars = [
            (
                "Treating CSTI4s",
                sti_daly_parts(country, poc_coverage),
                None,
            ),
            (
                "Averting HIV",
                hiv_daly_parts(idx, tr, country, poc_coverage),
                None,
            ),
        ]
        # total for each DALY bar = sum of its per-STI parts
        daly_bars = [
            (lbl, parts, _sum_parts(parts)) for (lbl, parts, _) in daly_bars
        ]
        draw_panel(
            axes[1][col],
            letters[1][col],
            daly_bars,
            f"DALYs averted to 2035,\n{country} (thousands)",
            ci_via_log,
            scale=1e-3,
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


def _sum_parts(parts):
    total = None
    for _, _, uf in parts:
        total = uf if total is None else total + uf
    return total


if __name__ == "__main__":
    with open(
        os.path.join(output_dir, "hiv_attributable_to_stis.ufloat.pkl"), "rb"
    ) as f:
        rows = pickle.load(f)
    keep_years = set(FIT_YEARS) | {ANCHOR_FROM, ANCHOR_TO, SINGLE_YEAR}
    rows = [r for r in rows if r["year"] in keep_years]

    make_figure(rows, 1.0, "figure_len_vs_sti_by_country")
    make_figure(
        rows,
        0.5,
        "figure_supplemental_len_vs_sti_by_country_half_poc_asymptomatic",
    )
