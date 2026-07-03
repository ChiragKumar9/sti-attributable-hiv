import os
import pickle

import numpy as np
import polars as pl
import yaml
from tqdm.auto import tqdm

from averted_burden import conditional_exposure
from averted_burden.delta_method import (
    LeafCache,
    ci_via_log,
    ci_via_logit,
    logit_normal_leaf,
    lognormal_leaf,
    point_mass,
)

data_dir = "data"
output_dir = "outputs"

STIS = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
ABX = ["Ciprofloxacin", "Cefixime", "Azithromycin", "Ceftriaxone"]
SEXES = ("Male", "Female", "MSM")

with open("params.yml", "r") as f:
    params = yaml.safe_load(f)
sti_symptom_params = params["sti_symptoms"]


# ----------------------------------------------------------------------
# UFloat helpers
# ----------------------------------------------------------------------
def umax0(x):
    """max(x, 0) for a UFloat -- a real sub-gradient kink, same semantics as a
    np.clip(lower_bound=0)."""
    if not hasattr(x, "nominal_value"):
        return max(x, 0.0)
    if x.nominal_value >= 0:
        return x
    return x - x  # exact zero, including zero variance


def safe_ratio(num, den):
    """num / den, returning a zero UFloat when the denominator's nominal value
    is zero or non-finite.

    The original computed transmission rates with polars division and then ran
    `.fill_nan(0).replace([inf, -inf], 0)` on the resulting columns -- i.e. a
    zero treatment-adjusted prevalence (or zero prevalence) was treated as a
    zero effective transmission rate. UFloat division by a zero-nominal
    denominator raises ZeroDivisionError instead of yielding inf, so we guard
    explicitly here to reproduce that intended behaviour.

    Still used for the transmission-rate denominators below; the scenario
    direct-averted burdens are no longer reconstructed by division (they are
    read straight from the attributable pickle as exact
    difference-of-counterfactuals), so the earlier attrib/(treatment factor)
    calls are gone.
    """
    den_nom = den.nominal_value if hasattr(den, "nominal_value") else den
    if den_nom == 0 or not np.isfinite(den_nom):
        # zero, with zero variance, regardless of whether num/den are UFloats
        if hasattr(num, "nominal_value"):
            return num - num
        return 0.0
    return num / den


# ----------------------------------------------------------------------
# HIV -> AIDS progression curve.
#
# The monotone-accumulate / conditionalise / repeat / sqrt preprocessing is
# IDENTICAL to the original and is done entirely in float space on the three
# reported survival arrays. Only afterwards is each per-year stage lifted to a
# leaf.
#
# HIV_TO_AIDS_CORRELATION = "independent"  (chosen):
#   each per-year stage is an independent logit_normal_leaf. Built once, globally
#   (the curve is a single published meta-analysis input, identical for every
#   location/scenario), so the SAME stage object is reused everywhere -- the
#   stages are independent of each other but each stage is perfectly shared
#   across locations and scenarios.
#
# To switch to the rho=1 ("matches original bound-carrying") model instead,
# build one shared standard-normal latent z and express each stage as a logit
# cluster member off that single z. A `logit_cluster_member` helper would need
# adding to delta_method.py; the hook is marked below.
# ----------------------------------------------------------------------
HIV_TO_AIDS_CORRELATION = "independent"


def _preprocess_hiv_to_aids(arr):
    """Reproduce the original float preprocessing for one survival array."""
    arr = np.minimum.accumulate(arr)
    arr = np.concatenate([[arr[0]], arr[1:] / arr[:-1]])  # conditionalise
    arr = np.repeat(arr, 2)  # 2-year chunks -> per year (duplicated)
    arr = np.sqrt(arr)  # 2-year survival -> 1-year survival
    return arr


def build_hiv_to_aids_curve(
    hiv_to_aids=np.array([0.82, 0.72, 0.64, 0.57, 0.26, 0.19, 0.0]),
    hiv_to_aids_upper=np.array([0.95, 0.88, 0.84, 0.89, 0.50, 0.48, 0.0]),
    hiv_to_aids_lower=np.array([0.70, 0.56, 0.44, 0.25, 0.03, 0.09, 0.0]),
):
    m = _preprocess_hiv_to_aids(hiv_to_aids)
    hi = _preprocess_hiv_to_aids(hiv_to_aids_upper)
    lo = _preprocess_hiv_to_aids(hiv_to_aids_lower)

    curve = []
    for i in range(len(m)):
        mean_i = float(m[i])
        # bounds may be out of [mean] order at the tails after the float
        # preprocessing; clamp defensively for the leaf builder.
        lo_i, hi_i = float(min(lo[i], m[i])), float(max(hi[i], m[i]))
        if mean_i <= 0:
            curve.append(point_mass(0.0, f"hiv2aids_{i}"))
            continue
        leaf = logit_normal_leaf(mean_i, lo_i, hi_i, f"hiv2aids_{i}")
        if leaf is None:
            leaf = point_mass(mean_i, f"hiv2aids_{i}")
        curve.append(leaf)
        # rho=1 hook: replace the two lines above with
        #   curve.append(logit_cluster_member(mean_i, lo_i, hi_i, z_aids))
        # where z_aids = new_cluster_latent("z_hiv2aids") is built once.
    return curve


HIV_TO_AIDS_CURVE = build_hiv_to_aids_curve()


# ----------------------------------------------------------------------
# Load the pickled, correlation-preserving UFloat rows from the attributable
# script (one aggregated row per country/location/region/sex/year). These now
# carry the exact direct-averted burdens (direct_ub_*, direct_2016_gc,
# direct_untreated_gc, direct_resistance_gc) computed upstream.
# ----------------------------------------------------------------------
with open(
    os.path.join(output_dir, "hiv_attributable_to_stis.ufloat.pkl"), "rb"
) as f:
    hiv_rows = pickle.load(f)

# transmission-increase RRs (HIV detection given STI). Positive, unbounded ->
# lognormal leaves, cached by (sti, sex-category).
sti_transmission_increase = pl.read_csv(
    os.path.join(output_dir, "meta_estimated_RRs_HIV_detection_given_STI.csv")
)
bacterial_names = {
    "Gonorrhea": "gc",
    "Chlamydia": "chlamydia",
    "Syphilis": "syphilis",
    "Trichomoniasis": "trichomoniasis",
}
sti_transmission_increase = sti_transmission_increase.with_columns(
    bacteria=pl.col("bacteria").replace(bacterial_names)
)

_increase_cache = LeafCache(lognormal_leaf)


def get_increase(sti, sex_cat):
    """sex_cat is 'Both' (used for Male/MSM) or 'Women' (used for Female)."""
    rows = sti_transmission_increase.filter(
        (pl.col("bacteria") == sti) & (pl.col("sex") == sex_cat)
    )
    return _increase_cache.get(
        (sti, sex_cat),
        rows["rr_mu"].item(),
        rows["rr_lower"].item(),
        rows["rr_upper"].item(),
        f"sti_increase_{sti}_{sex_cat}",
    )


# ----------------------------------------------------------------------
# Forward-transmission simulation, delta-method version.
#
# `data` maps (year, sex) -> dict with UFloat fields:
#   inc      : unaids_incidence_number
#   prev_ye  : unaids_prevalence_year_end_number
#   treat    : treatment_proportion          (UFloat in [0,1])
#   paf      : paf_{sti}_hiv                  (GBD paf, UFloat in [0,1])
#   direct   : the scenario's direct averted  (UFloat count)
# Returns indirect[(year, sex)] for year in years[1:].
# ----------------------------------------------------------------------
def calculate_indirect_averted_cases(
    years, data, inc_increase_male, inc_increase_female
):
    years = sorted(years)
    n = len(years)
    L = len(HIV_TO_AIDS_CURVE)

    def g(y, s, k):
        d = data.get((y, s))
        if d is None or d.get(k) is None:
            return point_mass(0.0, f"missing_{k}_{y}_{s}")
        return d[k]

    # transmission rates: next year's incidence over this year's transmitting
    # (untreated) prevalence. `safe_ratio` reproduces the original inf/nan -> 0.
    tr_M_F, tr_F_M, tr_MSM = {}, {}, {}
    for i, y in enumerate(years):
        if i == n - 1:
            break
        yn = years[i + 1]
        tr_M_F[y] = safe_ratio(
            g(yn, "Female", "inc"),
            g(y, "Male", "prev_ye") * (1 - g(y, "Male", "treat")),  # type: ignore
        )
        tr_F_M[y] = safe_ratio(
            g(yn, "Male", "inc"),
            g(y, "Female", "prev_ye") * (1 - g(y, "Female", "treat")),  # type: ignore
        )
        tr_MSM[y] = safe_ratio(
            g(yn, "MSM", "inc"),
            g(y, "MSM", "prev_ye") * (1 - g(y, "MSM", "treat")),  # type: ignore
        )

    curve = HIV_TO_AIDS_CURVE

    # Age-indexed transmitting pools: pool_*[a] holds the cohort now aged `a`
    # (0..L). Each cohort is decayed by its OWN age via the survival curve --
    # keyed to age, not calendar year -- so scenarios longer than the curve stay
    # correct. A cohort drops out once it would age past the curve (all
    # progressed to AIDS / dead by then). Only DIRECT cohorts enter the pool;
    # the secondary (indirect) cases they generate are recorded but do NOT
    # themselves transmit onward.
    def age_pool(pool):
        # cohort at age a survives to age a+1 with factor curve[a]; age-L drops
        return [point_mass(0.0, "age0")] + [
            pool[a] * curve[a] for a in range(L)
        ]

    pool_m = [point_mass(0.0, f"pm_{k}") for k in range(L + 1)]
    pool_f = [point_mass(0.0, f"pf_{k}") for k in range(L + 1)]
    pool_s = [point_mass(0.0, f"ps_{k}") for k in range(L + 1)]

    indirect = {}
    for idx in range(n - 1):
        y = years[idx]
        yn = years[idx + 1]

        # 1. age last year's survivors by one year
        pool_m = age_pool(pool_m)
        pool_f = age_pool(pool_f)
        pool_s = age_pool(pool_s)

        # 2. age-0 entrants: this year's fresh DIRECT averted cohort only.
        #    Only the untreated (non-viral-suppressed) share keeps transmitting.
        pool_m[0] = pool_m[0] + g(y, "Male", "direct") * (
            1 - g(y, "Male", "treat")
        )  # type: ignore
        pool_f[0] = pool_f[0] + g(y, "Female", "direct") * (
            1 - g(y, "Female", "treat")
        )  # type: ignore
        pool_s[0] = pool_s[0] + g(y, "MSM", "direct") * (
            1 - g(y, "MSM", "treat")
        )  # type: ignore

        # 3. total alive transmitting pool per sex (includes age-0)
        tot_m = sum(pool_m)  # type: ignore
        tot_f = sum(pool_f)  # type: ignore
        tot_s = sum(pool_s)  # type: ignore

        # 4. secondary infections generated for next year, by source sex.
        #    Recorded as indirectly-averted infections; NOT fed back into the
        #    pool (secondary cases do not transmit onward).
        indirect[(yn, "Female")] = tot_m * conditional_exposure.p_a_given_b(
            tr_M_F[y],
            g(y, "Male", "paf"),
            inc_increase_male,
            allow_invalid=True,
        )  # type: ignore
        indirect[(yn, "Male")] = tot_f * conditional_exposure.p_a_given_b(
            tr_F_M[y],
            g(y, "Female", "paf"),
            inc_increase_female,
            allow_invalid=True,
        )  # type: ignore
        indirect[(yn, "MSM")] = tot_s * conditional_exposure.p_a_given_b(
            tr_MSM[y],
            g(y, "MSM", "paf"),
            inc_increase_male,
            allow_invalid=True,
        )  # type: ignore

    return indirect


# ----------------------------------------------------------------------
# Build a per-location view of the merged UFloat rows and run each scenario.
# The scenario direct-averted burdens are now read straight from the pickle
# (exact difference-of-counterfactuals computed in the attributable script),
# so there is no attrib/(treatment factor) reconstruction here anymore.
# ----------------------------------------------------------------------

# index rows by location
rows_by_location = {}
for r in hiv_rows:
    rows_by_location.setdefault(r["location"], []).append(r)

output_rows = []  # one dict of UFloats per (location, year, sex)

for location, loc_rows in tqdm(
    rows_by_location.items(), total=len(rows_by_location), desc="Locations"
):
    region = loc_rows[0]["region"]
    years_all = sorted({r["year"] for r in loc_rows})

    # ---- per (year, sex) base quantities (UFloats straight from the pickle) ----
    base = {}
    for r in loc_rows:
        y, s = r["year"], r["sex"]
        entry = {
            "inc": r["unaids_incidence_number"],
            "prev_ye": r["unaids_prevalence_year_end_number"],
            "treat": r["treatment_proportion"],
            # exact direct-averted burdens, precomputed upstream as
            # difference-of-counterfactuals. Just read them.
            "direct_ub": {sti: r[f"direct_ub_{sti}"] for sti in STIS},
            "direct_2016": r["direct_2016_gc"],
            "direct_untreated": r["direct_untreated_gc"],
            "direct_resistance": r["direct_resistance_gc"],
        }
        for sti in STIS:
            # UNAIDS paf drives the transmission simulation, consistent with the
            # UNAIDS incidence/prevalence/attributable used everywhere else here
            entry[f"paf_{sti}"] = r[f"unaids_paf_{sti}_hiv"]
            # both pafs carried through for reporting, each under its own name
            entry[f"gbd_paf_{sti}"] = r[f"paf_{sti}_hiv"]
            entry[f"unaids_paf_{sti}"] = r[f"unaids_paf_{sti}_hiv"]
            entry[f"attrib_{sti}"] = r[
                f"unaids_hiv_incidence_number_attributable_to_{sti}"
            ]
        for abx in ABX:
            entry[f"{abx}_resistant_number"] = r[f"{abx}_resistant_number"]
        base[(y, s)] = entry

    # ---- scenario direct-averted UFloats now come straight from the
    #      attributable pickle (exact difference-of-counterfactuals).
    #      No reconstruction / division here. ----
    ub_direct = {sti: {} for sti in STIS}
    direct_2016 = {}
    direct_untreated = {}
    direct_resistance = {}
    for (y, s), e in base.items():
        for sti in STIS:
            ub_direct[sti][(y, s)] = e["direct_ub"][sti]
        direct_2016[(y, s)] = e["direct_2016"]
        direct_untreated[(y, s)] = e["direct_untreated"]
        direct_resistance[(y, s)] = e["direct_resistance"]

    # ---- helper to build the `data` map for a scenario run ----
    def make_data(year_filter, direct_map, sti):
        d = {}
        for (y, s), e in base.items():
            if not year_filter(y):
                continue
            d[(y, s)] = {
                "inc": e["inc"],
                "prev_ye": e["prev_ye"],
                "treat": e["treat"],
                "paf": e[f"paf_{sti}"],
                "direct": direct_map.get((y, s), point_mass(0.0, "no_direct")),
            }
        return d

    def years_for(year_filter):
        return sorted({y for y in years_all if year_filter(y)})

    incr = {
        sti: (get_increase(sti, "Both"), get_increase(sti, "Women"))
        for sti in STIS
    }

    # ---- scenario: upper bound, past (>= 2000) and future (>= 2025) ----
    ub_indirect_past = {sti: {} for sti in STIS}
    ub_indirect_future = {sti: {} for sti in STIS}
    for sti in STIS:
        inc_both, inc_women = incr[sti]
        ub_indirect_past[sti] = calculate_indirect_averted_cases(
            years_for(lambda y: y >= 2000),
            make_data(lambda y: y >= 2000, ub_direct[sti], sti),
            inc_both,
            inc_women,
        )
        ub_indirect_future[sti] = calculate_indirect_averted_cases(
            years_for(lambda y: y >= 2025),
            make_data(lambda y: y >= 2025, ub_direct[sti], sti),
            inc_both,
            inc_women,
        )

    # ---- scenario: 2016 gonorrhoea first-line change (2016..2023) ----
    inc_both_gc, inc_women_gc = incr["gc"]
    indirect_2016 = calculate_indirect_averted_cases(
        years_for(lambda y: 2016 <= y <= 2023),
        make_data(lambda y: 2016 <= y <= 2023, direct_2016, "gc"),
        inc_both_gc,
        inc_women_gc,
    )

    # ---- scenario: untreated pathway (2008..2023) ----
    indirect_untreated = calculate_indirect_averted_cases(
        years_for(lambda y: 2008 <= y <= 2023),
        make_data(lambda y: 2008 <= y <= 2023, direct_untreated, "gc"),
        inc_both_gc,
        inc_women_gc,
    )

    # ---- scenario: resistance pathway (2008..2023) ----
    indirect_resistance = calculate_indirect_averted_cases(
        years_for(lambda y: 2008 <= y <= 2023),
        make_data(lambda y: 2008 <= y <= 2023, direct_resistance, "gc"),
        inc_both_gc,
        inc_women_gc,
    )

    # ---- assemble per (year, sex) output, all still UFloats ----
    ZERO = point_mass(0.0, "zero")
    for (y, s), e in base.items():
        O = {
            "region": region,
            "location": location,
            "country_code": loc_rows[0]["country_code"],
            "sex": s,
            "year": y,
            # carried for downstream figures (reference incidence + percentages);
            # a count -> extracted via ci_via_log below.
            "unaids_incidence_number": e["inc"],
        }
        for sti in STIS:
            # both pafs reported, each correctly named
            O[f"paf_{sti}_hiv"] = e[f"gbd_paf_{sti}"]
            O[f"unaids_paf_{sti}_hiv"] = e[f"unaids_paf_{sti}"]
            d_ub = ub_direct[sti].get((y, s), ZERO)
            i_ub = ub_indirect_past[sti].get((y, s), ZERO)
            i_ub_fut = ub_indirect_future[sti].get((y, s), ZERO)
            # direct averted by treatment of {sti} (= B(1) - observed attrib),
            # named symmetrically with the indirect/total below
            O[f"direct_hiv_averted_{sti}_upper_bound"] = d_ub
            O[f"indirect_hiv_averted_{sti}_upper_bound"] = i_ub
            O[f"indirect_hiv_averted_{sti}_upper_bound_future"] = i_ub_fut
            O[f"hiv_averted_{sti}_upper_bound"] = d_ub + i_ub
            O[f"hiv_averted_{sti}_upper_bound_future"] = d_ub + i_ub_fut
            # raw observed attributable (unchanged name)
            O[f"unaids_hiv_incidence_number_attributable_to_{sti}"] = e[
                f"attrib_{sti}"
            ]
        for abx in ABX:
            O[f"{abx}_resistant_number"] = e[f"{abx}_resistant_number"]

        # 2016 gc change
        O["direct_hiv_averted_2016_gc_change"] = direct_2016.get((y, s), ZERO)
        O["indirect_hiv_averted_2016_gc_change"] = indirect_2016.get(
            (y, s), ZERO
        )
        O["hiv_averted_2016_gc_change"] = (
            O["direct_hiv_averted_2016_gc_change"]
            + O["indirect_hiv_averted_2016_gc_change"]
        )

        # untreated / resistance decomposition
        O["direct_hiv_averted_gc_untreated"] = direct_untreated.get(
            (y, s), ZERO
        )
        O["indirect_hiv_averted_gc_untreated"] = indirect_untreated.get(
            (y, s), ZERO
        )
        O["hiv_averted_gc_untreated"] = (
            O["direct_hiv_averted_gc_untreated"]
            + O["indirect_hiv_averted_gc_untreated"]
        )
        O["direct_hiv_averted_gc_resistance"] = direct_resistance.get(
            (y, s), ZERO
        )
        O["indirect_hiv_averted_gc_resistance"] = indirect_resistance.get(
            (y, s), ZERO
        )
        O["hiv_averted_gc_resistance"] = (
            O["direct_hiv_averted_gc_resistance"]
            + O["indirect_hiv_averted_gc_resistance"]
        )

        # cross-STI sums (UFloat sums preserve correlation between terms)
        O["direct_hiv_averted_upper_bound"] = sum(
            ub_direct[sti].get((y, s), ZERO) for sti in STIS
        )
        O["indirect_hiv_averted_upper_bound"] = sum(
            ub_indirect_past[sti].get((y, s), ZERO) for sti in STIS
        )
        O["indirect_hiv_averted_upper_bound_future"] = sum(
            ub_indirect_future[sti].get((y, s), ZERO) for sti in STIS
        )
        O["hiv_averted_upper_bound"] = (
            O["direct_hiv_averted_upper_bound"]
            + O["indirect_hiv_averted_upper_bound"]
        )
        O["hiv_averted_upper_bound_future"] = (
            O["direct_hiv_averted_upper_bound"]
            + O["indirect_hiv_averted_upper_bound_future"]
        )

        output_rows.append(O)


# ----------------------------------------------------------------------
# Pickle the raw per-(location, year, sex) UFloat rows too, not just the
# flat, already-extracted CSV below. Figure code that needs to sum across
# locations (to get year-level totals) or across locations+sexes (to get
# year-sex-level totals) has to do that summation on these raw UFloat
# objects -- summing the CSV's pre-extracted lower/upper bounds instead
# would repeat the same invalid error-propagation this pipeline was
# rewritten to avoid.
# ----------------------------------------------------------------------
with open(os.path.join(output_dir, "hiv_averted.ufloat.pkl"), "wb") as f:
    pickle.dump(output_rows, f)


# ----------------------------------------------------------------------
# Extract intervals once, at the very end.
#   * counts (averted / attributable / resistant numbers)  -> ci_via_log
#   * PAFs (proportions in [0,1])                           -> ci_via_logit
# ----------------------------------------------------------------------
PASSTHROUGH = ["region", "location", "country_code", "sex", "year"]
PAF_COLS = [f"paf_{sti}_hiv" for sti in STIS] + [
    f"unaids_paf_{sti}_hiv" for sti in STIS
]
COUNT_COLS = [c for c in output_rows[0] if c not in PASSTHROUGH + PAF_COLS]

flat_rows = []
for O in tqdm(output_rows, desc="Extracting intervals"):
    flat = {c: O[c] for c in PASSTHROUGH}
    for c in PAF_COLS:
        m, lo, hi = ci_via_logit(O[c])
        flat[c], flat[f"{c}_lower"], flat[f"{c}_upper"] = m, lo, hi
    for c in COUNT_COLS:
        m, lo, hi = ci_via_log(O[c])
        flat[c], flat[f"{c}_lower"], flat[f"{c}_upper"] = m, lo, hi
    flat_rows.append(flat)

hiv = pl.DataFrame(flat_rows).sort(["location", "sex", "year"])
hiv.write_csv(os.path.join(output_dir, "hiv_averted.csv"))


# ----------------------------------------------------------------------
# table_s2: select + rename the reporting columns (same mapping as before).
# ----------------------------------------------------------------------
def with_bounds(name):
    return [name, f"{name}_lower", f"{name}_upper"]


select_cols = (
    ["region", "location", "country_code", "sex", "year"]
    + [c for sti in STIS for c in with_bounds(f"paf_{sti}_hiv")]
    + [
        c
        for sti in STIS
        for c in with_bounds(
            f"unaids_hiv_incidence_number_attributable_to_{sti}"
        )
    ]
    + [c for abx in ABX for c in with_bounds(f"{abx}_resistant_number")]
    + [
        c
        for sti in STIS
        for c in with_bounds(f"direct_hiv_averted_{sti}_upper_bound")
    ]
    + [
        c
        for sti in STIS
        for c in with_bounds(f"indirect_hiv_averted_{sti}_upper_bound")
    ]
    + with_bounds("direct_hiv_averted_2016_gc_change")
    + with_bounds("indirect_hiv_averted_2016_gc_change")
    + with_bounds("direct_hiv_averted_gc_untreated")
    + with_bounds("indirect_hiv_averted_gc_untreated")
    + with_bounds("hiv_averted_gc_untreated")
    + with_bounds("direct_hiv_averted_gc_resistance")
    + with_bounds("indirect_hiv_averted_gc_resistance")
    + with_bounds("hiv_averted_gc_resistance")
)

table = hiv.select(select_cols)

mapping = {}
for abx in ABX:
    base_name = f"{abx}_resistant_number"
    new = f"hiv_gc_cases_with_{abx.lower()}_resistance"
    for suf in ["", "_lower", "_upper"]:
        mapping[base_name + suf] = new + suf
for sti in STIS:
    src = f"direct_hiv_averted_{sti}_upper_bound"
    dst = f"direct_hiv_incidence_number_averted_by_treatment_of_{sti}"
    for suf in ["", "_lower", "_upper"]:
        mapping[src + suf] = dst + suf
    isrc = f"indirect_hiv_averted_{sti}_upper_bound"
    idst = f"indirect_hiv_averted_by_treatment_of_{sti}"
    for suf in ["", "_lower", "_upper"]:
        mapping[isrc + suf] = idst + suf

table = table.rename(mapping)
table.write_csv(os.path.join(output_dir, "table_s2.csv"))
