import os
import pickle

import polars as pl
import yaml
from scipy import stats
from tqdm.auto import tqdm

from averted_burden import conditional_exposure
from averted_burden.delta_method import (
    LeafCache,
    ci_via_log,
    ci_via_logit,
    logit_normal_leaf,
    lognormal_cluster_member,
    lognormal_leaf,
    new_cluster_latent,
    point_mass,
)

output_dir = os.environ.get("OUTPUT_DIR", "outputs")
data_dir = "data"

STIS = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
ABX = ["Ciprofloxacin", "Cefixime", "Azithromycin", "Ceftriaxone"]

# read params
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

# Add this near the top after loading params
sti_symptom_params = params["sti_symptoms"]


def umax0(x):
    """max(x, 0) for a UFloat -- a real subgradient kink, same as the
    np.clip(lower_bound=0) calls in the original code."""
    if x.nominal_value >= 0:
        return x
    return x - x  # exact zero, including zero variance, like a true clip


# ----------------------------------------------------------------------
# Leaf caches: shared across every row that draws from the same upstream
# estimate, so uncertainty isn't diluted when many rows share one value.
# ----------------------------------------------------------------------

rr_causal_cache = LeafCache(lognormal_leaf)  # keyed by (sex, sti)
rr_assoc_cache = LeafCache(lognormal_leaf)  # keyed by (merging_sex, sti)
resistance_cache = LeafCache(
    logit_normal_leaf
)  # keyed by (location, year, abx)
tx_prop_cache = LeafCache(logit_normal_leaf)  # keyed by (location, year, sex)


def get_rr_causal(sex, sti, mean, lower, upper):
    return rr_causal_cache.get(
        (sex, sti), mean, lower, upper, f"rr_causal_{sti}_{sex}"
    )


def get_rr_assoc(sex, sti, mean, lower, upper):
    return rr_assoc_cache.get(
        (sex, sti), mean, lower, upper, f"rr_assoc_{sti}_{sex}"
    )


def get_resistance(location, year, abx, mean, lower, upper):
    """Antibiotic resistance is a [0,1] proportion, so use logit_normal_leaf."""
    return resistance_cache.get(
        (location, year, abx), mean, lower, upper, f"{abx}_{location}_{year}"
    )


def get_treatment_proportion(location, year, sex, mean, lower, upper):
    """treatment_proportion is a [0,1] coverage proportion, broadcast across
    age groups but estimated per (location, year, sex). Cache it at that grain
    so every age row reuses the same object (same reasoning as the resistance
    cache), and fall back to a fixed point mass if it is degenerate/missing."""
    leaf = tx_prop_cache.get(
        (location, year, sex),
        mean,
        lower,
        upper,
        f"tx_prop_{location}_{year}_{sex}",
    )
    if leaf is None:
        return point_mass(
            mean if mean is not None else 0.0,
            f"tx_prop_fallback_{location}_{year}_{sex}",
        )
    return leaf


# we want to assess the probability that an individual with an STI acquired hiv
# we need the RR of acquiring HIV given STI
rrs_causal = pl.read_csv(
    os.path.join(output_dir, "meta_estimated_RRs_causal_STI_given_HIV.csv")
)

hiv_sti = pl.read_csv(os.path.join(output_dir, "hiv_sti_with_gc_abx_r.csv"))

hiv_sti = hiv_sti.filter(
    (pl.col("cols_unaids_analysis")) | (pl.col("year") > 2023)
)
hiv_sti = hiv_sti.filter(pl.col("unaids_incidence_number").is_not_null())

# relabel the sex column to match the GBD data
rrs_causal = rrs_causal.with_columns(
    sex=pl.when(pl.col("sex") == "Heterosexual Women")
    .then(pl.lit("Female"))
    .when(pl.col("sex") == "Heterosexual Men")
    .then(pl.lit("Male"))
    .otherwise(pl.lit("MSM"))
)
# rename so we have the word causality in there
rrs_causal = rrs_causal.rename(
    {
        "rr_mu": "rr_causal",
        "rr_lower": "rr_causal_lower",
        "rr_upper": "rr_causal_upper",
    }
)
# pivot so we can get the bacteria name into the rr name
# but first make the bacteria names match what we have in the hiv_sti data
bacterial_names = {
    "Gonorrhea": "gc",
    "Chlamydia": "chlamydia",
    "Syphilis": "syphilis",
    "Trichomoniasis": "trichomoniasis",
}
rrs_causal = rrs_causal.with_columns(
    bacteria=pl.col("bacteria").replace(bacterial_names)
)
rrs_causal_prepivot = rrs_causal.__copy__()
rrs_causal = rrs_causal.pivot(
    on="bacteria", values=["rr_causal", "rr_causal_lower", "rr_causal_upper"]
)

# group up with main data
hiv_sti = hiv_sti.join(rrs_causal, on=["sex"], how="inner")

# get the coinfection RRs so we can estimate the actual prevalence of people with
# STI but no HIV
rrs_associative = pl.read_csv(
    os.path.join(output_dir, "meta_estimated_RRs_STI_HIV_coinfection.csv")
)

# relabel the sex column to match the GBD data
rrs_associative = rrs_associative.with_columns(
    merging_sex=pl.when(pl.col("sex") == "Heterosexual Women")
    .then(pl.lit("Female"))
    .otherwise(pl.lit("Male"))
)
# rename so we have the word associative in there
rrs_associative = rrs_associative.rename(
    {
        "rr_mu": "rr_associative",
        "rr_lower": "rr_associative_lower",
        "rr_upper": "rr_associative_upper",
    }
)
# pivot so we can get the bacteria name into the rr name
# but first make the bacteria names match what we have in the hiv_sti data
rrs_associative = rrs_associative.with_columns(
    bacteria=pl.col("bacteria").replace(bacterial_names)
)
rrs_associative = rrs_associative.pivot(
    on="bacteria",
    values=["rr_associative", "rr_associative_lower", "rr_associative_upper"],
)
rrs_associative = rrs_associative.drop("sex")

# join
hiv_sti = hiv_sti.with_columns(
    merging_sex=pl.when(pl.col("sex") == "MSM")
    .then(pl.lit("Male"))
    .otherwise(pl.col("sex"))
)
hiv_sti = hiv_sti.join(rrs_associative, on=["merging_sex"], how="inner")

# calculate the effective sti prevalence accounting for coinfection
female_coinfection = pl.read_csv(
    os.path.join(data_dir, "female_sti_coinfection_rates.csv")
)
female_coinfection = female_coinfection.filter(
    pl.col("Population").is_in(
        [
            "women, 15–24, Southern/Eastern Africa",
            "women, 15–24, South Africa",
            "women, 25–49, Southern/Eastern Africa",
            "women, 25–49, South Africa",
        ]
    )
).with_columns(
    sex=pl.lit("Female"),
    location_South_Africa=pl.when(
        pl.col("Population").str.contains("South Africa")
    )
    .then(pl.lit(True))
    .otherwise(pl.lit(False)),
    age=pl.when(pl.col("Population").str.contains("15–24"))
    .then(pl.lit("15-24"))
    .when(pl.col("Population").str.contains("25–49"))
    .then(pl.lit("25-49"))
    .otherwise(pl.lit(None)),
)

male_coinfection = pl.read_csv(
    os.path.join(data_dir, "male_sti_coinfection_rates.csv")
)

# Add Beta distribution bounds to male coinfection data
# Prior is beta(1, 1), lower is 2.5 percentile and upper is 97.5 percentile
prior_coinfected = 1
prior_not_coinfected = 1

male_coinfection = male_coinfection.with_columns(
    n_not_coinfected=pl.col("n STI 1") - pl.col("n coinfected"),
)
male_coinfection = male_coinfection.with_columns(
    n_coinfected_posterior=pl.col("n coinfected") + pl.lit(prior_coinfected),
    n_not_coinfected_posterior=pl.col("n_not_coinfected")
    + pl.lit(prior_not_coinfected),
)
male_coinfection = male_coinfection.with_columns(
    pl.struct(["n_coinfected_posterior", "n_not_coinfected_posterior"])
    .map_elements(
        lambda x: stats.beta.ppf(
            0.025, x["n_coinfected_posterior"], x["n_not_coinfected_posterior"]
        )
        * 100
        if (
            x["n_coinfected_posterior"] is not None
            and x["n_not_coinfected_posterior"] is not None
        )
        else None,
        return_dtype=pl.Float64,
    )
    .alias("Coinfection prevalence lower"),
)
male_coinfection = male_coinfection.with_columns(
    pl.struct(["n_coinfected_posterior", "n_not_coinfected_posterior"])
    .map_elements(
        lambda x: stats.beta.ppf(
            0.975, x["n_coinfected_posterior"], x["n_not_coinfected_posterior"]
        )
        * 100
        if (
            x["n_coinfected_posterior"] is not None
            and x["n_not_coinfected_posterior"] is not None
        )
        else None,
        return_dtype=pl.Float64,
    )
    .alias("Coinfection prevalence upper"),
)
male_coinfection = male_coinfection.drop(
    [
        "n_not_coinfected",
        "n_coinfected_posterior",
        "n_not_coinfected_posterior",
    ]
)

# the male data is missing anything for syphilis
# what we do is find the ratio of syphilis-other sti to the average rate of
# coinfection for women by location (and age), and then we can apply that ratio
# to the male data and create pseudo syphilis estimates
# the first thing we need to do is make the male data have as many
# age and location options as the female data does
# (this is just for ease of merging and looping later)
male_coinfection = male_coinfection.with_columns(
    location_South_Africa=pl.lit(True)
)
# and duplicate with location south africa as false
male_coinfection = pl.concat(
    [
        male_coinfection,
        male_coinfection.with_columns(location_South_Africa=pl.lit(False)),
    ],
    how="vertical",
)

male_coinfection = pl.concat(
    [
        male_coinfection.with_columns(age=pl.lit("15-24")),
        male_coinfection.with_columns(age=pl.lit("25-49")),
        male_coinfection.with_columns(age=pl.lit("50+")),
    ],
    how="vertical",
)

female_syphilis_ratios1 = (
    female_coinfection.filter(
        pl.col("STI 1 (for those infected with)") == "Syphilis"
    )
    .group_by(
        ["location_South_Africa", "STI 2 (how many are coinfected w)", "age"]
    )
    .agg(
        pl.col("Coinfection prevalence")
        .mean()
        .alias("syphilis_coinfection_rate")
    )
    .join(
        female_coinfection.filter(
            pl.col("STI 1 (for those infected with)") != "Syphilis"
        )
        .group_by(
            [
                "location_South_Africa",
                "STI 2 (how many are coinfected w)",
                "age",
            ]
        )
        .agg(
            pl.col("Coinfection prevalence")
            .mean()
            .alias("avg_coinfection_rate")
        ),
        on=[
            "location_South_Africa",
            "STI 2 (how many are coinfected w)",
            "age",
        ],
        how="left",
    )
    .with_columns(
        syphilis_ratio=(
            pl.col("syphilis_coinfection_rate")
            / pl.col("avg_coinfection_rate")
        )
    )
)

# now do effectively the same but for syphilis as sti 2 for each sti 1
# this way we can add STI-syphilis for men as well
female_syphilis_ratios2 = (
    female_coinfection.filter(
        pl.col("STI 2 (how many are coinfected w)") == "Syphilis"
    )
    .group_by(
        ["location_South_Africa", "STI 1 (for those infected with)", "age"]
    )
    .agg(
        pl.col("Coinfection prevalence")
        .mean()
        .alias("syphilis_coinfection_rate")
    )
    .join(
        # Get average coinfection rate for each STI
        female_coinfection.filter(
            pl.col("STI 2 (how many are coinfected w)") != "Syphilis"
        )
        .group_by(
            ["location_South_Africa", "STI 1 (for those infected with)", "age"]
        )
        .agg(
            pl.col("Coinfection prevalence")
            .mean()
            .alias("avg_coinfection_rate")
        ),
        on=["location_South_Africa", "STI 1 (for those infected with)", "age"],
        how="left",
    )
    .with_columns(
        syphilis_ratio=(
            pl.col("syphilis_coinfection_rate")
            / pl.col("avg_coinfection_rate")
        )
    )
)

male_coinfection2_mean = male_coinfection.group_by(
    ["location_South_Africa", "age", "STI 2 (how many are coinfected w)"]
).agg(pl.mean("Coinfection prevalence"))

# join with female_syphilis_ratios1
male_syphilis1 = male_coinfection2_mean.join(
    female_syphilis_ratios1,
    on=["location_South_Africa", "STI 2 (how many are coinfected w)", "age"],
)
male_syphilis1 = male_syphilis1.with_columns(
    pl.lit("Syphilis").alias("STI 1 (for those infected with)"),
    (pl.col("Coinfection prevalence") * pl.col("syphilis_ratio")).alias(
        "Coinfection prevalence"
    ),
).select(
    [
        "STI 1 (for those infected with)",
        "STI 2 (how many are coinfected w)",
        "Coinfection prevalence",
        "location_South_Africa",
        "age",
    ]
)

# vstack with the rest of our male data
male_coinfection = pl.concat(
    [male_coinfection, male_syphilis1], how="diagonal_relaxed"
)

# now we need to repeat the process for syphilis as sti2
male_coinfection1_mean = male_coinfection.group_by(
    ["location_South_Africa", "age", "STI 1 (for those infected with)"]
).agg(pl.mean("Coinfection prevalence"))

# join with female_syphilis_ratios2
male_syphilis2 = male_coinfection1_mean.join(
    female_syphilis_ratios2,
    on=["location_South_Africa", "STI 1 (for those infected with)", "age"],
)
male_syphilis2 = male_syphilis2.with_columns(
    pl.lit("Syphilis").alias("STI 2 (how many are coinfected w)"),
    (pl.col("Coinfection prevalence") * pl.col("syphilis_ratio")).alias(
        "Coinfection prevalence"
    ),
).select(
    [
        "STI 1 (for those infected with)",
        "STI 2 (how many are coinfected w)",
        "Coinfection prevalence",
        "location_South_Africa",
        "age",
    ]
)

# vstack with the rest of our male data
male_coinfection = pl.concat(
    [male_coinfection, male_syphilis2], how="diagonal_relaxed"
)
male_coinfection = male_coinfection.with_columns(sex=pl.lit("Male"))

# join
coinfection = pl.concat(
    [male_coinfection, female_coinfection], how="diagonal_relaxed"
)

# what we're going to do is remove people from the less risky STIs since we
# assume that the riskiest STI is what drives your risk
STIs_ranked_by_risk = rrs_causal_prepivot.sort(
    ["sex", "rr_causal"], descending=True
)

sti_names = {
    "gc": "Gonorrhea",
    "chlamydia": "Chlamydia",
    "syphilis": "Syphilis",
    "trichomoniasis": "Trichomoniasis",
}

# Pre-compute sex filters once
coinfection_female = coinfection.filter(pl.col("sex") == "Female")
coinfection_male = coinfection.filter(pl.col("sex") == "Male")

_coinfection_age_lookup = {"15-24": "15-24", "25-49": "25-49", "50+": "25-49"}

_female_coin_sea = {
    age: female_coinfection.filter(
        (pl.col("Population").str.contains("Southern/Eastern Africa"))
        & (pl.col("age") == age)
    )
    for age in ["15-24", "25-49"]
}
_female_coin_sa = {
    age: female_coinfection.filter(
        (pl.col("Population").str.contains("South Africa"))
        & (~pl.col("Population").str.contains("Southern"))
        & (pl.col("age") == age)
    )
    for age in ["15-24", "25-49"]
}
_male_coin_orig = (
    coinfection.filter(pl.col("sex") == "Male")
    .filter(pl.col("location_South_Africa"))
    .filter(pl.col("Overall prevalence of STI 2").is_not_null())
    .filter(pl.col("age") == "15-24")  # all age rows identical; pick one
)


def _get_rr(coin_df, sti1_disp, sti2_disp, coin_col="Coinfection prevalence"):
    # Return RR = coinfection_rate / study_prev
    rows = coin_df.filter(
        (pl.col("STI 1 (for those infected with)") == sti1_disp)
        & (pl.col("STI 2 (how many are coinfected w)") == sti2_disp)
    )
    if rows.height == 0 or rows["Overall prevalence of STI 2"][0] is None:
        return None
    study_prev = rows["Overall prevalence of STI 2"][0] / 100.0
    coin_rate = rows[coin_col][0] / 100.0
    return coin_rate / study_prev if study_prev > 0 else 0.0


for sti1 in ["gc", "chlamydia", "syphilis", "trichomoniasis"]:
    for sti2 in ["gc", "chlamydia", "syphilis", "trichomoniasis"]:
        if sti1 == sti2:
            continue
        s1, s2 = sti_names[sti1], sti_names[sti2]

        # male RRs are age-invariant
        rr_m = _get_rr(_male_coin_orig, s1, s2)
        rr_m_lo = _get_rr(
            _male_coin_orig, s1, s2, "Coinfection prevalence lower"
        )
        rr_m_hi = _get_rr(
            _male_coin_orig, s1, s2, "Coinfection prevalence upper"
        )

        for estimate, coin_col in [
            ("", "Coinfection prevalence"),
            ("_lower", "Coinfection prevalence lower"),
            ("_upper", "Coinfection prevalence upper"),
        ]:
            # build the expression starting from male/MSM baseline
            # males have no location or age split, so we start with
            # a single value for non-female rows
            _rr_m = {"": rr_m, "_lower": rr_m_lo, "_upper": rr_m_hi}[estimate]
            if _rr_m is None:
                _rr_m = (
                    _get_rr(_female_coin_sea["15-24"], s1, s2, coin_col) or 0.0
                )

            expr = pl.when(pl.col("sex") != "Female").then(
                (_rr_m * pl.col(f"{sti2}_prevalence{estimate}")).clip(
                    upper_bound=1.0
                )
            )

            # for females, add a branch per age group per location
            for age_group in ["15-24", "25-49", "50+"]:
                coin_age = _coinfection_age_lookup[age_group]
                rr_f_sea = (
                    _get_rr(_female_coin_sea[coin_age], s1, s2, coin_col)
                    or 0.0
                )
                rr_f_sa = (
                    _get_rr(_female_coin_sa[coin_age], s1, s2, coin_col) or 0.0
                )

                expr = (
                    expr.when(
                        (pl.col("sex") == "Female")
                        & (pl.col("age_group") == age_group)
                        & (pl.col("location") != "South Africa")
                    )
                    .then(
                        (
                            rr_f_sea * pl.col(f"{sti2}_prevalence{estimate}")
                        ).clip(upper_bound=1.0)
                    )
                    .when(
                        (pl.col("sex") == "Female")
                        & (pl.col("age_group") == age_group)
                        & (pl.col("location") == "South Africa")
                    )
                    .then(
                        (
                            rr_f_sa * pl.col(f"{sti2}_prevalence{estimate}")
                        ).clip(upper_bound=1.0)
                    )
                )

            hiv_sti = hiv_sti.with_columns(
                expr.otherwise(
                    (0.0 * pl.col(f"{sti2}_prevalence{estimate}")).clip(
                        upper_bound=1.0
                    )
                ).alias(f"{sti2}_given_{sti1}_coin{estimate}")
            )

# coinfection deduction loop
# age-specific coin columns
for sex in hiv_sti["sex"].unique():
    # Select appropriate coinfection dataframe
    # because sex in hiv_sti can be MSM, and for MSM we want to use
    # `coinfection_male`
    coinfection_sex = (
        coinfection_female if sex == "Female" else coinfection_male
    )

    # Get risk list once per sex
    risk = STIs_ranked_by_risk.filter(pl.col("sex") == sex)[
        "bacteria"
    ].to_list()

    for idx, bacteria in enumerate(risk[1:], start=1):
        higher_risk = risk[:idx]

        for location in hiv_sti["location"].unique():
            # Apply updates in a single pass
            mask = (pl.col("sex") == sex) & (pl.col("location") == location)

            hiv_sti = hiv_sti.with_columns(
                [
                    # Main prevalence
                    pl.when(mask)
                    .then(
                        pl.col(f"{bacteria}_prevalence")
                        * (
                            1
                            - pl.sum_horizontal(
                                [
                                    pl.col(f"{sti2}_given_{bacteria}_coin")
                                    for sti2 in higher_risk
                                ]
                            ).clip(upper_bound=1.0)
                        )
                    )
                    .otherwise(pl.col(f"{bacteria}_prevalence"))
                    .alias(f"{bacteria}_prevalence"),
                    # Lower bound
                    pl.when(mask)
                    .then(
                        pl.col(f"{bacteria}_prevalence_lower")
                        * (
                            1
                            - pl.sum_horizontal(
                                [
                                    pl.col(
                                        f"{sti2}_given_{bacteria}_coin_lower"
                                    )
                                    for sti2 in higher_risk
                                ]
                            ).clip(upper_bound=1.0)
                        )
                    )
                    .otherwise(pl.col(f"{bacteria}_prevalence_lower"))
                    .alias(f"{bacteria}_prevalence_lower"),
                    # Upper bound
                    pl.when(mask)
                    .then(
                        pl.col(f"{bacteria}_prevalence_upper")
                        * (
                            1
                            - pl.sum_horizontal(
                                [
                                    pl.col(
                                        f"{sti2}_given_{bacteria}_coin_upper"
                                    )
                                    for sti2 in higher_risk
                                ]
                            ).clip(upper_bound=1.0)
                        )
                    )
                    .otherwise(pl.col(f"{bacteria}_prevalence_upper"))
                    .alias(f"{bacteria}_prevalence_upper"),
                ]
            )

for sti in STIS:
    hiv_sti = hiv_sti.with_columns(
        [
            pl.min_horizontal(
                [
                    f"{sti}_prevalence_lower",
                    f"{sti}_prevalence",
                    f"{sti}_prevalence_upper",
                ]
            ).alias(f"{sti}_prevalence_lower"),
            pl.max_horizontal(
                [
                    f"{sti}_prevalence_lower",
                    f"{sti}_prevalence",
                    f"{sti}_prevalence_upper",
                ]
            ).alias(f"{sti}_prevalence_upper"),
            pl.concat_list(
                [
                    f"{sti}_prevalence_lower",
                    f"{sti}_prevalence",
                    f"{sti}_prevalence_upper",
                ]
            )
            .list.sort()
            .list.get(1)
            .alias(f"{sti}_prevalence"),
        ]
    )


def compute_attributable_block(
    sti_prevalence: dict,  # sti -> UFloat, post-coinfection-deduction prevalence
    hiv_prevalence_proportion,  # e.g. hiv_prevalence_number / population
    p_acquiring_hiv,
    incidence_number,  # e.g. hiv_incidence_number (for scaling attributable counts)
    rr_assoc: dict,  # sti -> UFloat
    rr_causal: dict,  # sti -> UFloat
    treatment_rate: dict,  # sti -> float (constant; zero variance)
    gc_treatment_effectiveness,
):
    """
    Audit notes:
    - All conditional probability calculations use the unchanged conditional_exposure
      functions, which implement the law of total probability correctly.
    - Treatment adjustment: prevalence_no_hiv_effective = prevalence_no_hiv * [(1-t) + t*(1-e)]
      This removes the fraction that is effectively treated. Correct.
    - Attributable risk: (P(HIV|STI) - P(HIV|no STI)) * P(STI) is the classic
      attributable risk formula. Correct.
    - PAF = attributable_risk / P(acquiring HIV). Correct.
    - The result dict now also carries prevalence_no_hiv (the PRE-treatment
      no-HIV prevalence) so the counterfactual direct-averted burdens can be
      re-evaluated exactly via attributable_at_multiplier below, rather than
      reconstructed by dividing attrib by a treatment factor (which linearly
      inverts the saturating p_a_given_b map and biases the counterfactual).
    """
    results = {}
    for sti in STIS:
        p_a = sti_prevalence[sti]

        # P(STI | HIV), P(STI and HIV), P(STI | no HIV)
        p_sti_given_hiv = conditional_exposure.p_a_given_b(
            p_a, hiv_prevalence_proportion, rr_assoc[sti]
        )
        p_sti_and_hiv = p_sti_given_hiv * hiv_prevalence_proportion
        # condition the joint P(STI and no HIV) on the susceptible pool, so this
        # shares the per-susceptible base of p_acquiring_hiv used below
        prevalence_no_hiv = (p_a - p_sti_and_hiv) / (
            1 - hiv_prevalence_proportion
        )

        # treatment adjustment
        if sti == "gc":
            effectiveness = gc_treatment_effectiveness
        else:
            # Fixed effectiveness for other STIs (no uncertainty)
            effectiveness = 1.0
        t = treatment_rate[sti]  # This is a float, not UFloat
        prevalence_no_hiv_effective = prevalence_no_hiv * (
            (1 - t) + t * (1 - effectiveness)
        )
        prevalence_no_hiv_effective = umax0(prevalence_no_hiv_effective)

        # attributable risk
        p_acq_given_sti = conditional_exposure.p_a_given_b(
            p_acquiring_hiv, prevalence_no_hiv_effective, rr_causal[sti]
        )
        p_acq_given_no_sti = conditional_exposure.p_a_given_not_b(
            p_acquiring_hiv, prevalence_no_hiv_effective, rr_causal[sti]
        )
        p_hiv_attributable = (p_acq_given_sti - p_acq_given_no_sti) * (
            prevalence_no_hiv_effective
        )
        p_hiv_attributable = umax0(p_hiv_attributable)

        paf = p_hiv_attributable / p_acquiring_hiv
        incidence_attributable = incidence_number * paf

        results[sti] = {
            "prevalence_no_hiv": prevalence_no_hiv,  # PRE-treatment (for counterfactuals)
            "prevalence_no_hiv_effective": prevalence_no_hiv_effective,
            "paf": paf,
            "incidence_attributable": incidence_attributable,
        }
    return results


def attributable_at_multiplier(
    prevalence_no_hiv, p_acquiring_hiv, rr_causal_sti, incidence_number, m
):
    """Exact attributable HIV incidence for one STI with treatment multiplier m
    applied to the PRE-treatment no-HIV prevalence. m is the transmitting-pool
    fraction:  m = (1 - t) + t * (1 - effectiveness).

    Passing m equal to the block's observed multiplier reproduces the block's
    incidence_attributable exactly (same conditional-exposure chain, same umax0
    clips). Counterfactual direct-averted burdens are then DIFFERENCES of this
    function at various m -- NOT attrib divided by a treatment factor. Division
    only linearly inverts p_a_given_b, which is a saturating (Michaelis-Menten)
    map in prevalence, so it overstates the counterfactual "no-treatment"
    burden, worst where (rr-1)*prevalence is large. Differencing exact
    evaluations removes that bias while keeping every UFloat leaf/correlation
    intact (same rr, incidence, prevalence objects reused)."""
    prev_eff = umax0(prevalence_no_hiv * m)
    p_acq_given_sti = conditional_exposure.p_a_given_b(
        p_acquiring_hiv, prev_eff, rr_causal_sti
    )
    p_acq_given_no_sti = conditional_exposure.p_a_given_not_b(
        p_acquiring_hiv, prev_eff, rr_causal_sti
    )
    p_hiv_attr = umax0((p_acq_given_sti - p_acq_given_no_sti) * prev_eff)
    return incidence_number * (p_hiv_attr / p_acquiring_hiv)


# ----------------------------------------------------------------------
# Row-by-row pass. Each row gets fresh independent leaves for its own
# prevalence/population-cluster quantities; RR and resistance leaves are
# pulled from the shared caches above by their natural grain.
# ----------------------------------------------------------------------

output_rows = []

for i, row in tqdm(
    enumerate(hiv_sti.iter_rows(named=True)),
    total=hiv_sti.height,
    desc="Processing rows",
):
    sex = row["sex"]
    merging_sex = "Male" if sex == "MSM" else sex
    location = row["location"]
    year = row["year"]

    sti_prev = {
        sti: logit_normal_leaf(
            row[f"{sti}_prevalence"],
            row[f"{sti}_prevalence_lower"],
            row[f"{sti}_prevalence_upper"],
            f"{sti}_prev_{i}",
        )
        for sti in STIS
    }

    # --- RR leaves (shared across age groups for the same sex) ---
    # RRs are positive, unbounded, so use lognormal_leaf
    rr_causal = {
        sti: get_rr_causal(
            sex,
            sti,
            row[f"rr_causal_{sti}"],
            row[f"rr_causal_lower_{sti}"],
            row[f"rr_causal_upper_{sti}"],
        )
        for sti in STIS
    }
    rr_assoc = {
        sti: get_rr_assoc(
            merging_sex,
            sti,
            row[f"rr_associative_{sti}"],
            row[f"rr_associative_lower_{sti}"],
            row[f"rr_associative_upper_{sti}"],
        )
        for sti in STIS
    }

    # --- treatment_proportion leaf (shared across age groups for the same
    #     location-year-sex; [0,1] coverage proportion, so logit_normal) ---
    treatment_proportion = get_treatment_proportion(
        location,
        year,
        sex,
        row["treatment_proportion"],
        row["treatment_proportion_lower"],
        row["treatment_proportion_upper"],
    )

    # --- antibiotic resistance leaves (shared across age groups & sex
    #     for the same location-year; pre-2006 rows have mean=lower=upper=0
    #     which logit_normal_leaf handles by returning a zero-variance leaf) ---
    # Resistance proportions are [0,1], so use logit_normal_leaf.
    #
    # FIX (was filling missing values to 0.0 BEFORE building leaves, which
    # meant a single missing drug in the post-2016 min() comparison forced
    # gc_treatment_failure_era to exactly 0 regardless of the other two
    # drugs -- i.e. "no data for Azithromycin" silently became "treatment
    # is perfectly effective". Now: build a leaf only for drugs that have
    # data, take min() over whatever's actually present, and only fall
    # back to a zero point-mass if NOTHING is present for this location-
    # year (matching the original script's fill_null(0) as a true last
    # resort, not a per-drug default).
    abx_leaf = {}
    for abx in ABX:
        if row[abx] is None:
            abx_leaf[abx] = None
        else:
            abx_leaf[abx] = get_resistance(
                location,
                year,
                abx,
                row[abx],
                row[f"{abx}_lower"],
                row[f"{abx}_upper"],
            )

    # standalone post-2016 gonorrhoea failure rate (min over the drugs that
    # actually have data). Used by the indirect script's 2016-change scenario
    # regardless of year, so compute it independently of the era branch and
    # with the same present-only protection.
    present_post2016 = [
        abx_leaf[a]
        for a in ("Azithromycin", "Cefixime", "Ceftriaxone")
        if abx_leaf[a] is not None
    ]
    gc_treatment_failure = (
        min(present_post2016, key=lambda u: u.nominal_value)
        if present_post2016
        else point_mass(0.0, f"gc_failure_min_fallback_{location}_{year}")
    )

    if year < 2016:
        gc_treatment_failure_era = abx_leaf["Ciprofloxacin"]
    else:
        gc_treatment_failure_era = (
            gc_treatment_failure if present_post2016 else None
        )

    if gc_treatment_failure_era is None:
        # no resistance data available at all for this location-year/era
        # branch -- fall back to a fixed zero-variance point mass, matching
        # the original script's fill_null(0) behavior
        gc_treatment_failure_era = point_mass(
            0.0, f"gc_failure_fallback_{location}_{year}"
        )

    # fill any individually-missing antibiotics with the same zero point
    # mass so the resistant-case multiplication below doesn't hit a None
    for abx in ABX:
        if abx_leaf[abx] is None:
            abx_leaf[abx] = point_mass(
                0.0, f"{abx}_fallback_{location}_{year}"
            )

    gc_treatment_effectiveness = 1 - gc_treatment_failure_era  # type: ignore

    # --- treatment-seeking rate per STI (frac_sought_treatment + the
    #     symptom params are fixed constants -- zero variance, as agreed) ---
    sex_key = "female" if sex == "Female" else "male"
    treatment_rate = {
        sti: sti_symptom_params[sti][sex_key] * row["frac_sought_treatment"]
        for sti in STIS
    }

    # --- GBD cluster: population, hiv_incidence_number, hiv_prevalence_number
    #     are perfectly correlated lognormal quantities sharing one latent ---
    # AUDIT: These are counts (positive, unbounded), so lognormal is correct
    z_gbd = new_cluster_latent(f"z_gbd_{i}")
    population = lognormal_cluster_member(
        row["population"],
        row["population_lower"],
        row["population_upper"],
        z_gbd,
    )
    hiv_incidence_number = lognormal_cluster_member(
        row["hiv_incidence_number"],
        row["hiv_incidence_number_lower"],
        row["hiv_incidence_number_upper"],
        z_gbd,
    )
    hiv_prevalence_number = lognormal_cluster_member(
        row["hiv_prevalence_number"],
        row["hiv_prevalence_number_lower"],
        row["hiv_prevalence_number_upper"],
        z_gbd,
    )

    # AUDIT: Division of correlated quantities preserves correlation
    hiv_prevalence_proportion = hiv_prevalence_number / population  # type: ignore
    # AUDIT: This is P(acquiring HIV) = incidence / (population - prevalence)
    # The denominator is "population at risk" (those without HIV)
    p_acquiring_hiv = hiv_incidence_number / (
        population - hiv_prevalence_number  # type: ignore
    )

    gbd_results = compute_attributable_block(
        sti_prevalence=sti_prev,
        hiv_prevalence_proportion=hiv_prevalence_proportion,
        p_acquiring_hiv=p_acquiring_hiv,
        incidence_number=hiv_incidence_number,
        rr_assoc=rr_assoc,
        rr_causal=rr_causal,
        treatment_rate=treatment_rate,
        gc_treatment_effectiveness=gc_treatment_effectiveness,
    )

    # --- UNAIDS cluster: un_pop, unaids_incidence_number,
    #     unaids_prevalence_number share their own latent ---
    z_un = new_cluster_latent(f"z_unaids_{i}")

    # AUDIT: Original code treats un_pop as having no CI (exact)
    # We still create a lognormal cluster member with identical upper/lower
    # which gives zero variance, but keeps it in the correlation structure
    un_pop = lognormal_cluster_member(
        row["un_pop"], row["un_pop"], row["un_pop"], z_un
    )
    unaids_incidence_number = lognormal_cluster_member(
        row["unaids_incidence_number"],
        row["unaids_incidence_number_lower"],
        row["unaids_incidence_number_upper"],
        z_un,
    )
    unaids_prevalence_number = lognormal_cluster_member(
        row["unaids_prevalence_number"],
        row["unaids_prevalence_number_lower"],
        row["unaids_prevalence_number_upper"],
        z_un,
    )
    # unaids_prevalence_year_end_number shares the SAME UNAIDS latent z_un, so
    # it stays correlated with the incidence/prevalence members. The indirect
    # script needs it for the transmission-rate denominator.
    unaids_prevalence_year_end_number = lognormal_cluster_member(
        row["unaids_prevalence_year_end_number"],
        row["unaids_prevalence_year_end_number_lower"],
        row["unaids_prevalence_year_end_number_upper"],
        z_un,
    )
    if unaids_prevalence_year_end_number is None:  # mean <= 0 fallback
        unaids_prevalence_year_end_number = point_mass(
            row["unaids_prevalence_year_end_number"] or 0.0,
            f"un_prev_ye_fallback_{i}",
        )
    unaids_prevalence_proportion = unaids_prevalence_number / un_pop  # type: ignore
    unaids_p_acquiring_hiv = unaids_incidence_number / (
        un_pop - unaids_prevalence_number  # type: ignore
    )

    unaids_results = compute_attributable_block(
        sti_prevalence=sti_prev,
        hiv_prevalence_proportion=unaids_prevalence_proportion,
        p_acquiring_hiv=unaids_p_acquiring_hiv,
        incidence_number=unaids_incidence_number,
        rr_assoc=rr_assoc,
        rr_causal=rr_causal,
        treatment_rate=treatment_rate,
        gc_treatment_effectiveness=gc_treatment_effectiveness,
    )

    # ------------------------------------------------------------------
    # EXACT counterfactual direct-averted burdens (UNAIDS cluster). These
    # used to live in the indirect script as attrib / (treatment factor)
    # reconstructions; they are moved here (where prevalence_no_hiv,
    # p_acquiring_hiv, rr_causal and the incidence leaves are still in scope)
    # and expressed as DIFFERENCES of attributable_at_multiplier evaluations,
    # so the saturation of p_a_given_b is handled at every treatment level.
    # ------------------------------------------------------------------
    def B_un(sti, m):
        return attributable_at_multiplier(
            unaids_results[sti]["prevalence_no_hiv"],
            unaids_p_acquiring_hiv,
            rr_causal[sti],
            unaids_incidence_number,
            m,
        )

    # per-STI universal-access upper bound: HIV currently averted by
    # symptomatic treatment = burden with NO treatment (m=1) minus the observed
    # (treated) attributable burden. Replaces `attrib * t/(1-t)`.
    direct_ub = {}
    for sti in STIS:
        attrib_sti = unaids_results[sti]["incidence_attributable"]
        direct_ub[sti] = umax0(B_un(sti, 1.0) - attrib_sti)

    # gc access / resistance decomposition -- multiplicative partition of the
    # observed gc-attributable HIV burden into the share caused by lack of
    # treatment access vs antibiotic resistance (as in the original script):
    #   total_burden     = observed / ((1 - t) + t * failure_era)
    #   untreated share  = (1 - t)         * total_burden
    #   resistance share = t * failure_era * total_burden
    # The era-appropriate failure rate is used (cipro pre-2016, min of
    # azithro/cefixime/ceftriaxone post-2016). The two shares sum exactly to the
    # observed attributable burden.
    t_gc = treatment_rate["gc"]
    attrib_gc_un = unaids_results["gc"]["incidence_attributable"]

    gc_hiv_total_burden = attrib_gc_un / (
        (1 - t_gc) + t_gc * gc_treatment_failure_era
    )
    direct_untreated_gc = (1 - t_gc) * gc_hiv_total_burden
    direct_resistance_gc = (
        t_gc * gc_treatment_failure_era * gc_hiv_total_burden
    )

    # 2016 gonorrhoea first-line change -- the REAL cipro -> post-2016 switch.
    # First-line failure was ciprofloxacin resistance; it becomes the post-2016
    # regimen's failure (min over the post-2016 drugs, = gc_treatment_failure).
    # The averted burden is the difference of the exact burdens under the two
    # first-line multipliers -- nonzero in every year (including pre-2016), not
    # just a universal-access upper bound restricted to 2016-2023.
    m_pre = (1 - t_gc) + t_gc * abx_leaf["Ciprofloxacin"]
    m_post = (1 - t_gc) + t_gc * gc_treatment_failure
    direct_2016_gc = umax0(B_un("gc", m_pre) - B_un("gc", m_post))

    # --- antibiotic-resistant case counts (reuses the SAME cached
    #     resistance leaf used above, so correlation with treatment
    #     effectiveness is automatic) ---
    # AUDIT: This multiplies GC-attributable HIV cases by resistance proportion
    # to get cases attributable to resistant GC. Correct.
    resistant_counts = {}
    unaids_resistant_counts = {}
    for abx in ABX:
        resistant_counts[abx] = (
            gbd_results["gc"]["incidence_attributable"] * abx_leaf[abx]
        )
        unaids_resistant_counts[abx] = (
            unaids_results["gc"]["incidence_attributable"] * abx_leaf[abx]
        )

    # ---- assemble the row's output, keeping the raw UFloat objects so
    #      the aggregation step downstream can sum() them directly before
    #      extracting any intervals (extracting intervals first and then
    #      summing would throw away the correlation structure) ----
    out_row = {
        "country_code": row["country_code"],
        "location": location,
        "region": row["region"],
        "sex": sex,
        "year": year,
        "age_group": row["age_group"],
        "cols_unaids_analysis": row["cols_unaids_analysis"],
        "hiv_incidence_number": hiv_incidence_number,
        "hiv_prevalence_number": hiv_prevalence_number,
        "unaids_incidence_number": unaids_incidence_number,
        "unaids_prevalence_number": unaids_prevalence_number,
        "unaids_prevalence_year_end_number": unaids_prevalence_year_end_number,
        "population": population,
        "un_pop": un_pop,
        # broadcast quantities carried forward for the merged indirect script
        "treatment_proportion": treatment_proportion,
        "frac_sought_treatment": row["frac_sought_treatment"],
        "gc_treatment_failure": gc_treatment_failure,
        "gc_treatment_failure_era": gc_treatment_failure_era,
    }
    for sti in STIS:
        out_row[f"paf_{sti}_hiv"] = gbd_results[sti]["paf"]
        out_row[f"hiv_incidence_number_attributable_to_{sti}"] = gbd_results[
            sti
        ]["incidence_attributable"]
        out_row[f"unaids_paf_{sti}_hiv"] = unaids_results[sti]["paf"]
        out_row[f"unaids_hiv_incidence_number_attributable_to_{sti}"] = (
            unaids_results[sti]["incidence_attributable"]
        )
    for abx in ABX:
        out_row[f"{abx}_resistant_number"] = resistant_counts[abx]
        out_row[f"unaids_{abx}_resistant_number"] = unaids_resistant_counts[
            abx
        ]

    # exact direct-averted burdens (UNAIDS cluster) -- consumed by the
    # indirect script, which no longer reconstructs them by division.
    for sti in STIS:
        out_row[f"direct_ub_{sti}"] = direct_ub[sti]
    out_row["direct_2016_gc"] = direct_2016_gc
    out_row["direct_untreated_gc"] = direct_untreated_gc
    out_row["direct_resistance_gc"] = direct_resistance_gc

    output_rows.append(out_row)

# ----------------------------------------------------------------------
# Write the age-stratified output, extracting mean/lower/upper from each
# UFloat via the domain-matched transform.
# ----------------------------------------------------------------------

# AUDIT: PAFs are proportions [0,1], so use ci_via_logit
# Counts are positive unbounded, so use ci_via_log
BOUNDED_COLS = [f"paf_{s}_hiv" for s in STIS] + [
    f"unaids_paf_{s}_hiv" for s in STIS
]
COUNT_COLS = (
    [
        "hiv_incidence_number",
        "hiv_prevalence_number",
        "unaids_incidence_number",
        "unaids_prevalence_number",
        "unaids_prevalence_year_end_number",
        "population",
        "un_pop",
    ]
    + [f"hiv_incidence_number_attributable_to_{s}" for s in STIS]
    + [f"unaids_hiv_incidence_number_attributable_to_{s}" for s in STIS]
    + [f"{abx}_resistant_number" for abx in ABX]
    + [f"unaids_{abx}_resistant_number" for abx in ABX]
)
# direct-averted counterfactual burdens (positive counts) -- summed across age,
# passed only through the handoff pickle (not the flat CSVs).
DIRECT_COLS = [f"direct_ub_{sti}" for sti in STIS] + [
    "direct_2016_gc",
    "direct_untreated_gc",
    "direct_resistance_gc",
]
PASSTHROUGH_COLS = [
    "country_code",
    "location",
    "region",
    "sex",
    "year",
    "age_group",
    "cols_unaids_analysis",
]

flat_rows = []
for r in output_rows:
    flat = {c: r[c] for c in PASSTHROUGH_COLS}
    for c in BOUNDED_COLS:
        mean, lower, upper = ci_via_logit(r[c])
        flat[c], flat[f"{c}_lower"], flat[f"{c}_upper"] = mean, lower, upper
    for c in COUNT_COLS:
        mean, lower, upper = ci_via_log(r[c])
        flat[c], flat[f"{c}_lower"], flat[f"{c}_upper"] = mean, lower, upper
    flat_rows.append(flat)

hiv_sti_out = pl.DataFrame(flat_rows)
hiv_sti_out.write_csv(
    os.path.join(output_dir, "hiv_attributable_to_stis_age_stratified.csv")
)

# ----------------------------------------------------------------------
# table_s3: age-stratified attribution table (both GBD and UNAIDS), the
# age-resolved companion to the country-level table_s2. Attribution outputs
# only -- averted-burden scenarios are computed post-age-aggregation and are
# age-agnostic, so they stay in table_s2. This is a straight select/rename of
# the age-stratified frame above, dropping the internal analysis flag.
# ----------------------------------------------------------------------
table_s3 = hiv_sti_out.drop("cols_unaids_analysis").sort(
    ["location", "sex", "age_group", "year"]
)
table_s3.write_csv(os.path.join(output_dir, "table_s3.csv"))

# Also pickle the raw per-age-row UFloat objects (output_rows) themselves,
# not just the flat CSV above. The age-pathogen figure panels need to group
# by age_group while summing across countries/years -- doing that on the
# already-extracted lower/upper bounds in the CSV would repeat the same
# invalid "sum the bounds" error propagation this script was written to
# avoid, so the figure code needs these raw UFloat objects instead.
with open(
    os.path.join(
        output_dir, "hiv_attributable_to_stis_age_stratified.ufloat.pkl"
    ),
    "wb",
) as f:
    pickle.dump(output_rows, f)

# ----------------------------------------------------------------------
# Aggregation across age groups: sum the raw UFloat objects (kept in
# `output_rows`, not the already-extracted flat_rows) within each
# country/sex/year group, THEN extract intervals. Summing UFloats this
# way automatically gives sum-of-variances for the independent per-row
# leaves (STI prevalence, GBD/UNAIDS clusters) while correctly NOT
# diluting the RR/resistance leaves, since those are the same cached
# object reused across every age-group row in the group.
# ----------------------------------------------------------------------

# AUDIT: The aggregation strategy is correct:
# 1. Sum UFloat objects directly (preserves correlation structure)
# 2. Extract intervals only after aggregation
# 3. Re-derive PAFs from aggregated counts (correct: PAF_total ≠ mean(PAF_age))

# broadcast quantities -- identical across the age rows in a group (and the
# SAME cached object), so pl.first-style "take first" keeps the shared leaf
# (and its correlations) for the merged indirect script.
BROADCAST_COLS = [
    "treatment_proportion",
    "gc_treatment_failure",
    "gc_treatment_failure_era",
]

group_keys = [
    "country_code",
    "location",
    "region",
    "sex",
    "year",
    "cols_unaids_analysis",
]
groups: dict = {}
for r in output_rows:
    key = tuple(r[k] for k in group_keys)
    groups.setdefault(key, []).append(r)

# 1) aggregated UFloat rows, kept raw for the merged indirect script
agg_ufloat_rows = []
for key, rows in groups.items():
    agg = dict(zip(group_keys, key))

    # Sum counts across age groups
    for c in COUNT_COLS:
        agg[c] = sum(r[c] for r in rows)

    # Sum the exact direct-averted burdens across age groups. Each per-age
    # value is already a difference-of-counterfactuals, so the aggregate
    # averted is the sum; the gc decomposition still sums to the aggregated
    # attrib_gc because it holds age by age.
    for c in DIRECT_COLS:
        agg[c] = sum(r[c] for r in rows)

    # AUDIT: Re-derive PAFs from aggregated attributable counts / aggregated
    # incidence, rather than averaging the per-age-group PAFs directly.
    # This is correct because PAF is a ratio, not a linear quantity.
    # PAF_total = sum(attributable_i) / sum(incidence_i) ≠ mean(PAF_i)
    for sti in STIS:
        agg[f"paf_{sti}_hiv"] = sum(
            r[f"hiv_incidence_number_attributable_to_{sti}"] for r in rows
        ) / sum(r["hiv_incidence_number"] for r in rows)
        agg[f"unaids_paf_{sti}_hiv"] = sum(
            r[f"unaids_hiv_incidence_number_attributable_to_{sti}"]
            for r in rows
        ) / sum(r["unaids_incidence_number"] for r in rows)

    # broadcast quantities: identical across the age rows in this group, and
    # the SAME cached object, so rows[0] keeps the shared leaf + correlations
    for c in BROADCAST_COLS:
        agg[c] = rows[0][c]
    agg["frac_sought_treatment"] = rows[0]["frac_sought_treatment"]  # float

    agg_ufloat_rows.append(agg)

# hand-off to the merged indirect script -- a single pickle dump preserves the
# whole object graph, so shared latents (and therefore every correlation built
# above) survive the boundary between the two scripts.
with open(
    os.path.join(output_dir, "hiv_attributable_to_stis.ufloat.pkl"), "wb"
) as f:
    pickle.dump(agg_ufloat_rows, f)

# 2) flat CSV (unchanged downstream behaviour) -- extract intervals now.
PAF_AGG_COLS = [f"paf_{s}_hiv" for s in STIS] + [
    f"unaids_paf_{s}_hiv" for s in STIS
]
# treatment_proportion / gc_treatment_failure(_era) are all [0,1] -> ci_via_logit
BOUNDED_BROADCAST = [
    "treatment_proportion",
    "gc_treatment_failure",
    "gc_treatment_failure_era",
]

agg_rows = []
for agg in agg_ufloat_rows:
    flat = {k: agg[k] for k in group_keys}
    flat["frac_sought_treatment"] = agg["frac_sought_treatment"]
    for c in COUNT_COLS:
        mean, lower, upper = ci_via_log(agg[c])
        flat[c], flat[f"{c}_lower"], flat[f"{c}_upper"] = mean, lower, upper
    for c in PAF_AGG_COLS + BOUNDED_BROADCAST:
        mean, lower, upper = ci_via_logit(agg[c])
        flat[c], flat[f"{c}_lower"], flat[f"{c}_upper"] = mean, lower, upper
    agg_rows.append(flat)

hiv_sti_agg = pl.DataFrame(agg_rows)
hiv_sti_agg.write_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))
