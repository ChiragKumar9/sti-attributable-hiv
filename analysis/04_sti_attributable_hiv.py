import os

import numpy as np
import polars as pl

from averted_burden import conditional_exposure

output_dir = "outputs"
data_dir = "data"

# we want to assess the probability that an individual with gc acquired hiv
# we need the RR of acquiring HIV given gonorrhea
rrs_causal = pl.read_csv(
    os.path.join(output_dir, "meta_estimated_RRs_causal_STI_given_HIV.csv")
)

hiv_sti = pl.read_csv(os.path.join(output_dir, "hiv_sti_with_gc_abx_r.csv"))

#use this code when subsetting to countries in both HIV datasets, comment out when not
hiv_sti = hiv_sti.filter(pl.col("cols_unaids_analysis"))

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
# TODO: we won't need to filter a specific age group but rather will match
# age distributions when we have the age-specific GBD data
female_coinfection = female_coinfection.filter(
    (pl.col("Population") == "women, 15–24, Southern/Eastern Africa")
    | (pl.col("Population") == "women, 15–24, South Africa")
).with_columns(
    sex=pl.lit("Female"),
    location_South_Africa=pl.when(
        pl.col("Population").str.contains("South Africa")
    )
    .then(pl.lit(True))
    .otherwise(pl.lit(False)),
    age=pl.when(pl.col("Population").str.contains("15–24"))
    .then(pl.lit("15-24"))
    .otherwise(pl.lit(None)),
)

male_coinfection = pl.read_csv(
    os.path.join(data_dir, "male_sti_coinfection_rates.csv")
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

# and add age
male_coinfection = male_coinfection.with_columns(age=pl.lit("15-24"))

# TODO: once we have age-specific data, we'll need this
# # and duplicate
# male_coinfection = pl.concat(
#     [male_coinfection,
#      male_coinfection.with_columns(age = pl.lit("25-49"))],
#         how="vertical"
# )

# for each of the age and location groups, calculate the female syphilis ratios
# Calculate the ratio of syphilis coinfection to average coinfection rate for females
# Group by location to get location-specific ratios
female_syphilis_ratios1 = (
    female_coinfection.filter(
        pl.col("STI 1 (for those infected with)") == "Syphilis"
    )
    .group_by(
        ["location_South_Africa", "STI 2 (how many are coinfected w)", "age"]
    )
    .agg(
        [
            # Specific syphilis coinfection rate for this STI
            pl.col("Coinfection prevalence")
            .mean()
            .alias("syphilis_coinfection_rate")
        ]
    )
    .join(
        # Get average coinfection rate for each STI
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
            [
                pl.col("Coinfection prevalence")
                .mean()
                .alias("avg_coinfection_rate")
            ]
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
        [
            # Specific syphilis coinfection rate for this STI
            pl.col("Coinfection prevalence")
            .mean()
            .alias("syphilis_coinfection_rate")
        ]
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
            [
                pl.col("Coinfection prevalence")
                .mean()
                .alias("avg_coinfection_rate")
            ]
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
    pl.col("Coinfection prevalence")
    * pl.col("syphilis_ratio").alias("Coinfection prevalence"),
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
    pl.col("Coinfection prevalence")
    * pl.col("syphilis_ratio").alias("Coinfection prevalence"),
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

# Calculation location-specific coinfection columns
# Instead of using absolute coinfection rates from the study directly, we compute
# a relative risk (RR = P(STI2|STI1)_study / P(STI2)_study) and then multiply by the
# local STI2 prevalence for each row. This makes coinfection rates vary by country/year.
# For females: separate RRs for South Africa vs Southern/Eastern Africa.
# For male syphillis data we still just use the values we impute
# TODO: add bounds for men
_female_coin_sea = female_coinfection.filter(
    pl.col("Population") == "women, 15–24, Southern/Eastern Africa"
)
_female_coin_sa = female_coinfection.filter(
    pl.col("Population") == "women, 15–24, South Africa"
)
# Male original rows only (exclude syphilis rows with Overall prevalence of STI 2)
_male_coin_orig = (
    coinfection
    .filter(pl.col("sex") == "Male")
    .filter(pl.col("location_South_Africa"))
    .filter(pl.col("Overall prevalence of STI 2").is_not_null())
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

        rr_f_sea    = _get_rr(_female_coin_sea, s1, s2) or 0.0
        rr_f_sea_lo = _get_rr(_female_coin_sea, s1, s2, "Coinfection prevalence lower") or 0.0
        rr_f_sea_hi = _get_rr(_female_coin_sea, s1, s2, "Coinfection prevalence upper") or 0.0
        rr_f_sa     = _get_rr(_female_coin_sa, s1, s2) or 0.0
        rr_f_sa_lo  = _get_rr(_female_coin_sa, s1, s2, "Coinfection prevalence lower") or 0.0
        rr_f_sa_hi  = _get_rr(_female_coin_sa, s1, s2, "Coinfection prevalence upper") or 0.0
        rr_m = _get_rr(_male_coin_orig, s1, s2)
        if rr_m is None:
            rr_m = rr_f_sea  # for male syphilis pairs

        for estimate, f_sea, f_sa, m in [
            ("",       rr_f_sea,    rr_f_sa,    rr_m),
            ("_lower", rr_f_sea_lo, rr_f_sa_lo, rr_m),  # TODO add bounds -  males: point estimate for all bounds since don't have bounds yet
            ("_upper", rr_f_sea_hi, rr_f_sa_hi, rr_m),
        ]:
            hiv_sti = hiv_sti.with_columns(
                pl.when((pl.col("sex") == "Female") & (pl.col("location") != "South Africa"))
                .then((f_sea * pl.col(f"{sti2}_prevalence{estimate}")).clip(upper_bound=1.0))
                .when((pl.col("sex") == "Female") & (pl.col("location") == "South Africa"))
                .then((f_sa * pl.col(f"{sti2}_prevalence{estimate}")).clip(upper_bound=1.0))
                .otherwise((m * pl.col(f"{sti2}_prevalence{estimate}")).clip(upper_bound=1.0))
                .alias(f"{sti2}_given_{sti1}_coin{estimate}")
            )

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
                        * (1 - pl.sum_horizontal([pl.col(f"{sti2}_given_{bacteria}_coin") for sti2 in higher_risk]).clip(upper_bound=1.0))
                    )
                    .otherwise(pl.col(f"{bacteria}_prevalence"))
                    .alias(f"{bacteria}_prevalence"),
                    # Lower bound
                    pl.when(mask)
                    .then(
                        pl.col(f"{bacteria}_prevalence_lower")
                        * (1 - pl.sum_horizontal([pl.col(f"{sti2}_given_{bacteria}_coin_lower") for sti2 in higher_risk]).clip(upper_bound=1.0))
                    )
                    .otherwise(pl.col(f"{bacteria}_prevalence_lower"))
                    .alias(f"{bacteria}_prevalence_lower"),
                    pl.when(mask)
                    .then(
                        pl.col(f"{bacteria}_prevalence_upper")
                        * (1 - pl.sum_horizontal([pl.col(f"{sti2}_given_{bacteria}_coin_upper") for sti2 in higher_risk]).clip(upper_bound=1.0))
                    )
                    .otherwise(pl.col(f"{bacteria}_prevalence_upper"))
                    .alias(f"{bacteria}_prevalence_upper"),
                ]
            )

# calculate the prevalence of STIs in people without HIV
# this is the eligible population that can develop HIV because of the STI
# we have P(STI) and RR(STI|HIV) associative, so we can calculate P(STI|HIV)
# and then multiply by P(HIV) to get P(STI and HIV), then subtract from P(STI)
# we have to do this for each of our STIs

STIs = ["gc", "chlamydia", "syphilis", "trichomoniasis"]

# ============================================================
# GBD analysis
# ============================================================

hiv_sti = hiv_sti.with_columns(
    [
        pl.struct(
            [
                f"{sti}_prevalence{estimate}",
                f"hiv_prevalence_number{estimate}",
                f"population{estimate}",
                f"rr_associative{estimate}_{sti}",
            ]
        )
        .map_elements(
            lambda x, s=sti, e=estimate: conditional_exposure.p_a_given_b(
                x[f"{s}_prevalence{e}"],
                x[f"hiv_prevalence_number{e}"] / x[f"population{e}"],
                x[f"rr_associative{e}_{s}"],
            ),
            return_dtype=pl.Float64,
        )
        .alias(f"p_{sti}_given_hiv{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# now calculate P(STI and HIV)
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"p_{sti}_given_hiv{estimate}")
            * (
                pl.col(f"hiv_prevalence_number{estimate}")
                / pl.col(f"population{estimate}")
            )
        ).alias(f"p_{sti}_and_hiv{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# now calculate STI prevalence in people without HIV
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"{sti}_prevalence{estimate}")
            - pl.col(f"p_{sti}_and_hiv{estimate}")
        ).alias(f"{sti}_prevalence_no_hiv{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# check that we don't have any negative prevalences
assert np.all(
    hiv_sti.select(
        [
            f"{sti}_prevalence_no_hiv{estimate}"
            for sti in STIs
            for estimate in ["", "_lower", "_upper"]
        ]
    )
    .min()
    .to_numpy()
    >= 0
)

# the multiplicative nature of this model means that for some highly uncertain
# places, we may see main <= lower <= upper or something like that
# print out places like this
for sti in STIs:
    print(
        hiv_sti.select(
            [
                "year",
                "sex",
                "location",
                f"{sti}_prevalence_no_hiv_lower",
                f"{sti}_prevalence_no_hiv",
                f"{sti}_prevalence_no_hiv_upper",
            ]
        )
        .with_columns(
            in_bounds=(
                pl.col(f"{sti}_prevalence_no_hiv_lower")
                <= pl.col(f"{sti}_prevalence_no_hiv")
            ).and_(
                pl.col(f"{sti}_prevalence_no_hiv")
                <= pl.col(f"{sti}_prevalence_no_hiv_upper")
            )
        )
        .filter(~pl.col("in_bounds"))
    )

# we notice that these are all places where quantification of population size
# is really tough
# so we apply a post-hoc correction where we reorder the values
for sti in STIs:
    hiv_sti = hiv_sti.with_columns(
        [
            # Calculate the minimum, median, and maximum of the three values
            pl.min_horizontal(
                [
                    f"{sti}_prevalence_no_hiv_lower",
                    f"{sti}_prevalence_no_hiv",
                    f"{sti}_prevalence_no_hiv_upper",
                ]
            ).alias(f"{sti}_prevalence_no_hiv_lower"),
            pl.max_horizontal(
                [
                    f"{sti}_prevalence_no_hiv_lower",
                    f"{sti}_prevalence_no_hiv",
                    f"{sti}_prevalence_no_hiv_upper",
                ]
            ).alias(f"{sti}_prevalence_no_hiv_upper"),
            pl.concat_list(
                [
                    f"{sti}_prevalence_no_hiv_lower",
                    f"{sti}_prevalence_no_hiv",
                    f"{sti}_prevalence_no_hiv_upper",
                ]
            )
            .list.sort()
            .list.get(1)
            .alias(f"{sti}_prevalence_no_hiv"),
        ]
    )

# use RR definition and law of total probability to calculate
# P(acquired HIV (attributable to STI) | has STI)
# The causal part is baked into using the causal RRs
hiv_sti = hiv_sti.with_columns(
    [
        pl.struct(
            [
                f"p_acquiring_hiv{estimate}",
                f"{sti}_prevalence_no_hiv{estimate}",
                f"rr_causal{estimate}_{sti}",
            ]
        )
        .map_elements(
            lambda x, s=sti, e=estimate: conditional_exposure.p_a_given_b(
                x[f"p_acquiring_hiv{e}"],
                x[f"{s}_prevalence_no_hiv{e}"],
                x[f"rr_causal{e}_{s}"],
            ),
            return_dtype=pl.Float64,
        )
        .alias(f"p_acquiring_hiv_given_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# we want to calculate the probability of acquiring hiv because of STI,
# not conditioned on having the STI

# in order to calculate the attributable probability of HIV, we need to
# subtract out the people who would have acquired HIV anyways -- not because
# of the STI
hiv_sti = hiv_sti.with_columns(
    [
        pl.struct(
            [
                f"p_acquiring_hiv{estimate}",
                f"{sti}_prevalence_no_hiv{estimate}",
                f"rr_causal{estimate}_{sti}",
            ]
        )
        .map_elements(
            lambda x, s=sti, e=estimate: conditional_exposure.p_a_given_not_b(
                x[f"p_acquiring_hiv{e}"],
                x[f"{s}_prevalence_no_hiv{e}"],
                x[f"rr_causal{e}_{s}"],
            ),
            return_dtype=pl.Float64,
        )
        .alias(f"p_acquiring_hiv_given_no_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

hiv_sti = hiv_sti.with_columns(
    [
        (
            (
                pl.col(f"p_acquiring_hiv_given_{sti}{estimate}")
                - pl.col(f"p_acquiring_hiv_given_no_{sti}{estimate}")
            )
            * pl.col(f"{sti}_prevalence_no_hiv{estimate}")
        ).alias(f"p_hiv_attributable_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# make sure no p_hiv_attributable go below 0
for sti in STIs:
    hiv_sti = hiv_sti.with_columns(
        pl.when(pl.col(f"p_hiv_attributable_{sti}{estimate}") < 0)
        .then(pl.lit(0))
        .otherwise(pl.col(f"p_hiv_attributable_{sti}{estimate}"))
        .alias(f"p_hiv_attributable_{sti}{estimate}")
        for estimate in ["", "_lower", "_upper"]
    )

# assert that the p_hiv_attributable for males for trichomoniasis is 0
assert np.all(
    hiv_sti.filter(pl.col("sex") != "Female")
    .select(
        [
            f"p_hiv_attributable_trichomoniasis{estimate}"
            for estimate in ["", "_lower", "_upper"]
        ]
    )
    .to_numpy()
    == 0
)

# now we want to calculate how much of hiv is attributable to the STI
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"p_hiv_attributable_{sti}{estimate}")
            / pl.col(f"p_acquiring_hiv{estimate}")
        ).alias(f"paf_{sti}_hiv{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# multiply the pafs by the hiv incidence number to get attributable infections
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"hiv_incidence_number{estimate}")
            * pl.col(f"paf_{sti}_hiv{estimate}")
        ).alias(f"hiv_incidence_number_attributable_to_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# now multiply GC attributable cases by resistance rates to get the number of
# cases resistant to a particular drug
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"hiv_incidence_number_attributable_to_gc{estimate}")
            * pl.col(f"{abx}{estimate}")
        ).alias(f"{abx}_resistant_number{estimate}")
        for abx in ["Ciprofloxacin", "Cefixime", "Azithromycin", "Ceftriaxone"]
        for estimate in ["", "_lower", "_upper"]
    ]
)
# ============================================================
# UNAIDS analysis
# uses unaids_prevalence_number, unaids_incidence_rate_uninfected,
# and unaids_incidence_number in place of the GBD equivalents
# ============================================================

hiv_sti = hiv_sti.with_columns(
    [
        pl.struct(
            [
                f"{sti}_prevalence{estimate}",
                f"unaids_prevalence_number{estimate}",
                f"population{estimate}",
                f"rr_associative{estimate}_{sti}",
            ]
        )
        .map_elements(
            lambda x, s=sti, e=estimate: conditional_exposure.p_a_given_b(
                x[f"{s}_prevalence{e}"],
                x[f"unaids_prevalence_number{e}"] / x[f"population{e}"],
                x[f"rr_associative{e}_{s}"],
            ),
            return_dtype=pl.Float64,
        )
        .alias(f"unaids_p_{sti}_given_hiv{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# now calculate P(STI and HIV)
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"unaids_p_{sti}_given_hiv{estimate}")
            * (
                pl.col(f"unaids_prevalence_number{estimate}")
                / pl.col(f"population{estimate}")
            )
        ).alias(f"unaids_p_{sti}_and_hiv{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# now calculate STI prevalence in people without HIV
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"{sti}_prevalence{estimate}")
            - pl.col(f"unaids_p_{sti}_and_hiv{estimate}")
        ).alias(f"unaids_{sti}_prevalence_no_hiv{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# check that we don't have any negative prevalences
assert np.all(
    hiv_sti.select(
        [
            f"unaids_{sti}_prevalence_no_hiv{estimate}"
            for sti in STIs
            for estimate in ["", "_lower", "_upper"]
        ]
    )
    .min()
    .to_numpy()
    >= 0
)
# the multiplicative nature of this model means that for some highly uncertain
# places, we may see main <= lower <= upper or something like that
# print out places like this
for sti in STIs:
    print(
        hiv_sti.select(
            [
                "year",
                "sex",
                "location",
                f"unaids_{sti}_prevalence_no_hiv_lower",
                f"unaids_{sti}_prevalence_no_hiv",
                f"unaids_{sti}_prevalence_no_hiv_upper",
            ]
        )
        .with_columns(
            in_bounds=(
                pl.col(f"unaids_{sti}_prevalence_no_hiv_lower")
                <= pl.col(f"unaids_{sti}_prevalence_no_hiv")
            ).and_(
                pl.col(f"unaids_{sti}_prevalence_no_hiv")
                <= pl.col(f"unaids_{sti}_prevalence_no_hiv_upper")
            )
        )
        .filter(~pl.col("in_bounds"))
    )
# we notice that these are all places where quantification of population size
# is really tough
# so we apply a post-hoc correction where we reorder the values
for sti in STIs:
    hiv_sti = hiv_sti.with_columns(
        [
        # Calculate the minimum, median, and maximum of the three values
            pl.min_horizontal(
                [
                    f"unaids_{sti}_prevalence_no_hiv_lower",
                    f"unaids_{sti}_prevalence_no_hiv",
                    f"unaids_{sti}_prevalence_no_hiv_upper",
                ]
            ).alias(f"unaids_{sti}_prevalence_no_hiv_lower"),
            pl.max_horizontal(
                [
                    f"unaids_{sti}_prevalence_no_hiv_lower",
                    f"unaids_{sti}_prevalence_no_hiv",
                    f"unaids_{sti}_prevalence_no_hiv_upper",
                ]
            ).alias(f"unaids_{sti}_prevalence_no_hiv_upper"),
            pl.concat_list(
                [
                    f"unaids_{sti}_prevalence_no_hiv_lower",
                    f"unaids_{sti}_prevalence_no_hiv",
                    f"unaids_{sti}_prevalence_no_hiv_upper",
                ]
            )
            .list.sort()
            .list.get(1)
            .alias(f"unaids_{sti}_prevalence_no_hiv"),
        ]
    )

# use RR definition and law of total probability to calculate
# P(acquired HIV (attributable to STI) | has STI)
# The causal part is baked into using the causal RRs
hiv_sti = hiv_sti.with_columns(
    [
        pl.struct(
            [
                f"unaids_incidence_rate_uninfected{estimate}",
                f"unaids_{sti}_prevalence_no_hiv{estimate}",
                f"rr_causal{estimate}_{sti}",
            ]
        )
        .map_elements(
            lambda x, s=sti, e=estimate: conditional_exposure.p_a_given_b(
                x[f"unaids_incidence_rate_uninfected{e}"],
                x[f"unaids_{s}_prevalence_no_hiv{e}"],
                x[f"rr_causal{e}_{s}"],
            ),
            return_dtype=pl.Float64,
        )
        .alias(f"unaids_p_acquiring_hiv_given_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# we want to calculate the probability of acquiring hiv because of STI,
# not conditioned on having the STI

# in order to calculate the attributable probability of HIV, we need to
# subtract out the people who would have acquired HIV anyways -- not because
# of the STI
hiv_sti = hiv_sti.with_columns(
    [
        pl.struct(
            [
                f"unaids_incidence_rate_uninfected{estimate}",
                f"unaids_{sti}_prevalence_no_hiv{estimate}",
                f"rr_causal{estimate}_{sti}",
            ]
        )
        .map_elements(
            lambda x, s=sti, e=estimate: conditional_exposure.p_a_given_not_b(
                x[f"unaids_incidence_rate_uninfected{e}"],
                x[f"unaids_{s}_prevalence_no_hiv{e}"],
                x[f"rr_causal{e}_{s}"],
            ),
            return_dtype=pl.Float64,
        )
        .alias(f"unaids_p_acquiring_hiv_given_no_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)


hiv_sti = hiv_sti.with_columns(
    [
        (
            (
                pl.col(f"unaids_p_acquiring_hiv_given_{sti}{estimate}")
                - pl.col(f"unaids_p_acquiring_hiv_given_no_{sti}{estimate}")
            )
            * pl.col(f"unaids_{sti}_prevalence_no_hiv{estimate}")
        ).alias(f"unaids_p_hiv_attributable_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# make sure no unaids_p_hiv_attributable go below 0
for sti in STIs:
    hiv_sti = hiv_sti.with_columns(
        pl.when(pl.col(f"unaids_p_hiv_attributable_{sti}{estimate}") < 0)
        .then(pl.lit(0))
        .otherwise(pl.col(f"unaids_p_hiv_attributable_{sti}{estimate}"))
        .alias(f"unaids_p_hiv_attributable_{sti}{estimate}")
        for estimate in ["", "_lower", "_upper"]
    )

# assert that unaids_p_hiv_attributable for males for trichomoniasis is 0
assert np.all(
    hiv_sti.filter(pl.col("sex") != "Female")
    .select(
        [
            f"unaids_p_hiv_attributable_trichomoniasis{estimate}"
            for estimate in ["", "_lower", "_upper"]
        ]
    )
    .to_numpy()
    == 0
)

# now we want to calculate how much of hiv is attributable to the STI
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"unaids_p_hiv_attributable_{sti}{estimate}")
            / pl.col(f"unaids_incidence_rate_uninfected{estimate}")
        ).alias(f"unaids_paf_{sti}_hiv{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# multiply the pafs by the unaids incidence number to get attributable infections
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"unaids_incidence_number{estimate}")
            * pl.col(f"unaids_paf_{sti}_hiv{estimate}")
        ).alias(f"unaids_hiv_incidence_number_attributable_to_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

# now multiply GC attributable cases by resistance rates to get the number of
# cases resistant to a particular drug
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"unaids_hiv_incidence_number_attributable_to_gc{estimate}")
            * pl.col(f"{abx}{estimate}")
        ).alias(f"unaids_{abx}_resistant_number{estimate}")
        for abx in ["Ciprofloxacin", "Cefixime", "Azithromycin", "Ceftriaxone"]
        for estimate in ["", "_lower", "_upper"]
    ]
)

hiv_sti.write_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))
