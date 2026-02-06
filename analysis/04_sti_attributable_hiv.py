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

# join
hiv_sti = hiv_sti.with_columns(
    merging_sex=pl.when(pl.col("sex") == "MSM")
    .then(pl.lit("Male"))
    .otherwise(pl.col("sex"))
)
hiv_sti = hiv_sti.join(rrs_associative, on=["merging_sex"], how="inner")

# calculate the effective sti prevalence accounting for coinfection
coinfection = pl.read_csv(os.path.join(data_dir, "sti_coinfection_rates.csv"))
coinfection = coinfection.filter(
    pl.col("Population") == "women, 15–24, Southern/Eastern Africa"
).with_columns(sex=pl.lit("Female"))
# make a male data column by taking the values present in the dataframe and
# multiplying them by 2/3
male_coinfection = coinfection.with_columns(
    (pl.col("Coinfection prevalence") * (2 / 3)).alias(
        "Coinfection prevalence"
    ),
    (pl.col("Coinfection prevalence lower") * (2 / 3)).alias(
        "Coinfection prevalence lower"
    ),
    (pl.col("Coinfection prevalence upper") * (2 / 3)).alias(
        "Coinfection prevalence upper"
    ),
    sex=pl.lit("Male"),
)

# join
coinfection = coinfection.vstack(male_coinfection)


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

for sex in hiv_sti["sex"].unique():
    # Select appropriate coinfection dataframe
    coinfection_sex = (
        coinfection_female if sex == "Female" else coinfection_male
    )

    # Get risk list once per sex
    risk = STIs_ranked_by_risk.filter(pl.col("sex") == sex)[
        "bacteria"
    ].to_list()

    for idx, bacteria in enumerate(risk[1:], start=1):
        # Pre-filter coinfection data for this bacteria
        bacteria_coinfections = coinfection_sex.filter(
            pl.col("STI 1 (for those infected with)") == sti_names[bacteria]
        ).filter(
            pl.col("STI 2 (how many are coinfected w)").is_in(
                [sti_names[b] for b in risk[:idx]]
            )
        )

        # Get all rates at once
        coinfection_rate = (
            bacteria_coinfections["Coinfection prevalence"].sum() / 100
        )
        coinfection_rate_lower = (
            bacteria_coinfections["Coinfection prevalence lower"].sum() / 100
        )
        coinfection_rate_upper = (
            bacteria_coinfections["Coinfection prevalence upper"].sum() / 100
        )

        # Apply updates in a single pass (without location loop)
        sex_mask = pl.col("sex") == sex

        hiv_sti = hiv_sti.with_columns(
            [
                # Main prevalence
                pl.when(sex_mask)
                .then(
                    pl.col(f"{bacteria}_prevalence") * (1 - coinfection_rate)
                )
                .otherwise(pl.col(f"{bacteria}_prevalence"))
                .alias(f"{bacteria}_prevalence"),
                # Lower bound (note: uses upper coinfection rate)
                pl.when(sex_mask)
                .then(
                    pl.col(f"{bacteria}_prevalence_lower")
                    * (1 - coinfection_rate_upper)
                )
                .otherwise(pl.col(f"{bacteria}_prevalence_lower"))
                .alias(f"{bacteria}_prevalence_lower"),
                # Upper bound (note: uses lower coinfection rate)
                pl.when(sex_mask)
                .then(
                    pl.col(f"{bacteria}_prevalence_upper")
                    * (1 - coinfection_rate_lower)
                )
                .otherwise(pl.col(f"{bacteria}_prevalence_upper"))
                .alias(f"{bacteria}_prevalence_upper"),
            ]
        )

# calculate the prevalence of STIs in people without HIV
# we have P(STI) and RR(STI|HIV) associative, so we can calculate P(STI|HIV)
# and then multiply by P(HIV) to get P(STI and HIV), then subtract from P(STI)
# we have to do this for each of our STIs

STIs = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
hiv_sti = hiv_sti.with_columns(
    [
        pl.struct(
            [
                f"{sti}_prevalence{estimate}",
                f"p_acquiring_hiv{estimate}",
                f"rr_associative{estimate}_{sti}",
            ]
        )
        .map_elements(
            lambda x, s=sti, e=estimate: conditional_exposure.p_a_given_b(
                x[f"{s}_prevalence{e}"],
                x[f"p_acquiring_hiv{e}"],
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
            * pl.col(f"p_acquiring_hiv{estimate}")
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
    > 0
)

# use RR definition and law of total probability to calculate
# P(acquired HIV (attributable to STI) | has STI)
# The causal part is baked into using the causal adjusted RRs
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
hiv_sti = hiv_sti.with_columns(
    [
        (
            pl.col(f"p_acquiring_hiv_given_{sti}{estimate}")
            * pl.col(f"{sti}_prevalence{estimate}")
        ).alias(f"p_hiv_attributable_{sti}{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
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
# NB that the two steps above are equivalent to calculating P(STI was causative|HIV)
# directly using Bayes theorem, but this step-by-step is more interpretable

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

hiv_sti.write_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))
