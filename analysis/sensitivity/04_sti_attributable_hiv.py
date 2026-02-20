import os

import numpy as np
import polars as pl

from averted_burden import conditional_exposure

output_dir = "outputs_unaids_sensitivity"

# we want to assess the probability that an individual with gc acquired hiv
# we need the RR of acquiring HIV given gonorrhea
rrs_causal = pl.read_csv(
    os.path.join(output_dir, "meta_estimated_RRs_causal_STI_given_HIV.csv")
)
hiv_sti = pl.read_csv(os.path.join(output_dir, "hiv_sti_with_gc_abx_r.csv"))

# Filter out rows with null values in critical columns
hiv_sti = hiv_sti.filter(
    pl.col("p_acquiring_hiv").is_not_null()
    & pl.col("p_acquiring_hiv_lower").is_not_null()
    & pl.col("p_acquiring_hiv_upper").is_not_null()
    & pl.col("gc_prevalence").is_not_null()
    & pl.col("chlamydia_prevalence").is_not_null()
    & pl.col("syphilis_prevalence").is_not_null()
    & pl.col("trichomoniasis_prevalence").is_not_null()
)

# relabel the sex column to match the GBD data
rrs_causal = rrs_causal.with_columns(
    sex=pl.when(pl.col("sex") == "Heterosexual Women")
    .then(pl.lit("Female"))
    .otherwise(pl.lit("Male"))
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
    sex=pl.when(pl.col("sex") == "Heterosexual Women")
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
hiv_sti = hiv_sti.join(rrs_associative, on=["sex"], how="inner")

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
