import os

import polars as pl

from averted_burden import conditional_exposure

output_dir = "outputs"

# we want to assess the probability that an individual with gc acquired hiv
# we need the RR of acquiring HIV given gonorrhea
rrs_causal = pl.read_csv(
    os.path.join(output_dir, "meta_estimated_RRs_causal_GC_given_HIV.csv")
)
hiv_gc = pl.read_csv(os.path.join(output_dir, "hiv_gc_with_abx.csv"))
# relabel the sex column to match the GBD data
rrs_causal = rrs_causal.with_columns(
    sex=pl.when(pl.col("sex") == "Heterosexual Women")
    .then(pl.lit("Female"))
    .otherwise(pl.lit("Male"))
).rename(
    {
        "rr_mu": "rr_mu_hiv_given_gc",
        "rr_lower": "rr_lower_hiv_given_gc",
        "rr_upper": "rr_upper_hiv_given_gc",
    }
)

# group up with main data
hiv_gc = hiv_gc.join(rrs_causal, on=["sex"], how="inner")

# use RR definition and law of total probability to calculate
# P(acquired HIV (attributable to GC) | has GC)
# The causal part is baked into using the causal adjusted RRs
hiv_gc = hiv_gc.with_columns(
    p_hiv_given_gc=pl.struct(
        ["p_hiv", "gc_prevalence", "rr_mu_hiv_given_gc"]
    ).map_elements(
        lambda x: conditional_exposure.p_a_given_b(
            x["p_hiv"], x["gc_prevalence"], x["rr_mu_hiv_given_gc"]
        ),
        return_dtype=pl.Float64,
    ),
    p_hiv_given_gc_lower=pl.struct(
        ["p_hiv_lower", "gc_prevalence_lower", "rr_lower_hiv_given_gc"]
    ).map_elements(
        lambda x: conditional_exposure.p_a_given_b(
            x["p_hiv_lower"],
            x["gc_prevalence_lower"],
            x["rr_lower_hiv_given_gc"],
        ),
        return_dtype=pl.Float64,
    ),
    p_hiv_given_gc_upper=pl.struct(
        ["p_hiv_upper", "gc_prevalence_upper", "rr_upper_hiv_given_gc"]
    ).map_elements(
        lambda x: conditional_exposure.p_a_given_b(
            x["p_hiv_upper"],
            x["gc_prevalence_upper"],
            x["rr_upper_hiv_given_gc"],
        ),
        return_dtype=pl.Float64,
    ),
)

# we want to calculate the probability of acquiring hiv because of gc,
# not conditioned on having gc
hiv_gc = hiv_gc.with_columns(
    p_hiv_attributable_gc=pl.col("p_hiv_given_gc") * pl.col("gc_prevalence"),
    p_hiv_attributable_gc_lower=pl.col("p_hiv_given_gc_lower")
    * pl.col("gc_prevalence_lower"),
    p_hiv_attributable_gc_upper=pl.col("p_hiv_given_gc_upper")
    * pl.col("gc_prevalence_upper"),
)

# now we want to calculate how much of hiv is attributable to gc
hiv_gc = hiv_gc.with_columns(
    paf_gc_hiv=pl.col("p_hiv_attributable_gc") / pl.col("p_hiv"),
    paf_gc_hiv_lower=pl.col("p_hiv_attributable_gc_lower")
    / pl.col("p_hiv_lower"),
    paf_gc_hiv_upper=pl.col("p_hiv_attributable_gc_upper")
    / pl.col("p_hiv_upper"),
)
# NB that the two steps above are equivalent to calculating P(GC|HIV) directly
# using Bayes theorem, but this is more interpretable

# multiply the pafs by the hiv incidence (number and rate) to get attributable
# infections

hiv_gc = hiv_gc.with_columns(
    hiv_incidence_number_attributable=pl.col("hiv_incidence_number")
    * pl.col("paf_gc_hiv"),
    hiv_incidence_number_attributable_lower=pl.col(
        "hiv_incidence_number_lower"
    )
    * pl.col("paf_gc_hiv_lower"),
    hiv_incidence_number_attributable_upper=pl.col(
        "hiv_incidence_number_upper"
    )
    * pl.col("paf_gc_hiv_upper"),
    # same for rate
    hiv_incidence_rate_attributable=pl.col("hiv_incidence_rate")
    * pl.col("paf_gc_hiv"),
    hiv_incidence_rate_attributable_lower=pl.col("hiv_incidence_rate_lower")
    * pl.col("paf_gc_hiv_lower"),
    hiv_incidence_rate_attributable_upper=pl.col("hiv_incidence_rate_upper")
    * pl.col("paf_gc_hiv_upper"),
)

# print out the dataframe sorted by paf
print(
    hiv_gc.sort(by="paf_gc_hiv", descending=True)[
        [
            "sex",
            "year",
            "location",
            "paf_gc_hiv",
            "paf_gc_hiv_lower",
            "paf_gc_hiv_upper",
        ]
    ]
)
# print out the dataframe sorted by number attributable
print(
    hiv_gc.sort(by="hiv_incidence_number_attributable", descending=True)[
        [
            "sex",
            "year",
            "location",
            "hiv_incidence_number_attributable",
            "paf_gc_hiv",
            "hiv_incidence_number_attributable_lower",
            "hiv_incidence_number_attributable_upper",
        ]
    ]
)

# now multiply by resistance rates to get the number of cases resistant to a
# particular drug

hiv_gc = hiv_gc.with_columns(
    Ciprofloxacin_resistant_number=pl.col("hiv_incidence_number_attributable")
    * pl.col("Ciprofloxacin"),
    Ciprofloxacin_resistant_number_lower=pl.col(
        "hiv_incidence_number_attributable_lower"
    )
    * pl.col("Ciprofloxacin_lower"),
    Ciprofloxacin_resistant_number_upper=pl.col(
        "hiv_incidence_number_attributable_upper"
    )
    * pl.col("Ciprofloxacin_upper"),
    Ciprofloxacin_resistant_rate=pl.col("hiv_incidence_rate_attributable")
    * pl.col("Ciprofloxacin"),
    Ciprofloxacin_resistant_rate_lower=pl.col(
        "hiv_incidence_rate_attributable_lower"
    )
    * pl.col("Ciprofloxacin_lower"),
    Ciprofloxacin_resistant_rate_upper=pl.col(
        "hiv_incidence_rate_attributable_upper"
    )
    * pl.col("Ciprofloxacin_upper"),
    # Cefixime
    Cefixime_resistant_number=pl.col("hiv_incidence_number_attributable")
    * pl.col("Cefixime"),
    Cefixime_resistant_number_lower=pl.col(
        "hiv_incidence_number_attributable_lower"
    )
    * pl.col("Cefixime_lower"),
    Cefixime_resistant_number_upper=pl.col(
        "hiv_incidence_number_attributable_upper"
    )
    * pl.col("Cefixime_upper"),
    Cefixime_resistant_rate=pl.col("hiv_incidence_rate_attributable")
    * pl.col("Cefixime"),
    Cefixime_resistant_rate_lower=pl.col(
        "hiv_incidence_rate_attributable_lower"
    )
    * pl.col("Cefixime_lower"),
    Cefixime_resistant_rate_upper=pl.col(
        "hiv_incidence_rate_attributable_upper"
    )
    * pl.col("Cefixime_upper"),
    # Azithromycin
    Azithromycin_resistant_number=pl.col("hiv_incidence_number_attributable")
    * pl.col("Azithromycin"),
    Azithromycin_resistant_number_lower=pl.col(
        "hiv_incidence_number_attributable_lower"
    )
    * pl.col("Azithromycin_lower"),
    Azithromycin_resistant_number_upper=pl.col(
        "hiv_incidence_number_attributable_upper"
    )
    * pl.col("Azithromycin_upper"),
    Azithromycin_resistant_rate=pl.col("hiv_incidence_rate_attributable")
    * pl.col("Azithromycin"),
    Azithromycin_resistant_rate_lower=pl.col(
        "hiv_incidence_rate_attributable_lower"
    )
    * pl.col("Azithromycin_lower"),
    Azithromycin_resistant_rate_upper=pl.col(
        "hiv_incidence_rate_attributable_upper"
    )
    * pl.col("Azithromycin_upper"),
)

hiv_gc.write_csv(os.path.join(output_dir, "hiv_attributable_to_gc.csv"))
