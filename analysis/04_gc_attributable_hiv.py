import os

import polars as pl

from averted_burden import population_attributable_fraction

output_dir = "outputs"

rrs = pl.read_csv(os.path.join(output_dir, "meta_estimated_RRs.csv"))

hiv_gc = pl.read_csv(os.path.join(output_dir, "hiv_gc_with_abx.csv"))

# join the two
# adjust the rrs to have the sex labels be the same as GBD
rrs = rrs.with_columns(
    sex=pl.when(pl.col("sex") == "Heterosexual Women")
    .then(pl.lit("Female"))
    .otherwise(pl.lit("Male"))
)
# rename the rr_mu elements to be informative
rrs = rrs.rename(
    {
        "rr_mu": "rr_mu_hiv_given_gc",
        "rr_lower": "rr_lower_hiv_given_gc",
        "rr_upper": "rr_upper_hiv_given_gc",
    }
)

# join the data up
hiv_gc = hiv_gc.join(rrs, on=["sex"])

# calculate the population attributable fractions

hiv_gc = hiv_gc.with_columns(
    paf_gc_hiv=pl.struct(
        ["gc_prevalence_hiv_pos", "rr_mu_hiv_given_gc"]
    ).map_elements(
        lambda x: population_attributable_fraction.paf(
            x["rr_mu_hiv_given_gc"], x["gc_prevalence_hiv_pos"]
        ),
        return_dtype=pl.Float64,
    ),
    paf_gc_hiv_lower=pl.struct(
        ["gc_prevalence_hiv_pos_lower", "rr_lower_hiv_given_gc"]
    ).map_elements(
        lambda x: population_attributable_fraction.paf(
            x["rr_lower_hiv_given_gc"], x["gc_prevalence_hiv_pos_lower"]
        ),
        return_dtype=pl.Float64,
    ),
    paf_gc_hiv_upper=pl.struct(
        ["gc_prevalence_hiv_pos_upper", "rr_upper_hiv_given_gc"]
    ).map_elements(
        lambda x: population_attributable_fraction.paf(
            x["rr_upper_hiv_given_gc"], x["gc_prevalence_hiv_pos_upper"]
        ),
        return_dtype=pl.Float64,
    ),
)

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
