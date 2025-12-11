import os

import polars as pl

from averted_burden import conditional_exposure, rr_meta_estimate

data_dir = "data"
output_dir = "outputs"

# read data
incidence = pl.read_csv(os.path.join(data_dir, "ihme_incidence.csv"))
prevalence = pl.read_csv(os.path.join(data_dir, "ihme_prevalence.csv"))

# aggregate data by location, summing over ages
incidence = incidence.group_by(
    ["measure", "location", "sex", "cause", "metric", "year"]
).agg(pl.sum("val"), pl.sum("lower"), pl.sum("upper"))

prevalence = prevalence.group_by(
    ["measure", "location", "sex", "cause", "metric", "year"]
).agg(pl.sum("val"), pl.sum("lower"), pl.sum("upper"))

# we want number and rate of incident hiv cases by country
incidence_hiv = incidence.filter((pl.col("cause") == "HIV/AIDS"))

# group by sex and year, summing over locations
incidence_hiv = incidence_hiv.group_by(
    ["sex", "year", "measure", "metric", "location"]
).agg(
    pl.sum("val"),
    pl.sum("lower"),
    pl.sum("upper"),
)

# hiv rate and number by sex, location, year
incidence_hiv = incidence_hiv.filter(pl.col("metric") != "Percent")

# can calculate effective population size by dividing number by rate
incidence_hiv_number = (
    incidence_hiv.filter(pl.col("metric") == "Number")
    .rename(
        {
            "val": "hiv_incidence_number",
            "lower": "hiv_incidence_number_lower",
            "upper": "hiv_incidence_number_upper",
        }
    )
    .drop(["measure", "metric"])
)

incidence_hiv_rate = (
    incidence_hiv.filter(pl.col("metric") == "Rate")
    .rename(
        {
            "val": "hiv_incidence_rate",
            "lower": "hiv_incidence_rate_lower",
            "upper": "hiv_incidence_rate_upper",
        }
    )
    .drop(["measure", "metric"])
)

# join the two together
incidence_hiv = incidence_hiv_number.join(
    incidence_hiv_rate, on=["sex", "year", "location"], how="inner"
)

# calculate an effective population size
incidence_hiv = incidence_hiv.with_columns(
    # recall that the rate is per 100,000 population
    population=(pl.col("hiv_incidence_number") / pl.col("hiv_incidence_rate"))
    * 100000
)

# we need prevalence of gc
prevalence_gc = prevalence.filter(
    (pl.col("cause") == "Gonococcal infection")
    & (pl.col("metric") == "Number")
)

prevalence_gc = (
    prevalence_gc.rename(
        {
            "val": "gc_prevalence",
            "lower": "gc_prevalence_lower",
            "upper": "gc_prevalence_upper",
        }
    )
    .with_columns(
        gc_prevalence=pl.col("gc_prevalence"),
        gc_prevalence_lower=pl.col("gc_prevalence_lower"),
        gc_prevalence_upper=pl.col("gc_prevalence_upper"),
    )
    .drop(["measure", "metric", "cause"])
)

# join to incidence data
data = incidence_hiv.join(
    prevalence_gc, on=["sex", "year", "location"], how="inner"
)

# convert gc prevalence to be per effective population
data = data.with_columns(
    gc_prevalence=pl.col("gc_prevalence") / pl.col("population"),
    gc_prevalence_lower=pl.col("gc_prevalence_lower") / pl.col("population"),
    gc_prevalence_upper=pl.col("gc_prevalence_upper") / pl.col("population"),
)

# we want to calculate the prevalence of gc among hiv+ individuals
# we scale the overall prevalence of gc by the risk of gc among hiv+ individuals

rrs = pl.read_csv(os.path.join(data_dir, "RRs_GC_associated_with_HIV.csv"))

rrs = rrs.with_columns(
    log_val=pl.col("val").log(),
    log_lower=pl.col("lower").log(),
    log_upper=pl.col("upper").log(),
).with_columns(sigma=(pl.col("log_upper") - pl.col("log_lower")) / (2 * 1.96))

rr_mu, rr_lower, rr_upper = rr_meta_estimate.meta_estimate_rrs(
    means=rrs["log_val"].to_list(),
    sigmas=rrs["sigma"].to_list(),
    group_assignments=rrs["sex"].to_list(),
)

results = pl.DataFrame(
    {
        "sex": list(set(rrs["sex"].to_list())),
        "rr_mu_gc_hiv_coinfection": rr_mu,
        "rr_lower_gc_hiv_coinfection": rr_lower,
        "rr_upper_gc_hiv_coinfection": rr_upper,
    }
).with_columns(
    # make the sex names to be the same as GBD data
    sex=pl.when(pl.col("sex") == "Heterosexual Women")
    .then(pl.lit("Female"))
    .otherwise(pl.lit("Male"))
)

# group up with main data
# note that this drops data by "both" sexes
data = data.join(results, on=["sex"], how="inner")

# adjust prevalence rates to be among hiv+ individuals
# first we need the probability of hiv incidence
data = data.with_columns(
    p_hiv=pl.col("hiv_incidence_number") / pl.col("population"),
    p_hiv_lower=pl.col("hiv_incidence_number_lower") / pl.col("population"),
    p_hiv_upper=pl.col("hiv_incidence_number_upper") / pl.col("population"),
)

data = data.with_columns(
    gc_prevalence_hiv_pos=pl.struct(
        ["gc_prevalence", "p_hiv", "rr_mu_gc_hiv_coinfection"]
    ).map_elements(
        lambda x: conditional_exposure.p_a_given_b(
            x["gc_prevalence"],
            x["p_hiv"],
            x["rr_mu_gc_hiv_coinfection"],
        ),
        return_dtype=pl.Float64,
    ),
    gc_prevalence_hiv_pos_lower=pl.struct(
        ["gc_prevalence_lower", "p_hiv_lower", "rr_lower_gc_hiv_coinfection"]
    ).map_elements(
        lambda x: conditional_exposure.p_a_given_b(
            x["gc_prevalence_lower"],
            x["p_hiv_lower"],
            x["rr_lower_gc_hiv_coinfection"],
        ),
        return_dtype=pl.Float64,
    ),
    gc_prevalence_hiv_pos_upper=pl.struct(
        ["gc_prevalence_upper", "p_hiv_upper", "rr_upper_gc_hiv_coinfection"]
    ).map_elements(
        lambda x: conditional_exposure.p_a_given_b(
            x["gc_prevalence_upper"],
            x["p_hiv_upper"],
            x["rr_upper_gc_hiv_coinfection"],
        ),
        return_dtype=pl.Float64,
    ),
)

# save the assembled data
data.write_csv(os.path.join(output_dir, "hiv_gc.csv"))
