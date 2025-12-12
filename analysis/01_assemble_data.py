import os

import polars as pl

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

# join up with population data
population = pl.read_csv(os.path.join(data_dir, "ihme_population.csv"))

population = (
    population.group_by(["location", "year", "sex"])
    .agg(
        pl.sum("val"),
        pl.sum("lower"),
        pl.sum("upper"),
    )
    .rename(
        {
            "val": "population",
            "lower": "population_lower",
            "upper": "population_upper",
        }
    )
)

incidence_hiv = incidence_hiv.join(
    # note that this will drop data on "both" sexes
    population,
    on=["sex", "year", "location"],
    how="inner",
)

# to get the true rate of acquiring HIV, we need the prevalence of HIV
prevalence_hiv = pl.read_csv(os.path.join(data_dir, "ihme_hiv_prevalence.csv"))
prevalence_hiv = prevalence_hiv.filter(pl.col("cause") == "HIV/AIDS")
prevalence_hiv = prevalence_hiv.filter(pl.col("metric") == "Number")
prevalence_hiv = prevalence_hiv.filter(pl.col("measure") == "Prevalence")
# sum over age groups
prevalence_hiv = (
    prevalence_hiv.group_by(["location", "year", "sex"])
    .agg(
        pl.sum("val"),
        pl.sum("lower"),
        pl.sum("upper"),
    )
    .rename(
        {
            "val": "hiv_prevalence",
            "lower": "hiv_prevalence_lower",
            "upper": "hiv_prevalence_upper",
        }
    )
)

# join to incidence data
incidence_hiv = incidence_hiv.join(
    prevalence_hiv, on=["sex", "year", "location"], how="inner"
)

# calculate the probability of a susceptible person acquiring HIV
# we want to calculate the number of incident infections over the susceptible
# population
# susceptible population = total population - prevalent infections
# but, the prevalent infections already include some of the incident infections
# IHME reports mid-year prevalence, and incidence is over the full year,
# so we have to subtract out half of the incidence
incidence_hiv = incidence_hiv.with_columns(
    p_hiv=pl.col("hiv_incidence_number")
    / (
        pl.col("population")
        - pl.col("hiv_prevalence")
        - 0.5 * pl.col("hiv_incidence_number")
    ),
    p_hiv_lower=pl.col("hiv_incidence_number_lower")
    / (
        pl.col("population_lower")
        - pl.col("hiv_prevalence_lower")
        - 0.5 * pl.col("hiv_incidence_number_lower")
    ),
    p_hiv_upper=pl.col("hiv_incidence_number_upper")
    / (
        pl.col("population_upper")
        - pl.col("hiv_prevalence_upper")
        - 0.5 * pl.col("hiv_incidence_number_upper")
    ),
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
    gc_prevalence_lower=pl.col("gc_prevalence_lower")
    / pl.col("population_lower"),
    gc_prevalence_upper=pl.col("gc_prevalence_upper")
    / pl.col("population_upper"),
)

# save the assembled data
data.write_csv(os.path.join(output_dir, "hiv_gc.csv"))
