import os

import polars as pl

data_dir = "data"
output_dir = "outputs"

# read data
incidence = pl.read_csv(os.path.join(data_dir, "ihme_incidence.csv"))
prevalence = pl.read_csv(os.path.join(data_dir, "ihme_prevalence.csv"))

# aggregate data by location, summing over ages
incidence = incidence.filter(pl.col("metric") == "Number")
incidence = incidence.group_by(
    ["measure", "location", "sex", "cause", "metric", "year"]
).agg(pl.sum("val"), pl.sum("lower"), pl.sum("upper"))

prevalence = prevalence.group_by(
    ["measure", "location", "sex", "cause", "metric", "year"]
).agg(pl.sum("val"), pl.sum("lower"), pl.sum("upper"))

# we want number incident hiv cases by country
incidence_hiv = incidence.filter((pl.col("cause") == "HIV/AIDS"))

# group by sex and year, summing over locations
incidence_hiv = incidence_hiv.group_by(
    ["sex", "year", "measure", "metric", "location"]
).agg(
    pl.sum("val"),
    pl.sum("lower"),
    pl.sum("upper"),
)

incidence_hiv = incidence_hiv.rename(
    {
        "val": "hiv_incidence_number",
        "lower": "hiv_incidence_number_lower",
        "upper": "hiv_incidence_number_upper",
    }
).drop(["measure", "metric"])

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
            "val": "hiv_prevalence_number",
            "lower": "hiv_prevalence_number_lower",
            "upper": "hiv_prevalence_number_upper",
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
    p_acquiring_hiv=pl.col("hiv_incidence_number")
    / (
        pl.col("population")
        - pl.col("hiv_prevalence_number")
        - 0.5 * pl.col("hiv_incidence_number")
    ),
    p_acquiring_hiv_lower=pl.col("hiv_incidence_number_lower")
    / (
        pl.col("population_lower")
        - pl.col("hiv_prevalence_number_lower")
        - 0.5 * pl.col("hiv_incidence_number_lower")
    ),
    p_acquiring_hiv_upper=pl.col("hiv_incidence_number_upper")
    / (
        pl.col("population_upper")
        - pl.col("hiv_prevalence_number_upper")
        - 0.5 * pl.col("hiv_incidence_number_upper")
    ),
)

# we need prevalence of gc
# we have to do this separately from the other STIs because the data are in a
# separate CSV
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

# we also want to join up with the other sti data
prevalence_sti = pl.read_csv(os.path.join(data_dir, "ihme_sti_prevalence.csv"))
prevalence_sti = prevalence_sti.filter(pl.col("metric") == "Number")
prevalence_sti = prevalence_sti.group_by(
    ["location", "sex", "cause", "year"]
).sum()
prevalence_sti = prevalence_sti.drop(["measure", "metric", "age"])
# pivot wider so we have columns for each sti
prevalence_sti = prevalence_sti.pivot(
    on="cause", values=["val", "upper", "lower"]
)
# rename
prevalence_sti = prevalence_sti.rename(
    {
        "val_Chlamydial infection": "chlamydia_prevalence",
        "val_Trichomoniasis": "trichomoniasis_prevalence",
        "val_Syphilis": "syphilis_prevalence",
        "upper_Chlamydial infection": "chlamydia_prevalence_upper",
        "upper_Trichomoniasis": "trichomoniasis_prevalence_upper",
        "upper_Syphilis": "syphilis_prevalence_upper",
        "lower_Chlamydial infection": "chlamydia_prevalence_lower",
        "lower_Trichomoniasis": "trichomoniasis_prevalence_lower",
        "lower_Syphilis": "syphilis_prevalence_lower",
    }
)

# join with existing dataframe
data = data.join(prevalence_sti, on=["sex", "year", "location"], how="inner")

# convert sti prevalence to be per effective population
data = data.with_columns(
    gc_prevalence=pl.col("gc_prevalence") / pl.col("population"),
    gc_prevalence_lower=pl.col("gc_prevalence_lower")
    / pl.col("population_lower"),
    gc_prevalence_upper=pl.col("gc_prevalence_upper")
    / pl.col("population_upper"),
    chlamydia_prevalence=pl.col("chlamydia_prevalence") / pl.col("population"),
    chlamydia_prevalence_lower=pl.col("chlamydia_prevalence_lower")
    / pl.col("population_lower"),
    chlamydia_prevalence_upper=pl.col("chlamydia_prevalence_upper")
    / pl.col("population_upper"),
    syphilis_prevalence=pl.col("syphilis_prevalence") / pl.col("population"),
    syphilis_prevalence_lower=pl.col("syphilis_prevalence_lower")
    / pl.col("population_lower"),
    syphilis_prevalence_upper=pl.col("syphilis_prevalence_upper")
    / pl.col("population_upper"),
    trichomoniasis_prevalence=pl.col("trichomoniasis_prevalence")
    / pl.col("population"),
    trichomoniasis_prevalence_lower=pl.col("trichomoniasis_prevalence_lower")
    / pl.col("population_lower"),
    trichomoniasis_prevalence_upper=pl.col("trichomoniasis_prevalence_upper")
    / pl.col("population_upper"),
)

# finally, we need hiv treatment rates over time
hiv_treatment_rate = pl.read_csv(
    os.path.join(data_dir, "hiv_treatment_rate.csv")
)
# divide by 100 to make proportions
hiv_treatment_rate = hiv_treatment_rate.with_columns(
    val=pl.col("val") / 100,
    lower=pl.col("lower") / 100,
    upper=pl.col("upper") / 100,
)
# rename to be more informative
hiv_treatment_rate = hiv_treatment_rate.rename(
    {
        "val": "treatment_proportion",
        "lower": "treatment_proportion_lower",
        "upper": "treatment_proportion_upper",
    }
)
# join to existing data on basis of year and sex
data = data.join(hiv_treatment_rate, on=["year", "sex"], how="left")
# since any missing values are pre 2023, impute hiv treatment rate as 0
data = data.with_columns(
    treatment_proportion=pl.col("treatment_proportion").fill_null(0),
    treatment_proportion_lower=pl.col("treatment_proportion_lower").fill_null(
        0
    ),
    treatment_proportion_upper=pl.col("treatment_proportion_upper").fill_null(
        0
    ),
)

# save the assembled data
data.write_csv(os.path.join(output_dir, "hiv_sti.csv"))
