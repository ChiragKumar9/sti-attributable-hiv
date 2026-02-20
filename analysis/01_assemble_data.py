import os

import polars as pl
import yaml

data_dir = "data"
output_dir = "outputs"

# read parameters
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

countries_to_regions = {
    "Somalia": "Eastern",
    "Central African Republic": "Central",
    "Ghana": "Western",
    "Mauritania": "Western",
    "Mauritius": "Eastern",
    "Equatorial Guinea": "Central",
    "Benin": "Western",
    "Kenya": "Eastern",
    "Eritrea": "Eastern",
    "Cameroon": "Western",
    "Angola": "Central",
    "Botswana": "Southern",
    "Mozambique": "Eastern",
    "Niger": "Western",
    "Democratic Republic of the Congo": "Central",
    "Mali": "Western",
    "Gabon": "Central",
    "Guinea": "Western",
    "Namibia": "Southern",
    "Djibouti": "Eastern",
    "United Republic of Tanzania": "Eastern",
    "South Africa": "Southern",
    "Madagascar": "Eastern",
    "Côte d'Ivoire": "Western",
    "Gambia": "Western",
    "Rwanda": "Eastern",
    "Comoros": "Eastern",
    "Zambia": "Eastern",
    "Senegal": "Western",
    "Sao Tome and Principe": "Western",
    "Lesotho": "Southern",
    "Ethiopia": "Eastern",
    "Eswatini": "Southern",
    "Burkina Faso": "Western",
    "Cabo Verde": "Western",
    "South Sudan": "Eastern",
    "Congo": "Central",
    "Liberia": "Western",
    "Guinea-Bissau": "Western",
    "Nigeria": "Western",
    "Burundi": "Eastern",
    "Togo": "Western",
    "Uganda": "Eastern",
    "Zimbabwe": "Southern",
    "Malawi": "Eastern",
    "Sierra Leone": "Western",
    "Chad": "Western",
}

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
# note that gbd themselves say that incidence is wrt to mid year population size
incidence_hiv = incidence_hiv.with_columns(
    hiv_prevalence_year_end_number=pl.col("hiv_prevalence_number")
    + 0.5 * pl.col("hiv_incidence_number"),
    hiv_prevalence_year_end_number_lower=pl.col("hiv_prevalence_number_lower")
    + 0.5 * pl.col("hiv_incidence_number_lower"),
    hiv_prevalence_year_end_number_upper=pl.col("hiv_prevalence_number_upper")
    + 0.5 * pl.col("hiv_incidence_number_upper"),
)

incidence_hiv = incidence_hiv.with_columns(
    p_acquiring_hiv=pl.col("hiv_incidence_number")
    / (pl.col("population") - pl.col("hiv_prevalence_number")),
    p_acquiring_hiv_lower=pl.col("hiv_incidence_number_lower")
    / (pl.col("population_lower") - pl.col("hiv_prevalence_number_lower")),
    p_acquiring_hiv_upper=pl.col("hiv_incidence_number_upper")
    / (pl.col("population_upper") - pl.col("hiv_prevalence_number_upper")),
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
    prevalence_gc,
    left_on=["sex", "year", "location"],
    right_on=["sex", "year", "location"],
    how="inner",
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
data = data.join(
    prevalence_sti,
    left_on=["sex", "year", "location"],
    right_on=["sex", "year", "location"],
    how="inner",
)

# convert sti prevalence to be per effective population
# we don't care to do the 0.5 adjustment here because all that matters is the
# proportion of the population with the STI (and also STI prevalences are
# rather constant over time)
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
    os.path.join(data_dir, "art-coverage-african-countries.csv")
)

hiv_treatment_rate = hiv_treatment_rate.rename(
    {
        "Year": "year",
        "CountryName": "location",
        "CoverageVal": "val",
    }
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

# make country names consistent
hiv_treatment_rate = hiv_treatment_rate.with_columns(
    location=pl.when(pl.col("location").str.contains(", "))
    .then(pl.col("location").str.split(", ").list.reverse().list.join(" "))
    .otherwise(pl.col("location"))
)

# join to existing data on basis of year and location
data = data.join(hiv_treatment_rate, on=["year", "location"], how="left")

# fill in any missing values with the average value for that year
data = data.with_columns(
    treatment_proportion=pl.col("treatment_proportion").fill_null(
        pl.col("treatment_proportion").mean().over("year")
    ),
    treatment_proportion_lower=pl.col("treatment_proportion_lower").fill_null(
        pl.col("treatment_proportion_lower").mean().over("year")
    ),
    treatment_proportion_upper=pl.col("treatment_proportion_upper").fill_null(
        pl.col("treatment_proportion_upper").mean().over("year")
    ),
)

# fill any remaining missing values with 0 because they are pre data
data = data.with_columns(
    treatment_proportion=pl.col("treatment_proportion").fill_null(0),
    treatment_proportion_lower=pl.col("treatment_proportion_lower").fill_null(
        0
    ),
    treatment_proportion_upper=pl.col("treatment_proportion_upper").fill_null(
        0
    ),
)

# map countries to regions
data = data.with_columns(
    region=pl.col("location").replace(countries_to_regions)
)

# let's take out part of the male population and make it msm
msm_fraction = params["msm_fraction"]
msm_data = (
    data.filter(pl.col("sex") == "Male")
    .with_columns(
        # take hiv incidence and prevalence numbers and scale them by the msm fraction
        **{
            col: pl.col(col) * msm_fraction
            for col in [
                "hiv_incidence_number",
                "hiv_incidence_number_lower",
                "hiv_incidence_number_upper",
                "hiv_prevalence_number",
                "hiv_prevalence_number_lower",
                "hiv_prevalence_number_upper",
                "hiv_prevalence_year_end_number",
                "hiv_prevalence_year_end_number_lower",
                "hiv_prevalence_year_end_number_upper",
                "population",
                "population_lower",
                "population_upper",
            ]
        }
    )
    .with_columns(sex=pl.lit("MSM"))
)

# now we need to similarly adjust the incidence and prevalence for the remaining population
non_msm_data = data.filter(pl.col("sex") == "Male").with_columns(
    # take all columns besides year, sex, location and multiply by (1 - msm fraction)
    **{
        col: pl.col(col) * (1 - msm_fraction)
        for col in [
            "hiv_incidence_number",
            "hiv_incidence_number_lower",
            "hiv_incidence_number_upper",
            "hiv_prevalence_number",
            "hiv_prevalence_number_lower",
            "hiv_prevalence_number_upper",
            "hiv_prevalence_year_end_number",
            "hiv_prevalence_year_end_number_lower",
            "hiv_prevalence_year_end_number_upper",
            "population",
            "population_lower",
            "population_upper",
        ]
    }
)

female_data = data.filter(pl.col("sex") == "Female")

# join back together
data = pl.concat([msm_data, non_msm_data, female_data])

# save the assembled data
data.write_csv(os.path.join(output_dir, "hiv_sti.csv"))
