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

# country name to iso3 code mapping, with manual overrides for problematic IHME names
country_codes_df = pl.read_csv(os.path.join(data_dir, "country_codes.csv"))
_code_lookup = {row["name"]: row["alpha-3"] for row in country_codes_df.to_dicts()}
_manual_iso3 = {
    "United Republic of Tanzania": "TZA",
    "Democratic Republic of the Congo": "COD",
    "Côte d'Ivoire": "CIV",
    "Cote d'Ivoire": "CIV",
    "Congo": "COG",
    "Sao Tome and Principe": "STP",
    "Cabo Verde": "CPV",
    "Eswatini": "SWZ",
    "South Sudan": "SSD",
    "Gambia": "GMB",
    "Guinea-Bissau": "GNB",
}
_code_lookup.update(_manual_iso3)
ihme_name_to_iso3 = _code_lookup

# convert country names in countries_to_regions to iso3 codes
countries_to_regions = {
    ihme_name_to_iso3[country]: region
    for country, region in countries_to_regions.items()
}

def read_unaids_wide(filepath, sex_label, value_col_name):
    """reads wide format UNAIDs data and returns long format with cleaned numeric values, and a column for country codes"""
    df = pl.read_csv(filepath, infer_schema_length=0) 
    df = df.rename({col: col.strip() for col in df.columns}) # remove white space in colnames

    # differentiate year, lower bound, and upper bound columns
    year_cols = [c for c in df.columns if c.isdigit()]
    lower_cols = [c for c in df.columns if c.endswith("_lower") and c.split("_")[0].isdigit()]
    upper_cols = [c for c in df.columns if c.endswith("_upper") and c.split("_")[0].isdigit()]

    # pivot main values from wide to long, on country and year
    val_long = df.select(["Country"] + year_cols)
    val_long = val_long.unpivot(index="Country", variable_name="year", value_name="val_raw")
    val_long = val_long.with_columns(pl.col("year").cast(pl.Int64)) 

    # same for lower bounds
    lower_long = df.select(["Country"] + lower_cols)
    lower_long = lower_long.unpivot(index="Country", variable_name="yr", value_name="lower_raw")
    lower_long = lower_long.with_columns(
        pl.col("yr").str.replace("_lower", "").cast(pl.Int64).alias("year")
    )
    lower_long = lower_long.drop("yr")

    # same for upper bounds
    upper_long = df.select(["Country"] + upper_cols)
    upper_long = upper_long.unpivot(index="Country", variable_name="yr", value_name="upper_raw")
    upper_long = upper_long.with_columns(
        pl.col("yr").str.replace("_upper", "").cast(pl.Int64).alias("year")
    )
    upper_long = upper_long.drop("yr")

    long = val_long.join(lower_long, on=["Country", "year"]).join(upper_long, on=["Country", "year"])

    # raw data has "..." for missing data
    # assumed values when < reported
    # if <100 --> make 50 
    # if <200 --> make 150
    def clean_col(col_name):
        return (
            pl.when(pl.col(col_name).is_null() | (pl.col(col_name).str.strip_chars() == "..."))
            .then(None)
            .when(pl.col(col_name).str.contains("<100"))
            .then(pl.lit(50.0))
            .when(pl.col(col_name).str.contains("<200")) 
            .then(pl.lit(150.0))
            .when(pl.col(col_name).str.contains("<500")) 
            .then(pl.lit(350.0))
            .when(pl.col(col_name).str.contains("<0.01"))
            .then(pl.lit(0.005))
            .when(pl.col(col_name).str.contains("<0.1"))
            .then(pl.lit(0.05))
            .otherwise(pl.col(col_name).str.replace_all(" ", "").cast(pl.Float64, strict=False))
            )

    long = long.with_columns(
        clean_col("val_raw").alias(value_col_name),
        clean_col("lower_raw").alias(f"{value_col_name}_lower"),
        clean_col("upper_raw").alias(f"{value_col_name}_upper"),
        pl.lit(sex_label).alias("sex"),
        pl.col("Country").replace(ihme_name_to_iso3).alias("country_code"),
    )

    # note - some non SSA countries don't map to iso3 codes, but this doesn't matter for us
    long = long.drop(["val_raw", "lower_raw", "upper_raw", "Country"])
    return long

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

# join UN population data (15+, Median variant, Male/Female by country-year)
un_pop = (
    pl.read_csv(os.path.join(data_dir, "un_pop_data.csv"))
    .rename({"Location": "location", "Sex": "sex", "Time": "year", "Value": "un_pop"})
    .select(["location", "sex", "year", "un_pop"])
)
incidence_hiv = incidence_hiv.join(un_pop, on=["location", "sex", "year"], how="left")

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
prevalence_sti = prevalence_sti.drop(["measure", "metric", "age"])
prevalence_sti = prevalence_sti.group_by(
    ["location", "sex", "cause", "year"]
).sum()
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

# add a column for country codes to ihme data before joining with treatment data
data = data.with_columns(
    country_code=pl.col("location").replace(ihme_name_to_iso3)
)
missing_codes = data.filter(pl.col("country_code") == pl.col("location")).select("location").unique()
assert len(missing_codes) == 0, f"No ISO3 code found for the following GBD countries: {missing_codes['location'].to_list()}"
data = data.select(["country_code"] + [c for c in data.columns if c != "country_code"]) # move country code to be first col

# finally, we need hiv treatment rates over time
hiv_treatment_rate = pl.read_csv(
    os.path.join(data_dir, "art_coverage_treatment_unaids_final.csv")
)

hiv_treatment_rate = hiv_treatment_rate.rename(
    {
        "Year": "year",
        "CountryName": "location",
        "CountryCode": "country_code",
        "Sex": "sex",
        "CoverageVal": "val",
    }
)

# map sex values to be consistent with HIV data
hiv_treatment_rate = hiv_treatment_rate.with_columns(
    pl.col("sex").replace(
        {
            "male": "Male",
            "female": "Female",
        }
    )
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
# join to existing data on basis of year and country_code and sex
# drop location from treatment data since we already have it from IHME
data = data.join(
    hiv_treatment_rate.drop("location"), on=["year", "country_code", "sex"], how="left"
)

# assumption check that this makes sense, could alternatively fill in with average across all countries in this year
# fill in missing djibuti value for 2003 with average halfway between 2002 and 2004 value
djibouti_2003_fill = (
    data.filter(
        (pl.col("country_code") == "DJI") & pl.col("year").is_in([2002, 2004])
    )
    .group_by("sex")
    .agg(
        pl.col("treatment_proportion")
        .mean()
        .alias("treatment_proportion_fill"),
        pl.col("treatment_proportion_lower")
        .mean()
        .alias("treatment_proportion_lower_fill"),
        pl.col("treatment_proportion_upper")
        .mean()
        .alias("treatment_proportion_upper_fill"),
    )
    .with_columns(
        pl.lit("DJI").alias("country_code"), pl.lit(2003).alias("year")
    )
)
data = (
    data.join(djibouti_2003_fill, on=["country_code", "year", "sex"], how="left")
    .with_columns(
        treatment_proportion=pl.coalesce(
            ["treatment_proportion", "treatment_proportion_fill"]
        ),
        treatment_proportion_lower=pl.coalesce(
            ["treatment_proportion_lower", "treatment_proportion_lower_fill"]
        ),
        treatment_proportion_upper=pl.coalesce(
            ["treatment_proportion_upper", "treatment_proportion_upper_fill"]
        ),
    )
    .drop(
        [
            "treatment_proportion_fill",
            "treatment_proportion_lower_fill",
            "treatment_proportion_upper_fill",
        ]
    )
)

# map countries to regions
data = data.with_columns(
    region=pl.col("country_code").replace(countries_to_regions)
)
# fill in any missing values with the average value for that year
# note - missing countries: Mauritius, Equatorial Guinea, Sao Tome and Principe
data = data.with_columns(
    treatment_proportion=pl.col("treatment_proportion").fill_null(
        pl.col("treatment_proportion").mean().over(["year", "sex", "region"])
    ),
    treatment_proportion_lower=pl.col("treatment_proportion_lower").fill_null(
        pl.col("treatment_proportion_lower")
        .mean()
        .over(["year", "sex", "region"])
    ),
    treatment_proportion_upper=pl.col("treatment_proportion_upper").fill_null(
        pl.col("treatment_proportion_upper")
        .mean()
        .over(["year", "sex", "region"])
    ),
)

# read and join UNAIDS incidence numbers, prevalence numbers, and incidence rate per uninfected
unaids_inc_num = pl.concat([
    read_unaids_wide(os.path.join(data_dir, "UNAIDS-raw-data-HIV/inc_num_adult_female.csv"), "Female", "unaids_incidence_number"),
    read_unaids_wide(os.path.join(data_dir, "UNAIDS-raw-data-HIV/inc_num_adult_male.csv"), "Male", "unaids_incidence_number"),
])
unaids_prev_num = pl.concat([
    read_unaids_wide(os.path.join(data_dir, "UNAIDS-raw-data-HIV/prev_num_adult_female.csv"), "Female", "unaids_prevalence_number"),
    read_unaids_wide(os.path.join(data_dir, "UNAIDS-raw-data-HIV/prev_num_adult_male.csv"), "Male", "unaids_prevalence_number"),
])
unaids_inc_rate = pl.concat([
    read_unaids_wide(os.path.join(data_dir, "UNAIDS-raw-data-HIV/inc_rateUnin_adult_female.csv"), "Female", "unaids_incidence_rate_uninfected"),
    read_unaids_wide(os.path.join(data_dir, "UNAIDS-raw-data-HIV/inc_rateUnin_adult_male.csv"), "Male", "unaids_incidence_rate_uninfected"),
])

unaids_data = (
    unaids_inc_num
    .join(unaids_prev_num, on=["country_code", "sex", "year"], how="full", coalesce=True)
    .join(unaids_inc_rate, on=["country_code", "sex", "year"], how="full", coalesce=True)
)

data = data.join(unaids_data, on=["country_code", "sex", "year"], how="left")

# divide incidence rate by 1000 since original data is per 1000 uninfected population
data = data.with_columns(
    unaids_incidence_rate_uninfected=pl.col("unaids_incidence_rate_uninfected") / 1000,
    unaids_incidence_rate_uninfected_lower=pl.col("unaids_incidence_rate_uninfected_lower") / 1000,
    unaids_incidence_rate_uninfected_upper=pl.col("unaids_incidence_rate_uninfected_upper") / 1000,
)

# add a column that mnually calcultes p_acquiring_hiv using UNAIDS numbers, to check agains the hiv inc rate they report
# use un pop data to use here - no bounds should be added
data = data.with_columns(
    unaids_p_acquiring_hiv=pl.col("unaids_incidence_number")
    / (pl.col("un_pop") - pl.col("unaids_prevalence_number")),
    unaids_p_acquiring_hiv_lower=pl.col("unaids_incidence_number_lower")
    / (pl.col("un_pop") - pl.col("unaids_prevalence_number_lower")),
    unaids_p_acquiring_hiv_upper=pl.col("unaids_incidence_number_upper")
    / (pl.col("un_pop") - pl.col("unaids_prevalence_number_upper")),
)

# add a column that is a true or false for whether this row will be used in the UNAIDS analysis
# false for the following countries: Liberia, Equatorial Guinea, São Tomé and Príncipe, Mauritius, Somalia
countries_to_exclude = ["LBR", "GNQ", "STP", "MUS", "SOM"]
# countries_to_exclude = ["LBR", "GNQ", "STP", "MUS", "MRT", "SOM", 
#                         "ERI", "COM", "CPV", "DJI", "GNB", "GMB",
#                          "MDG", "BEN", "ETH", "NER", "BDI", "GAB"]

data = data.with_columns(
    cols_unaids_analysis=pl.when(pl.col("country_code").is_in(countries_to_exclude))
        .then(pl.lit(False))
        .otherwise(pl.lit(True))
)

# compute unaids year-end prevalence to use later
data = data.with_columns(
    unaids_prevalence_year_end_number=pl.col("unaids_prevalence_number")
    + 0.5 * pl.col("unaids_incidence_number"),
    unaids_prevalence_year_end_number_lower=pl.col("unaids_prevalence_number_lower")
    + 0.5 * pl.col("unaids_incidence_number_lower"),
    unaids_prevalence_year_end_number_upper=pl.col("unaids_prevalence_number_upper")
    + 0.5 * pl.col("unaids_incidence_number_upper"),
)

# Add in DHS treatment seeking rate for someone with STI symptoms, by region and by sex
dhs_treatment_seeking = pl.read_csv(os.path.join(data_dir, "dhs_treatment_seeking_symptom.csv"))

# map country name to iso3; 3 DHS names differ from IHME names
_dhs_name_to_iso3 = dict(ihme_name_to_iso3)  
_dhs_name_to_iso3["Tanzania"] = "TZA"
_dhs_name_to_iso3["DRC"] = "COD"
_dhs_name_to_iso3["Sao Tome"] = "STP"

dhs_treatment_seeking = dhs_treatment_seeking.with_columns(
    iso3=pl.col("country").replace(_dhs_name_to_iso3)
).with_columns(
    region=pl.col("iso3").replace(countries_to_regions)
)
# calculate mean from numbers of respodnents in each country, summed by regiuon
dhs_treatment_seeking = dhs_treatment_seeking.group_by("region", "sex").agg(
    symptom_treat_seek_rate=pl.sum("n_sought_treatment") / pl.sum("n_sti_respondents"),
).select(["region", "sex", "symptom_treat_seek_rate"])

# clean sex names to match
dhs_treatment_seeking = dhs_treatment_seeking.with_columns(
    sex=pl.when(pl.col("sex") == "male").then(pl.lit("Male"))
    .when(pl.col("sex") == "female").then(pl.lit("Female"))
    .otherwise(pl.lit("Error: unexpected sex value"))
)
data = data.join(dhs_treatment_seeking, on=["region", "sex"], how="left")

assert data["symptom_treat_seek_rate"].null_count() == 0, \
    "Missing DHS treatment seeking rate for some rows"

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
                "unaids_incidence_number",
                "unaids_incidence_number_lower",
                "unaids_incidence_number_upper",
                "unaids_prevalence_number",
                "unaids_prevalence_number_lower",
                "unaids_prevalence_number_upper",
                "unaids_prevalence_year_end_number",
                "unaids_prevalence_year_end_number_lower",
                "unaids_prevalence_year_end_number_upper",
                "population",
                "population_lower",
                "population_upper",
                "un_pop",
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
            "unaids_incidence_number",
            "unaids_incidence_number_lower",
            "unaids_incidence_number_upper",
            "unaids_prevalence_number",
            "unaids_prevalence_number_lower",
            "unaids_prevalence_number_upper",
            "unaids_prevalence_year_end_number",
            "unaids_prevalence_year_end_number_lower",
            "unaids_prevalence_year_end_number_upper",
            "population",
            "population_lower",
            "population_upper",
            "un_pop",
        ]
    }
)

female_data = data.filter(pl.col("sex") == "Female")

# join back together
data = pl.concat([msm_data, non_msm_data, female_data])
data = data.drop("cause") # cause column is all hiv, remenant from merge
# save the assembled data
data.write_csv(os.path.join(output_dir, "hiv_sti.csv"))

unaids_proj = pl.read_csv(os.path.join(data_dir, "unaids_future_hiv_projections_all_years.csv"))
unaids_proj = unaids_proj.filter(pl.col("metric") == "new_infections")

# For 2020-2024 use the "98-98-99 treatment, 2024 prevention" column
# for 2025-2030 use the "Historical trend" column
unaids_proj = unaids_proj.with_columns(
    pl.when(pl.col("year") <= 2024)
    .then(pl.col("98-98-99 treatment, 2024 prevention"))
    .otherwise(pl.col("Historical trend"))
    .alias("value")
)

# Map country name to iso3 code then iso3 to region
unaids_proj = unaids_proj.with_columns(
    country_code=pl.col("country").replace(ihme_name_to_iso3),
)
unaids_proj = unaids_proj.with_columns(
    region=pl.col("country_code").replace(countries_to_regions),
)

# Keep only SSA countries 
unaids_proj = unaids_proj.filter(
    pl.col("region").is_in(["Western", "Eastern", "Central", "Southern"])
)

unaids_proj = unaids_proj.select(["country", "country_code", "region", "year", "value"])

unaids_proj.write_csv(os.path.join(output_dir, "unaids_future_hiv_projections_formatted.csv"))

# Sex-split projections: apply historical sex fractions to projected totals
# Compute SSA-wide female/male fractions from UNAIDS historical data by year
# Recombine Male + MSM to get back total male UNAIDS incidence
sex_fractions = (
    data
    .group_by("year")
    .agg(
        female_inc=pl.col("unaids_incidence_number").filter(pl.col("sex") == "Female").sum(),
        male_inc=pl.col("unaids_incidence_number").filter(pl.col("sex").is_in(["Male", "MSM"])).sum(),
    )
    .with_columns(
        female_fraction=pl.col("female_inc") / (pl.col("female_inc") + pl.col("male_inc")),
        male_fraction=pl.col("male_inc") / (pl.col("female_inc") + pl.col("male_inc")),
    )
    .select(["year", "female_fraction", "male_fraction"])
)

# For each projected year Y, get the sex ratio from a historical year using progressive decay
# the year of the sex ratio is
#   2023 - (Y - 2024)  i.e.  2024 -> 2023, 2025 -> 2022, 2026 -> 2021, ...
# For historical years already in the data (Y <= 2023), use Y's own ratio.
last_hist_year = 2023
unaids_proj_sex = unaids_proj.with_columns(
    sex_ratio_year=pl.when(pl.col("year") <= last_hist_year)
        .then(pl.col("year"))
        .otherwise(pl.lit(2 * last_hist_year + 1) - pl.col("year"))
        .cast(pl.Int64)
)

unaids_proj_sex = unaids_proj_sex.join(
    sex_fractions, left_on="sex_ratio_year", right_on="year", how="left"
)

unaids_proj_by_sex = pl.concat([
    unaids_proj_sex.with_columns(
        sex=pl.lit("Female"),
        value=pl.col("value") * pl.col("female_fraction"),
    ).select(["country", "country_code", "region", "year", "sex", "value"]),
    unaids_proj_sex.with_columns(
        sex=pl.lit("Male"),
        value=pl.col("value") * pl.col("male_fraction"),
    ).select(["country", "country_code", "region", "year", "sex", "value"]),
])

unaids_proj_by_sex.write_csv(os.path.join(output_dir, "unaids_future_hiv_projections_by_sex.csv"))
