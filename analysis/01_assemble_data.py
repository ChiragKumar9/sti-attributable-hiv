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
_code_lookup = {
    row["name"]: row["alpha-3"] for row in country_codes_df.to_dicts()
}
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
    df = df.rename(
        {col: col.strip() for col in df.columns}
    )  # remove white space in colnames

    # differentiate year, lower bound, and upper bound columns
    year_cols = [c for c in df.columns if c.isdigit()]
    lower_cols = [
        c
        for c in df.columns
        if c.endswith("_lower") and c.split("_")[0].isdigit()
    ]
    upper_cols = [
        c
        for c in df.columns
        if c.endswith("_upper") and c.split("_")[0].isdigit()
    ]

    # pivot main values from wide to long, on country and year
    val_long = df.select(["Country"] + year_cols)
    val_long = val_long.unpivot(
        index="Country", variable_name="year", value_name="val_raw"
    )
    val_long = val_long.with_columns(pl.col("year").cast(pl.Int64))

    # same for lower bounds
    lower_long = df.select(["Country"] + lower_cols)
    lower_long = lower_long.unpivot(
        index="Country", variable_name="yr", value_name="lower_raw"
    )
    lower_long = lower_long.with_columns(
        pl.col("yr").str.replace("_lower", "").cast(pl.Int64).alias("year")
    )
    lower_long = lower_long.drop("yr")

    # same for upper bounds
    upper_long = df.select(["Country"] + upper_cols)
    upper_long = upper_long.unpivot(
        index="Country", variable_name="yr", value_name="upper_raw"
    )
    upper_long = upper_long.with_columns(
        pl.col("yr").str.replace("_upper", "").cast(pl.Int64).alias("year")
    )
    upper_long = upper_long.drop("yr")

    long = val_long.join(lower_long, on=["Country", "year"]).join(
        upper_long, on=["Country", "year"]
    )

    # raw data has "..." for missing data
    # assumed values when < reported
    # if <100 --> make 50
    # if <200 --> make 150
    def clean_col(col_name):
        return (
            pl.when(
                pl.col(col_name).is_null()
                | (pl.col(col_name).str.strip_chars() == "...")
            )
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
            .otherwise(
                pl.col(col_name)
                .str.replace_all(" ", "")
                .cast(pl.Float64, strict=False)
            )
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
    .rename(
        {
            "Location": "location",
            "Sex": "sex",
            "Time": "year",
            "Value": "un_pop",
        }
    )
    .select(["location", "sex", "year", "un_pop"])
)
incidence_hiv = incidence_hiv.join(
    un_pop, on=["location", "sex", "year"], how="left"
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
missing_codes = (
    data.filter(pl.col("country_code") == pl.col("location"))
    .select("location")
    .unique()
)
assert len(missing_codes) == 0, (
    f"No ISO3 code found for the following GBD countries: {missing_codes['location'].to_list()}"
)
data = data.select(
    ["country_code"] + [c for c in data.columns if c != "country_code"]
)  # move country code to be first col

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
    hiv_treatment_rate.drop("location"),
    on=["year", "country_code", "sex"],
    how="left",
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
    data.join(
        djibouti_2003_fill, on=["country_code", "year", "sex"], how="left"
    )
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
unaids_inc_num = pl.concat(
    [
        read_unaids_wide(
            os.path.join(
                data_dir, "UNAIDS-raw-data-HIV/inc_num_adult_female.csv"
            ),
            "Female",
            "unaids_incidence_number",
        ),
        read_unaids_wide(
            os.path.join(
                data_dir, "UNAIDS-raw-data-HIV/inc_num_adult_male.csv"
            ),
            "Male",
            "unaids_incidence_number",
        ),
    ]
)
unaids_prev_num = pl.concat(
    [
        read_unaids_wide(
            os.path.join(
                data_dir, "UNAIDS-raw-data-HIV/prev_num_adult_female.csv"
            ),
            "Female",
            "unaids_prevalence_number",
        ),
        read_unaids_wide(
            os.path.join(
                data_dir, "UNAIDS-raw-data-HIV/prev_num_adult_male.csv"
            ),
            "Male",
            "unaids_prevalence_number",
        ),
    ]
)
unaids_inc_rate = pl.concat(
    [
        read_unaids_wide(
            os.path.join(
                data_dir, "UNAIDS-raw-data-HIV/inc_rateUnin_adult_female.csv"
            ),
            "Female",
            "unaids_incidence_rate_uninfected",
        ),
        read_unaids_wide(
            os.path.join(
                data_dir, "UNAIDS-raw-data-HIV/inc_rateUnin_adult_male.csv"
            ),
            "Male",
            "unaids_incidence_rate_uninfected",
        ),
    ]
)

unaids_data = unaids_inc_num.join(
    unaids_prev_num,
    on=["country_code", "sex", "year"],
    how="full",
    coalesce=True,
).join(
    unaids_inc_rate,
    on=["country_code", "sex", "year"],
    how="full",
    coalesce=True,
)

data = data.join(unaids_data, on=["country_code", "sex", "year"], how="left")

#  Manually interpolate Somalia UNAIDS HIV incidence numbers (2023 by sex not published)

# 2022 values given: For each sex: val = <200 --> 150, lower = <100 --> 50, upper = <200 --> 150
# 2023 values: Combined both sexes: 282.17 (from unaids future projections)
# assume that 2022 sex ratio (1:1 based on the precision of data we have) carries through to 2023, giving us 141.085 for each
# round to 150 to be consistent with the level of precision of the 2022 data, and to avoid implying false precision from a modelled input
# carry through bounds from 2022 for consistency as well

# Also carry forward the 2022 incidence rate (pre-/1000 value: 0.05, i.e. <0.1 per 1000 uninfected).
# Justification: both 2022 and 2023 Somalia rates are censored at <0.1 per 1000. Any true year-on-year
# change in incidence rate at this level would be masked by the <0.1 coding assumption
values = {
    "unaids_incidence_number": 150.0,
    "unaids_incidence_number_lower": 50.0,
    "unaids_incidence_number_upper": 150.0,
    "unaids_incidence_rate_uninfected": 0.05,  # <0.1 per 1000; divided by 1000 below → 0.00005
    "unaids_incidence_rate_uninfected_lower": 0.05,
    "unaids_incidence_rate_uninfected_upper": 0.05,
}
for col, val in values.items():
    data = data.with_columns(
        pl.when((pl.col("country_code") == "SOM") & (pl.col("year") == 2023))
        .then(pl.lit(val))
        .otherwise(pl.col(col))
        .alias(col)
    )

# divide incidence rate by 1000 since original data is per 1000 uninfected population
data = data.with_columns(
    unaids_incidence_rate_uninfected=pl.col("unaids_incidence_rate_uninfected")
    / 1000,
    unaids_incidence_rate_uninfected_lower=pl.col(
        "unaids_incidence_rate_uninfected_lower"
    )
    / 1000,
    unaids_incidence_rate_uninfected_upper=pl.col(
        "unaids_incidence_rate_uninfected_upper"
    )
    / 1000,
)

# add a column that mnually calculates p_acquiring_hiv using UNAIDS numbers, to check agains the hiv inc rate they report
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
# false for the following countries: Liberia, Equatorial Guinea, São Tomé and Príncipe
# Cabo Verde, Cote dIvoire
countries_to_exclude = ["LBR", "GNQ", "STP", "CPV", "CIV"]

# TODO - why are we excluding Cote dIvoire
data = data.with_columns(
    cols_unaids_analysis=pl.when(
        pl.col("country_code").is_in(countries_to_exclude)
    )
    .then(pl.lit(False))
    .otherwise(pl.lit(True))
)

# compute unaids year-end prevalence to use later
data = data.with_columns(
    unaids_prevalence_year_end_number=pl.col("unaids_prevalence_number")
    + 0.5 * pl.col("unaids_incidence_number"),
    unaids_prevalence_year_end_number_lower=pl.col(
        "unaids_prevalence_number_lower"
    )
    + 0.5 * pl.col("unaids_incidence_number_lower"),
    unaids_prevalence_year_end_number_upper=pl.col(
        "unaids_prevalence_number_upper"
    )
    + 0.5 * pl.col("unaids_incidence_number_upper"),
)

# Add in DHS treatment seeking rate for someone with STI symptoms, by region and by sex
dhs_treatment_seeking = pl.read_csv(
    os.path.join(data_dir, "dhs_treatment_seeking_symptoms_hivNeg.csv")
)
# map country name to iso3; 3 DHS names differ from IHME names
_dhs_name_to_iso3 = dict(ihme_name_to_iso3)
_dhs_name_to_iso3["Tanzania"] = "TZA"
_dhs_name_to_iso3["DRC"] = "COD"
_dhs_name_to_iso3["Sao Tome"] = "STP"
_dhs_name_to_iso3["Cote d'Ivoire"] = "CIV"

dhs_treatment_seeking = dhs_treatment_seeking.with_columns(
    iso3=pl.col("country").replace(_dhs_name_to_iso3)
).with_columns(region=pl.col("iso3").replace(countries_to_regions))

# filter out nulls
dhs_treatment_seeking = dhs_treatment_seeking.filter(
    pl.col("pct_sought_treatment_wtd").is_not_null()
)

# calculate mean from numbers of respondents in each country, summed by regions
dhs_treatment_seeking = (
    dhs_treatment_seeking.group_by("region", "sex", "year")
    .agg(frac_sought_treatment=pl.mean("pct_sought_treatment_wtd") / 100)
    .select(["region", "sex", "year", "frac_sought_treatment"])
)

# clean sex names to match
dhs_treatment_seeking = dhs_treatment_seeking.with_columns(
    sex=pl.when(pl.col("sex") == "male")
    .then(pl.lit("Male"))
    .when(pl.col("sex") == "female")
    .then(pl.lit("Female"))
    .otherwise(pl.lit("Error: unexpected sex value"))
)

# assert that there are no instances of unexpected sex
assert (
    dhs_treatment_seeking.filter(
        pl.col("sex") == "Error: unexpected sex value"
    ).height
    == 0
), "There are unexpected sex values in the DHS treatment seeking data"

data = data.join(
    dhs_treatment_seeking, on=["region", "sex", "year"], how="left"
)

# interpolate missing treatment seeking rates by first interpolating between succesive
# years within each country, then filling any remaining missing values with the average
# for that year
data = data.sort("sex", "region", "year")

data = data.with_columns(
    pl.col("frac_sought_treatment")
    .interpolate(method="linear")
    .over(["sex", "region"])
)

data = data.with_columns(
    pl.col("frac_sought_treatment").fill_null(
        pl.col("frac_sought_treatment").mean().over(["sex", "year"])
    )
)

assert (
    data.filter(pl.col("year") >= 2000)["frac_sought_treatment"].null_count()
    == 0
), "Missing DHS treatment seeking rate for some rows"

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
data = data.drop("cause")  # cause column is all hiv, remenant from merge

# Adjust GC and chlamydia prevalence for MSM vs non-MSM males using published MSM prevalence estimates.
# PR = MSM_prev / all_male_prev (from source data / GBD data)
# where f is the 15% MSM fraction:
# all_male_prev = f * MSM_prev + (1-f) * non_MSM_prev
# Solving for each group:
#   non_MSM_prev = all_male_prev / (1 - f + f * PR)
#   MSM_prev     = PR * all_male_prev / (1 - f + f * PR)
# Use lower/upper bounds on PR


def _msm(pr, f, all_male_prev):
    return (all_male_prev * pr) / (1 - f + f * pr)


def _non_msm(pr, f, all_male_prev):
    return all_male_prev / (1 - f + f * pr)


msm_sti_prev = pl.read_csv(os.path.join(data_dir, "MSM_STI_prev.csv"))
msm_sti_prev = msm_sti_prev.rename(
    {col: col.strip() for col in msm_sti_prev.columns}
)
msm_sti_prev = msm_sti_prev.with_columns(
    pl.col("Pathogen").str.strip_chars(),
    pl.col("Population").str.strip_chars(),
)

# calculate the PR values - swap upper and lower bounds for non MSM since higher PR means lower non MSM prev
gc_msm_row = msm_sti_prev.filter(
    (pl.col("Pathogen") == "GC") & (pl.col("Population") == "MSM, 15-49")
)
gc_all_row = msm_sti_prev.filter(
    (pl.col("Pathogen") == "GC") & (pl.col("Population") == "All men, 15-49")
)
pr_gc = gc_msm_row["Value"].item() / gc_all_row["Value"].item()
pr_gc_lower = gc_msm_row["Lower"].item() / gc_all_row["Upper"].item()
pr_gc_upper = gc_msm_row["Upper"].item() / gc_all_row["Lower"].item()

chlamydia_msm_row = msm_sti_prev.filter(
    (pl.col("Pathogen") == "Chlamydia")
    & (pl.col("Population") == "MSM, 15-49")
)
chlamydia_all_row = msm_sti_prev.filter(
    (pl.col("Pathogen") == "Chlamydia")
    & (pl.col("Population") == "All men, 15-49")
)
pr_chlamydia = (
    chlamydia_msm_row["Value"].item() / chlamydia_all_row["Value"].item()
)
pr_chlamydia_lower = (
    chlamydia_msm_row["Lower"].item() / chlamydia_all_row["Upper"].item()
)
pr_chlamydia_upper = (
    chlamydia_msm_row["Upper"].item() / chlamydia_all_row["Lower"].item()
)

# GC
data = data.with_columns(
    pl.when(pl.col("sex") == "MSM")
    .then(_msm(pr_gc, msm_fraction, pl.col("gc_prevalence")))
    .when(pl.col("sex") == "Male")
    .then(_non_msm(pr_gc, msm_fraction, pl.col("gc_prevalence")))
    .otherwise(pl.col("gc_prevalence"))
    .alias("gc_prevalence")
)

data = data.with_columns(
    pl.when(pl.col("sex") == "MSM")
    .then(_msm(pr_gc_lower, msm_fraction, pl.col("gc_prevalence_lower")))
    .when(pl.col("sex") == "Male")
    .then(_non_msm(pr_gc_upper, msm_fraction, pl.col("gc_prevalence_lower")))
    .otherwise(pl.col("gc_prevalence_lower"))
    .alias("gc_prevalence_lower")
)

data = data.with_columns(
    pl.when(pl.col("sex") == "MSM")
    .then(_msm(pr_gc_upper, msm_fraction, pl.col("gc_prevalence_upper")))
    .when(pl.col("sex") == "Male")
    .then(_non_msm(pr_gc_lower, msm_fraction, pl.col("gc_prevalence_upper")))
    .otherwise(pl.col("gc_prevalence_upper"))
    .alias("gc_prevalence_upper")
)


# Chlamydia
data = data.with_columns(
    pl.when(pl.col("sex") == "MSM")
    .then(_msm(pr_chlamydia, msm_fraction, pl.col("chlamydia_prevalence")))
    .when(pl.col("sex") == "Male")
    .then(_non_msm(pr_chlamydia, msm_fraction, pl.col("chlamydia_prevalence")))
    .otherwise(pl.col("chlamydia_prevalence"))
    .alias("chlamydia_prevalence")
)

data = data.with_columns(
    pl.when(pl.col("sex") == "MSM")
    .then(
        _msm(
            pr_chlamydia_lower,
            msm_fraction,
            pl.col("chlamydia_prevalence_lower"),
        )
    )
    .when(pl.col("sex") == "Male")
    .then(
        _non_msm(
            pr_chlamydia_upper,
            msm_fraction,
            pl.col("chlamydia_prevalence_lower"),
        )
    )
    .otherwise(pl.col("chlamydia_prevalence_lower"))
    .alias("chlamydia_prevalence_lower")
)

data = data.with_columns(
    pl.when(pl.col("sex") == "MSM")
    .then(
        _msm(
            pr_chlamydia_upper,
            msm_fraction,
            pl.col("chlamydia_prevalence_upper"),
        )
    )
    .when(pl.col("sex") == "Male")
    .then(
        _non_msm(
            pr_chlamydia_lower,
            msm_fraction,
            pl.col("chlamydia_prevalence_upper"),
        )
    )
    .otherwise(pl.col("chlamydia_prevalence_upper"))
    .alias("chlamydia_prevalence_upper")
)

# save the assembled data
data.write_csv(os.path.join(output_dir, "hiv_sti.csv"))

unaids_proj = pl.read_csv(
    os.path.join(data_dir, "unaids_future_hiv_projections_all_years.csv")
)
unaids_proj = unaids_proj.filter(pl.col("metric") == "new_infections")

# Use the 98-98-99 treatment scenario for everything
unaids_proj = unaids_proj.with_columns(
    value=pl.when(pl.col("year") <= 2024)
    .then(pl.col("95-95-95 treatment, 2024 level of prevention"))
    .otherwise(pl.col("95-95-95 treatment, 2024 level of prevention"))
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

unaids_proj = unaids_proj.select(
    ["country", "country_code", "region", "year", "value"]
)

unaids_proj.write_csv(
    os.path.join(output_dir, "unaids_future_hiv_projections_formatted.csv")
)

# Sex-split projections: apply historical sex fractions to projected totals
# Compute SSA-wide female/male fractions from UNAIDS historical data by year
# Recombine Male + MSM to get back total male UNAIDS incidence
sex_fractions = (
    data.group_by("year")
    .agg(
        female_inc=pl.col("unaids_incidence_number")
        .filter(pl.col("sex") == "Female")
        .sum(),
        male_inc=pl.col("unaids_incidence_number")
        .filter(pl.col("sex").is_in(["Male", "MSM"]))
        .sum(),
    )
    .with_columns(
        female_fraction=pl.col("female_inc")
        / (pl.col("female_inc") + pl.col("male_inc")),
        male_fraction=pl.col("male_inc")
        / (pl.col("female_inc") + pl.col("male_inc")),
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

unaids_proj_by_sex = pl.concat(
    [
        unaids_proj_sex.with_columns(
            sex=pl.lit("Female"),
            value=pl.col("value") * pl.col("female_fraction"),
        ).select(
            ["country", "country_code", "region", "year", "sex", "value"]
        ),
        unaids_proj_sex.with_columns(
            sex=pl.lit("Male"),
            value=pl.col("value") * pl.col("male_fraction"),
        ).select(
            ["country", "country_code", "region", "year", "sex", "value"]
        ),
    ]
)

# add UN population into future projections file covers all years in the projections file (historical and future)
un_pop_for_proj = pl.concat(
    [
        # historical (1990-2023)
        pl.read_csv(os.path.join(data_dir, "un_pop_data.csv"))
        .rename(
            {
                "Location": "country",
                "Sex": "sex",
                "Time": "year",
                "Value": "un_pop",
            }
        )
        .select(["country", "sex", "year", "un_pop"]),
        # future projections (2024-2030)
        pl.read_csv(os.path.join(data_dir, "UN_pop_projections.csv"))
        .with_columns(
            sex=pl.col("sex").replace({"female": "Female", "male": "Male"}),
            val=pl.col("val").cast(pl.Float64),
        )
        .rename({"location": "country", "val": "un_pop"})
        .select(["country", "sex", "year", "un_pop"]),
    ]
)
unaids_proj_by_sex = unaids_proj_by_sex.join(
    un_pop_for_proj, on=["country", "sex", "year"], how="left"
).rename({"value": "unaids_inc_num_proj"})

unaids_proj_by_sex.write_csv(
    os.path.join(output_dir, "unaids_future_hiv_projections_by_sex.csv")
)

# we want to make an overall dataframe with all the historical data and the
# projections together for our counterfactual analyses

# we want to drop any countries from the projection dataset for which we don't
# have historical data
unaids_proj_by_sex = unaids_proj_by_sex.filter(
    pl.col("country_code").is_in(
        data.filter(pl.col("cols_unaids_analysis"))["country_code"]
        .unique()
        .to_list()
    )
)

# let's create an MSM projection using the same method as before for the
# projections

unads_proj_msm = unaids_proj_by_sex.filter(
    pl.col("sex") == "Male"
).with_columns(
    sex=pl.lit("MSM"),
    unaids_inc_num_proj=pl.col("unaids_inc_num_proj") * msm_fraction,
)
unaids_proj_male = unaids_proj_by_sex.filter(
    pl.col("sex") == "Male"
).with_columns(
    unaids_inc_num_proj=pl.col("unaids_inc_num_proj") * (1 - msm_fraction)
)
unaids_proj_combined = pl.concat(
    [
        unaids_proj_by_sex.filter(pl.col("sex") == "Female"),
        unaids_proj_male,
        unads_proj_msm,
    ]
)

# rename country to location so that the merge is nicer
unaids_proj_combined = unaids_proj_combined.rename({"country": "location"})
# add in the regions
unaids_proj_combined = unaids_proj_combined.with_columns(
    region=pl.col("country_code").replace(countries_to_regions)
)

# now combine with historical data
data = pl.concat([data, unaids_proj_combined], how="diagonal_relaxed")

# we need to forward fill variables
# we have three strategies for oding this -- last value carried forward, linear
# interpolation, or people propagation for incidence numbers

data = data.sort(["country_code", "sex", "year"])

# for STI prevalences, let's do linear interpolation over the last 5 years of data
sti_cols = [
    "gc_prevalence",
    "gc_prevalence_lower",
    "gc_prevalence_upper",
    "chlamydia_prevalence",
    "chlamydia_prevalence_lower",
    "chlamydia_prevalence_upper",
    "syphilis_prevalence",
    "syphilis_prevalence_lower",
    "syphilis_prevalence_upper",
    "trichomoniasis_prevalence",
    "trichomoniasis_prevalence_lower",
    "trichomoniasis_prevalence_upper",
]

# we're actually going to do the same method for population because we don't have
# future population projections

pop_cols = [
    "un_pop",
]

# Fit linear trend using only data from 2019 onwards, then extrapolate
recent_cutoff = 2014

for col in [*sti_cols, *pop_cols]:
    data = (
        data.with_columns(
            # Mark recent non-null values for fitting
            _year_fit=pl.when(
                (pl.col("year") >= recent_cutoff) & pl.col(col).is_not_null()
            )
            .then(pl.col("year"))
            .otherwise(None),
            _val_fit=pl.when(
                (pl.col("year") >= recent_cutoff) & pl.col(col).is_not_null()
            )
            .then(pl.col(col))
            .otherwise(None),
        )
        .with_columns(
            # Calculate means for fitting
            _year_mean=pl.col("_year_fit")
            .mean()
            .over(["country_code", "sex"]),
            _val_mean=pl.col("_val_fit").mean().over(["country_code", "sex"]),
        )
        .with_columns(
            # Calculate slope
            _slope=(
                (
                    (pl.col("_year_fit") - pl.col("_year_mean"))
                    * (pl.col("_val_fit") - pl.col("_val_mean"))
                )
                .sum()
                .over(["country_code", "sex"])
                / ((pl.col("_year_fit") - pl.col("_year_mean")).pow(2))
                .sum()
                .over(["country_code", "sex"])
            )
        )
        .with_columns(
            # Calculate intercept and apply linear model
            _intercept=pl.col("_val_mean")
            - pl.col("_slope") * pl.col("_year_mean"),
        )
        .with_columns(
            # Fill nulls with linear extrapolation
            **{
                col: pl.coalesce(
                    [
                        pl.col(col),
                        pl.col("_slope") * pl.col("year")
                        + pl.col("_intercept"),
                    ]
                )
            }
        )
        .drop(
            [
                "_year_fit",
                "_val_fit",
                "_year_mean",
                "_val_mean",
                "_slope",
                "_intercept",
            ]
        )
    )

# for treatment proportions and frac sought treatment, let's do last value carried
# forward
treatment_cols = [
    "treatment_proportion",
    "treatment_proportion_lower",
    "treatment_proportion_upper",
]

data = data.with_columns(
    **{
        col: pl.col(col)
        .fill_null(strategy="forward")
        .over(["country_code", "sex"])
        for col in treatment_cols
    }
)

# frac sought treatment is null filled with 100%
data = data.with_columns(
    frac_sought_treatment=pl.col("frac_sought_treatment").fill_null(1)
)

# note that the incidence values for projections are in unaids_inc_num_proj
# projections don't have bounds, so use point estimate for lower/upper
data = data.with_columns(
    unaids_incidence_number=pl.when(
        pl.col("unaids_inc_num_proj").is_not_null()
    )
    .then(pl.col("unaids_inc_num_proj"))
    .otherwise(pl.col("unaids_incidence_number")),
    unaids_incidence_number_lower=pl.when(
        pl.col("unaids_inc_num_proj").is_not_null()
    )
    .then(pl.col("unaids_inc_num_proj"))
    .otherwise(pl.col("unaids_incidence_number_lower")),
    unaids_incidence_number_upper=pl.when(
        pl.col("unaids_inc_num_proj").is_not_null()
    )
    .then(pl.col("unaids_inc_num_proj"))
    .otherwise(pl.col("unaids_incidence_number_upper")),
)

# now we need to calculate unaids prevalence and unaids prevalence year end
# prevalence = prevalence + incidence - deaths
# assume that without treatment, hiv is 100% fatal within 10 years
# but we are only extrapolating over a few years, so we assume that the number of
# deaths is negligable

# so prevalence = prevalence in the last year + incidence
# so we forward fill prevalence, and then we add the cumulative sum of incidence
data = data.with_columns(
    unaids_prevalence_number=pl.col("unaids_prevalence_number")
    .fill_null(strategy="forward")
    .over(["country_code", "sex"]),
    unaids_prevalence_number_lower=pl.col("unaids_prevalence_number_lower")
    .fill_null(strategy="forward")
    .over(["country_code", "sex"]),
    unaids_prevalence_number_upper=pl.col("unaids_prevalence_number_upper")
    .fill_null(strategy="forward")
    .over(["country_code", "sex"]),
    cumulative_incidence=pl.col("unaids_inc_num_proj")
    .fill_null(0)
    .cum_sum()
    .over(["country_code", "sex"]),
).with_columns(
    unaids_prevalence_number=pl.col("unaids_prevalence_number")
    + pl.col("cumulative_incidence"),
    unaids_prevalence_number_lower=pl.col("unaids_prevalence_number_lower")
    + pl.col("cumulative_incidence"),
    unaids_prevalence_number_upper=pl.col("unaids_prevalence_number_upper")
    + pl.col("cumulative_incidence"),
)

# calculate year end prevalence
data = data.with_columns(
    unaids_prevalence_year_end_number=pl.col("unaids_prevalence_number")
    + 0.5 * pl.col("unaids_inc_num_proj").fill_null(0),
    unaids_prevalence_year_end_number_lower=pl.col(
        "unaids_prevalence_number_lower"
    )
    + 0.5 * pl.col("unaids_inc_num_proj").fill_null(0),
    unaids_prevalence_year_end_number_upper=pl.col(
        "unaids_prevalence_number_upper"
    )
    + 0.5 * pl.col("unaids_inc_num_proj").fill_null(0),
)

# calculate p_acquiring_hiv for projection years only
data = data.with_columns(
    unaids_p_acquiring_hiv=pl.when(pl.col("unaids_inc_num_proj").is_not_null())
    .then(
        pl.col("unaids_inc_num_proj")
        / (pl.col("un_pop") - pl.col("unaids_prevalence_number"))
    )
    .otherwise(pl.col("unaids_p_acquiring_hiv")),
    unaids_p_acquiring_hiv_lower=pl.when(
        pl.col("unaids_inc_num_proj").is_not_null()
    )
    .then(
        pl.col("unaids_inc_num_proj")
        / (pl.col("un_pop") - pl.col("unaids_prevalence_number_lower"))
    )
    .otherwise(pl.col("unaids_p_acquiring_hiv_lower")),
    unaids_p_acquiring_hiv_upper=pl.when(
        pl.col("unaids_inc_num_proj").is_not_null()
    )
    .then(
        pl.col("unaids_inc_num_proj")
        / (pl.col("un_pop") - pl.col("unaids_prevalence_number_upper"))
    )
    .otherwise(pl.col("unaids_p_acquiring_hiv_upper")),
)

# we don't care about these values for the gbd dataset since we don't do
# projections for those quantities
# but we need to fill in something for them so that we don't
# have nulls
# let's just forward fill them
gbd_columns = [
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
    "p_acquiring_hiv",
    "p_acquiring_hiv_lower",
    "p_acquiring_hiv_upper",
]

# forward fill gbd columns
data = data.with_columns(
    **{
        col: pl.col(col)
        .fill_null(strategy="forward")
        .over(["country_code", "sex"])
        for col in gbd_columns
    }
)

# export csv
data.write_csv(os.path.join(output_dir, "hiv_sti_with_projections.csv"))
