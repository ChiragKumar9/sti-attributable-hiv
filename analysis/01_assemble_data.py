import os

import polars as pl
import yaml

data_dir = "data"
output_dir = "outputs"

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


def assign_age_group(age_col: pl.Expr) -> pl.Expr:
    lower_bound = age_col.str.extract(r"(\d+)", 1).cast(pl.Int64)
    return (
        pl.when(lower_bound < 25)
        .then(pl.lit("15-24"))
        .when(lower_bound < 50)
        .then(pl.lit("25-49"))
        .otherwise(pl.lit("50+"))
    )


population = pl.read_csv(
    os.path.join(data_dir, "ihme_full_population_age.csv")
)
population = population.with_columns(age_group=assign_age_group(pl.col("age")))
population = population.group_by(["location", "year", "sex", "age_group"]).agg(
    pl.sum("val").alias("population"),
    pl.sum("lower").alias("population_lower"),
    pl.sum("upper").alias("population_upper"),
)

total_population = population.group_by(["location", "year", "sex"]).agg(
    pl.sum("population").alias("total_population")
)
population = population.join(
    total_population, on=["location", "year", "sex"], how="left"
)
population = population.with_columns(
    gbd_age_fraction=(pl.col("population") / pl.col("total_population"))
)

hiv_data = pl.read_csv(os.path.join(data_dir, "IHME-GBD_2023_DATA-HIV.csv"))
hiv_data = hiv_data.filter(pl.col("metric") == "Number")
hiv_data = hiv_data.with_columns(age_group=assign_age_group(pl.col("age")))
hiv_data = hiv_data.group_by(
    ["measure", "location", "sex", "year", "age_group"]
).agg(
    pl.sum("val"),
    pl.sum("lower"),
    pl.sum("upper"),
)

incidence_hiv = (
    hiv_data.filter(pl.col("measure") == "Incidence")
    .rename(
        {
            "val": "hiv_incidence_number",
            "lower": "hiv_incidence_number_lower",
            "upper": "hiv_incidence_number_upper",
        }
    )
    .drop("measure")
)

prevalence_hiv = (
    hiv_data.filter(pl.col("measure") == "Prevalence")
    .rename(
        {
            "val": "hiv_prevalence_number",
            "lower": "hiv_prevalence_number_lower",
            "upper": "hiv_prevalence_number_upper",
        }
    )
    .drop("measure")
)

incidence_hiv = incidence_hiv.join(
    population.drop("total_population"),
    on=["location", "sex", "year", "age_group"],
    how="inner",
)

incidence_hiv = incidence_hiv.join(
    prevalence_hiv,
    on=["location", "sex", "year", "age_group"],
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
incidence_hiv = incidence_hiv.with_columns(
    un_pop=(pl.col("un_pop") * pl.col("gbd_age_fraction"))
)

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


def read_sti_csv(sti_name: str) -> pl.DataFrame:
    col_prefix = "gc" if sti_name == "Gonorrhea" else sti_name.lower()
    df = pl.read_csv(
        os.path.join(data_dir, f"IHME-GBD_2023_DATA-{sti_name}.csv")
    )
    df = df.filter(
        (pl.col("measure") == "Prevalence") & (pl.col("metric") == "Number")
    )
    df = df.with_columns(age_group=assign_age_group(pl.col("age")))
    df = df.group_by(["location", "sex", "year", "age_group"]).agg(
        pl.sum("val").alias(f"{col_prefix}_prevalence"),
        pl.sum("lower").alias(f"{col_prefix}_prevalence_lower"),
        pl.sum("upper").alias(f"{col_prefix}_prevalence_upper"),
    )
    return df


gc_prev = read_sti_csv("Gonorrhea")
chlamydia_prev = read_sti_csv("Chlamydia")
syphilis_prev = read_sti_csv("Syphilis")
trichomoniasis_prev = read_sti_csv("Trichomoniasis")

data = incidence_hiv.join(
    gc_prev, on=["location", "sex", "year", "age_group"], how="inner"
)
data = data.join(
    chlamydia_prev, on=["location", "sex", "year", "age_group"], how="inner"
)
data = data.join(
    syphilis_prev, on=["location", "sex", "year", "age_group"], how="inner"
)
data = data.join(
    trichomoniasis_prev,
    on=["location", "sex", "year", "age_group"],
    how="inner",
)

sti_prefixes = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
data = data.with_columns(
    [
        (pl.col(f"{s}_prevalence") / pl.col("population")).alias(
            f"{s}_prevalence"
        )
        for s in sti_prefixes
    ]
    + [
        (pl.col(f"{s}_prevalence_lower") / pl.col("population_lower")).alias(
            f"{s}_prevalence_lower"
        )
        for s in sti_prefixes
    ]
    + [
        (pl.col(f"{s}_prevalence_upper") / pl.col("population_upper")).alias(
            f"{s}_prevalence_upper"
        )
        for s in sti_prefixes
    ]
)

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
)

# we need the HIV treatment rates over time by country and sex
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
hiv_treatment_rate = hiv_treatment_rate.with_columns(
    pl.col("sex").replace({"male": "Male", "female": "Female"})
)
hiv_treatment_rate = hiv_treatment_rate.with_columns(
    val=pl.col("val") / 100,
    lower=pl.col("lower") / 100,
    upper=pl.col("upper") / 100,
)
hiv_treatment_rate = hiv_treatment_rate.rename(
    {
        "val": "treatment_proportion",
        "lower": "treatment_proportion_lower",
        "upper": "treatment_proportion_upper",
    }
)
hiv_treatment_rate = hiv_treatment_rate.with_columns(
    treatment_proportion=pl.col("treatment_proportion")
    * params["art_efficacy"],
    treatment_proportion_lower=pl.col("treatment_proportion_lower")
    * params["art_efficacy"],
    treatment_proportion_upper=pl.col("treatment_proportion_upper")
    * params["art_efficacy"],
)

# join to existing data on basis of year and country_code and sex
# drop location from treatment data since we already have it from IHME
data = data.join(
    hiv_treatment_rate.drop("location"),
    on=["year", "country_code", "sex"],
    how="left",
)

# manually fill in missing treatment proportion data in djibouti using average
# of neighboring years
djibouti_2003_fill = (
    data.filter(
        (pl.col("country_code") == "DJI") & pl.col("year").is_in([2002, 2004])
    )
    .group_by(["sex", "age_group"])
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
        djibouti_2003_fill,
        on=["country_code", "year", "sex", "age_group"],
        how="left",
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

data = data.with_columns(
    region=pl.col("country_code").replace(countries_to_regions)
)

data = data.with_columns(
    treatment_proportion=pl.col("treatment_proportion").fill_null(
        pl.col("treatment_proportion")
        .mean()
        .over(["year", "sex", "region", "age_group"])
    ),
    treatment_proportion_lower=pl.col("treatment_proportion_lower").fill_null(
        pl.col("treatment_proportion_lower")
        .mean()
        .over(["year", "sex", "region", "age_group"])
    ),
    treatment_proportion_upper=pl.col("treatment_proportion_upper").fill_null(
        pl.col("treatment_proportion_upper")
        .mean()
        .over(["year", "sex", "region", "age_group"])
    ),
)


def read_unaids_wide(filepath, sex_label, value_col_name):
    df = pl.read_csv(filepath, infer_schema_length=0)
    df = df.rename({col: col.strip() for col in df.columns})
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
    val_long = (
        df.select(["Country"] + year_cols)
        .unpivot(index="Country", variable_name="year", value_name="val_raw")
        .with_columns(pl.col("year").cast(pl.Int64))
    )
    lower_long = (
        df.select(["Country"] + lower_cols)
        .unpivot(index="Country", variable_name="yr", value_name="lower_raw")
        .with_columns(
            pl.col("yr").str.replace("_lower", "").cast(pl.Int64).alias("year")
        )
        .drop("yr")
    )
    upper_long = (
        df.select(["Country"] + upper_cols)
        .unpivot(index="Country", variable_name="yr", value_name="upper_raw")
        .with_columns(
            pl.col("yr").str.replace("_upper", "").cast(pl.Int64).alias("year")
        )
        .drop("yr")
    )
    long = val_long.join(lower_long, on=["Country", "year"]).join(
        upper_long, on=["Country", "year"]
    )

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
    long = long.drop(["val_raw", "lower_raw", "upper_raw", "Country"])
    return long


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

unaids_count_cols = [
    "unaids_incidence_number",
    "unaids_incidence_number_lower",
    "unaids_incidence_number_upper",
    "unaids_prevalence_number",
    "unaids_prevalence_number_lower",
    "unaids_prevalence_number_upper",
]
data = data.with_columns(
    [
        (pl.col(col) * pl.col("gbd_age_fraction")).alias(col)
        for col in unaids_count_cols
    ]
)

#  Manually interpolate Somalia UNAIDS HIV incidence numbers (2023 by sex not published)

# 2022 values given: For each sex: val = <200 --> 150, lower = <100 --> 50, upper = <200 --> 150
# 2023 values: Combined both sexes: 282.17 (from unaids future projections)
# assume that 2022 sex ratio (1:1 based on the precision of data we have) carries through to 2023, giving us 141.085 for each
# round to 150 to be consistent with the level of precision of the 2022 data, and to avoid implying false precision from a modelled input
# carry through bounds from 2022 for consistency as well

# Also carry forward the 2022 incidence rate (pre-/1000 value: 0.05, i.e. <0.1 per 1000 uninfected).
# Justification: both 2022 and 2023 Somalia rates are censored at <0.1 per 1000. Any true year-on-year
# change in incidence rate at this level would be masked by the <0.1 coding assumption

som_2023_values = {
    "unaids_incidence_number": 150.0,
    "unaids_incidence_number_lower": 50.0,
    "unaids_incidence_number_upper": 150.0,
}
for col, val in som_2023_values.items():
    data = data.with_columns(
        pl.when((pl.col("country_code") == "SOM") & (pl.col("year") == 2023))
        .then(pl.lit(val) * pl.col("gbd_age_fraction"))
        .otherwise(pl.col(col))
        .alias(col)
    )

data = data.with_columns(
    pl.when((pl.col("country_code") == "SOM") & (pl.col("year") == 2023))
    .then(pl.lit(0.05))
    .otherwise(pl.col("unaids_incidence_rate_uninfected"))
    .alias("unaids_incidence_rate_uninfected"),
    pl.when((pl.col("country_code") == "SOM") & (pl.col("year") == 2023))
    .then(pl.lit(0.05))
    .otherwise(pl.col("unaids_incidence_rate_uninfected_lower"))
    .alias("unaids_incidence_rate_uninfected_lower"),
    pl.when((pl.col("country_code") == "SOM") & (pl.col("year") == 2023))
    .then(pl.lit(0.05))
    .otherwise(pl.col("unaids_incidence_rate_uninfected_upper"))
    .alias("unaids_incidence_rate_uninfected_upper"),
)

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
# Cabo Verde
countries_to_exclude = ["LBR", "GNQ", "STP", "CPV"]
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

# Add in DHS treatment seeking rate for someone with STI symptoms, by country and sex
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

dhs_treatment_seeking = dhs_treatment_seeking.filter(
    pl.col("pct_sought_treatment_wtd").is_not_null()
)

# Convert to fraction and clean sex names
dhs_treatment_seeking = dhs_treatment_seeking.with_columns(
    frac_sought_treatment=pl.col("pct_sought_treatment_wtd") / 100,
    sex=pl.when(pl.col("sex") == "male")
    .then(pl.lit("Male"))
    .when(pl.col("sex") == "female")
    .then(pl.lit("Female"))
    .otherwise(pl.lit("Error: unexpected sex value")),
)
assert (
    dhs_treatment_seeking.filter(
        pl.col("sex") == "Error: unexpected sex value"
    ).height
    == 0
)

dhs_country = dhs_treatment_seeking.select(
    ["iso3", "sex", "year", "frac_sought_treatment"]
).rename(
    {
        "iso3": "country_code",
        "frac_sought_treatment": "frac_sought_treatment_country",
    }
)

# Calculate regional averages and interpolate
dhs_regional = (
    dhs_treatment_seeking.group_by("region", "sex", "year")
    .agg(frac_sought_treatment_region=pl.mean("frac_sought_treatment"))
    .sort("region", "sex", "year")
    .with_columns(
        frac_sought_treatment_region=pl.col("frac_sought_treatment_region")
        .interpolate(method="linear")
        .over(["region", "sex"])
    )
)

# Calculate Africa-wide averages
dhs_africa = dhs_treatment_seeking.group_by("sex", "year").agg(
    frac_sought_treatment_africa=pl.mean("frac_sought_treatment")
)

# Create a complete grid of sex and years to ensure all years are present
sex_values = dhs_africa.select("sex").unique()
year_min = dhs_africa.select(pl.col("year").min()).item()
year_max = dhs_africa.select(pl.col("year").max()).item()
complete_grid = sex_values.join(
    pl.DataFrame({"year": range(year_min, year_max + 1)}), how="cross"
)
dhs_africa = (
    complete_grid.join(dhs_africa, on=["sex", "year"], how="left")
    .sort("sex", "year")
    .with_columns(
        frac_sought_treatment_africa=pl.col("frac_sought_treatment_africa")
        .interpolate(method="linear")
        .over(["sex"])
    )
)

# Join all three levels to main data
data = (
    data.join(dhs_country, on=["country_code", "sex", "year"], how="left")
    .join(dhs_regional, on=["region", "sex", "year"], how="left")
    .join(dhs_africa, on=["sex", "year"], how="left")
)

data = data.sort("country_code", "sex", "age_group", "year")

# Interpolate country-level data within each country-sex group
data = data.with_columns(
    frac_sought_treatment_country_interp=pl.col(
        "frac_sought_treatment_country"
    )
    .interpolate(method="linear")
    .over(["country_code", "sex", "age_group"])
)

# Apply fallback hierarchy: country -> region -> africa
data = data.with_columns(
    frac_sought_treatment=pl.coalesce(
        [
            pl.col("frac_sought_treatment_country_interp"),
            pl.col("frac_sought_treatment_region"),
            pl.col("frac_sought_treatment_africa"),
        ]
    )
).drop(
    [
        "frac_sought_treatment_country",
        "frac_sought_treatment_country_interp",
        "frac_sought_treatment_region",
        "frac_sought_treatment_africa",
    ]
)

assert (
    data.filter(pl.col("year") >= 2000)["frac_sought_treatment"].null_count()
    == 0
), "Missing DHS treatment seeking rate for some rows"

# let's take out part of the male population and make it msm
msm_fraction = params["msm_fraction"]

msm_cols = [
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

msm_data = (
    data.filter(pl.col("sex") == "Male")
    .with_columns(**{col: pl.col(col) * msm_fraction for col in msm_cols})
    .with_columns(sex=pl.lit("MSM"))
)
non_msm_data = data.filter(pl.col("sex") == "Male").with_columns(
    **{col: pl.col(col) * (1 - msm_fraction) for col in msm_cols}
)
female_data = data.filter(pl.col("sex") == "Female")
data = pl.concat([msm_data, non_msm_data, female_data])

msm_sti_prev = pl.read_csv(os.path.join(data_dir, "MSM_STI_prev.csv"))

gc_msm_row = msm_sti_prev.filter(
    (pl.col("Pathogen") == "GC") & (pl.col("Population") == "MSM, 15-49")
)
gc_all_row = msm_sti_prev.filter(
    (pl.col("Pathogen") == "GC") & (pl.col("Population") == "All men, 15-49")
)
pr_gc = gc_msm_row["Value"].item() / gc_all_row["Value"].item()
pr_gc_upper = gc_msm_row["Lower"].item() / gc_all_row["Lower"].item()
pr_gc_lower = gc_msm_row["Upper"].item() / gc_all_row["Upper"].item()

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
pr_chlamydia_upper = (
    chlamydia_msm_row["Lower"].item() / chlamydia_all_row["Lower"].item()
)
pr_chlamydia_lower = (
    chlamydia_msm_row["Upper"].item() / chlamydia_all_row["Upper"].item()
)


def _msm(pr, f, all_male_prev):
    return pr * all_male_prev


def _non_msm(pr, f, all_male_prev):
    return all_male_prev * (1 - f * pr) / (1 - f)


for prev_col, pr_val, pr_lo, pr_hi in [
    ("gc_prevalence", pr_gc, pr_gc_lower, pr_gc_upper),
    (
        "chlamydia_prevalence",
        pr_chlamydia,
        pr_chlamydia_lower,
        pr_chlamydia_upper,
    ),
]:
    data = data.with_columns(
        pl.when(pl.col("sex") == "MSM")
        .then(_msm(pr_val, msm_fraction, pl.col(prev_col)))
        .when(pl.col("sex") == "Male")
        .then(_non_msm(pr_val, msm_fraction, pl.col(prev_col)))
        .otherwise(pl.col(prev_col))
        .alias(prev_col)
    )
    data = data.with_columns(
        pl.when(pl.col("sex") == "MSM")
        .then(_msm(pr_lo, msm_fraction, pl.col(f"{prev_col}_lower")))
        .when(pl.col("sex") == "Male")
        .then(_non_msm(pr_hi, msm_fraction, pl.col(f"{prev_col}_lower")))
        .otherwise(pl.col(f"{prev_col}_lower"))
        .alias(f"{prev_col}_lower")
    )
    data = data.with_columns(
        pl.when(pl.col("sex") == "MSM")
        .then(_msm(pr_hi, msm_fraction, pl.col(f"{prev_col}_upper")))
        .when(pl.col("sex") == "Male")
        .then(_non_msm(pr_lo, msm_fraction, pl.col(f"{prev_col}_upper")))
        .otherwise(pl.col(f"{prev_col}_upper"))
        .alias(f"{prev_col}_upper")
    )

data.write_csv(os.path.join(output_dir, "hiv_sti.csv"))

unaids_proj = pl.read_csv(
    os.path.join(data_dir, "unaids_future_hiv_projections_all_years.csv")
)
unaids_proj = unaids_proj.filter(pl.col("metric") == "new_infections")
unaids_proj = unaids_proj.with_columns(
    value=pl.col("95-95-95 treatment, 2024 level of prevention")
)
unaids_proj = unaids_proj.with_columns(
    country_code=pl.col("country").replace(ihme_name_to_iso3),
)
unaids_proj = unaids_proj.with_columns(
    region=pl.col("country_code").replace(countries_to_regions),
)
unaids_proj = unaids_proj.filter(
    pl.col("region").is_in(["Western", "Eastern", "Central", "Southern"])
)
unaids_proj = unaids_proj.select(
    ["country", "country_code", "region", "year", "value"]
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
            sex=pl.lit("Male"), value=pl.col("value") * pl.col("male_fraction")
        ).select(
            ["country", "country_code", "region", "year", "sex", "value"]
        ),
    ]
)


last_gbd_year = 2023

gbd_age_fractions_for_proj = (
    data.filter(pl.col("sex").is_in(["Female", "Male"]))
    .select(["country_code", "sex", "year", "age_group", "gbd_age_fraction"])
    .unique()
)

last_gbd_fractions = gbd_age_fractions_for_proj.filter(
    pl.col("year") == last_gbd_year
).drop("year")

proj_years = unaids_proj_by_sex["year"].unique().to_list()
historical_proj_years = [y for y in proj_years if y <= last_gbd_year]
future_proj_years = [y for y in proj_years if y > last_gbd_year]

proj_historical_age = unaids_proj_by_sex.filter(
    pl.col("year").is_in(historical_proj_years)
).join(
    gbd_age_fractions_for_proj,
    on=["country_code", "sex", "year"],
    how="left",
)

proj_future_age = unaids_proj_by_sex.filter(
    pl.col("year").is_in(future_proj_years)
).join(
    last_gbd_fractions,
    on=["country_code", "sex"],
    how="left",
)

unaids_proj_by_sex_age = pl.concat(
    [proj_historical_age, proj_future_age], how="diagonal_relaxed"
)
unaids_proj_by_sex_age = unaids_proj_by_sex_age.with_columns(
    unaids_inc_num_proj=pl.col("value") * pl.col("gbd_age_fraction")
).drop("value")

un_pop_for_proj = pl.concat(
    [
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
        pl.read_csv(os.path.join(data_dir, "UN_pop_projections.csv"))
        .with_columns(
            sex=pl.col("sex").replace({"female": "Female", "male": "Male"}),
            val=pl.col("val").cast(pl.Float64),
        )
        .rename({"location": "country", "val": "un_pop"})
        .select(["country", "sex", "year", "un_pop"]),
    ]
)

unaids_proj_by_sex_age = unaids_proj_by_sex_age.join(
    un_pop_for_proj, on=["country", "sex", "year"], how="left"
).with_columns(un_pop=pl.col("un_pop") * pl.col("gbd_age_fraction"))

unaids_proj_by_sex_age = unaids_proj_by_sex_age.filter(
    pl.col("country_code").is_in(
        data.filter(pl.col("cols_unaids_analysis"))["country_code"]
        .unique()
        .to_list()
    )
)

unaids_proj_msm = unaids_proj_by_sex_age.filter(
    pl.col("sex") == "Male"
).with_columns(
    sex=pl.lit("MSM"),
    unaids_inc_num_proj=pl.col("unaids_inc_num_proj") * msm_fraction,
    un_pop=pl.col("un_pop") * msm_fraction,
)
unaids_proj_male = unaids_proj_by_sex_age.filter(
    pl.col("sex") == "Male"
).with_columns(
    unaids_inc_num_proj=pl.col("unaids_inc_num_proj") * (1 - msm_fraction),
    un_pop=pl.col("un_pop") * (1 - msm_fraction),
)
unaids_proj_combined = pl.concat(
    [
        unaids_proj_by_sex_age.filter(pl.col("sex") == "Female"),
        unaids_proj_male,
        unaids_proj_msm,
    ]
)

# rename country to location so that the merge is nicer
unaids_proj_combined = unaids_proj_combined.rename({"country": "location"})
unaids_proj_combined = unaids_proj_combined.with_columns(
    region=pl.col("country_code").replace(countries_to_regions)
)

data = pl.concat([data, unaids_proj_combined], how="diagonal_relaxed")
data = data.sort(["country_code", "sex", "age_group", "year"])

# for STI prevalences, let's do linear interpolation over the last years of data
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

# Fit linear trend using only data from 2015 onwards, then extrapolate
recent_cutoff = 2015
for col in sti_cols:
    data = (
        data.with_columns(
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
            _year_mean=pl.col("_year_fit")
            .mean()
            .over(["country_code", "sex", "age_group"]),
            _val_mean=pl.col("_val_fit")
            .mean()
            .over(["country_code", "sex", "age_group"]),
        )
        .with_columns(
            _slope=(
                (
                    (pl.col("_year_fit") - pl.col("_year_mean"))
                    * (pl.col("_val_fit") - pl.col("_val_mean"))
                )
                .sum()
                .over(["country_code", "sex", "age_group"])
                / ((pl.col("_year_fit") - pl.col("_year_mean")).pow(2))
                .sum()
                .over(["country_code", "sex", "age_group"])
            )
        )
        .with_columns(
            _intercept=pl.col("_val_mean")
            - pl.col("_slope") * pl.col("_year_mean"),
        )
        .with_columns(
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

# For treatment, use our knowledge of the 95-95-95 targets to set the art efficacy
#  by 2030 as being 95**3
TARGET_COVERAGE = 0.95**3
TARGET_YEAR = 2030
LAST_OBSERVED_YEAR = 2023
treatment_cols = [
    "treatment_proportion",
    "treatment_proportion_lower",
    "treatment_proportion_upper",
]

data = data.with_columns(
    **{
        col: pl.col(col)
        .fill_null(strategy="forward")
        .over(["country_code", "sex", "age_group"])
        for col in treatment_cols
    }
)

for col in treatment_cols:
    data = data.with_columns(
        pl.when(pl.col("year") <= LAST_OBSERVED_YEAR)
        .then(pl.col(col))
        .otherwise(
            pl.col(col)
            + (
                pl.max_horizontal(pl.col(col), pl.lit(TARGET_COVERAGE))
                - pl.col(col)
            )
            * (pl.col("year") - LAST_OBSERVED_YEAR)
            / (TARGET_YEAR - LAST_OBSERVED_YEAR)
        )
        .alias(col)
    )

data = data.with_columns(
    frac_sought_treatment=pl.col("frac_sought_treatment").fill_null(1)
)

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

data = data.with_columns(
    unaids_prevalence_number=pl.col("unaids_prevalence_number")
    .fill_null(strategy="forward")
    .over(["country_code", "sex", "age_group"]),
    unaids_prevalence_number_lower=pl.col("unaids_prevalence_number_lower")
    .fill_null(strategy="forward")
    .over(["country_code", "sex", "age_group"]),
    unaids_prevalence_number_upper=pl.col("unaids_prevalence_number_upper")
    .fill_null(strategy="forward")
    .over(["country_code", "sex", "age_group"]),
    cumulative_incidence=(
        pl.col("unaids_inc_num_proj").fill_null(0)
        * (
            pl.col("treatment_proportion")
            + (1 - pl.col("treatment_proportion")) * 0.82
        )
    )
    .cum_sum()
    .over(["country_code", "sex", "age_group"]),
).with_columns(
    unaids_prevalence_number=pl.col("unaids_prevalence_number")
    + pl.col("cumulative_incidence"),
    unaids_prevalence_number_lower=pl.col("unaids_prevalence_number_lower")
    + pl.col("cumulative_incidence"),
    unaids_prevalence_number_upper=pl.col("unaids_prevalence_number_upper")
    + pl.col("cumulative_incidence"),
)

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

# even though we don't use the GBD columns for future projections, we need to do
# some forward filling for them so our loop does not crash
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

data = data.with_columns(
    **{
        col: pl.col(col)
        .fill_null(strategy="forward")
        .over(["country_code", "sex", "age_group"])
        for col in gbd_columns
    }
)

data.write_csv(os.path.join(output_dir, "hiv_sti_with_projections.csv"))
