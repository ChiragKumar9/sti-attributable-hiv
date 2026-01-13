import os

import polars as pl

data_dir = "data"
output_dir = "outputs_unaids_sensitivity"

# HELPER FUNCTIONS

def filter_measure_and_alias(data, measure, alias_prefix):
    """Filter data by measure and alias value/upper/lower columns with prefix."""
    return data.filter(pl.col("measure") == measure).select([
        "country", "year", "sex", "age",
        pl.col("value").alias(f"{alias_prefix}_value"),
        pl.col("upper").alias(f"{alias_prefix}_upper"),
        pl.col("lower").alias(f"{alias_prefix}_lower"),
    ])


def calculate_adult_by_subtraction(data, measure_name):
    """Calculate adult values by subtracting child from all ages for Male/Female."""
    all_ages = data.filter(pl.col("age") == "all ages").select([
        "country", "year", "sex",
        pl.col("value").alias("value_all"),
        pl.col("upper").alias("upper_all"),
        pl.col("lower").alias("lower_all"),
    ])

    child = data.filter(pl.col("age") == "child").select([
        "country", "year", "sex",
        pl.col("value").alias("value_child"),
        pl.col("upper").alias("upper_child"),
        pl.col("lower").alias("lower_child"),
    ])

    adult = all_ages.join(child, on=["country", "year", "sex"], how="inner").with_columns(
        age=pl.lit("adult"),
        measure=pl.lit(measure_name),
        value=pl.col("value_all") - pl.col("value_child"),
        upper=pl.col("upper_all") - pl.col("lower_child"),
        lower=pl.col("lower_all") - pl.col("upper_child"),
    ).select(["country", "year", "sex", "age", "measure", "value", "upper", "lower"])

    return adult


# ==============================================================================
# DATASET 1: POPULATION DATA

population = pl.read_csv(os.path.join(data_dir, "all_age_ihme_population.csv"))

# Aggregate adult age groups (15-49, 50-69, 70+) into one adult category
population_adults = population.filter(
    pl.col("age").is_in(["15-49 years", "50-69 years", "70+ years"])
).group_by(["measure", "location", "sex", "year"]).agg(
    pl.sum("val"),
    pl.sum("upper"),
    pl.sum("lower")
).with_columns(age=pl.lit("adult"))

# Keep child (0-14) and all ages as-is, rename
population_other = population.filter(
    pl.col("age").is_in(["0-14 years", "All ages"])
).with_columns(
    age=pl.when(pl.col("age") == "0-14 years")
        .then(pl.lit("child"))
        .otherwise(pl.lit("all ages"))
).select(["measure", "location", "sex", "year", "val", "upper", "lower", "age"])

# Combine adult, child, all ages
population = pl.concat([population_adults, population_other])
population = population.rename({
    "location": "country",
    "val": "value"
}).select(["country", "year", "sex", "age", "measure", "value", "upper", "lower"])


# DATASET 2: HIV INCIDENCE RATE (per 1000)

incidence_hiv = pl.read_csv(os.path.join(data_dir, "UNAIDS_HIV_incidence_rate.csv"))

# Clean
incidence_hiv = incidence_hiv.with_columns(
    pl.col("val").cast(pl.Float64, strict=False),
    pl.col("lower").cast(pl.Float64, strict=False),
    pl.col("upper").cast(pl.Float64, strict=False),
)

incidence_hiv = incidence_hiv.filter(
    pl.col("val").is_not_null() &
    pl.col("lower").is_not_null() &
    pl.col("upper").is_not_null()
)

# Keep only country-year combinations that have all three sex options
incidence_hiv = incidence_hiv.with_columns(
    sex_count=pl.col("sex").n_unique().over(["location", "year"])
)
incidence_hiv = incidence_hiv.filter(pl.col("sex_count") == 3).drop("sex_count")

# Divide by 1000 to get rates in same units as GBD data
incidence_hiv = incidence_hiv.with_columns(
    (pl.col("val") / 1000).alias("val"),
    (pl.col("lower") / 1000).alias("lower"),
    (pl.col("upper") / 1000).alias("upper")
)

# Convert sex codes to standard format
incidence_hiv = incidence_hiv.with_columns(
    sex=pl.when(pl.col("sex") == "SEX_FMLE")
        .then(pl.lit("Female"))
        .when(pl.col("sex") == "SEX_MLE")
        .then(pl.lit("Male"))
        .when(pl.col("sex") == "SEX_BTSX")
        .then(pl.lit("Both"))
        .otherwise(pl.col("sex"))
)

# Convert country codes to country names
country_codes = pl.read_csv(os.path.join(data_dir, "country_codes.csv"))
country_codes = country_codes.select([pl.col("alpha-3"), pl.col("name")])
incidence_hiv = incidence_hiv.join(
    country_codes,
    left_on="location",
    right_on="alpha-3",
    how="left"
)

# Change to long format
incidence_hiv = incidence_hiv.select([
    pl.col("name").alias("country"),
    pl.col("year"),
    pl.col("sex"),
    pl.lit("all ages").alias("age"),
    pl.lit("incidence_rate").alias("measure"),
    pl.col("val").alias("value"),
    pl.col("upper"),
    pl.col("lower"),
])

# DATASET 3: HIV INCIDENCE NUMBERS (by age)
incidence_hiv_by_age = pl.read_csv(os.path.join(data_dir, "UNAIDS_HIV_incidence_num.csv"))

# Rename and change to long format
incidence_hiv_by_age = incidence_hiv_by_age.rename({
    "estimate": "value",
    "location": "country"
}).with_columns(
    sex=pl.lit("Both"),
    measure=pl.lit("incidence_number"),
    age=pl.when(pl.col("age") == "all ages")
        .then(pl.lit("all ages"))
        .otherwise(pl.lit("child"))
)

incidence_hiv_by_age = incidence_hiv_by_age.select([
    "country", "year", "sex", "age", "measure", "value", "upper", "lower"
])

# Calculate adult incidence numbers (all ages - child)
incidence_both_only = incidence_hiv_by_age.filter(pl.col("sex") == "Both")
incidence_adult_calc = calculate_adult_by_subtraction(incidence_both_only, "incidence_number")

incidence_hiv_by_age = pl.concat([incidence_hiv_by_age, incidence_adult_calc])

# Split child incidence 50/50 between Male and Female (assumption)
incidence_child_split = incidence_hiv_by_age.filter(pl.col("age") == "child")
incidence_child_male = incidence_child_split.with_columns(
    sex=pl.lit("Male"),
    value=pl.col("value") / 2,
    upper=pl.col("upper") / 2,
    lower=pl.col("lower") / 2,
)
incidence_child_female = incidence_child_split.with_columns(
    sex=pl.lit("Female"),
    value=pl.col("value") / 2,
    upper=pl.col("upper") / 2,
    lower=pl.col("lower") / 2,
)

incidence_hiv_by_age = pl.concat([
    incidence_hiv_by_age,
    incidence_child_male,
    incidence_child_female
])

# DATASET 4: HIV PREVALENCE (by age/sex)
prevalence_hiv = pl.read_csv(os.path.join(data_dir, "UNAIDS_HIV_prevalence.csv"))

#Clean data
prevalence_hiv = prevalence_hiv.rename({
    "location": "country",
    "Estimate children": "child_Both_val",
    "Low children": "child_Both_lower",
    "High children": "child_Both_upper",
    "Estimate women adult": "adult_Female_val",
    "Low women  adult": "adult_Female_lower",
    "High women adult": "adult_Female_upper",
    "Estimate adult": "adult_Both_val",
    "Low adult": "adult_Both_lower",
    "High adult": "adult_Both_upper"
})

prevalence_hiv = prevalence_hiv.with_columns(
    pl.col("child_Both_lower").cast(pl.Float64, strict=False),
    pl.col("child_Both_upper").cast(pl.Float64, strict=False),
    pl.col("adult_Female_lower").cast(pl.Float64, strict=False),
    pl.col("adult_Female_upper").cast(pl.Float64, strict=False),
    pl.col("adult_Both_lower").cast(pl.Float64, strict=False),
    pl.col("adult_Both_upper").cast(pl.Float64, strict=False),
)

prevalence_hiv = prevalence_hiv.filter(pl.col("child_Both_val").is_not_null())

# Calculate missing sex-age combinations
prevalence_hiv = prevalence_hiv.with_columns(
    # Child Male/Female (split 50/50) (assumption)
    child_Male_val=pl.col("child_Both_val") / 2,
    child_Male_lower=pl.col("child_Both_lower") / 2,
    child_Male_upper=pl.col("child_Both_upper") / 2,
    child_Female_val=pl.col("child_Both_val") / 2,
    child_Female_lower=pl.col("child_Both_lower") / 2,
    child_Female_upper=pl.col("child_Both_upper") / 2,
    # Adult Male (subtract Female from Both)
    adult_Male_val=pl.col("adult_Both_val") - pl.col("adult_Female_val"),
    adult_Male_lower=pl.col("adult_Both_lower") - pl.col("adult_Female_upper"),
    adult_Male_upper=pl.col("adult_Both_upper") - pl.col("adult_Female_lower"),
)

# Calculate all ages for each sex
prevalence_hiv = prevalence_hiv.with_columns(
    all_ages_Male_val=pl.col("child_Male_val") + pl.col("adult_Male_val"),
    all_ages_Male_lower=pl.col("child_Male_lower") + pl.col("adult_Male_lower"),
    all_ages_Male_upper=pl.col("child_Male_upper") + pl.col("adult_Male_upper"),
    all_ages_Female_val=pl.col("child_Female_val") + pl.col("adult_Female_val"),
    all_ages_Female_lower=pl.col("child_Female_lower") + pl.col("adult_Female_lower"),
    all_ages_Female_upper=pl.col("child_Female_upper") + pl.col("adult_Female_upper"),
    all_ages_Both_val=pl.col("child_Both_val") + pl.col("adult_Both_val"),
    all_ages_Both_lower=pl.col("child_Both_lower") + pl.col("adult_Both_lower"),
    all_ages_Both_upper=pl.col("child_Both_upper") + pl.col("adult_Both_upper"),
)

# Pivot to long format
age_sex_combos = [
    ("child", "Male"), ("child", "Female"), ("child", "Both"),
    ("adult", "Male"), ("adult", "Female"), ("adult", "Both"),
    ("all ages", "Male"), ("all ages", "Female"), ("all ages", "Both"),
]

prevalence_long_list = []
for age, sex in age_sex_combos:
    age_label = age.replace(" ", "_")
    col_prefix = f"{age_label}_{sex}"

    df_subset = prevalence_hiv.select([
        "country", "year",
        pl.lit(sex).alias("sex"),
        pl.lit(age).alias("age"),
        pl.lit("prevalence").alias("measure"),
        pl.col(f"{col_prefix}_val").alias("value"),
        pl.col(f"{col_prefix}_upper").alias("upper"),
        pl.col(f"{col_prefix}_lower").alias("lower"),
    ])
    prevalence_long_list.append(df_subset)

prevalence_hiv_long = pl.concat(prevalence_long_list)

# COMBINE ALL DATASETS
combined_data = pl.concat([
    population,
    incidence_hiv,
    incidence_hiv_by_age,
    prevalence_hiv_long
])

countries_to_keep = [
    "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde",
    "Cameroon", "Central African Republic", "Chad", "Comoros", "Congo",
    "Côte d'Ivoire", "Democratic Republic of the Congo", "Djibouti",
    "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
    "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mozambique", "Namibia",
    "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal",
    "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Togo",
    "Uganda", "United Republic of Tanzania", "Zambia", "Zimbabwe"
]
combined_data = combined_data.filter(pl.col("country").is_in(countries_to_keep))

# ==============================================================================
# CALCULATE MISSING VALUES
# 1) uninfected population
# Formula 1: uninfected_pop = population - prevalence - 0.5 * incidence_number
# (for rows with incidence_number: child Male/Female/Both, adult Both, all ages Both)
# Formula 2: uninfected_pop = (pop - prev) / (1 + 0.5 * incidence_rate)
# (for rows with incidence_rate: all ages Male/Female)

# Get population and prevalence 
pop_vals = filter_measure_and_alias(combined_data, "Population", "pop")
prev_vals = filter_measure_and_alias(combined_data, "prevalence", "prev")

# Formula 1: Using incidence_number (child rows + adult Both + all ages Both)
inc_num_vals = filter_measure_and_alias(combined_data, "incidence_number", "inc")

uninfected_calc_1 = pop_vals.join(prev_vals, on=["country", "year", "sex", "age"], how="inner")
uninfected_calc_1 = uninfected_calc_1.join(inc_num_vals, on=["country", "year", "sex", "age"], how="inner")

uninfected_pop_1 = uninfected_calc_1.with_columns(
    measure=pl.lit("uninfected_pop"),
    value=pl.col("pop_value") - pl.col("prev_value") - 0.5 * pl.col("inc_value"),
    upper=pl.col("pop_upper") - pl.col("prev_lower") - 0.5 * pl.col("inc_lower"),
    lower=pl.col("pop_lower") - pl.col("prev_upper") - 0.5 * pl.col("inc_upper"),
).select(["country", "year", "sex", "age", "measure", "value", "upper", "lower"])

# Formula 2: Using incidence_rate (all ages Male/Female only)
inc_rate_vals = filter_measure_and_alias(combined_data, "incidence_rate", "inc_rate")

uninfected_calc_2 = pop_vals.join(prev_vals, on=["country", "year", "sex", "age"], how="inner")
uninfected_calc_2 = uninfected_calc_2.join(inc_rate_vals, on=["country", "year", "sex", "age"], how="inner")

# Only keep all ages Male/Female
uninfected_calc_2 = uninfected_calc_2.filter(
    (pl.col("age") == "all ages") &
    (pl.col("sex").is_in(["Male", "Female"]))
)

uninfected_pop_2 = uninfected_calc_2.with_columns(
    measure=pl.lit("uninfected_pop"),
    value=(pl.col("pop_value") - pl.col("prev_value")) / (1 + 0.5 * pl.col("inc_rate_value")),
    upper=(pl.col("pop_upper") - pl.col("prev_lower")) / (1 + 0.5 * pl.col("inc_rate_lower")),
    lower=(pl.col("pop_lower") - pl.col("prev_upper")) / (1 + 0.5 * pl.col("inc_rate_upper")),
).select(["country", "year", "sex", "age", "measure", "value", "upper", "lower"])

# Combine both calculation methods
uninfected_pop = pl.concat([uninfected_pop_1, uninfected_pop_2])

# Calculate adult Male/Female by subtraction (all ages - child)
uninf_male_female = uninfected_pop.filter(pl.col("sex").is_in(["Male", "Female"]))
uninf_adult_mf = calculate_adult_by_subtraction(uninf_male_female, "uninfected_pop")

# Combine all uninfected_pop calculations
uninfected_pop_all = pl.concat([uninfected_pop, uninf_adult_mf])

# Add uninfected_pop to combined data
combined_data = pl.concat([combined_data, uninfected_pop_all])

# CALCULATE INCIDENCE RATE VALUES (FOR CHILD AND ADULT BOTH)
#formula: incidence rate = incidence num / uninfected pop

uninfected_pop_vals = filter_measure_and_alias(combined_data, "uninfected_pop", "unin_pop")
inc_vals = filter_measure_and_alias(combined_data, "incidence_number", "inc")

incidence_rate_calc = uninfected_pop_vals.join(inc_vals, on=["country", "year", "sex", "age"], how="inner")

# Calculate incidence rate for rows where all components are present
# This should be: child male, child female, child Both, adult both
incidence_rates = incidence_rate_calc.with_columns(
    measure=pl.lit("incidence_rate"),
    value=pl.col("inc_value") / pl.col("unin_pop_value"),
    upper=pl.col("inc_upper") / pl.col("unin_pop_upper"),
    lower=pl.col("inc_lower") / pl.col("unin_pop_lower"),
).select(["country", "year", "sex", "age", "measure", "value", "upper", "lower"])

# Exclude rows already calculated above (exclude all ages rows)
incidence_rates = incidence_rates.filter(pl.col("age") != "all ages")
combined_data = pl.concat([combined_data, incidence_rates])

# CALCULATE INCIDENCE NUMBER VALUES
# formula: incidence number = incidence rate * uninfected pop

uninfected_pop_vals = filter_measure_and_alias(combined_data, "uninfected_pop", "unin_pop")
inc_rate_vals = filter_measure_and_alias(combined_data, "incidence_rate", "inc_rate")

incidence_num_calc = uninfected_pop_vals.join(inc_rate_vals, on=["country", "year", "sex", "age"], how="inner")

# Calculate incidence number for rows where all components are present
# This should be: all ages male, all ages female
incidence_numbers = incidence_num_calc.with_columns(
    measure=pl.lit("incidence_number"),
    value=pl.col("inc_rate_value") * pl.col("unin_pop_value"),
    upper=pl.col("inc_rate_upper") * pl.col("unin_pop_upper"),
    lower=pl.col("inc_rate_lower") * pl.col("unin_pop_lower"),
).select(["country", "year", "sex", "age", "measure", "value", "upper", "lower"])

# Only include the following two rows: all ages male, all ages female
incidence_numbers = incidence_numbers.filter(pl.col("age") == "all ages")
incidence_numbers = incidence_numbers.filter(pl.col("sex").is_in(["Male", "Female"]))

# Calculate adult male and adult female incidence numbers, subracting
inc_num_male_female = pl.concat([
    incidence_numbers,
    combined_data.filter(
        (pl.col("measure") == "incidence_number") &
        (pl.col("age") == "child") &
        (pl.col("sex").is_in(["Male", "Female"]))
    )
])
inc_num_adult_mf = calculate_adult_by_subtraction(inc_num_male_female, "incidence_number")

incidence_numbers_all = pl.concat([incidence_numbers, inc_num_adult_mf])

combined_data = pl.concat([combined_data, incidence_numbers_all])

# CALCULATE FINAL INCIDENCE RATE VALUES (FOR ADULT MALE/FEMALE)
# formula: incidence rate = incidence num / uninfected pop

uninfected_pop_vals = filter_measure_and_alias(combined_data, "uninfected_pop", "unin_pop")
inc_num_vals = filter_measure_and_alias(combined_data, "incidence_number", "inc")

incidence_rate_final_calc = uninfected_pop_vals.join(inc_num_vals, on=["country", "year", "sex", "age"], how="inner")

# Only keep adult Male/Female
incidence_rate_final_calc = incidence_rate_final_calc.filter(
    (pl.col("age") == "adult") &
    (pl.col("sex").is_in(["Male", "Female"]))
)
incidence_rates_final = incidence_rate_final_calc.with_columns(
    measure=pl.lit("incidence_rate"),
    value=pl.col("inc_value") / pl.col("unin_pop_value"),
    upper=pl.col("inc_upper") / pl.col("unin_pop_lower"),
    lower=pl.col("inc_lower") / pl.col("unin_pop_upper"),
).select(["country", "year", "sex", "age", "measure", "value", "upper", "lower"])

combined_data = pl.concat([combined_data, incidence_rates_final])

# REFORMAT FOR OUTPUT
# Filter for adult Male and Female only
# Keep: hiv_incidence_number, population, prevalence, p_acquiring_hiv (incidence_rate)

final_data = combined_data.filter(
    (pl.col("age") == "adult") &
    (pl.col("sex").is_in(["Male", "Female"]))
)

final_data = final_data.pivot(
    on="measure",
    values=["value", "upper", "lower"],
    index=["country", "year", "sex"]
)

final_data = final_data.rename({
    "country": "location",
    "value_incidence_number": "hiv_incidence_number",
    "lower_incidence_number": "hiv_incidence_number_lower",
    "upper_incidence_number": "hiv_incidence_number_upper",
    "value_Population": "population",
    "lower_Population": "population_lower",
    "upper_Population": "population_upper",
    "value_prevalence": "hiv_prevalence_number",
    "lower_prevalence": "hiv_prevalence_number_lower",
    "upper_prevalence": "hiv_prevalence_number_upper",
    "value_incidence_rate": "p_acquiring_hiv",
    "lower_incidence_rate": "p_acquiring_hiv_lower",
    "upper_incidence_rate": "p_acquiring_hiv_upper",
})

hiv_final_data = final_data.select([
    "sex", "year", "location",
    "hiv_incidence_number", "hiv_incidence_number_lower", "hiv_incidence_number_upper",
    "population", "population_lower", "population_upper",
    "hiv_prevalence_number", "hiv_prevalence_number_lower", "hiv_prevalence_number_upper",
    "p_acquiring_hiv", "p_acquiring_hiv_lower", "p_acquiring_hiv_upper",
])

# ==============================================================================
# ADD IN STI PREVALENCE AND HIV TREATMENT RATES

prevalence = pl.read_csv(os.path.join(data_dir, "ihme_prevalence.csv"))

# we need prevalence of gc
# we have to do this separately from the other STIs because the data are in a
# separate CSV
prevalence_gc = prevalence.filter(
    (pl.col("cause") == "Gonococcal infection")
    & (pl.col("metric") == "Number")
)

# Aggregate across all age groups to get total prevalence
prevalence_gc = prevalence_gc.group_by(
    ["location", "sex", "year"]
).agg(pl.sum("val"), pl.sum("lower"), pl.sum("upper"))

prevalence_gc = prevalence_gc.rename({
    "val": "gc_prevalence",
    "lower": "gc_prevalence_lower",
    "upper": "gc_prevalence_upper",
})

# join to incidence data - use left join to keep all HIV data
data = hiv_final_data.join(
    prevalence_gc, on=["sex", "year", "location"], how="left"
)

# we also want to join up with the other sti data
prevalence_sti = pl.read_csv(os.path.join(data_dir, "ihme_sti_prevalence.csv"))
prevalence_sti = prevalence_sti.filter(pl.col("metric") == "Number")

# Aggregate across all age groups
prevalence_sti = prevalence_sti.group_by(
    ["location", "sex", "cause", "year"]
).agg(pl.sum("val"), pl.sum("lower"), pl.sum("upper"))

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

# join with existing dataframe - use left join to keep all data
data = data.join(prevalence_sti, on=["sex", "year", "location"], how="left")

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

# save the assembled data
data.write_csv(os.path.join(output_dir, "hiv_sti.csv"))