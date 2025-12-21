import os

import polars as pl

data_dir = "data"
output_dir = "outputs"

countries_to_regions = {
    "Somalia": "Eastern",
    "Central African Republic": "Central",
    "Ghana": "Western",
    "Mauritania": "Western",
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

# read spectrum sti data
spectrum_drug_resistance = pl.read_csv(
    os.path.join(data_dir, "spectrum_sti.csv")
)

region_drug_resistance = pl.read_csv(
    os.path.join(data_dir, "resistance_percent_region_africa.csv")
)

# turn reported percentages into proportions
spectrum_drug_resistance = spectrum_drug_resistance.with_columns(
    Ciprofloxacin=pl.col("Ciprofloxacin") / 100,
    Cefixime=pl.col("Cefixime") / 100,
    Azithromycin=pl.col("Azithromycin") / 100,
)
spectrum_drug_resistance = spectrum_drug_resistance.with_columns(
    region=pl.lit("Southern")
)

# calculate how much more regions are drug resistance than southern
# divide val, lower, upper for Eastern, Western, and Central by Southern values
region_drug_resistance = region_drug_resistance.with_columns(
    val=pl.col("val")
    / region_drug_resistance.filter(pl.col("region") == "Southern")["val"],
    lower=pl.col("lower")
    / region_drug_resistance.filter(pl.col("region") == "Southern")["val"],
    upper=pl.col("upper")
    / region_drug_resistance.filter(pl.col("region") == "Southern")["val"],
)

# make a resistance data frame
resistance = []
for region in region_drug_resistance["region"].unique():
    # create a spectrum_drug_resistance data frame where we take the values
    # and multiply by the relative rate
    rate_change = region_drug_resistance.filter(pl.col("region") == region)[
        "val"
    ]
    rate_change_lower = region_drug_resistance.filter(
        pl.col("region") == region
    )["lower"]
    rate_change_upper = region_drug_resistance.filter(
        pl.col("region") == region
    )["upper"]
    resistance.append(
        spectrum_drug_resistance.with_columns(
            region=pl.lit(region)
        ).with_columns(
            Ciprofloxacin=pl.col("Ciprofloxacin") * rate_change,
            Cefixime=pl.col("Cefixime") * rate_change,
            Azithromycin=pl.col("Azithromycin") * rate_change,
            Ciprofloxacin_lower=pl.col("Ciprofloxacin") * rate_change_lower,
            Cefixime_lower=pl.col("Cefixime") * rate_change_lower,
            Azithromycin_lower=pl.col("Azithromycin") * rate_change_lower,
            Ciprofloxacin_upper=pl.col("Ciprofloxacin") * rate_change_upper,
            Cefixime_upper=pl.col("Cefixime") * rate_change_upper,
            Azithromycin_upper=pl.col("Azithromycin") * rate_change_upper,
        )
    )

resistance = pl.concat(resistance)

resistance.write_csv(
    os.path.join(output_dir, "estimated_resistance_rates.csv")
)

# read in our hiv sti data
hiv_sti = pl.read_csv(os.path.join(output_dir, "hiv_sti.csv"))

# map countries to regions
hiv_sti = hiv_sti.with_columns(
    region=pl.col("location").replace(countries_to_regions)
)

# merge
hiv_sti = hiv_sti.join(
    resistance,
    on=["year", "region"],
    how="left",  # so we don't discard the years for which we don't have data
)

hiv_sti.write_csv(os.path.join(output_dir, "hiv_sti_with_gc_abx_r.csv"))
