import os

import polars as pl
from scipy import stats

data_dir = "data"
output_dir = "outputs"

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

region_drug_resistance = pl.read_csv(
    os.path.join(data_dir, "resistance_percent_region_africa.csv")
)

gasp = pl.read_csv(
    os.path.join(data_dir, "FINAL-GC-resistance - All AMR data.csv")
)

# goal is to assemble resistance data for each country, region, and year
# first let's do some data cleaning
# for years that are reported as two years, take the latter
gasp = gasp.with_columns(
    year=pl.when(pl.col("Year").str.contains("-"))
    .then(pl.col("Year").str.split("-").list.get(-1))
    .otherwise(pl.col("Year"))
)

gasp = gasp.select(
    [
        "Source",
        "Country",
        "year",
        "Azithromycin_num_resistant",
        "Azithromycin_num_tested",
        "Cefixime_num_resistant",
        "Cefixime_num_tested",
        "Ciprofloxacin_num_resistant",
        "Ciprofloxacin_num_tested",
        "Ceftriaxone_num_resistant",
        "Ceftriaxone_num_tested",
    ]
)

gasp = gasp.with_columns(
    year=pl.col("year").cast(pl.Int64),
    Azithromycin_num_resistant=pl.col("Azithromycin_num_resistant").cast(
        pl.Float64, strict=False
    ),
    Azithromycin_num_tested=pl.col("Azithromycin_num_tested").cast(
        pl.Float64, strict=False
    ),
    Cefixime_num_resistant=pl.col("Cefixime_num_resistant").cast(
        pl.Float64, strict=False
    ),
    Cefixime_num_tested=pl.col("Cefixime_num_tested").cast(
        pl.Float64, strict=False
    ),
    Ciprofloxacin_num_resistant=pl.col("Ciprofloxacin_num_resistant").cast(
        pl.Float64, strict=False
    ),
    Ciprofloxacin_num_tested=pl.col("Ciprofloxacin_num_tested").cast(
        pl.Float64, strict=False
    ),
    Ceftriaxone_num_resistant=pl.col("Ceftriaxone_num_resistant").cast(
        pl.Float64, strict=False
    ),
    Ceftriaxone_num_tested=pl.col("Ceftriaxone_num_tested").cast(
        pl.Float64, strict=False
    ),
)

# group by country and year
gasp = gasp.drop("Source")  # drop source from gasp before summing
gasp = gasp.group_by(["Country", "year"]).sum()

# we know that any cases num_tested is 0, that really means no data
gasp = gasp.with_columns(
    Azithromycin_num_tested=pl.when(pl.col("Azithromycin_num_tested") == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("Azithromycin_num_tested")),
    Cefixime_num_tested=pl.when(pl.col("Cefixime_num_tested") == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("Cefixime_num_tested")),
    Ciprofloxacin_num_tested=pl.when(pl.col("Ciprofloxacin_num_tested") == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("Ciprofloxacin_num_tested")),
    Ceftriaxone_num_tested=pl.when(pl.col("Ceftriaxone_num_tested") == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("Ceftriaxone_num_tested")),
)
# similarly, when the num tested is null, the num resistant should be null
gasp = gasp.with_columns(
    Azithromycin_num_resistant=pl.when(
        pl.col("Azithromycin_num_tested").is_null()
    )
    .then(pl.lit(None))
    .otherwise(pl.col("Azithromycin_num_resistant")),
    Cefixime_num_resistant=pl.when(pl.col("Cefixime_num_tested").is_null())
    .then(pl.lit(None))
    .otherwise(pl.col("Cefixime_num_resistant")),
    Ciprofloxacin_num_resistant=pl.when(
        pl.col("Ciprofloxacin_num_tested").is_null()
    )
    .then(pl.lit(None))
    .otherwise(pl.col("Ciprofloxacin_num_resistant")),
    Ceftriaxone_num_resistant=pl.when(
        pl.col("Ceftriaxone_num_tested").is_null()
    )
    .then(pl.lit(None))
    .otherwise(pl.col("Ceftriaxone_num_resistant")),
)

# assign regions to countries
# some country names are special
gasp = gasp.with_columns(
    Country=pl.when(pl.col("Country") == "Cote d'Ivoire")
    .then(pl.lit("Côte d'Ivoire"))
    .when(pl.col("Country") == "Tanzania")
    .then(pl.lit("United Republic of Tanzania"))
    .otherwise(pl.col("Country"))
)
gasp = gasp.with_columns(
    region=pl.col("Country").replace(countries_to_regions)
)

# for each country and year, calculate resistance percent using Bayesian Beta framework
prior_susceptible = 1
prior_resistant = 1

gasp = gasp.with_columns(
    Azithromycin_num_susceptible=pl.col("Azithromycin_num_tested")
    - pl.col("Azithromycin_num_resistant"),
    Cefixime_num_susceptible=pl.col("Cefixime_num_tested")
    - pl.col("Cefixime_num_resistant"),
    Ciprofloxacin_num_susceptible=pl.col("Ciprofloxacin_num_tested")
    - pl.col("Ciprofloxacin_num_resistant"),
    Ceftriaxone_num_susceptible=pl.col("Ceftriaxone_num_tested")
    - pl.col("Ceftriaxone_num_resistant"),
)

gasp = gasp.with_columns(
    Azithromycin_num_resistant_posterior=pl.col("Azithromycin_num_resistant")
    + pl.lit(prior_resistant),
    Azithromycin_num_susceptible_posterior=pl.col(
        "Azithromycin_num_susceptible"
    )
    + pl.lit(prior_susceptible),
    Cefixime_num_resistant_posterior=pl.col("Cefixime_num_resistant")
    + pl.lit(prior_resistant),
    Cefixime_num_susceptible_posterior=pl.col("Cefixime_num_susceptible")
    + pl.lit(prior_susceptible),
    Ciprofloxacin_num_resistant_posterior=pl.col("Ciprofloxacin_num_resistant")
    + pl.lit(prior_resistant),
    Ciprofloxacin_num_susceptible_posterior=pl.col(
        "Ciprofloxacin_num_susceptible"
    )
    + pl.lit(prior_susceptible),
    Ceftriaxone_num_resistant_posterior=pl.col("Ceftriaxone_num_resistant")
    + pl.lit(prior_resistant),
    Ceftriaxone_num_susceptible_posterior=pl.col("Ceftriaxone_num_susceptible")
    + pl.lit(prior_susceptible),
)

gasp = gasp.with_columns(
    Azithromycin=pl.col("Azithromycin_num_resistant_posterior")
    / (
        pl.col("Azithromycin_num_resistant_posterior")
        + pl.col("Azithromycin_num_susceptible_posterior")
    ),
    Cefixime=pl.col("Cefixime_num_resistant_posterior")
    / (
        pl.col("Cefixime_num_resistant_posterior")
        + pl.col("Cefixime_num_susceptible_posterior")
    ),
    Ciprofloxacin=pl.col("Ciprofloxacin_num_resistant_posterior")
    / (
        pl.col("Ciprofloxacin_num_resistant_posterior")
        + pl.col("Ciprofloxacin_num_susceptible_posterior")
    ),
    Ceftriaxone=pl.col("Ceftriaxone_num_resistant_posterior")
    / (
        pl.col("Ceftriaxone_num_resistant_posterior")
        + pl.col("Ceftriaxone_num_susceptible_posterior")
    ),
)

gasp = gasp.with_columns(
    [
        pl.struct(
            [
                f"{abx}_num_resistant_posterior",
                f"{abx}_num_susceptible_posterior",
            ]
        )
        .map_elements(
            lambda x, abx=abx: stats.beta.ppf(
                0.025,
                x[f"{abx}_num_resistant_posterior"],
                x[f"{abx}_num_susceptible_posterior"],
            )
            if (
                x[f"{abx}_num_resistant_posterior"] is not None
                and x[f"{abx}_num_susceptible_posterior"] is not None
            )
            else None,
            return_dtype=pl.Float64,
        )
        .alias(f"{abx}_lower")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

gasp = gasp.with_columns(
    [
        pl.struct(
            [
                f"{abx}_num_resistant_posterior",
                f"{abx}_num_susceptible_posterior",
            ]
        )
        .map_elements(
            lambda x, abx=abx: stats.beta.ppf(
                0.975,
                x[f"{abx}_num_resistant_posterior"],
                x[f"{abx}_num_susceptible_posterior"],
            )
            if (
                x[f"{abx}_num_resistant_posterior"] is not None
                and x[f"{abx}_num_susceptible_posterior"] is not None
            )
            else None,
            return_dtype=pl.Float64,
        )
        .alias(f"{abx}_upper")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

years_of_data = gasp["year"].sort().unique().to_list()


# we want to make a dataframe for values across the region
gasp_region = (
    gasp.drop("Country")
    .group_by(["region", "year"])
    .sum()
    .select(
        [
            "region",
            "year",
            "Azithromycin_num_resistant",
            "Azithromycin_num_tested",
            "Cefixime_num_resistant",
            "Cefixime_num_tested",
            "Ciprofloxacin_num_resistant",
            "Ciprofloxacin_num_tested",
            "Ceftriaxone_num_resistant",
            "Ceftriaxone_num_tested",
        ]
    )
)

# we know 0s must be nulls
gasp_region = gasp_region.with_columns(
    Azithromycin_num_tested=pl.when(pl.col("Azithromycin_num_tested") == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("Azithromycin_num_tested")),
    Cefixime_num_tested=pl.when(pl.col("Cefixime_num_tested") == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("Cefixime_num_tested")),
    Ciprofloxacin_num_tested=pl.when(pl.col("Ciprofloxacin_num_tested") == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("Ciprofloxacin_num_tested")),
    Ceftriaxone_num_tested=pl.when(pl.col("Ceftriaxone_num_tested") == 0)
    .then(pl.lit(None))
    .otherwise(pl.col("Ceftriaxone_num_tested")),
)

# we know when tested is null, resistant is null
gasp_region = gasp_region.with_columns(
    Azithromycin_num_resistant=pl.when(
        pl.col("Azithromycin_num_tested").is_null()
    )
    .then(pl.lit(None))
    .otherwise(pl.col("Azithromycin_num_resistant")),
    Cefixime_num_resistant=pl.when(pl.col("Cefixime_num_tested").is_null())
    .then(pl.lit(None))
    .otherwise(pl.col("Cefixime_num_resistant")),
    Ciprofloxacin_num_resistant=pl.when(
        pl.col("Ciprofloxacin_num_tested").is_null()
    )
    .then(pl.lit(None))
    .otherwise(pl.col("Ciprofloxacin_num_resistant")),
    Ceftriaxone_num_resistant=pl.when(
        pl.col("Ceftriaxone_num_tested").is_null()
    )
    .then(pl.lit(None))
    .otherwise(pl.col("Ceftriaxone_num_resistant")),
)

# calculate susceptible
gasp_region = gasp_region.with_columns(
    Azithromycin_num_susceptible=pl.col("Azithromycin_num_tested")
    - pl.col("Azithromycin_num_resistant"),
    Cefixime_num_susceptible=pl.col("Cefixime_num_tested")
    - pl.col("Cefixime_num_resistant"),
    Ciprofloxacin_num_susceptible=pl.col("Ciprofloxacin_num_tested")
    - pl.col("Ciprofloxacin_num_resistant"),
    Ceftriaxone_num_susceptible=pl.col("Ceftriaxone_num_tested")
    - pl.col("Ceftriaxone_num_resistant"),
)

# calculate posterior values
gasp_region = gasp_region.with_columns(
    Azithromycin_num_resistant_posterior=pl.col("Azithromycin_num_resistant")
    + pl.lit(prior_resistant),
    Azithromycin_num_susceptible_posterior=pl.col(
        "Azithromycin_num_susceptible"
    )
    + pl.lit(prior_susceptible),
    Cefixime_num_resistant_posterior=pl.col("Cefixime_num_resistant")
    + pl.lit(prior_resistant),
    Cefixime_num_susceptible_posterior=pl.col("Cefixime_num_susceptible")
    + pl.lit(prior_susceptible),
    Ciprofloxacin_num_resistant_posterior=pl.col("Ciprofloxacin_num_resistant")
    + pl.lit(prior_resistant),
    Ciprofloxacin_num_susceptible_posterior=pl.col(
        "Ciprofloxacin_num_susceptible"
    )
    + pl.lit(prior_susceptible),
    Ceftriaxone_num_resistant_posterior=pl.col("Ceftriaxone_num_resistant")
    + pl.lit(prior_resistant),
    Ceftriaxone_num_susceptible_posterior=pl.col("Ceftriaxone_num_susceptible")
    + pl.lit(prior_susceptible),
)

# we want to fill in the NAs for the region-wide data
# so that we can use this region-wide data to fill in the NAs for the country data

gasp_region = gasp_region.with_columns(
    Azithromycin=pl.col("Azithromycin_num_resistant_posterior")
    / (
        pl.col("Azithromycin_num_resistant_posterior")
        + pl.col("Azithromycin_num_susceptible_posterior")
    ),
    Cefixime=pl.col("Cefixime_num_resistant_posterior")
    / (
        pl.col("Cefixime_num_resistant_posterior")
        + pl.col("Cefixime_num_susceptible_posterior")
    ),
    Ciprofloxacin=pl.col("Ciprofloxacin_num_resistant_posterior")
    / (
        pl.col("Ciprofloxacin_num_resistant_posterior")
        + pl.col("Ciprofloxacin_num_susceptible_posterior")
    ),
    Ceftriaxone=pl.col("Ceftriaxone_num_resistant_posterior")
    / (
        pl.col("Ceftriaxone_num_resistant_posterior")
        + pl.col("Ceftriaxone_num_susceptible_posterior")
    ),
)

gasp_region = gasp_region.with_columns(
    [
        pl.struct(
            [
                f"{abx}_num_resistant_posterior",
                f"{abx}_num_susceptible_posterior",
            ]
        )
        .map_elements(
            lambda x, abx=abx: stats.beta.ppf(
                0.025,
                x[f"{abx}_num_resistant_posterior"],
                x[f"{abx}_num_susceptible_posterior"],
            )
            if (
                x[f"{abx}_num_resistant_posterior"] is not None
                and x[f"{abx}_num_susceptible_posterior"] is not None
            )
            else None,
            return_dtype=pl.Float64,
        )
        .alias(f"{abx}_lower")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

gasp_region = gasp_region.with_columns(
    [
        pl.struct(
            [
                f"{abx}_num_resistant_posterior",
                f"{abx}_num_susceptible_posterior",
            ]
        )
        .map_elements(
            lambda x, abx=abx: stats.beta.ppf(
                0.975,
                x[f"{abx}_num_resistant_posterior"],
                x[f"{abx}_num_susceptible_posterior"],
            )
            if (
                x[f"{abx}_num_resistant_posterior"] is not None
                and x[f"{abx}_num_susceptible_posterior"] is not None
            )
            else None,
            return_dtype=pl.Float64,
        )
        .alias(f"{abx}_upper")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

for year in years_of_data:
    for region in gasp_region["region"].unique():
        for abx in [
            "Azithromycin",
            "Cefixime",
            "Ciprofloxacin",
            "Ceftriaxone",
        ]:
            # ask if there is data for this year and region
            data_point = gasp_region.filter(
                (pl.col("year") == year) & (pl.col("region") == region)
            )[f"{abx}"].to_list()
            if len(data_point) == 0 or data_point[0] is None:
                # there is no data for this year and region
                # is there data for this year for another region?
                other_regions_data = gasp_region.filter(
                    (pl.col("year") == year)
                )[
                    ["region", f"{abx}", f"{abx}_lower", f"{abx}_upper"]
                ].drop_nulls()
                if other_regions_data.height > 0:
                    # take the mean
                    mean_value = other_regions_data[f"{abx}"].mean()
                    lower = other_regions_data[f"{abx}_lower"].mean()
                    upper = other_regions_data[f"{abx}_upper"].mean()
                    # extrapolate to this region
                    # use the region drug resistance data to find out the
                    # relative rate between these regions and our region of interest
                    rate_region = region_drug_resistance.filter(
                        pl.col("region") == region
                    )["val"]
                    rate_other_regions = region_drug_resistance.filter(
                        pl.col("region").is_in(
                            other_regions_data["region"].to_list()
                        )
                    )["val"]
                    rate_other_regions_lower = region_drug_resistance.filter(
                        pl.col("region").is_in(
                            other_regions_data["region"].to_list()
                        )
                    )["lower"]
                    rate_other_regions_upper = region_drug_resistance.filter(
                        pl.col("region").is_in(
                            other_regions_data["region"].to_list()
                        )
                    )["upper"]

                    relative_rate = rate_region / rate_other_regions.mean()
                    relative_rate_lower = (
                        rate_region / rate_other_regions_upper.mean()
                    )
                    relative_rate_upper = (
                        rate_region / rate_other_regions_lower.mean()
                    )

                    extrapolated_value = mean_value * relative_rate
                    extrapolated_lower = lower * relative_rate_lower
                    extrapolated_upper = upper * relative_rate_upper

                    # now put these values in our dataframe
                    # but we have to put the values in differently depending
                    # on whether the year exists or not
                    if len(data_point) != 0:
                        # year existed, it was just null
                        gasp_region = gasp_region.with_columns(
                            pl.when(
                                (pl.col("year") == year)
                                & (pl.col("region") == region)
                            )
                            .then(pl.lit(extrapolated_value))
                            .otherwise(pl.col(f"{abx}"))
                            .alias(f"{abx}"),
                            pl.when(
                                (pl.col("year") == year)
                                & (pl.col("region") == region)
                            )
                            .then(pl.lit(extrapolated_lower))
                            .otherwise(pl.col(f"{abx}_lower"))
                            .alias(f"{abx}_lower"),
                            pl.when(
                                (pl.col("year") == year)
                                & (pl.col("region") == region)
                            )
                            .then(pl.lit(extrapolated_upper))
                            .otherwise(pl.col(f"{abx}_upper"))
                            .alias(f"{abx}_upper"),
                        )
                    else:
                        # year does not exist, we have to add a new row
                        new_row = pl.DataFrame(
                            {
                                "region": region,
                                "year": year,
                                f"{abx}": extrapolated_value,
                                f"{abx}_lower": extrapolated_lower,
                                f"{abx}_upper": extrapolated_upper,
                            }
                        )
                        gasp_region = pl.concat(
                            [gasp_region, new_row], how="diagonal_relaxed"
                        )


gasp_region = gasp_region.sort(["region", "year"])

# fill in any NAs with linear interpolation
gasp_region = gasp_region.with_columns(
    [
        pl.col(abx).interpolate(method="linear").over("region").alias(abx)
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

gasp_region = gasp_region.with_columns(
    [
        pl.col(f"{abx}_lower")
        .interpolate(method="linear")
        .over("region")
        .alias(f"{abx}_lower")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

gasp_region = gasp_region.with_columns(
    [
        pl.col(f"{abx}_upper")
        .interpolate(method="linear")
        .over("region")
        .alias(f"{abx}_upper")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

# we have to make a set of rows for the central region because there is no gasp
# surveillance data for any country in that region
# tether the central region to the southern region
gasp_region_central = gasp_region.filter(
    pl.col("region") == "Southern"
).with_columns(region=pl.lit("Central"))
# figure out the relative rate
relative_rate = (
    region_drug_resistance.filter(pl.col("region") == "Central")[
        "val"
    ].to_list()[0]
    / region_drug_resistance.filter(pl.col("region") == "Southern")[
        "val"
    ].to_list()[0]
)
relative_rate_lower = (
    region_drug_resistance.filter(pl.col("region") == "Central")[
        "lower"
    ].to_list()[0]
    / region_drug_resistance.filter(pl.col("region") == "Southern")[
        "upper"
    ].to_list()[0]
)
relative_rate_upper = (
    region_drug_resistance.filter(pl.col("region") == "Central")[
        "upper"
    ].to_list()[0]
    / region_drug_resistance.filter(pl.col("region") == "Southern")[
        "lower"
    ].to_list()[0]
)

gasp_region_central = gasp_region_central.with_columns(
    [
        pl.col(abx) * pl.lit(relative_rate).alias(abx)
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

gasp_region_central = gasp_region_central.with_columns(
    [
        pl.col(f"{abx}_lower")
        * pl.lit(relative_rate_lower).alias(f"{abx}_lower")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

gasp_region_central = gasp_region_central.with_columns(
    [
        pl.col(f"{abx}_upper")
        * pl.lit(relative_rate_upper).alias(f"{abx}_upper")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

# concatenate
gasp_region = pl.concat([gasp_region, gasp_region_central], how="vertical")

# now a similar process for the country-level data
# except now if the county-year data is missing, we fill in with the region-year
# data if available
# what this is really doing is letting the region-year data supercede the country-year
# data
gasp_country = gasp.join(
    gasp_region, on=["region", "year"], how="left", suffix="_region"
)

gasp_country = gasp_country.with_columns(
    [
        pl.when(pl.col(abx).is_null())
        .then(pl.col(f"{abx}_region"))
        .otherwise(pl.col(abx))
        .alias(abx)
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

gasp_country = gasp_country.with_columns(
    [
        pl.when(pl.col(f"{abx}_lower").is_null())
        .then(pl.col(f"{abx}_lower_region"))
        .otherwise(pl.col(f"{abx}_lower"))
        .alias(f"{abx}_lower")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

gasp_country = gasp_country.with_columns(
    [
        pl.when(pl.col(f"{abx}_upper").is_null())
        .then(pl.col(f"{abx}_upper_region"))
        .otherwise(pl.col(f"{abx}_upper"))
        .alias(f"{abx}_upper")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

# now we have two dataframes: gasp_country and gasp_region
# we want to combine them into one resistance dataframe
gasp_country = gasp_country.rename({"Country": "location"}).select(
    [
        "location",
        "year",
        "Azithromycin",
        "Azithromycin_lower",
        "Azithromycin_upper",
        "Cefixime",
        "Cefixime_lower",
        "Cefixime_upper",
        "Ciprofloxacin",
        "Ciprofloxacin_lower",
        "Ciprofloxacin_upper",
        "Ceftriaxone",
        "Ceftriaxone_lower",
        "Ceftriaxone_upper",
    ]
)

gasp_region = gasp_region.rename({"region": "location"}).select(
    [
        "location",
        "year",
        "Azithromycin",
        "Azithromycin_lower",
        "Azithromycin_upper",
        "Cefixime",
        "Cefixime_lower",
        "Cefixime_upper",
        "Ciprofloxacin",
        "Ciprofloxacin_lower",
        "Ciprofloxacin_upper",
        "Ceftriaxone",
        "Ceftriaxone_lower",
        "Ceftriaxone_upper",
    ]
)

resistance = pl.concat([gasp_country, gasp_region], how="vertical")

resistance.write_csv(
    os.path.join(output_dir, "estimated_resistance_rates.csv")
)

# read in our hiv sti data
hiv_sti = pl.read_csv(os.path.join(output_dir, "hiv_sti_with_projections.csv"))

# merge first with the country-level data
hiv_sti = hiv_sti.join(
    gasp_country,
    on=["year", "location"],
    how="left",  # so we don't discard the years for which we don't have data
)

# merge next with the region level data
hiv_sti = hiv_sti.join(
    gasp_region,
    left_on=["year", "region"],
    right_on=["year", "location"],
    how="left",
    suffix="_region",
)

# let the region-level data fill in any NAs
hiv_sti = hiv_sti.with_columns(
    [
        pl.when(pl.col(abx).is_null())
        .then(pl.col(f"{abx}_region"))
        .otherwise(pl.col(abx))
        .alias(abx)
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

hiv_sti = hiv_sti.with_columns(
    [
        pl.when(pl.col(f"{abx}_lower").is_null())
        .then(pl.col(f"{abx}_lower_region"))
        .otherwise(pl.col(f"{abx}_lower"))
        .alias(f"{abx}_lower")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

hiv_sti = hiv_sti.with_columns(
    [
        pl.when(pl.col(f"{abx}_upper").is_null())
        .then(pl.col(f"{abx}_upper_region"))
        .otherwise(pl.col(f"{abx}_upper"))
        .alias(f"{abx}_upper")
        for abx in ["Azithromycin", "Cefixime", "Ciprofloxacin", "Ceftriaxone"]
    ],
)

# drop the region-level columns
hiv_sti = hiv_sti.drop(
    [
        "Azithromycin_region",
        "Azithromycin_lower_region",
        "Azithromycin_upper_region",
        "Cefixime_region",
        "Cefixime_lower_region",
        "Cefixime_upper_region",
        "Ciprofloxacin_region",
        "Ciprofloxacin_lower_region",
        "Ciprofloxacin_upper_region",
        "Ceftriaxone_region",
        "Ceftriaxone_lower_region",
        "Ceftriaxone_upper_region",
    ]
)

hiv_sti.write_csv(os.path.join(output_dir, "hiv_sti_with_gc_abx_r.csv"))
