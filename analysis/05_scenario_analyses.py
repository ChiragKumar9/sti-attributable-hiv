import os

import polars as pl

data_dir = "data"
output_dir = "outputs"

proportion_treated = 0.73  # need to put this data in csv

hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_gc.csv"))

# for the fake scenario we are doing where there is no cipro resistance
hiv = hiv.with_columns(
    hiv_averted=pl.col("hiv_incidence_number_attributable")
    - (
        pl.col("Ciprofloxacin_resistant_number")
        * (1 / (1 + pl.col("rr_mu_hiv_given_gc")))
    ),
    hiv_averted_lower=pl.col("hiv_incidence_number_attributable_lower")
    - (
        pl.col("Ciprofloxacin_resistant_number_lower")
        * (1 / (1 + pl.col("rr_lower_hiv_given_gc")))
    ),
    hiv_averted_upper=pl.col("hiv_incidence_number_attributable_upper")
    - (
        pl.col("Ciprofloxacin_resistant_number_upper")
        * (1 / (1 + pl.col("rr_upper_hiv_given_gc")))
    ),
)

# now we want to estimate indirect effects

hiv = hiv.sort(by="year")

hiv = (
    hiv.with_columns(
        # year-over-year change in hiv as a proxy for transmission
        yoy_change=(
            pl.col("hiv_incidence_number")
            - pl.col("hiv_incidence_number").shift(1).over(["location", "sex"])
        )
        / pl.col("hiv_incidence_number").shift(1).over(["location", "sex"]),
    )
    .with_columns(yoy_change=pl.col("yoy_change").abs())
    .with_columns(
        indirect_effect=pl.cum_sum("hiv_averted")
        * pl.col("yoy_change")
        * proportion_treated,
        indirect_effect_lower=pl.cum_sum("hiv_averted_lower")
        * pl.col("yoy_change")
        * proportion_treated,
        indirect_effect_upper=pl.cum_sum("hiv_averted_upper")
        * pl.col("yoy_change")
        * proportion_treated,
    )
    .with_columns(
        total_averted=pl.col("hiv_averted") + pl.col("indirect_effect"),
        total_averted_lower=pl.col("hiv_averted_lower")
        + pl.col("indirect_effect_lower"),
        total_averted_upper=pl.col("hiv_averted_upper")
        + pl.col("indirect_effect_upper"),
    )
)

hiv.write_csv(
    os.path.join(output_dir, "hiv_attributable_to_gc_no_cipro_resistance.csv")
)
