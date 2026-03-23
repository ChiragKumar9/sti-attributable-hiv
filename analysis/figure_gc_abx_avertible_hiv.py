import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl
from matplotlib import rc

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 25))
    if nrows * ncols == 1:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        ax.set_axisbelow(True)
        return fig, ax
    for temp in ax.flatten():  # type: ignore
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        # offset the spines
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        temp.set_axisbelow(True)
    return fig, ax


def sum_preserve_null(column: str) -> pl.Expr:
    """Custom aggregation function to sum while preserving nulls."""
    return (
        pl.when(pl.col(column).is_null().all())
        .then(None)
        .otherwise(pl.sum(column))
        .alias(column)
    )


def plot_attributable_hiv_burden_drug_resistance(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year"])
        .agg(
            sum_preserve_null("Ciprofloxacin_resistant_number"),
            sum_preserve_null("Ciprofloxacin_resistant_number_lower"),
            sum_preserve_null("Ciprofloxacin_resistant_number_upper"),
            sum_preserve_null("Cefixime_resistant_number"),
            sum_preserve_null("Cefixime_resistant_number_lower"),
            sum_preserve_null("Cefixime_resistant_number_upper"),
            sum_preserve_null("Azithromycin_resistant_number"),
            sum_preserve_null("Azithromycin_resistant_number_lower"),
            sum_preserve_null("Azithromycin_resistant_number_upper"),
            sum_preserve_null("Ceftriaxone_resistant_number"),
            sum_preserve_null("Ceftriaxone_resistant_number_lower"),
            sum_preserve_null("Ceftriaxone_resistant_number_upper"),
            pl.sum("hiv_incidence_number_attributable_to_gc"),
            pl.sum("hiv_incidence_number_attributable_to_gc_upper"),
            pl.sum("hiv_incidence_number_attributable_to_gc_lower"),
        )
        .filter(pl.col("year") >= 2008)
    ).with_columns(
        # Cefixime pre 2009 should be None
        Cefixime_resistant_number=pl.when((pl.col("year") < 2009))
        .then(pl.lit(None))
        .otherwise(pl.col("Cefixime_resistant_number")),
        Cefixime_resistant_number_lower=pl.when((pl.col("year") < 2009))
        .then(pl.lit(None))
        .otherwise(pl.col("Cefixime_resistant_number_lower")),
        Cefixime_resistant_number_upper=pl.when((pl.col("year") < 2009))
        .then(pl.lit(None))
        .otherwise(pl.col("Cefixime_resistant_number_upper")),
    )
    # mask all values at the attributable incidence
    hiv = hiv.with_columns(
        cipro=pl.when(
            pl.col("Ciprofloxacin_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number")),
        cipro_lower=pl.when(
            pl.col("Ciprofloxacin_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number_lower")),
        cipro_upper=pl.when(
            pl.col("Ciprofloxacin_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number_upper")),
        cef=pl.when(
            pl.col("Cefixime_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Cefixime_resistant_number")),
        cef_lower=pl.when(
            pl.col("Cefixime_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Cefixime_resistant_number_lower")),
        cef_upper=pl.when(
            pl.col("Cefixime_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Cefixime_resistant_number_upper")),
        azithro=pl.when(
            pl.col("Azithromycin_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Azithromycin_resistant_number")),
        azithro_lower=pl.when(
            pl.col("Azithromycin_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Azithromycin_resistant_number_lower")),
        azithro_upper=pl.when(
            pl.col("Azithromycin_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Azithromycin_resistant_number_upper")),
        ceft=pl.when(
            pl.col("Ceftriaxone_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Ceftriaxone_resistant_number")),
        ceft_lower=pl.when(
            pl.col("Ceftriaxone_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Ceftriaxone_resistant_number_lower")),
        ceft_upper=pl.when(
            pl.col("Ceftriaxone_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Ceftriaxone_resistant_number_upper")),
    )

    # filter to years with resistance data
    hiv = hiv.filter(pl.col("year") <= 2023)

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv["year"],
        hiv["cipro"],
        linewidth=3,
        label="Ciprofloxacin",
        color="red",
    )

    ax.fill_between(
        hiv["year"],
        hiv["cipro_lower"],
        hiv["cipro_upper"],
        alpha=0.3,
        color="red",
    )

    ax.plot(
        hiv["year"], hiv["cef"], linewidth=3, label="Cefixime", color="blue"
    )

    ax.fill_between(
        hiv["year"],
        hiv["cef_lower"],
        hiv["cef_upper"],
        alpha=0.3,
        color="blue",
    )

    ax.plot(
        hiv["year"],
        hiv["azithro"],
        linewidth=3,
        label="Azithromycin",
        color="green",
    )

    ax.fill_between(
        hiv["year"],
        hiv["azithro_lower"],
        hiv["azithro_upper"],
        alpha=0.3,
        color="green",
    )

    ax.plot(
        hiv["year"],
        hiv["ceft"],
        linewidth=3,
        label="Ceftriaxone",
        color="purple",
    )

    ax.fill_between(
        hiv["year"],
        hiv["ceft_lower"],
        hiv["ceft_upper"],
        alpha=0.3,
        color="purple",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence (N)")
    ax.set_yscale("log")
    # put y ticks in normal notation
    ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
    ax.set_yticklabels(["1", "10", "100", "1,000", "10,000", "100,000"])
    ax.legend(loc=(0.5, 1.0), edgecolor="black")


def plot_attributable_hiv_burden_drug_resistance_region(hiv, ax, fig):
    # we want to enforce an ordering for the regions
    regions = ["Western", "Eastern", "Central", "Southern"]
    hiv = hiv.with_columns(region=pl.col("region").cast(pl.Enum(regions)))

    # summing over sex
    hiv = hiv.group_by(["year", "region"], maintain_order=True).agg(
        sum_preserve_null("Ciprofloxacin_resistant_number"),
        sum_preserve_null("Ciprofloxacin_resistant_number_lower"),
        sum_preserve_null("Ciprofloxacin_resistant_number_upper"),
        sum_preserve_null("Cefixime_resistant_number"),
        sum_preserve_null("Cefixime_resistant_number_lower"),
        sum_preserve_null("Cefixime_resistant_number_upper"),
        sum_preserve_null("Azithromycin_resistant_number"),
        sum_preserve_null("Azithromycin_resistant_number_lower"),
        sum_preserve_null("Azithromycin_resistant_number_upper"),
        sum_preserve_null("Ceftriaxone_resistant_number"),
        sum_preserve_null("Ceftriaxone_resistant_number_lower"),
        sum_preserve_null("Ceftriaxone_resistant_number_upper"),
        pl.sum("hiv_incidence_number_attributable_to_gc"),
        pl.sum("hiv_incidence_number_attributable_to_gc_upper"),
        pl.sum("hiv_incidence_number_attributable_to_gc_lower"),
    )
    # filter to years with resistance data
    hiv = hiv.filter(pl.col("year") <= 2023)
    # sort by year
    hiv = hiv.sort(by=["region", "year"])
    # take the last value in each region -- the most recent year with data
    hiv = hiv.group_by(["region"], maintain_order=True).agg(
        pl.last("Ciprofloxacin_resistant_number"),
        pl.last("Ciprofloxacin_resistant_number_lower"),
        pl.last("Ciprofloxacin_resistant_number_upper"),
        pl.last("Cefixime_resistant_number"),
        pl.last("Cefixime_resistant_number_lower"),
        pl.last("Cefixime_resistant_number_upper"),
        pl.last("Azithromycin_resistant_number"),
        pl.last("Azithromycin_resistant_number_lower"),
        pl.last("Azithromycin_resistant_number_upper"),
        pl.last("Ceftriaxone_resistant_number"),
        pl.last("Ceftriaxone_resistant_number_lower"),
        pl.last("Ceftriaxone_resistant_number_upper"),
        pl.last("hiv_incidence_number_attributable_to_gc"),
        pl.last("hiv_incidence_number_attributable_to_gc_upper"),
        pl.last("hiv_incidence_number_attributable_to_gc_lower"),
    )
    # mask all values at the attributable incidence
    hiv = hiv.with_columns(
        cipro=pl.when(
            pl.col("Ciprofloxacin_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number")),
        cipro_lower=pl.when(
            pl.col("Ciprofloxacin_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number_lower")),
        cipro_upper=pl.when(
            pl.col("Ciprofloxacin_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number_upper")),
        cef=pl.when(
            pl.col("Cefixime_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Cefixime_resistant_number")),
        cef_lower=pl.when(
            pl.col("Cefixime_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Cefixime_resistant_number_lower")),
        cef_upper=pl.when(
            pl.col("Cefixime_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Cefixime_resistant_number_upper")),
        azithro=pl.when(
            pl.col("Azithromycin_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Azithromycin_resistant_number")),
        azithro_lower=pl.when(
            pl.col("Azithromycin_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Azithromycin_resistant_number_lower")),
        azithro_upper=pl.when(
            pl.col("Azithromycin_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Azithromycin_resistant_number_upper")),
        ceft=pl.when(
            pl.col("Ceftriaxone_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Ceftriaxone_resistant_number")),
        ceft_lower=pl.when(
            pl.col("Ceftriaxone_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Ceftriaxone_resistant_number_lower")),
        ceft_upper=pl.when(
            pl.col("Ceftriaxone_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Ceftriaxone_resistant_number_upper")),
    )

    drugs = ["cipro", "cef", "azithro", "ceft"]
    sti_names = {
        "cef": "Cefixime",
        "cipro": "Ciprofloxacin",
        "azithro": "Azithromycin",
        "ceft": "Ceftriaxone",
    }

    colors = {
        "cef": "blue",
        "cipro": "red",
        "azithro": "green",
        "ceft": "purple",
    }

    x = np.arange(len(regions))

    width = 0.2
    for i, pathogen in enumerate(drugs):
        offset = width * (i - 1.5)  # center the group of bars
        ax.bar(
            x + offset,
            hiv[pathogen],
            yerr=[
                hiv[pathogen] - hiv[f"{pathogen}_lower"],
                hiv[f"{pathogen}_upper"] - hiv[pathogen],
            ],
            capsize=5,
            width=width,
            label=sti_names[pathogen],
            color=colors[pathogen],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=-45, ha="left")
    ax.set_xlabel("Region")
    ax.set_ylabel("HIV incidence, 2023 (N)")
    ax.set_yscale("log")
    # put y ticks in normal notation
    ax.set_yticks([0.01, 0.1, 1, 10, 100, 1000, 10000])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100", "1,000", "10,000"])
    ax.legend(loc=(0.5, 1.0), edgecolor="black")


def plot_averted_hiv_region(hiv, ax, scenario_name, year, fig, forward=False):
    if forward:
        scalar = -1.0
    else:
        scalar = 1.0

    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum(f"hiv_averted_{scenario_name}"),
            pl.sum(f"hiv_averted_{scenario_name}_lower"),
            pl.sum(f"hiv_averted_{scenario_name}_upper"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            total=pl.lit(scalar) * pl.col(f"hiv_averted_{scenario_name}"),
            total_lower=pl.lit(scalar)
            * pl.col(f"hiv_averted_{scenario_name}_lower"),
            total_upper=pl.lit(scalar)
            * pl.col(f"hiv_averted_{scenario_name}_upper"),
        )
    )

    hiv = hiv.filter(pl.col("year") >= year)

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv.filter(pl.col("region") == "Western")["year"],
        hiv.filter(pl.col("region") == "Western")["total"],
        linewidth=3,
        label="Western",
        color="darkorange",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Western")["year"],
        hiv.filter(pl.col("region") == "Western")["total_lower"],
        hiv.filter(pl.col("region") == "Western")["total_upper"],
        alpha=0.3,
        color="darkorange",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Eastern")["year"],
        hiv.filter(pl.col("region") == "Eastern")["total"],
        linewidth=3,
        label="Eastern",
        color="olivedrab",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Eastern")["year"],
        hiv.filter(pl.col("region") == "Eastern")["total_lower"],
        hiv.filter(pl.col("region") == "Eastern")["total_upper"],
        alpha=0.3,
        color="olivedrab",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["total"],
        linewidth=3,
        label="Central",
        color="teal",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["total_lower"],
        hiv.filter(pl.col("region") == "Central")["total_upper"],
        alpha=0.3,
        color="teal",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["total"],
        linewidth=3,
        label="Southern",
        color="maroon",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["total_lower"],
        hiv.filter(pl.col("region") == "Southern")["total_upper"],
        alpha=0.3,
        color="maroon",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Change in HIV incidence (N)")
    ax.set_yscale("symlog")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    if forward:
        ax.legend(loc=(0.6, 0.6), edgecolor="black")
    else:
        ax.legend(loc=(0.6, 0.95), edgecolor="black")
        # minor x axis ticks every year
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        # minor y axis ticks at custom values
        ax.set_ylim(20)
        ax.yaxis.set_minor_locator(
            ticker.FixedLocator(
                scalar
                * np.array(
                    [
                        20,
                        30,
                        40,
                        50,
                        60,
                        70,
                        80,
                        90,
                        200,
                        300,
                        400,
                        500,
                        600,
                        700,
                        800,
                        900,
                        2000,
                        3000,
                        4000,
                        5000,
                        6000,
                        7000,
                        8000,
                        9000,
                        20000,
                        30000,
                        40000,
                        50000,
                        60000,
                        70000,
                        80000,
                        90000,
                        200000,
                    ]
                )  # type: ignore
            )
        )


def plot_access_vs_resistance(hiv, ax, year, fig):
    # aggregate globally across all locations and sexes
    hiv = (
        hiv.filter(pl.col("year") >= year)
        .group_by(["year"])
        .agg(
            pl.sum("hiv_averted_gc_untreated"),
            pl.sum("hiv_averted_gc_untreated_lower"),
            pl.sum("hiv_averted_gc_untreated_upper"),
            pl.sum("hiv_averted_gc_resistance"),
            pl.sum("hiv_averted_gc_resistance_lower"),
            pl.sum("hiv_averted_gc_resistance_upper"),
        )
        # plot as negative numbers -- averted cases are reductions
        .with_columns(
            untreated=pl.col("hiv_averted_gc_untreated") * -1.0,
            untreated_lower=pl.col("hiv_averted_gc_untreated_upper") * -1.0,
            untreated_upper=pl.col("hiv_averted_gc_untreated_lower") * -1.0,
            resistance=pl.col("hiv_averted_gc_resistance") * -1.0,
            resistance_lower=pl.col("hiv_averted_gc_resistance_upper") * -1.0,
            resistance_upper=pl.col("hiv_averted_gc_resistance_lower") * -1.0,
        )
        .sort(by="year")
    )

    ax.plot(
        hiv["year"],
        hiv["untreated"],
        linewidth=3,
        label="Universal access",
        color="darkgoldenrod",
    )
    ax.fill_between(
        hiv["year"],
        hiv["untreated_lower"],
        hiv["untreated_upper"],
        alpha=0.3,
        color="darkgoldenrod",
    )

    ax.plot(
        hiv["year"],
        hiv["resistance"],
        linewidth=3,
        label="Susceptible to first treatment",
        color="brown",
    )
    ax.fill_between(
        hiv["year"],
        hiv["resistance_lower"],
        hiv["resistance_upper"],
        alpha=0.3,
        color="brown",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Change in HIV incidence (N)")
    ax.set_yscale("symlog")
    ax.legend(loc=(0.03, 0.9), edgecolor="black")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )


def plot_access_vs_resistance_sex(hiv, ax, year, fig):
    # compute proportion of total averted that is attributable to lack of
    # access: untreated / (untreated + resistance), grouped by region and year
    hiv = (
        hiv.filter(pl.col("year") >= year)
        .group_by(["year", "sex"])
        .agg(
            pl.sum("hiv_averted_gc_untreated"),
            pl.sum("hiv_averted_gc_untreated_lower"),
            pl.sum("hiv_averted_gc_untreated_upper"),
            pl.sum("hiv_averted_gc_resistance"),
            pl.sum("hiv_averted_gc_resistance_lower"),
            pl.sum("hiv_averted_gc_resistance_upper"),
        )
        .with_columns(
            # proportion = untreated / (untreated + resistance)
            # for the lower bound of the proportion, use lower untreated and
            # upper resistance (most conservative split toward resistance)
            # for the upper bound, use upper untreated and lower resistance
            proportion=100
            * pl.col("hiv_averted_gc_untreated")
            / (
                pl.col("hiv_averted_gc_untreated")
                + pl.col("hiv_averted_gc_resistance")
            ),
            proportion_lower=100
            * pl.col("hiv_averted_gc_untreated_lower")
            / (
                pl.col("hiv_averted_gc_untreated_lower")
                + pl.col("hiv_averted_gc_resistance_lower")
            ),
            proportion_upper=100
            * pl.col("hiv_averted_gc_untreated_upper")
            / (
                pl.col("hiv_averted_gc_untreated_upper")
                + pl.col("hiv_averted_gc_resistance_upper")
            ),
        )
        .sort(by="year")
    )

    sex_colors = {
        "Female": "magenta",
        "Male": "midnightblue",
        "MSM": "slategrey",
    }

    for sex, color in sex_colors.items():
        region_df = hiv.filter(pl.col("sex") == sex)
        ax.plot(
            region_df["year"],
            region_df["proportion"],
            linewidth=3,
            label=sex,
            color=color,
        )
        ax.fill_between(
            region_df["year"],
            region_df["proportion_lower"],
            region_df["proportion_upper"],
            alpha=0.3,
            color=color,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "GC-attributable HIV avertible\nby antibiotic access alone (%)"
    )
    ax.legend(edgecolor="black")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(None, 100)


def plot_access_vs_resistance_region(hiv, ax, year, fig):
    # compute proportion of total averted that is attributable to lack of
    # access: untreated / (untreated + resistance), grouped by region and year
    hiv = (
        hiv.filter(pl.col("year") >= year)
        .group_by(["year", "region"])
        .agg(
            pl.sum("hiv_averted_gc_untreated"),
            pl.sum("hiv_averted_gc_untreated_lower"),
            pl.sum("hiv_averted_gc_untreated_upper"),
            pl.sum("hiv_averted_gc_resistance"),
            pl.sum("hiv_averted_gc_resistance_lower"),
            pl.sum("hiv_averted_gc_resistance_upper"),
        )
        .with_columns(
            # proportion = untreated / (untreated + resistance)
            # for the lower bound of the proportion, use lower untreated and
            # upper resistance (most conservative split toward resistance)
            # for the upper bound, use upper untreated and lower resistance
            proportion=100
            * pl.col("hiv_averted_gc_untreated")
            / (
                pl.col("hiv_averted_gc_untreated")
                + pl.col("hiv_averted_gc_resistance")
            ),
            proportion_lower=100
            * pl.col("hiv_averted_gc_untreated_lower")
            / (
                pl.col("hiv_averted_gc_untreated_lower")
                + pl.col("hiv_averted_gc_resistance_lower")
            ),
            proportion_upper=100
            * pl.col("hiv_averted_gc_untreated_upper")
            / (
                pl.col("hiv_averted_gc_untreated_upper")
                + pl.col("hiv_averted_gc_resistance_upper")
            ),
        )
        .sort(by="year")
    )

    region_colors = {
        "Western": "darkorange",
        "Eastern": "olivedrab",
        "Central": "teal",
        "Southern": "maroon",
    }

    for region, color in region_colors.items():
        region_df = hiv.filter(pl.col("region") == region)
        ax.plot(
            region_df["year"],
            region_df["proportion"],
            linewidth=3,
            label=region,
            color=color,
        )
        ax.fill_between(
            region_df["year"],
            region_df["proportion_lower"],
            region_df["proportion_upper"],
            alpha=0.3,
            color=color,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(
        "GC-attributable HIV avertible\nby antibiotic access alone (%)"
    )
    ax.legend(edgecolor="black")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_ylim(None, 100)


if __name__ == "__main__":
    fig, ax = setup_plot(3, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))

    ax[0, 0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    plot_attributable_hiv_burden_drug_resistance(
        hiv,
        ax[0, 0],  # type: ignore
        fig,
    )

    ax[0, 1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_drug_resistance_region(
        hiv,
        ax[0, 1],  # type: ignore
        fig,
    )

    hiv = pl.read_csv(
        os.path.join(output_dir, "hiv_averted.csv"),
    )

    hiv = hiv.filter(pl.col("year") <= 2023)

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_access_vs_resistance(
        hiv,
        ax[1, 0],  # type: ignore
        2009,
        fig,
    )

    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_access_vs_resistance_sex(
        hiv,
        ax[1, 1],  # type: ignore
        2009,
        fig,
    )

    ax[2, 0].text(  # type: ignore
        -0.25,
        1.08,
        "e",
        transform=ax[2, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_access_vs_resistance_region(
        hiv,
        ax[2, 0],  # type: ignore
        2009,
        fig,
    )

    ax[2, 1].text(  # type: ignore
        -0.25,
        1.08,
        "f",
        transform=ax[2, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_hiv_region(hiv, ax[2, 1], "2016_gc_change", 2017, fig)  # type: ignore

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_hiv_averted_appropriate_gc_abx.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_hiv_averted_appropriate_gc_abx.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
