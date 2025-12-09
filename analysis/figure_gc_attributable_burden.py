import os

import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl
from matplotlib import rc

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15))
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


def plot_attributable_hiv_burden_sex(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year", "sex"])
        .agg(
            pl.sum("hiv_incidence_number_attributable"),
            pl.sum("hiv_incidence_number_attributable_upper"),
            pl.sum("hiv_incidence_number_attributable_lower"),
            pl.sum("population"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            hiv_incidence_rate=pl.col("hiv_incidence_number_attributable")
            / pl.col("hiv_incidence_number")
            * 100,
            hiv_incidence_rate_upper=pl.col(
                "hiv_incidence_number_attributable_upper"
            )
            / pl.col("hiv_incidence_number_lower")
            * 100,
            hiv_incidence_rate_lower=pl.col(
                "hiv_incidence_number_attributable_lower"
            )
            / pl.col("hiv_incidence_number_upper")
            * 100,
        )
    )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv.filter(pl.col("sex") == "Male")["year"],
        hiv.filter(pl.col("sex") == "Male")["hiv_incidence_rate"],
        linewidth=3,
        label="Male",
        color="midnightblue",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Male")["year"],
        hiv.filter(pl.col("sex") == "Male")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("sex") == "Male")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="midnightblue",
    )

    ax.plot(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["hiv_incidence_rate"],
        linewidth=3,
        label="Female",
        color="magenta",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("sex") == "Female")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="magenta",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence attributable\nto gonorrhea (%)")
    ax.legend()
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_ylim(0)


def plot_attributable_hiv_burden_region(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum("hiv_incidence_number_attributable"),
            pl.sum("hiv_incidence_number_attributable_upper"),
            pl.sum("hiv_incidence_number_attributable_lower"),
            pl.sum("population"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            hiv_incidence_rate=pl.col("hiv_incidence_number_attributable")
            / pl.col("hiv_incidence_number")
            * 100,
            hiv_incidence_rate_upper=pl.col(
                "hiv_incidence_number_attributable_upper"
            )
            / pl.col("hiv_incidence_number_lower")
            * 100,
            hiv_incidence_rate_lower=pl.col(
                "hiv_incidence_number_attributable_lower"
            )
            / pl.col("hiv_incidence_number_upper")
            * 100,
        )
    )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv.filter(pl.col("region") == "Western")["year"],
        hiv.filter(pl.col("region") == "Western")["hiv_incidence_rate"],
        linewidth=3,
        label="Western",
        color="darkorange",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Western")["year"],
        hiv.filter(pl.col("region") == "Western")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Western")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="darkorange",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Eastern")["year"],
        hiv.filter(pl.col("region") == "Eastern")["hiv_incidence_rate"],
        linewidth=3,
        label="Eastern",
        color="olivedrab",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Eastern")["year"],
        hiv.filter(pl.col("region") == "Eastern")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Eastern")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="olivedrab",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["hiv_incidence_rate"],
        linewidth=3,
        label="Central",
        color="royalblue",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Central")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="royalblue",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate"],
        linewidth=3,
        label="Southern",
        color="purple",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="purple",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence attributable\nto gonorrhea (%)")
    ax.legend()
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_ylim(0)


def plot_attributable_hiv_burden_map(hiv, africa, ax, fig):
    hiv = (
        hiv.group_by(["year", "location"])
        .agg(
            pl.sum("hiv_incidence_number_attributable"),
            pl.sum("hiv_incidence_number_attributable_upper"),
            pl.sum("hiv_incidence_number_attributable_lower"),
            pl.sum("population"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            percentage_attributable=pl.col("hiv_incidence_number_attributable")
            / pl.col("hiv_incidence_number")
            * 100,
            percentage_attributable_upper=pl.col(
                "hiv_incidence_number_attributable_upper"
            )
            / pl.col("hiv_incidence_number_lower")
            * 100,
            percentage_attributable_lower=pl.col(
                "hiv_incidence_number_attributable_lower"
            )
            / pl.col("hiv_incidence_number_upper")
            * 100,
        )
        .filter(
            # take the last year of data
            pl.col("year") == pl.max("year")
        )
    )

    # join up with the africa shapefile
    africa = africa.merge(
        hiv.to_pandas(), left_on="ADM0_NAME", right_on="location", how="left"
    )
    has_data_mask = africa["percentage_attributable"].notna()
    no_data_mask = africa["percentage_attributable"].isna()

    africa[no_data_mask].plot(
        ax=ax, color="lightgrey", linewidth=0.3, edgecolor="grey"
    )

    # Plot counties with data
    africa[has_data_mask].plot(
        column="percentage_attributable",
        cmap="Reds",
        linewidth=0.3,
        ax=ax,
        edgecolor="grey",
        legend=False,
        missing_kwds={"color": "lightgrey"},
    )

    cbar = fig.colorbar(
        ax.collections[-1],
        ax=ax,
        shrink=0.7,  # Size of colorbar relative to axes
        aspect=20,  # Ratio of long to short dimensions
        pad=0.0,  # Distance between axes and colorbar
        extend="neither",  # Add arrow at top ('neither', 'both', 'min', 'max')
        orientation="horizontal",  # 'vertical' or 'horizontal'
    )

    # Customize the colorbar
    cbar.set_label(
        f"Gonorrhea-attributable HIV incidence in {pl.col('year').max()} (%)",
        rotation=0,
        labelpad=0,
        fontsize=28,
    )

    ax.set_axis_off()


def plot_attributable_hiv_burden_drug_resistance(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum("Ciprofloxacin_resistant_number"),
            pl.sum("Ciprofloxacin_resistant_number_lower"),
            pl.sum("Ciprofloxacin_resistant_number_upper"),
            pl.sum("Cefixime_resistant_number"),
            pl.sum("Cefixime_resistant_number_lower"),
            pl.sum("Cefixime_resistant_number_upper"),
            pl.sum("Azithromycin_resistant_number"),
            pl.sum("Azithromycin_resistant_number_lower"),
            pl.sum("Azithromycin_resistant_number_upper"),
            pl.sum("population"),
            pl.sum("hiv_incidence_number_attributable"),
            pl.sum("hiv_incidence_number_attributable_upper"),
            pl.sum("hiv_incidence_number_attributable_lower"),
        )
        .with_columns(
            cipro=pl.col("Ciprofloxacin_resistant_number")
            / pl.col("hiv_incidence_number_attributable")
            * 100,
            cipro_lower=pl.col("Ciprofloxacin_resistant_number_upper")
            / pl.col("hiv_incidence_number_attributable_lower")
            * 100,
            cipro_upper=pl.col("Ciprofloxacin_resistant_number_lower")
            / pl.col("hiv_incidence_number_attributable_upper")
            * 100,
            cef=pl.col("Cefixime_resistant_number")
            / pl.col("hiv_incidence_number_attributable")
            * 100,
            cef_lower=pl.col("Cefixime_resistant_number_upper")
            / pl.col("hiv_incidence_number_attributable_lower")
            * 100,
            cef_upper=pl.col("Cefixime_resistant_number_lower")
            / pl.col("hiv_incidence_number_attributable_upper")
            * 100,
            azithro=pl.col("Azithromycin_resistant_number")
            / pl.col("hiv_incidence_number_attributable")
            * 100,
            azithro_lower=pl.col("Azithromycin_resistant_number_upper")
            / pl.col("hiv_incidence_number_attributable_lower")
            * 100,
            azithro_upper=pl.col("Azithromycin_resistant_number_lower")
            / pl.col("hiv_incidence_number_attributable_upper")
            * 100,
        )
    )
    # mask all values at 100
    hiv = hiv.with_columns(
        cipro=pl.when(pl.col("cipro") > 100)
        .then(100)
        .otherwise(pl.col("cipro")),
        cipro_lower=pl.when(pl.col("cipro_lower") > 100)
        .then(100)
        .otherwise(pl.col("cipro_lower")),
        cipro_upper=pl.when(pl.col("cipro_upper") > 100)
        .then(100)
        .otherwise(pl.col("cipro_upper")),
        cef=pl.when(pl.col("cef") > 100).then(100).otherwise(pl.col("cef")),
        cef_lower=pl.when(pl.col("cef_lower") > 100)
        .then(100)
        .otherwise(pl.col("cef_lower")),
        cef_upper=pl.when(pl.col("cef_upper") > 100)
        .then(100)
        .otherwise(pl.col("cef_upper")),
        azithro=pl.when(pl.col("azithro") > 100)
        .then(100)
        .otherwise(pl.col("azithro")),
        azithro_lower=pl.when(pl.col("azithro_lower") > 100)
        .then(100)
        .otherwise(pl.col("azithro_lower")),
        azithro_upper=pl.when(pl.col("azithro_upper") > 100)
        .then(100)
        .otherwise(pl.col("azithro_upper")),
    )

    # subset the data to only include post 2007 because resistance data is not available before then
    hiv = hiv.filter(pl.col("year") >= 2006)
    # also note that 0s are not 0s -- they are nulls when they are part of the timeseries
    hiv = hiv.with_columns(
        cipro=pl.when(pl.col("Ciprofloxacin_resistant_number") == 0)
        .then(None)
        .otherwise(pl.col("cipro")),
        cipro_lower=pl.when(
            pl.col("Ciprofloxacin_resistant_number_upper") == 0
        )
        .then(None)
        .otherwise(pl.col("cipro_lower")),
        cipro_upper=pl.when(
            pl.col("Ciprofloxacin_resistant_number_lower") == 0
        )
        .then(None)
        .otherwise(pl.col("cipro_upper")),
        # cef = pl.when(pl.col("Cefixime_resistant_number") == 0).then(None).otherwise(pl.col("cef")),
        # cef_lower = pl.when(pl.col("Cefixime_resistant_number_upper") == 0).then(None).otherwise(pl.col("cef_lower")),
        # cef_upper = pl.when(pl.col("Cefixime_resistant_number_lower") == 0).then(None).otherwise(pl.col("cef_upper")),
        azithro=pl.when(pl.col("Azithromycin_resistant_number") == 0)
        .then(None)
        .otherwise(pl.col("azithro")),
        azithro_lower=pl.when(
            pl.col("Azithromycin_resistant_number_upper") == 0
        )
        .then(None)
        .otherwise(pl.col("azithro_lower")),
        azithro_upper=pl.when(
            pl.col("Azithromycin_resistant_number_lower") == 0
        )
        .then(None)
        .otherwise(pl.col("azithro_upper")),
    )

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

    ax.set_xlabel("Year")
    ax.set_ylabel("Resistance among\ngonorrhea-attributable HIV (%)")
    ax.legend()
    ax.set_ylim(0)


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_gc.csv"))

    # by sex
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
    plot_attributable_hiv_burden_sex(hiv, ax[0, 0], fig)  # type: ignore

    # by region
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
    plot_attributable_hiv_burden_region(hiv, ax[0, 1], fig)  # type: ignore

    africa = gpd.read_file(os.path.join(data_dir, "afr_g2014_2013_0.shp"))
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
    plot_attributable_hiv_burden_map(hiv, africa, ax[1, 0], fig)  # type: ignore

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
    plot_attributable_hiv_burden_drug_resistance(hiv, ax[1, 1], fig)  # type: ignore

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_attributable_hiv.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_attributable_hiv.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
