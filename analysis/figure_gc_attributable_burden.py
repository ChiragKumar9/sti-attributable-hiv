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
            pl.sum("hiv_incidence_number_attributable_to_gc"),
            pl.sum("hiv_incidence_number_attributable_to_gc_upper"),
            pl.sum("hiv_incidence_number_attributable_to_gc_lower"),
            pl.sum("population"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            hiv_incidence_rate=pl.col(
                "hiv_incidence_number_attributable_to_gc"
            )
            / pl.col("hiv_incidence_number")
            * 100,
            hiv_incidence_rate_upper=pl.col(
                "hiv_incidence_number_attributable_to_gc_upper"
            )
            / pl.col("hiv_incidence_number_upper")
            * 100,
            hiv_incidence_rate_lower=pl.col(
                "hiv_incidence_number_attributable_to_gc_lower"
            )
            / pl.col("hiv_incidence_number_lower")
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
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_ylim(0)


def plot_attributable_hiv_burden_region(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum("hiv_incidence_number_attributable_to_gc"),
            pl.sum("hiv_incidence_number_attributable_to_gc_upper"),
            pl.sum("hiv_incidence_number_attributable_to_gc_lower"),
            pl.sum("population"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            hiv_incidence_rate=pl.col(
                "hiv_incidence_number_attributable_to_gc"
            )
            / pl.col("hiv_incidence_number")
            * 100,
            hiv_incidence_rate_upper=pl.col(
                "hiv_incidence_number_attributable_to_gc_upper"
            )
            / pl.col("hiv_incidence_number_upper")
            * 100,
            hiv_incidence_rate_lower=pl.col(
                "hiv_incidence_number_attributable_to_gc_lower"
            )
            / pl.col("hiv_incidence_number_lower")
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
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_ylim(0)


def plot_attributable_hiv_burden_map(hiv, mode, africa, ax, fig):
    if mode == "paf":
        hiv = (
            hiv.group_by(["year", "location"])
            .agg(
                pl.sum("hiv_incidence_number_attributable_to_gc"),
                pl.sum("hiv_incidence_number_attributable_to_gc_upper"),
                pl.sum("hiv_incidence_number_attributable_to_gc_lower"),
                pl.sum("hiv_incidence_number"),
                pl.sum("hiv_incidence_number_upper"),
                pl.sum("hiv_incidence_number_lower"),
            )
            .with_columns(
                percentage_attributable=pl.col(
                    "hiv_incidence_number_attributable_to_gc"
                )
                / pl.col("hiv_incidence_number")
                * 100,
                percentage_attributable_upper=pl.col(
                    "hiv_incidence_number_attributable_to_gc_upper"
                )
                / pl.col("hiv_incidence_number_upper")
                * 100,
                percentage_attributable_lower=pl.col(
                    "hiv_incidence_number_attributable_to_gc_lower"
                )
                / pl.col("hiv_incidence_number_lower")
                * 100,
            )
        )

        column = "percentage_attributable"
        legend = "%"
    elif mode == "incidence":
        hiv = (
            hiv.group_by(["year", "location"])
            .agg(
                pl.sum("hiv_incidence_number_attributable_to_gc"),
                pl.sum("hiv_incidence_number_attributable_to_gc_upper"),
                pl.sum("hiv_incidence_number_attributable_to_gc_lower"),
                pl.sum("population"),
                pl.sum("population_upper"),
                pl.sum("population_lower"),
            )
            .with_columns(
                incidence_attributable=pl.col(
                    "hiv_incidence_number_attributable_to_gc"
                )
                / pl.col("population")
                * 100000,
                incidence_attributable_upper=pl.col(
                    "hiv_incidence_number_attributable_to_gc_upper"
                )
                / pl.col("population_upper")
                * 100000,
                incidence_attributable_lower=pl.col(
                    "hiv_incidence_number_attributable_to_gc_lower"
                )
                / pl.col("population_lower")
                * 100000,
            )
        )

        column = "incidence_attributable"
        legend = "per 100,000"
    else:
        raise ValueError("mode must be 'paf' or 'incidence'")

    # join up with the africa shapefile
    africa = africa.merge(
        hiv.to_pandas(), left_on="ADM0_NAME", right_on="location", how="left"
    )
    has_data_mask = africa[column].notna()
    no_data_mask = africa[column].isna()

    africa[no_data_mask].plot(
        ax=ax, color="lightgrey", linewidth=0.3, edgecolor="grey"
    )

    # Plot counties with data
    africa[has_data_mask].plot(
        column=column,
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
        f"HIV incidence attributable to \ngonorrhea in 2023 ({legend})",
        rotation=0,
        labelpad=0,
        fontsize=28,
    )

    ax.set_axis_off()


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))

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
    plot_attributable_hiv_burden_map(
        hiv,
        "paf",
        africa,
        ax[1, 0],  # type: ignore
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
    plot_attributable_hiv_burden_map(
        hiv,
        "incidence",
        africa,
        ax[1, 1],  # type: ignore
        fig,
    )

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
