import os

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


def plot_hiv_burden(hiv, ax, label, fig):
    hiv = (
        hiv.group_by(["year", "sex"])
        .agg(
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
            pl.sum("population"),
            pl.sum("population_lower"),
            pl.sum("population_upper"),
        )
        .with_columns(
            hiv_incidence_rate=pl.col("hiv_incidence_number")
            / pl.col("population")
            * 100000,
            hiv_incidence_rate_upper=pl.col("hiv_incidence_number_upper")
            / pl.col("population_upper")
            * 100000,
            hiv_incidence_rate_lower=pl.col("hiv_incidence_number_lower")
            / pl.col("population_lower")
            * 100000,
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
    ax.set_ylabel(f"HIV incidence per 100,000\nin {label} Africa")
    ax.legend()
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_ylim(0)


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_gc.csv"))
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
    plot_hiv_burden(
        hiv.filter(pl.col("region") == "Western"),
        ax[0, 0],  # type: ignore
        "Western",
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
    plot_hiv_burden(
        hiv.filter(pl.col("region") == "Eastern"),
        ax[0, 1],  # type: ignore
        "Eastern",
        fig,
    )

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
    plot_hiv_burden(
        hiv.filter(pl.col("region") == "Central"),
        ax[1, 0],  # type: ignore
        "Central",
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
    plot_hiv_burden(
        hiv.filter(pl.col("region") == "Southern"),
        ax[1, 1],  # type: ignore
        "Southern",
        fig,
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_across_africa_trends.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_across_africa_trends.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
