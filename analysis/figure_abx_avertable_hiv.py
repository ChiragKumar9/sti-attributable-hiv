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


def plot_averted_hiv(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum("hiv_averted"),
            pl.sum("hiv_averted_lower"),
            pl.sum("hiv_averted_upper"),
            pl.sum("total_averted"),
            pl.sum("total_averted_lower"),
            pl.sum("total_averted_upper"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            direct=pl.col("hiv_averted")
            / pl.col("hiv_incidence_number")
            * 100,
            direct_lower=pl.col("hiv_averted_lower")
            / pl.col("hiv_incidence_number_lower")
            * 100,
            direct_upper=pl.col("hiv_averted_upper")
            / pl.col("hiv_incidence_number_upper")
            * 100,
            total=pl.col("total_averted")
            / pl.col("hiv_incidence_number")
            * 100,
            total_lower=pl.col("total_averted_lower")
            / pl.col("hiv_incidence_number_lower")
            * 100,
            total_upper=pl.col("total_averted_upper")
            / pl.col("hiv_incidence_number_upper")
            * 100,
        )
    )
    # cap values at 100%
    hiv = hiv.with_columns(
        direct=pl.when(pl.col("direct") > 100)
        .then(100)
        .otherwise(pl.col("direct")),
        direct_lower=pl.when(pl.col("direct_lower") > 100)
        .then(100)
        .otherwise(pl.col("direct_lower")),
        direct_upper=pl.when(pl.col("direct_upper") > 100)
        .then(100)
        .otherwise(pl.col("direct_upper")),
        total=pl.when(pl.col("total") > 100)
        .then(100)
        .otherwise(pl.col("total")),
        total_lower=pl.when(pl.col("total_lower") > 100)
        .then(100)
        .otherwise(pl.col("total_lower")),
        total_upper=pl.when(pl.col("total_upper") > 100)
        .then(100)
        .otherwise(pl.col("total_upper")),
    )

    # subset the data to only include post 2007 because resistance data is not available before then
    hiv = hiv.filter(pl.col("year") >= 2006)

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv["year"], hiv["direct"], linewidth=3, label="Direct", color="grey"
    )

    ax.fill_between(
        hiv["year"],
        hiv["direct_lower"],
        hiv["direct_upper"],
        alpha=0.3,
        color="grey",
    )

    ax.plot(
        hiv["year"], hiv["total"], linewidth=3, label="Total", color="brown"
    )

    ax.fill_between(
        hiv["year"],
        hiv["total_lower"],
        hiv["total_upper"],
        alpha=0.3,
        color="brown",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Avertable HIV incidence (%)")
    ax.legend()
    ax.set_ylim(0)


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(
        os.path.join(
            output_dir, "hiv_attributable_to_gc_no_cipro_resistance.csv"
        ),
        schema_overrides={
            "hiv_averted": pl.Float64,
            "hiv_averted_lower": pl.Float64,
            "hiv_averted_upper": pl.Float64,
            "indirect_effect": pl.Float64,
            "indirect_effect_lower": pl.Float64,
            "indirect_effect_upper": pl.Float64,
            "total_averted": pl.Float64,
            "total_averted_lower": pl.Float64,
            "total_averted_upper": pl.Float64,
            "hiv_incidence_number": pl.Float64,
            "hiv_incidence_number_upper": pl.Float64,
            "hiv_incidence_number_lower": pl.Float64,
        },
    )

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
    plot_averted_hiv(hiv, ax[0, 1], fig)  # type: ignore

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

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_avertible_hiv.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_avertible_hiv.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
