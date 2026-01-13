import os

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import rc

data_dir = "data"
output_dir = "outputs_unaids_sensitivity"
fig_dir = "figures-sensitivity"

font = {"family": "serif", "size": 28}
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


def plot_best_case(hiv, ax, year, label, fig, legend=False):
    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum("hiv_averted_upper_bound"),
            pl.sum("hiv_averted_upper_bound_lower"),
            pl.sum("hiv_averted_upper_bound_upper"),
            pl.sum("direct_hiv_averted_upper_bound"),
            pl.sum("direct_hiv_averted_upper_bound_lower"),
            pl.sum("direct_hiv_averted_upper_bound_upper"),
            pl.sum("population"),
            pl.sum("population_upper"),
            pl.sum("population_lower"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            direct=pl.col("direct_hiv_averted_upper_bound")
            / pl.col("population")
            * 100000,
            direct_lower=pl.col("direct_hiv_averted_upper_bound_lower")
            / pl.col("population_lower")
            * 100000,
            direct_upper=pl.col("direct_hiv_averted_upper_bound_upper")
            / pl.col("population_upper")
            * 100000,
            total=pl.col("hiv_averted_upper_bound")
            / pl.col("population")
            * 100000,
            total_lower=pl.col("hiv_averted_upper_bound_lower")
            / pl.col("population_lower")
            * 100000,
            total_upper=pl.col("hiv_averted_upper_bound_upper")
            / pl.col("population_upper")
            * 100000,
            reference=pl.col("hiv_incidence_number")
            / pl.col("population")
            * 100000,
            reference_lower=pl.col("hiv_incidence_number_lower")
            / pl.col("population_lower")
            * 100000,
            reference_upper=pl.col("hiv_incidence_number_upper")
            / pl.col("population_upper")
            * 100000,
        )
        .with_columns(
            direct=pl.col("reference") - pl.col("direct"),
            direct_lower=pl.col("reference_upper") - pl.col("direct_lower"),
            direct_upper=pl.col("reference_lower") - pl.col("direct_upper"),
            total=pl.col("reference") - pl.col("total"),
            total_lower=pl.col("reference_upper") - pl.col("total_lower"),
            total_upper=pl.col("reference_lower") - pl.col("total_upper"),
        )
    )

    hiv = hiv.filter(pl.col("year") >= year)

    hiv = hiv.sort(by="year")
    print(
        hiv.filter(pl.col("year") == pl.max("year")).select(
            [
                "total",
                "total_lower",
                "total_upper",
                "reference",
                "reference_lower",
                "reference_upper",
            ]
        )
    )

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

    ax.plot(
        hiv["year"],
        hiv["reference"],
        linewidth=3,
        label="Current",
        color="red",
    )

    ax.fill_between(
        hiv["year"],
        hiv["reference_lower"],
        hiv["reference_upper"],
        alpha=0.3,
        color="red",
    )

    ax.axhline(
        0.1 * hiv.filter(pl.col("year") == year)["reference"][0],
        linestyle="--",
        color="#009CDE",
        label="SDG target",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(f"HIV incidence per 100,000\nin {label}")
    ax.set_xticks([2010, 2015, 2020])
    if legend:
        ax.legend()
    ax.set_ylim(0)


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(
        os.path.join(output_dir, "hiv_averted.csv"),
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
    plot_best_case(
        hiv.filter(pl.col("region") == "Western"),
        ax[0, 0],  # type: ignore
        2010,
        "Western Africa",
        fig,
        legend=True,
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
    plot_best_case(
        hiv.filter(pl.col("region") == "Eastern"),
        ax[0, 1],  # type: ignore
        2010,
        "Eastern Africa",
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
    plot_best_case(
        hiv.filter(pl.col("region") == "Central"),
        ax[1, 0],  # type: ignore
        2010,
        "Central Africa",
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
    plot_best_case(
        hiv.filter(pl.col("region") == "Southern"),
        ax[1, 1],  # type: ignore
        2010,
        "Southern Africa",
        fig,
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_across_africa_reaching_sdg_target.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_across_africa_reaching_sdg_target.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
