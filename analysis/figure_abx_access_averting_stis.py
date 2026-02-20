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
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 15))
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


def plot_best_case(hiv, ax, year, fig, forward=False):
    if forward:
        scalar = -1.0
        # this may change based on how the future sims are set up?
        stub = "upper_bound"
    else:
        scalar = 1.0
        stub = "upper_bound"

    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum(f"hiv_averted_{stub}"),
            pl.sum(f"hiv_averted_{stub}_lower"),
            pl.sum(f"hiv_averted_{stub}_upper"),
            pl.sum(f"direct_hiv_averted_{stub}"),
            pl.sum(f"direct_hiv_averted_{stub}_lower"),
            pl.sum(f"direct_hiv_averted_{stub}_upper"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            direct=pl.col(f"direct_hiv_averted_{stub}"),
            direct_lower=pl.col(f"direct_hiv_averted_{stub}_lower"),
            direct_upper=pl.col(f"direct_hiv_averted_{stub}_upper"),
            total=pl.col(f"hiv_averted_{stub}"),
            total_lower=pl.col(f"hiv_averted_{stub}_lower"),
            total_upper=pl.col(f"hiv_averted_{stub}_upper"),
            reference=pl.col("hiv_incidence_number"),
            reference_lower=pl.col("hiv_incidence_number_lower"),
            reference_upper=pl.col("hiv_incidence_number_upper"),
        )
    )
    if forward:
        hiv = hiv.with_columns(
            direct=pl.col("reference") + (pl.lit(scalar) * pl.col("direct")),
            direct_lower=pl.col("reference_lower")
            + (pl.lit(scalar) * pl.col("direct_upper")),
            direct_upper=pl.col("reference_upper")
            + (pl.lit(scalar) * pl.col("direct_lower")),
            total=pl.col("reference") + (pl.lit(scalar) * pl.col("total")),
            total_lower=pl.col("reference_lower")
            + (pl.lit(scalar) * pl.col("total_upper")),
            total_upper=pl.col("reference_upper")
            + (pl.lit(scalar) * pl.col("total_lower")),
        )
    else:
        hiv = hiv.with_columns(
            direct=pl.col("reference") + (pl.lit(scalar) * pl.col("direct")),
            direct_lower=pl.col("reference_lower")
            + (pl.lit(scalar) * pl.col("direct_lower")),
            direct_upper=pl.col("reference_upper")
            + (pl.lit(scalar) * pl.col("direct_upper")),
            total=pl.col("reference") + (pl.lit(scalar) * pl.col("total")),
            total_lower=pl.col("reference_lower")
            + (pl.lit(scalar) * pl.col("total_lower")),
            total_upper=pl.col("reference_upper")
            + (pl.lit(scalar) * pl.col("total_upper")),
        )

    # if forward, we need to plot the blue sdg target
    # to do this, we need to get the value in 2010
    if forward:
        reference_2010 = hiv.filter(pl.col("year") == 2010)["reference"][0]

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
        hiv["year"],
        hiv["direct"],
        linewidth=3,
        label="Direct",
        color="goldenrod",
    )

    ax.fill_between(
        hiv["year"],
        hiv["direct_lower"],
        hiv["direct_upper"],
        alpha=0.3,
        color="goldenrod",
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
        color="black",
    )

    ax.fill_between(
        hiv["year"],
        hiv["reference_lower"],
        hiv["reference_upper"],
        alpha=0.3,
        color="black",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence (N)")
    if forward:
        loc = (0.65, 0.65)
    else:
        loc = (0, 0)
    ax.legend(loc=loc)
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    if forward:
        ax.axhline(
            0.1 * reference_2010,  # type: ignore
            linestyle="--",
            color="#009CDE",
            label="SDG target",
        )
        ax.set_xticks([2025, 2030])
    else:
        ax.set_xticks([2005, 2010, 2015, 2020])
    # minor x ticks every year
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # need consistent y axis ticks
    if forward:
        ax.set_yticks([0, 500000, 1000000, 1500000, 2000000])
    else:
        ax.set_yticks([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000])
    # minor y ticks every 250000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(250000))


def plot_averted_sex(hiv, ax, year, fig, forward=False):
    if forward:
        scalar = -1.0
        stub = "upper_bound"
    else:
        scalar = 1.0
        stub = "upper_bound"

    hiv = (
        hiv.group_by(["year", "sex"])
        .agg(
            pl.sum(f"hiv_averted_{stub}"),
            pl.sum(f"hiv_averted_{stub}_lower"),
            pl.sum(f"hiv_averted_{stub}_upper"),
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_lower"),
            pl.sum("hiv_incidence_number_upper"),
        )
        .with_columns(
            total=pl.col(f"hiv_averted_{stub}") * pl.lit(scalar),
            total_lower=pl.col(f"hiv_averted_{stub}_lower") * pl.lit(scalar),
            total_upper=pl.col(f"hiv_averted_{stub}_upper") * pl.lit(scalar),
        )
    )

    hiv = hiv.filter(pl.col("year") >= year)

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv.filter(pl.col("sex") == "Male")["year"],
        hiv.filter(pl.col("sex") == "Male")["total"],
        linewidth=3,
        label="Heterosexual male",
        color="midnightblue",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Male")["year"],
        hiv.filter(pl.col("sex") == "Male")["total_lower"],
        hiv.filter(pl.col("sex") == "Male")["total_upper"],
        alpha=0.3,
        color="midnightblue",
    )

    ax.plot(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["total"],
        linewidth=3,
        label="Heterosexual female",
        color="magenta",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["total_lower"],
        hiv.filter(pl.col("sex") == "Female")["total_upper"],
        alpha=0.3,
        color="magenta",
    )

    ax.plot(
        hiv.filter(pl.col("sex") == "MSM")["year"],
        hiv.filter(pl.col("sex") == "MSM")["total"],
        linewidth=3,
        label="MSM",
        color="slategrey",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "MSM")["year"],
        hiv.filter(pl.col("sex") == "MSM")["total_lower"],
        hiv.filter(pl.col("sex") == "MSM")["total_upper"],
        alpha=0.3,
        color="slategrey",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Change in HIV incidence (N)")
    if forward:
        loc = (0.4, 0.7)
    else:
        loc = (0, 0)
    ax.legend(loc=loc)
    ax.set_yscale("symlog")
    # make the numbers appear in non scientific notation
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    if forward:
        ax.set_xticks([2025, 2030])
    else:
        ax.set_xticks([2005, 2010, 2015, 2020])
    # minor x ticks every year
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # set custom y axis minor ticks
    ax.yaxis.set_minor_locator(
        ticker.FixedLocator(
            scalar
            * np.array(
                [
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
                    300000,
                    400000,
                    500000,
                    600000,
                    700000,
                    800000,
                    900000,
                ]
            )  # type: ignore
        )
    )


def plot_averted_region(hiv, ax, year, fig, forward=False):
    if forward:
        scalar = -1.0
        stub = "upper_bound"
    else:
        scalar = 1.0
        stub = "upper_bound"

    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum(f"hiv_averted_{stub}"),
            pl.sum(f"hiv_averted_{stub}_lower"),
            pl.sum(f"hiv_averted_{stub}_upper"),
        )
        .with_columns(
            total=pl.col(f"hiv_averted_{stub}") * pl.lit(scalar),
            total_lower=pl.col(f"hiv_averted_{stub}_lower") * pl.lit(scalar),
            total_upper=pl.col(f"hiv_averted_{stub}_upper") * pl.lit(scalar),
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
    if forward:
        loc = (0.6, 0.6)
    else:
        loc = (0, 0)
    ax.legend(loc=loc)
    ax.set_yscale("symlog")
    # make the numbers appear in non scientific notation
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    if forward:
        ax.set_xticks([2025, 2030])
    else:
        ax.set_xticks([2005, 2010, 2015, 2020])
    # minor x ticks every year
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # set custom y axis minor ticks
    ax.yaxis.set_minor_locator(
        ticker.FixedLocator(
            scalar
            * np.array(
                [
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
                    300000,
                    400000,
                    500000,
                    600000,
                    700000,
                    800000,
                    900000,
                ]
            )  # type: ignore
        )
    )


if __name__ == "__main__":
    fig, ax = setup_plot(2, 3)  # type: ignore

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
        hiv,
        ax[0, 0],  # type: ignore
        2006,
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
    plot_averted_sex(
        hiv,
        ax[0, 1],  # type: ignore
        2006,
        fig,
    )

    ax[0, 2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[0, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_region(
        hiv,
        ax[0, 2],  # type: ignore
        2006,
        fig,
    )

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_best_case(
        hiv,
        ax[1, 0],  # type: ignore
        2020,
        fig,
        forward=True,
    )

    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "e",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_sex(
        hiv,
        ax[1, 1],  # type: ignore
        2020,
        fig,
        forward=True,
    )

    ax[1, 2].text(  # type: ignore
        -0.25,
        1.08,
        "f",
        transform=ax[1, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_region(
        hiv,
        ax[1, 2],  # type: ignore
        2020,
        fig,
        forward=True,
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_hiv_averted_via_antibiotic_access.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_hiv_averted_via_antibiotic_access.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
