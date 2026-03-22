import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def plot_best_case(hiv, ax, year, label, fig, legend=False, forward=False):
    # if forward, we need to plot the blue sdg target
    # to do this, we need to get the value in 2010
    if forward:
        reference_2010 = hiv.filter(pl.col("year") == 2010)[
            "unaids_hiv_incidence_number"
        ][0]

    if forward:
        scalar = -1.0
        # this may change based on how the future sims are set up?
        stub = "upper_bound_future"
        hiv = hiv.filter(pl.col("year") >= 2023)
    else:
        scalar = 1.0
        stub = "upper_bound"
        hiv = hiv.filter(pl.col("year") <= 2023)

    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum(f"hiv_averted_{stub}"),
            pl.sum(f"hiv_averted_{stub}_lower"),
            pl.sum(f"hiv_averted_{stub}_upper"),
            pl.sum("direct_hiv_averted_upper_bound"),
            pl.sum("direct_hiv_averted_upper_bound_lower"),
            pl.sum("direct_hiv_averted_upper_bound_upper"),
            pl.sum("unaids_hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
        )
        .with_columns(
            direct=pl.col("direct_hiv_averted_upper_bound"),
            direct_lower=pl.col("direct_hiv_averted_upper_bound_lower"),
            direct_upper=pl.col("direct_hiv_averted_upper_bound_upper"),
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

    hiv = hiv.filter(pl.col("year") >= year)

    hiv = hiv.sort(by="year")
    print(
        hiv.filter(pl.col("year") == 2023).select(
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

    if forward:
        ax.axhline(
            0.1 * reference_2010,  # type: ignore
            linestyle="--",
            color="#009CDE",
            label="SDG target",
        )
        ax.set_xticks([2025, 2030])
        # minor tick every 1 year
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    else:
        ax.set_xticks([1990, 2000, 2010, 2020])
        # minor ticks every two years
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

    ax.set_xlabel("Year")
    ax.set_ylabel(f"HIV incidence in\n{label} (N)")
    if legend:
        ax.legend()
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )


if __name__ == "__main__":
    for forward in [True, False]:
        if forward:
            # TODO: we will change this in the future
            year = 1991
        else:
            year = 1991

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
            year,
            "Western Africa",
            fig,
            legend=True,
            forward=forward,
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
            year,
            "Eastern Africa",
            fig,
            forward=forward,
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
            year,
            "Central Africa",
            fig,
            forward=forward,
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
            year,
            "Southern Africa",
            fig,
            forward=forward,
        )

        fig.tight_layout()

        # save
        if forward:
            stub = "future"
        else:
            stub = "historical"

        fig.savefig(
            os.path.join(
                fig_dir,
                f"figure_supplemental_across_africa_hiv_averted_abx_access_{stub}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(
                fig_dir,
                f"figure_supplemental_across_africa_hiv_averted_abx_access_{stub}.pdf",
            ),
            dpi=300,
            bbox_inches="tight",
        )
