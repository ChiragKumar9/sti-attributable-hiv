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
        for spine in ax.spines.values():
            spine.set_position(("outward", 5))
        ax.set_axisbelow(True)
        return fig, ax
    for temp in ax.flatten():  # type: ignore
        temp.spines["top"].set_visible(False)
        temp.spines["right"].set_visible(False)
        temp.get_xaxis().tick_bottom()
        temp.get_yaxis().tick_left()
        temp.tick_params(axis="x", direction="out")
        temp.tick_params(axis="y", direction="out")
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        temp.set_axisbelow(True)
    return fig, ax


def plot_best_case(
    hiv, ax, year, fig, forward=False, reference=None, legend=False
):
    if forward:
        scalar = -1.0
        stub = "upper_bound_future"
        hiv = hiv.filter(pl.col("year") >= 2026)
    else:
        scalar = 1.0
        stub = "upper_bound"
        hiv = hiv.filter(pl.col("year") <= 2024)

    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum(f"hiv_averted_{stub}"),
            pl.sum(f"hiv_averted_{stub}_lower"),
            pl.sum(f"hiv_averted_{stub}_upper"),
            pl.sum("direct_hiv_averted_upper_bound"),
            pl.sum("direct_hiv_averted_upper_bound_lower"),
            pl.sum("direct_hiv_averted_upper_bound_upper"),
            pl.sum("unaids_incidence_number"),
            pl.sum("unaids_incidence_number_upper"),
            pl.sum("unaids_incidence_number_lower"),
        )
        .with_columns(
            direct=pl.col("direct_hiv_averted_upper_bound"),
            direct_lower=pl.col("direct_hiv_averted_upper_bound_lower"),
            direct_upper=pl.col("direct_hiv_averted_upper_bound_upper"),
            total=pl.col(f"hiv_averted_{stub}"),
            total_lower=pl.col(f"hiv_averted_{stub}_lower"),
            total_upper=pl.col(f"hiv_averted_{stub}_upper"),
            reference=pl.col("unaids_incidence_number"),
            reference_lower=pl.col("unaids_incidence_number_lower"),
            reference_upper=pl.col("unaids_incidence_number_upper"),
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
    print("HIV values in 2023")
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

    print("Total HIV averted")
    print(hiv["total"].sum() - hiv["reference"].sum())
    print(hiv["total_lower"].sum() - hiv["reference_lower"].sum())
    print(hiv["total_upper"].sum() - hiv["reference_upper"].sum())

    print("Total HIV averted as percentage of reference")
    print(
        (hiv["total"].sum() - hiv["reference"].sum()) / hiv["reference"].sum()
    )
    print(
        (hiv["total_lower"].sum() - hiv["reference_lower"].sum())
        / hiv["reference_lower"].sum()
    )
    print(
        (hiv["total_upper"].sum() - hiv["reference_upper"].sum())
        / hiv["reference_upper"].sum()
    )

    ax.plot(
        hiv["year"],
        hiv["direct"],
        linewidth=3,
        label="Direct",
        color="#275DAD",
    )

    ax.fill_between(
        hiv["year"],
        hiv["direct_lower"],
        hiv["direct_upper"],
        alpha=0.3,
        color="#275DAD",
    )

    ax.plot(
        hiv["year"], hiv["total"], linewidth=3, label="Total", color="darkred"
    )

    ax.fill_between(
        hiv["year"],
        hiv["total_lower"],
        hiv["total_upper"],
        alpha=0.3,
        color="darkred",
    )

    ax.plot(
        hiv["year"],
        hiv["reference"],
        linewidth=3,
        label="Historic" if not forward else "Projected",
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
    if legend:
        if forward:
            loc = (0.55, 0.75)
        else:
            loc = (0.6, 0.7)
        ax.legend(loc=loc, edgecolor="black")
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10000))

    if reference is not None:
        ax.axhline(
            0.1 * reference,  # type: ignore
            linestyle="--",
            color="#009CDE",
            label="SDG target",
        )
        ax.set_xticks([2026, 2030])
    else:
        ax.set_xticks([2000, 2010, 2020])


if __name__ == "__main__":
    regions = [
        ("Western", "a", "Western Africa"),
        ("Eastern", "b", "Eastern Africa"),
        ("Central", "c", "Central Africa"),
        ("Southern", "d", "Southern Africa"),
    ]

    for forward in [True, False]:
        year = 1991

        fig, ax = setup_plot(2, 2)  # type: ignore

        hiv = pl.read_csv(
            os.path.join(output_dir, "hiv_averted.csv"),
        )

        for i, (region, panel_label, region_label) in enumerate(regions):
            row, col = divmod(i, 2)

            # Get 2010 reference value for SDG target line (forward only)
            reference_2010 = None
            if forward:
                reference_2010 = hiv.filter(
                    (pl.col("region") == region) & (pl.col("year") == 2010)
                )["unaids_incidence_number"].sum()

            ax[row, col].text(  # type: ignore
                -0.25,
                1.08,
                panel_label,
                transform=ax[row, col].transAxes,  # type: ignore
                fontsize=24,
                fontweight="bold",
                va="top",
                ha="right",
            )

            plot_best_case(
                hiv.filter(pl.col("region") == region),
                ax[row, col],  # type: ignore
                year,
                fig,
                forward=forward,
                reference=reference_2010,
                legend=(i == 0),
            )

            # Override y-axis label to include region name
            ax[row, col].set_ylabel(f"HIV incidence in\n{region_label} (N)")  # type: ignore

        fig.tight_layout()

        stub = "future" if forward else "historical"

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
