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
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 7.5))
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


def plot_best_case_sex(hiv, ax, fig, sex_label, reference, legend=False):
    scalar = -1.0
    stub = "upper_bound_future"

    hiv = hiv.filter(pl.col("year") >= 2026)

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
        .with_columns(
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
        .with_columns(
            # lower bounds cannot hit 0, so put them at ten or something nominal
            direct_lower=pl.when(pl.col("direct_lower") <= 10)
            .then(pl.lit(5000))
            .otherwise(pl.col("direct_lower")),
            total_lower=pl.when(pl.col("total_lower") <= 10)
            .then(pl.lit(5000))
            .otherwise(pl.col("total_lower")),
        )
    )

    hiv = hiv.sort(by="year")

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
        hiv["year"],
        hiv["total"],
        linewidth=3,
        label="Total",
        color="darkred",
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
        label="Projected",
        color="black",
    )
    ax.fill_between(
        hiv["year"],
        hiv["reference_lower"],
        hiv["reference_upper"],
        alpha=0.3,
        color="black",
    )

    ax.axhline(
        0.1 * reference,
        linestyle="--",
        color="#009CDE",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(f"HIV incidence in {sex_label} (N)")
    ax.set_ylim(0)
    ax.set_xticks([2026, 2030])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10000))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    if legend:
        ax.legend(loc=(0.55, 0.75), edgecolor="black")


if __name__ == "__main__":
    sexes = [
        (["Male", "MSM"], "a", "males"),
        (["Female"], "b", "females"),
    ]

    fig, ax = setup_plot(1, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_averted.csv"))

    for i, (sex_groups, panel_label, sex_label) in enumerate(sexes):
        hiv_sex = hiv.filter(pl.col("sex").is_in(sex_groups))

        reference_2010 = hiv_sex.filter(pl.col("year") == 2010)[
            "unaids_incidence_number"
        ].sum()

        ax[i].text(  # type: ignore
            -0.25,
            1.08,
            panel_label,
            transform=ax[i].transAxes,  # type: ignore
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="right",
        )

        plot_best_case_sex(
            hiv_sex,
            ax[i],  # type: ignore
            fig,
            sex_label,
            reference_2010,
            legend=(i == 0),
        )

    fig.tight_layout()

    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_sdg_target_by_sex.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_supplemental_sdg_target_by_sex.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
