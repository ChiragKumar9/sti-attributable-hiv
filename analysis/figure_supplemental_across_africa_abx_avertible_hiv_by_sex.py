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


sex_colors = {
    "Male": "midnightblue",
    "Female": "magenta",
    "MSM": "slategrey",
}

sex_labels = {
    "Male": "Heterosexual male",
    "Female": "Heterosexual female",
    "MSM": "MSM",
}


def plot_averted_by_sex(hiv, ax, region_label, legend=False):
    scalar = 1.0
    stub = "upper_bound_future"

    hiv = hiv.filter(pl.col("year") >= 2026)

    hiv = (
        hiv.group_by(["year", "sex"])
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

    hiv = hiv.sort(by="year")

    for sex, color in sex_colors.items():
        hiv_sex = hiv.filter(pl.col("sex") == sex)
        if hiv_sex.is_empty():
            continue

        ax.plot(
            hiv_sex["year"],
            hiv_sex["total"],
            linewidth=3,
            label=sex_labels[sex],
            color=color,
        )
        ax.fill_between(
            hiv_sex["year"],
            hiv_sex["total_lower"],
            hiv_sex["total_upper"],
            alpha=0.3,
            color=color,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(f"HIV averted in\n{region_label} (N)")
    ax.set_ylim(0)
    ax.set_xticks([2026, 2030])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10000))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    if legend:
        ax.legend(loc=(0.35, 0.75), edgecolor="black")


if __name__ == "__main__":
    regions = [
        ("Western", "a", "Western Africa"),
        ("Eastern", "b", "Eastern Africa"),
        ("Central", "c", "Central Africa"),
        ("Southern", "d", "Southern Africa"),
    ]

    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_averted.csv"))

    for i, (region, panel_label, region_label) in enumerate(regions):
        row, col = divmod(i, 2)

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

        plot_averted_by_sex(
            hiv.filter(pl.col("region") == region),
            ax[row, col],  # type: ignore
            region_label,
            legend=(i == 0),
        )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_across_africa_abx_avertible_hiv_by_sex.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_across_africa_abx_avertible_hiv_by_sex.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
