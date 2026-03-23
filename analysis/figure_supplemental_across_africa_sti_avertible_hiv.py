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


def plot_avertible_hiv_burden_sex(
    hiv, sti, ax, label, fig, forward, legend=False
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
        hiv.group_by(["year", "sex"])
        .agg(
            pl.sum(f"hiv_averted_{sti}_{stub}"),
            pl.sum(f"hiv_averted_{sti}_{stub}_lower"),
            pl.sum(f"hiv_averted_{sti}_{stub}_upper"),
        )
        .with_columns(
            total=pl.col(f"hiv_averted_{sti}_{stub}") * pl.lit(scalar),
            total_lower=pl.col(f"hiv_averted_{sti}_{stub}_lower")
            * pl.lit(scalar),
            total_upper=pl.col(f"hiv_averted_{sti}_{stub}_upper")
            * pl.lit(scalar),
        )
    )

    hiv = hiv.sort(by="year")
    sexes = hiv["sex"].unique().to_list()

    if "Male" in sexes:
        ax.plot(
            hiv.filter(pl.col("sex") == "Male")["year"],
            hiv.filter(pl.col("sex") == "Male")["total"],
            linewidth=3,
            label="Male",
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
        label="Female",
        color="magenta",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["total_lower"],
        hiv.filter(pl.col("sex") == "Female")["total_upper"],
        alpha=0.3,
        color="magenta",
    )

    if "MSM" in sexes:
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
    ax.set_ylabel(label)
    if legend:
        ax.legend(edgecolor="black")
    if forward:
        ax.set_xticks([2026, 2030])
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.set_ylim(top=0)
    else:
        ax.set_xticks([1990, 2000, 2010, 2020])
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.set_ylim(0)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10000))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )


if __name__ == "__main__":
    sti_map = {
        "gc": "gonorrhea",
        "chlamydia": "chlamydia",
        "syphilis": "syphilis",
        "trichomoniasis": "trichomoniasis",
    }
    regions = [
        ("Western", "a", "Western Africa"),
        ("Eastern", "b", "Eastern Africa"),
        ("Central", "c", "Central Africa"),
        ("Southern", "d", "Southern Africa"),
    ]
    units = "N"

    for forward in [True, False]:
        name = "future" if forward else "historical"

        hiv = pl.read_csv(os.path.join(output_dir, "hiv_averted.csv"))

        for sti, sti_label in sti_map.items():
            fig, ax = setup_plot(2, 2)  # type: ignore

            # For trichomoniasis, exclude male sex groups
            hiv_sti = (
                hiv.filter(pl.col("sex") != "Male").filter(
                    pl.col("sex") != "MSM"
                )
                if sti == "trichomoniasis"
                else hiv
            )

            base_label = f"Change in HIV incidence from\ntreating {sti_label}"

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
                plot_avertible_hiv_burden_sex(
                    hiv_sti.filter(pl.col("region") == region),
                    sti,
                    ax[row, col],  # type: ignore
                    base_label + f" in {region_label} ({units})",
                    fig,
                    forward,
                    legend=(i == 0),
                )

            fig.tight_layout()

            fig.savefig(
                os.path.join(
                    fig_dir,
                    f"figure_supplemental_{sti}_avertible_hiv_region_sex_{name}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(
                    fig_dir,
                    f"figure_supplemental_{sti}_avertible_hiv_region_sex_{name}.pdf",
                ),
                dpi=300,
                bbox_inches="tight",
            )
