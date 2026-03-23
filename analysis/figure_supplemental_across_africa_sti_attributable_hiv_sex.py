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


def plot_attributable_hiv_burden_sex(
    hiv, sti, ax, label, numbers, fig, legend=False
):
    hiv = hiv.group_by(["year", "sex"]).agg(
        pl.sum(f"hiv_incidence_number_attributable_to_{sti}"),
        pl.sum(f"hiv_incidence_number_attributable_to_{sti}_upper"),
        pl.sum(f"hiv_incidence_number_attributable_to_{sti}_lower"),
        pl.sum("population"),
        pl.sum("hiv_incidence_number"),
        pl.sum("hiv_incidence_number_upper"),
        pl.sum("hiv_incidence_number_lower"),
    )
    if numbers:
        hiv = hiv.with_columns(
            hiv_incidence_rate=pl.col(
                f"hiv_incidence_number_attributable_to_{sti}"
            ),
            hiv_incidence_rate_upper=pl.col(
                f"hiv_incidence_number_attributable_to_{sti}_upper"
            ),
            hiv_incidence_rate_lower=pl.col(
                f"hiv_incidence_number_attributable_to_{sti}_lower"
            ),
        )
    else:
        hiv = hiv.with_columns(
            hiv_incidence_rate=pl.col(
                f"hiv_incidence_number_attributable_to_{sti}"
            )
            / pl.col("hiv_incidence_number")
            * 100,
            hiv_incidence_rate_upper=pl.col(
                f"hiv_incidence_number_attributable_to_{sti}_upper"
            )
            / pl.col("hiv_incidence_number_upper")
            * 100,
            hiv_incidence_rate_lower=pl.col(
                f"hiv_incidence_number_attributable_to_{sti}_lower"
            )
            / pl.col("hiv_incidence_number_lower")
            * 100,
        )

    hiv = hiv.sort(by="year")
    sexes = hiv["sex"].unique().to_list()

    if "Male" in sexes:
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

    if "MSM" in sexes:
        ax.plot(
            hiv.filter(pl.col("sex") == "MSM")["year"],
            hiv.filter(pl.col("sex") == "MSM")["hiv_incidence_rate"],
            linewidth=3,
            label="MSM",
            color="slategrey",
        )

        ax.fill_between(
            hiv.filter(pl.col("sex") == "MSM")["year"],
            hiv.filter(pl.col("sex") == "MSM")["hiv_incidence_rate_lower"],
            hiv.filter(pl.col("sex") == "MSM")["hiv_incidence_rate_upper"],
            alpha=0.3,
            color="slategrey",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    if legend:
        ax.legend(edgecolor="black")
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_ylim(0)
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

    for numbers in [False, True]:
        units = "N" if numbers else "%"
        for sti in sti_map.keys():
            fig, ax = setup_plot(2, 2)  # type: ignore

            hiv = pl.read_csv(
                os.path.join(output_dir, "hiv_attributable_to_stis.csv")
            )
            hiv = hiv.filter(pl.col("year") <= 2023)

            if sti == "trichomoniasis":
                # remove men with trich
                hiv = hiv.filter(pl.col("sex") != "Male")
                hiv = hiv.filter(pl.col("sex") != "MSM")
            base_label = f"HIV incidence attributable to\n{sti_map[sti]}"

            # western africa
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
            plot_attributable_hiv_burden_sex(
                hiv.filter(pl.col("region") == "Western"),
                sti,
                ax[0, 0],  # type: ignore
                base_label + f"in Western Africa ({units})",
                numbers,
                fig,
                legend=True,
            )

            # eastern africa
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
            plot_attributable_hiv_burden_sex(
                hiv.filter(pl.col("region") == "Eastern"),
                sti,
                ax[0, 1],  # type: ignore
                base_label + f"in Eastern Africa ({units})",
                numbers,
                fig,
            )

            # central africa
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
            plot_attributable_hiv_burden_sex(
                hiv.filter(pl.col("region") == "Central"),
                sti,
                ax[1, 0],  # type: ignore
                base_label + f"in Central Africa ({units})",
                numbers,
                fig,
            )

            # southern africa
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
            plot_attributable_hiv_burden_sex(
                hiv.filter(pl.col("region") == "Southern"),
                sti,
                ax[1, 1],  # type: ignore
                base_label + f"in Southern Africa ({units})",
                numbers,
                fig,
            )

            fig.tight_layout()

            # save
            name = "numbers" if numbers else "percentages"
            fig.savefig(
                os.path.join(
                    fig_dir,
                    f"figure_supplemental_{sti}_attributable_hiv_region_sex_{name}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(
                    fig_dir,
                    f"figure_supplemental_{sti}_attributable_hiv_region_sex_{name}.pdf",
                ),
                dpi=300,
                bbox_inches="tight",
            )
