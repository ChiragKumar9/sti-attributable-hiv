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


def plot_gc_attributable_hiv(hiv, ax):
    # simple plot of indirect estimates of gonorrhea-attributable HIV over time

    hiv = hiv.group_by("year").agg(
        pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper_bound"),
        pl.sum(
            "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_lower"
        ),
        pl.sum(
            "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_upper"
        ),
        pl.sum("indirect_hiv_averted_gc_upper_bound"),
        pl.sum("indirect_hiv_averted_gc_upper_bound_lower"),
        pl.sum("indirect_hiv_averted_gc_upper_bound_upper"),
    )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv["year"],
        hiv["unaids_hiv_incidence_number_attributable_to_gc_upper_bound"],
        linewidth=3,
        label="Direct",
        color="k",
    )

    ax.fill_between(
        hiv["year"],
        hiv[
            "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_lower"
        ],
        hiv[
            "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_upper"
        ],
        alpha=0.3,
        color="k",
    )

    ax.plot(
        hiv["year"],
        hiv["unaids_hiv_incidence_number_attributable_to_gc_upper_bound"]
        + hiv["indirect_hiv_averted_gc_upper_bound"],
        linewidth=3,
        label="Total",
        color="brown",
    )

    ax.fill_between(
        hiv["year"],
        hiv["unaids_hiv_incidence_number_attributable_to_gc_upper_bound_lower"]
        + hiv["indirect_hiv_averted_gc_upper_bound_lower"],
        hiv["unaids_hiv_incidence_number_attributable_to_gc_upper_bound_upper"]
        + hiv["indirect_hiv_averted_gc_upper_bound_upper"],
        alpha=0.3,
        color="brown",
    )

    ax.axvline(2016, color="grey")

    ax.set_xlabel("Year")
    ax.set_xlim(2010, 2023)
    ax.legend()
    ax.set_xticks([2010, 2015, 2020])
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100000))
    ax.set_ylabel(
        "Change in HIV incidence attributable\nto gonorrhea infection (N)"
    )


def plot_gc_attributable_hiv_treatment(hiv, ax):
    # simple plot of indirect estimates of gonorrhea-attributable HIV over time

    hiv.filter(pl.col("year") >= 2017)
    hiv.filter(pl.col("year") <= 2023)

    hiv = hiv.group_by("year").agg(
        pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper_bound"),
        pl.sum(
            "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_lower"
        ),
        pl.sum(
            "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_upper"
        ),
        pl.sum("indirect_hiv_averted_gc_upper_bound"),
        pl.sum("indirect_hiv_averted_gc_upper_bound_lower"),
        pl.sum("indirect_hiv_averted_gc_upper_bound_upper"),
        pl.sum("indirect_hiv_averted_2016_gc_change"),
        pl.sum("indirect_hiv_averted_2016_gc_change_lower"),
        pl.sum("indirect_hiv_averted_2016_gc_change_upper"),
        pl.sum("direct_hiv_averted_2016_gc_change_lower"),
        pl.sum("direct_hiv_averted_2016_gc_change_upper"),
        pl.sum("direct_hiv_averted_2016_gc_change"),
    )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv["year"],
        hiv["unaids_hiv_incidence_number_attributable_to_gc_upper_bound"]
        + hiv["indirect_hiv_averted_gc_upper_bound"],
        linewidth=3,
        label="Antibiotic eventually treats",
        color="brown",
    )

    ax.fill_between(
        hiv["year"],
        hiv["unaids_hiv_incidence_number_attributable_to_gc_upper_bound_lower"]
        + hiv["indirect_hiv_averted_gc_upper_bound_lower"],
        hiv["unaids_hiv_incidence_number_attributable_to_gc_upper_bound_upper"]
        + hiv["indirect_hiv_averted_gc_upper_bound_upper"],
        alpha=0.3,
        color="brown",
    )

    ax.plot(
        hiv["year"],
        hiv["indirect_hiv_averted_2016_gc_change"]
        + hiv["direct_hiv_averted_2016_gc_change"],
        linewidth=3,
        label="First treatment works",
        color="green",
    )

    ax.fill_between(
        hiv["year"],
        hiv["indirect_hiv_averted_2016_gc_change_lower"]
        + hiv["direct_hiv_averted_2016_gc_change_lower"],
        hiv["indirect_hiv_averted_2016_gc_change_upper"]
        + hiv["direct_hiv_averted_2016_gc_change_upper"],
        alpha=0.3,
        color="green",
    )

    print(
        (
            hiv["indirect_hiv_averted_2016_gc_change"].sum()
            + hiv["direct_hiv_averted_2016_gc_change"].sum()
        )
        / (
            hiv[
                "unaids_hiv_incidence_number_attributable_to_gc_upper_bound"
            ].sum()
            + hiv["indirect_hiv_averted_gc_upper_bound"].sum()
        )
    )

    print(
        (
            hiv["indirect_hiv_averted_2016_gc_change_lower"].sum()
            + hiv["direct_hiv_averted_2016_gc_change_lower"].sum()
        )
        / (
            hiv[
                "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_lower"
            ].sum()
            + hiv["indirect_hiv_averted_gc_upper_bound_lower"].sum()
        )
    )

    print(
        (
            hiv["indirect_hiv_averted_2016_gc_change_upper"].sum()
            + hiv["direct_hiv_averted_2016_gc_change_upper"].sum()
        )
        / (
            hiv[
                "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_upper"
            ].sum()
            + hiv["indirect_hiv_averted_gc_upper_bound_upper"].sum()
        )
    )

    ax.set_xlabel("Year")
    ax.set_xlim(2017, 2023)
    ax.legend()
    ax.set_xticks([2018, 2020, 2022])
    ax.set_ylim(0, 500000)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100000))
    ax.set_ylabel(
        "Change in HIV incidence attributable\nto gonorrhea infection (N)"
    )


if __name__ == "__main__":
    fig, ax = setup_plot(1, 2)  # type: ignore

    year = 2016

    hiv = pl.read_csv(
        os.path.join(output_dir, "hiv_averted.csv"),
    )

    ax[0].text(  # type: ignore
        -0.25,
        1.08,
        "a",
        transform=ax[0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_gc_attributable_hiv(hiv, ax[0])  # type: ignore

    ax[1].text(  # type: ignore
        -0.25,
        1.08,
        "b",
        transform=ax[1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_gc_attributable_hiv_treatment(hiv, ax[1])  # type: ignore

    fig.tight_layout()

    # save

    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_comparing_access_and_resistance.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_comparing_access_and_resistance.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
