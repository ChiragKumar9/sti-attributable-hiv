import os

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import rc
from matplotlib.ticker import FuncFormatter

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


def sum_preserve_null(column: str) -> pl.Expr:
    """Custom aggregation function to sum while preserving nulls."""
    return (
        pl.when(pl.col(column).is_null().all())
        .then(None)
        .otherwise(pl.sum(column))
        .alias(column)
    )


def plot_attributable_hiv_burden_drug_resistance(
    hiv, ax, label, fig, legend=False
):
    hiv = hiv.group_by(["year"]).agg(
        sum_preserve_null("Ciprofloxacin_resistant_number"),
        sum_preserve_null("Ciprofloxacin_resistant_number_lower"),
        sum_preserve_null("Ciprofloxacin_resistant_number_upper"),
        sum_preserve_null("Cefixime_resistant_number"),
        sum_preserve_null("Cefixime_resistant_number_lower"),
        sum_preserve_null("Cefixime_resistant_number_upper"),
        sum_preserve_null("Azithromycin_resistant_number"),
        sum_preserve_null("Azithromycin_resistant_number_lower"),
        sum_preserve_null("Azithromycin_resistant_number_upper"),
        sum_preserve_null("Ceftriaxone_resistant_number"),
        sum_preserve_null("Ceftriaxone_resistant_number_lower"),
        sum_preserve_null("Ceftriaxone_resistant_number_upper"),
        pl.sum("hiv_incidence_number_attributable_to_gc"),
        pl.sum("hiv_incidence_number_attributable_to_gc_upper"),
        pl.sum("hiv_incidence_number_attributable_to_gc_lower"),
    )
    # mask all values at the attributable incidence
    hiv = hiv.with_columns(
        cipro=pl.when(
            pl.col("Ciprofloxacin_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number")),
        cipro_lower=pl.when(
            pl.col("Ciprofloxacin_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number_lower")),
        cipro_upper=pl.when(
            pl.col("Ciprofloxacin_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Ciprofloxacin_resistant_number_upper")),
        cef=pl.when(
            pl.col("Cefixime_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Cefixime_resistant_number")),
        cef_lower=pl.when(
            pl.col("Cefixime_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Cefixime_resistant_number_lower")),
        cef_upper=pl.when(
            pl.col("Cefixime_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Cefixime_resistant_number_upper")),
        azithro=pl.when(
            pl.col("Azithromycin_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Azithromycin_resistant_number")),
        azithro_lower=pl.when(
            pl.col("Azithromycin_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Azithromycin_resistant_number_lower")),
        azithro_upper=pl.when(
            pl.col("Azithromycin_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Azithromycin_resistant_number_upper")),
        ceft=pl.when(
            pl.col("Ceftriaxone_resistant_number")
            > pl.col("hiv_incidence_number_attributable_to_gc")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc"))
        .otherwise(pl.col("Ceftriaxone_resistant_number")),
        ceft_lower=pl.when(
            pl.col("Ceftriaxone_resistant_number_lower")
            > pl.col("hiv_incidence_number_attributable_to_gc_lower")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_lower"))
        .otherwise(pl.col("Ceftriaxone_resistant_number_lower")),
        ceft_upper=pl.when(
            pl.col("Ceftriaxone_resistant_number_upper")
            > pl.col("hiv_incidence_number_attributable_to_gc_upper")
        )
        .then(pl.col("hiv_incidence_number_attributable_to_gc_upper"))
        .otherwise(pl.col("Ceftriaxone_resistant_number_upper")),
    )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv["year"],
        hiv["cipro"],
        linewidth=3,
        label="Ciprofloxacin",
        color="red",
    )

    ax.fill_between(
        hiv["year"],
        hiv["cipro_lower"],
        hiv["cipro_upper"],
        alpha=0.3,
        color="red",
    )

    ax.plot(
        hiv["year"], hiv["cef"], linewidth=3, label="Cefixime", color="blue"
    )

    ax.fill_between(
        hiv["year"],
        hiv["cef_lower"],
        hiv["cef_upper"],
        alpha=0.3,
        color="blue",
    )

    ax.plot(
        hiv["year"],
        hiv["azithro"],
        linewidth=3,
        label="Azithromycin",
        color="green",
    )

    ax.fill_between(
        hiv["year"],
        hiv["azithro_lower"],
        hiv["azithro_upper"],
        alpha=0.3,
        color="green",
    )

    ax.plot(
        hiv["year"],
        hiv["ceft"],
        linewidth=3,
        label="Ceftriaxone",
        color="purple",
    )

    ax.fill_between(
        hiv["year"],
        hiv["ceft_lower"],
        hiv["ceft_upper"],
        alpha=0.3,
        color="purple",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(f"HIV incidence\nin {label} (N)")
    if legend:
        ax.legend()
    ax.set_yscale("log")
    # make sure y axis ticks are not in scientific notation
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda y, _: "{:g}".format(y))
    )


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))

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
    plot_attributable_hiv_burden_drug_resistance(
        hiv.filter(pl.col("region") == "Western"),
        ax[0, 0],  # type: ignore
        "Western Africa",
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
    plot_attributable_hiv_burden_drug_resistance(
        hiv.filter(pl.col("region") == "Eastern"),
        ax[0, 1],  # type: ignore
        "Eastern Africa",
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
    plot_attributable_hiv_burden_drug_resistance(
        hiv.filter(pl.col("region") == "Central"),
        ax[1, 0],  # type: ignore
        "Central Africa",
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
    plot_attributable_hiv_burden_drug_resistance(
        hiv.filter(pl.col("region") == "Southern"),
        ax[1, 1],  # type: ignore
        "Southern Africa",
        fig,
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_abx_resistance_gc_attributable_hiv.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_abx_resistance_gc_attributable_hiv.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
