import os

import matplotlib.pyplot as plt
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


def sum_preserve_null(column: str) -> pl.Expr:
    """Custom aggregation function to sum while preserving nulls."""
    return (
        pl.when(pl.col(column).is_null().all())
        .then(None)
        .otherwise(pl.sum(column))
        .alias(column)
    )


def plot_attributable_hiv_burden_drug_resistance(hiv, ax, label, fig):
    hiv = (
        hiv.group_by(["year"])
        .agg(
            sum_preserve_null("Ciprofloxacin_resistant_number"),
            sum_preserve_null("Ciprofloxacin_resistant_number_lower"),
            sum_preserve_null("Ciprofloxacin_resistant_number_upper"),
            sum_preserve_null("Cefixime_resistant_number"),
            sum_preserve_null("Cefixime_resistant_number_lower"),
            sum_preserve_null("Cefixime_resistant_number_upper"),
            sum_preserve_null("Azithromycin_resistant_number"),
            sum_preserve_null("Azithromycin_resistant_number_lower"),
            sum_preserve_null("Azithromycin_resistant_number_upper"),
            pl.sum("hiv_incidence_number_attributable"),
            pl.sum("hiv_incidence_number_attributable_upper"),
            pl.sum("hiv_incidence_number_attributable_lower"),
        )
        .with_columns(
            cipro=pl.col("Ciprofloxacin_resistant_number")
            / pl.col("hiv_incidence_number_attributable")
            * 100,
            cipro_lower=pl.col("Ciprofloxacin_resistant_number_lower")
            / pl.col("hiv_incidence_number_attributable_lower")
            * 100,
            cipro_upper=pl.col("Ciprofloxacin_resistant_number_upper")
            / pl.col("hiv_incidence_number_attributable_upper")
            * 100,
            cef=pl.col("Cefixime_resistant_number")
            / pl.col("hiv_incidence_number_attributable")
            * 100,
            cef_lower=pl.col("Cefixime_resistant_number_lower")
            / pl.col("hiv_incidence_number_attributable_lower")
            * 100,
            cef_upper=pl.col("Cefixime_resistant_number_upper")
            / pl.col("hiv_incidence_number_attributable_upper")
            * 100,
            azithro=pl.col("Azithromycin_resistant_number")
            / pl.col("hiv_incidence_number_attributable")
            * 100,
            azithro_lower=pl.col("Azithromycin_resistant_number_lower")
            / pl.col("hiv_incidence_number_attributable_lower")
            * 100,
            azithro_upper=pl.col("Azithromycin_resistant_number_upper")
            / pl.col("hiv_incidence_number_attributable_upper")
            * 100,
        )
    )
    # mask all values at 100
    hiv = hiv.with_columns(
        cipro=pl.when(pl.col("cipro") > 100)
        .then(100)
        .otherwise(pl.col("cipro")),
        cipro_lower=pl.when(pl.col("cipro_lower") > 100)
        .then(100)
        .otherwise(pl.col("cipro_lower")),
        cipro_upper=pl.when(pl.col("cipro_upper") > 100)
        .then(100)
        .otherwise(pl.col("cipro_upper")),
        cef=pl.when(pl.col("cef") > 100).then(100).otherwise(pl.col("cef")),
        cef_lower=pl.when(pl.col("cef_lower") > 100)
        .then(100)
        .otherwise(pl.col("cef_lower")),
        cef_upper=pl.when(pl.col("cef_upper") > 100)
        .then(100)
        .otherwise(pl.col("cef_upper")),
        azithro=pl.when(pl.col("azithro") > 100)
        .then(100)
        .otherwise(pl.col("azithro")),
        azithro_lower=pl.when(pl.col("azithro_lower") > 100)
        .then(100)
        .otherwise(pl.col("azithro_lower")),
        azithro_upper=pl.when(pl.col("azithro_upper") > 100)
        .then(100)
        .otherwise(pl.col("azithro_upper")),
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

    ax.set_xlabel("Year")
    ax.set_ylabel(
        f"Resistance among gonorrhea-attributable\nHIV ({label}) (%)"
    )
    ax.legend()
    ax.set_ylim(0)


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_gc.csv"))

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
            "figure_supplemental_across_africa_attributable_hiv_resistance.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_across_africa_attributable_hiv_resistance.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
