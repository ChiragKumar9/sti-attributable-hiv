import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rc

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

font = {"family": "Nimbus Roman", "size": 28}
rc("font", **font)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 10))
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


def plot_drug_resistance(estimated_resistance_rates, ax, fig):
    # plot the estimated resistance rates averaged across all of africa
    estimated_resistance_rates = estimated_resistance_rates.group_by(
        [
            "year",
        ]
    ).agg(
        pl.mean("Ciprofloxacin"),
        pl.mean("Cefixime"),
        pl.mean("Azithromycin"),
        pl.mean("Ciprofloxacin_lower"),
        pl.mean("Cefixime_lower"),
        pl.mean("Azithromycin_lower"),
        pl.mean("Ciprofloxacin_upper"),
        pl.mean("Cefixime_upper"),
        pl.mean("Azithromycin_upper"),
    )

    estimated_resistance_rates = estimated_resistance_rates.sort("year")
    # plot each of the drugs and error bars
    ax.plot(
        estimated_resistance_rates["year"],
        estimated_resistance_rates["Ciprofloxacin"] * 100,
        label="Ciprofloxacin",
        linewidth=3,
        color="red",
    )
    ax.fill_between(
        estimated_resistance_rates["year"],
        estimated_resistance_rates["Ciprofloxacin_lower"] * 100,
        estimated_resistance_rates["Ciprofloxacin_upper"] * 100,
        color="red",
        alpha=0.3,
    )

    ax.plot(
        estimated_resistance_rates["year"],
        estimated_resistance_rates["Cefixime"] * 100,
        label="Cefixime",
        linewidth=3,
        color="blue",
    )
    ax.fill_between(
        estimated_resistance_rates["year"],
        estimated_resistance_rates["Cefixime_lower"] * 100,
        estimated_resistance_rates["Cefixime_upper"] * 100,
        color="blue",
        alpha=0.3,
    )

    ax.plot(
        estimated_resistance_rates["year"],
        estimated_resistance_rates["Azithromycin"] * 100,
        label="Azithromycin",
        linewidth=3,
        color="green",
    )

    ax.fill_between(
        estimated_resistance_rates["year"],
        estimated_resistance_rates["Azithromycin_lower"] * 100,
        estimated_resistance_rates["Azithromycin_upper"] * 100,
        color="green",
        alpha=0.3,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Gonorrhea resistance (%)")
    ax.legend()
    ax.set_xticks([2010, 2015])
    ax.set_ylim(0)


def plot_rr_estimates(rrs, aggregated_rrs, ax, fig):
    # error bars showing the RR estimates from each study by sex
    # combine the two dataframes into one
    aggregated_rrs = aggregated_rrs.rename(
        {"rr_mu": "val", "rr_lower": "lower", "rr_upper": "upper"}
    ).with_columns(paper=pl.lit("Aggregated"))

    rrs = pl.concat([rrs, aggregated_rrs], how="diagonal_relaxed")

    male_data = rrs.filter(pl.col("sex") == "Heterosexual Men")
    female_data = rrs.filter(pl.col("sex") == "Heterosexual Women")

    xs = np.arange(0, male_data.shape[0])

    offset = 0.1
    ax.errorbar(
        xs - offset,
        male_data["val"],
        yerr=[
            male_data["val"].to_numpy() - male_data["lower"].to_numpy(),
            male_data["upper"].to_numpy() - male_data["val"].to_numpy(),
        ],
        fmt="o",
        label="Male",
        color="midnightblue",
        elinewidth=1,
        capsize=3,
    )

    ax.errorbar(
        xs + offset,
        female_data["val"],
        yerr=[
            female_data["val"].to_numpy() - female_data["lower"].to_numpy(),
            female_data["upper"].to_numpy() - female_data["val"].to_numpy(),
        ],
        fmt="o",
        label="Female",
        color="magenta",
        elinewidth=1,
        capsize=3,
    )

    ax.legend(loc="upper left")
    ax.set_xlabel("Study")
    ax.set_xticks(
        xs, rrs["paper"].unique(maintain_order=True), rotation=-45, ha="left"
    )

    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_ylim(1)

    ax.set_ylabel("RR(HIV | Gonorrhea)")


def plot_hiv_burden(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year", "sex"])
        .agg(
            pl.sum("hiv_incidence_number"),
            pl.sum("hiv_incidence_number_upper"),
            pl.sum("hiv_incidence_number_lower"),
            pl.sum("population"),
            pl.sum("population_upper"),
            pl.sum("population_lower"),
        )
        .with_columns(
            hiv_incidence_rate=pl.col("hiv_incidence_number")
            / pl.col("population")
            * 100000,
            hiv_incidence_rate_upper=pl.col("hiv_incidence_number_upper")
            / pl.col("population_upper")
            * 100000,
            hiv_incidence_rate_lower=pl.col("hiv_incidence_number_lower")
            / pl.col("population_lower")
            * 100000,
        )
    )

    hiv = hiv.sort(by="year")

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

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence per 100,000\nin sub-Saharan Africa")
    ax.legend()
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_ylim(0)


if __name__ == "__main__":
    estimated_resistance_rates = pl.read_csv(
        os.path.join(output_dir, "estimated_resistance_rates.csv")
    )

    fig, ax = setup_plot(1, 3)  # type: ignore

    # estimated drug resistance rates
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
    plot_drug_resistance(estimated_resistance_rates, ax[0], fig)  # type: ignore

    # estimates of the RRs
    rrs = pl.read_csv(os.path.join(data_dir, "RRs_causal_GC_given_HIV.csv"))
    aggregated_rrs = pl.read_csv(
        os.path.join(output_dir, "meta_estimated_RRs_causal_GC_given_HIV.csv")
    )
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
    plot_rr_estimates(rrs, aggregated_rrs, ax[1], fig)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_gc.csv"))
    ax[2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_hiv_burden(hiv, ax[2], fig)  # type: ignore

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_overall_burden_gc_resistance_hiv.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_overall_burden_gc_resistance_hiv.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
