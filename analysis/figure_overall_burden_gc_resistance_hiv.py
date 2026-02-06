import os

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib import rc

data_dir = "data"
output_dir = "outputs"
fig_dir = "figures"

with open("params.yml") as f:
    params = yaml.safe_load(f)

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


def plot_hiv_burden_by_sex(hiv, ax, fig):
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


def plot_hiv_burden_by_region(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year", "region"])
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
        hiv.filter(pl.col("region") == "Western")["year"],
        hiv.filter(pl.col("region") == "Western")["hiv_incidence_rate"],
        linewidth=3,
        label="Western",
        color="darkorange",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Western")["year"],
        hiv.filter(pl.col("region") == "Western")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Western")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="darkorange",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Eastern")["year"],
        hiv.filter(pl.col("region") == "Eastern")["hiv_incidence_rate"],
        linewidth=3,
        label="Eastern",
        color="olivedrab",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Eastern")["year"],
        hiv.filter(pl.col("region") == "Eastern")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Eastern")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="olivedrab",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["hiv_incidence_rate"],
        linewidth=3,
        label="Central",
        color="royalblue",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Central")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="royalblue",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate"],
        linewidth=3,
        label="Southern",
        color="teal",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="teal",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence per 100,000")
    ax.set_ylim(10)
    ax.legend(loc=(0, 0))
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_yscale("log")
    ax.set_yticks([100, 1000])
    ax.set_yticklabels(["100", "1,000"])


def plot_rr_estimates(aggregated_rrs, ax, fig):
    # error bars showing the RR estimates from each study by sex
    # combine the two dataframes into one
    aggregated_rrs = aggregated_rrs.rename(
        {"rr_mu": "val", "rr_lower": "lower", "rr_upper": "upper"}
    )

    female_data = aggregated_rrs.filter(pl.col("sex") == "Heterosexual Women")
    male_data = aggregated_rrs.filter(pl.col("sex") != "Heterosexual Women")
    # for the male data, we have some estimates for MSM and some for heterosexual men
    msm_data = male_data.filter(pl.col("sex") == "MSM")
    heterosexual_male_data = male_data.filter(
        pl.col("sex") == "Heterosexual Men"
    )
    # join the two
    male_data = heterosexual_male_data.join(
        msm_data, on=["bacteria"], how="left", suffix="_msm"
    )
    # take the weighted average of the two estimates for each bacteria, using the msm_fraction parameter
    male_data = male_data.with_columns(
        val=pl.col("val") * (1 - params["msm_fraction"])
        + pl.col("val_msm") * params["msm_fraction"],
        lower=pl.col("lower") * (1 - params["msm_fraction"])
        + pl.col("lower_msm") * params["msm_fraction"],
        upper=pl.col("upper") * (1 - params["msm_fraction"])
        + pl.col("upper_msm") * params["msm_fraction"],
    )

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
        capsize=10,
        capthick=2,
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
        capsize=10,
        capthick=2,
    )

    ax.legend(loc="upper right")
    ax.set_xlabel("STI")
    ax.set_xticks(
        xs,
        aggregated_rrs["bacteria"].unique(maintain_order=True),
        rotation=-45,
        ha="left",
    )

    ax.set_yticks([1, 3, 5, 7, 9, 11])
    ax.axhline(1, color="grey", linestyle="--")

    ax.set_ylabel("Causal RR(HIV acquisition | STI infection)")


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
        pl.mean("Ceftriaxone"),
        pl.mean("Ciprofloxacin_lower"),
        pl.mean("Cefixime_lower"),
        pl.mean("Azithromycin_lower"),
        pl.mean("Ceftriaxone_lower"),
        pl.mean("Ciprofloxacin_upper"),
        pl.mean("Cefixime_upper"),
        pl.mean("Azithromycin_upper"),
        pl.mean("Ceftriaxone_upper"),
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

    ax.plot(
        estimated_resistance_rates["year"],
        estimated_resistance_rates["Ceftriaxone"] * 100,
        label="Ceftriaxone",
        linewidth=3,
        color="purple",
    )

    ax.fill_between(
        estimated_resistance_rates["year"],
        estimated_resistance_rates["Ceftriaxone_lower"] * 100,
        estimated_resistance_rates["Ceftriaxone_upper"] * 100,
        color="purple",
        alpha=0.3,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Gonorrhea resistance (%)")
    ax.legend()
    ax.set_xticks([2010, 2015, 2020])
    ax.set_ylim(0)


if __name__ == "__main__":
    fig, ax = setup_plot(2, 2)  # type: ignore

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_sti.csv"))
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
    plot_hiv_burden_by_sex(hiv, ax[0, 0], fig)  # type: ignore

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
    plot_hiv_burden_by_region(hiv, ax[0, 1], fig)  # type: ignore

    # estimates of the RRs
    aggregated_rrs = pl.read_csv(
        os.path.join(output_dir, "meta_estimated_RRs_causal_STI_given_HIV.csv")
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
    plot_rr_estimates(aggregated_rrs, ax[1, 0], fig)  # type: ignore

    estimated_resistance_rates = pl.read_csv(
        os.path.join(output_dir, "estimated_resistance_rates.csv")
    )

    # estimated drug resistance rates
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
    plot_drug_resistance(estimated_resistance_rates, ax[1, 1], fig)  # type: ignore

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
