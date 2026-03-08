import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
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


def plot_hiv_burden_by_sex(hiv, ax, fig):
    # Aggregate UNAIDS incidence; combine Male + MSM into a single Male total
    hiv_agg = (
        hiv.group_by(["year", "sex"])
        .agg(
            pl.sum("unaids_incidence_number").alias("incidence"),
            pl.sum("unaids_incidence_number_upper").alias("incidence_upper"),
            pl.sum("unaids_incidence_number_lower").alias("incidence_lower"),
        )
        .sort("year")
    )

    male_hist = (
        hiv_agg.filter(pl.col("sex").is_in(["Male", "MSM"]))
        .group_by("year")
        .agg(
            pl.sum("incidence"),
            pl.sum("incidence_upper"),
            pl.sum("incidence_lower"),
        )
        .sort("year")
    )
    female_hist = hiv_agg.filter(pl.col("sex") == "Female").sort("year")

    ax.plot(
        male_hist["year"],
        male_hist["incidence"],
        linewidth=3,
        label="Male",
        color="midnightblue",
    )
    ax.fill_between(
        male_hist["year"],
        male_hist["incidence_lower"],
        male_hist["incidence_upper"],
        alpha=0.3,
        color="midnightblue",
    )

    ax.plot(
        female_hist["year"],
        female_hist["incidence"],
        linewidth=3,
        label="Female",
        color="magenta",
    )
    ax.fill_between(
        female_hist["year"],
        female_hist["incidence_lower"],
        female_hist["incidence_upper"],
        alpha=0.3,
        color="magenta",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence in\nsub-Saharan Africa (N)")
    ax.legend(loc=(0.5, 0.73))
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_yticks([0, 500000, 1000000, 1500000])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    # y axis minor ticks every 100,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100000))
    # x axis ticks every 2 years
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_hiv_burden_by_region(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum("unaids_incidence_number"),
            pl.sum("unaids_incidence_number_upper"),
            pl.sum("unaids_incidence_number_lower"),
        )
        .rename(
            {
                "unaids_incidence_number": "incidence",
                "unaids_incidence_number_upper": "incidence_upper",
                "unaids_incidence_number_lower": "incidence_lower",
            }
        )
    )
    hiv = hiv.sort(by="year")

    region_colors = {
        "Western": "darkorange",
        "Eastern": "olivedrab",
        "Central": "teal",
        "Southern": "maroon",
    }

    for region, color in region_colors.items():
        hist = hiv.filter(pl.col("region") == region)
        ax.plot(
            hist["year"],
            hist["incidence"],
            linewidth=3,
            label=region,
            color=color,
        )
        ax.fill_between(
            hist["year"],
            hist["incidence_lower"],
            hist["incidence_upper"],
            alpha=0.3,
            color=color,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence (N)")
    ax.legend(loc=(0.5, 0.68))
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_yticks([0, 250000, 500000, 750000, 1000000])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    ax.set_ylim(0)
    # minor y ticks every 50,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50000))
    # x axis ticks every 2 years
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_rr_estimates(aggregated_rrs, ax, fig):
    # error bars showing the RR estimates from each study by sex
    # combine the two dataframes into one
    aggregated_rrs = aggregated_rrs.rename(
        {"rr_mu": "val", "rr_lower": "lower", "rr_upper": "upper"}
    )

    female_data = aggregated_rrs.filter(pl.col("sex") == "Heterosexual Women")
    heterosexual_male_data = aggregated_rrs.filter(
        pl.col("sex") == "Heterosexual Men"
    )
    msm_data = aggregated_rrs.filter(pl.col("sex") == "MSM")
    # we have to get all the RRs in the same order
    female_data = female_data.sort("bacteria")
    heterosexual_male_data = heterosexual_male_data.sort("bacteria")
    msm_data = msm_data.sort("bacteria")
    # for heterosexual male and msm, filter out trichomoniasis since the RRs
    # are inferred/1.0
    heterosexual_male_data = heterosexual_male_data.filter(
        pl.col("bacteria") != "Trichomoniasis"
    )
    msm_data = msm_data.filter(pl.col("bacteria") != "Trichomoniasis")

    male_xs = np.arange(0, heterosexual_male_data.shape[0])
    female_xs = np.arange(0, female_data.shape[0])

    offset = 0.2
    ax.errorbar(
        male_xs - offset,
        heterosexual_male_data["val"],
        yerr=[
            heterosexual_male_data["val"].to_numpy()
            - heterosexual_male_data["lower"].to_numpy(),
            heterosexual_male_data["upper"].to_numpy()
            - heterosexual_male_data["val"].to_numpy(),
        ],
        fmt="o",
        label="Heterosexual male",
        color="midnightblue",
        elinewidth=1,
        capsize=10,
        capthick=2,
    )

    ax.errorbar(
        female_xs,
        female_data["val"],
        yerr=[
            female_data["val"].to_numpy() - female_data["lower"].to_numpy(),
            female_data["upper"].to_numpy() - female_data["val"].to_numpy(),
        ],
        fmt="o",
        label="Heterosexual female",
        color="magenta",
        elinewidth=1,
        capsize=10,
        capthick=2,
    )

    ax.errorbar(
        male_xs + offset,
        msm_data["val"],
        yerr=[
            msm_data["val"].to_numpy() - msm_data["lower"].to_numpy(),
            msm_data["upper"].to_numpy() - msm_data["val"].to_numpy(),
        ],
        fmt="o",
        label="MSM",
        color="slategrey",
        elinewidth=1,
        capsize=10,
        capthick=2,
    )

    ax.set_ylim(0.1)
    ax.legend(loc=(0.35, 0))
    ax.set_xlabel("STI")
    ax.set_xticks(
        female_xs,
        female_data["bacteria"].unique(maintain_order=True),
        rotation=-45,
        ha="left",
    )

    ax.set_yscale("log")
    ax.set_yticks([1, 10])
    ax.set_yticklabels(["1", "10"])
    ax.axhline(1, color="grey", linestyle="--")
    ax.set_xlabel("STI")
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
    # y axis ticks every 10%
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))


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
