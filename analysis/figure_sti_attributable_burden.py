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

# make a figures dir if it doesn't exist
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


def setup_plot(nrows, ncols):
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 54))
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


def plot_attributable_hiv_burden_source(hiv, ax, fig):
    hiv_gbd_base = hiv.filter(
        pl.col("cols_unaids_analysis")
    )  # restrict GBD data to UNAIDS-analysis countries for comparability
    gbd_locs = set(
        hiv_gbd_base["location"].unique().to_list()
    )  # save this here to check later with unaids countries we use

    hiv_gbd = (
        hiv_gbd_base.group_by(["year"])
        .agg(
            pl.sum("hiv_incidence_number_attributable_to_gc"),
            pl.sum("hiv_incidence_number_attributable_to_gc_upper"),
            pl.sum("hiv_incidence_number_attributable_to_gc_lower"),
            pl.sum("hiv_incidence_number_attributable_to_syphilis"),
            pl.sum("hiv_incidence_number_attributable_to_syphilis_upper"),
            pl.sum("hiv_incidence_number_attributable_to_syphilis_lower"),
            pl.sum("hiv_incidence_number_attributable_to_trichomoniasis"),
            pl.sum(
                "hiv_incidence_number_attributable_to_trichomoniasis_upper"
            ),
            pl.sum(
                "hiv_incidence_number_attributable_to_trichomoniasis_lower"
            ),
            pl.sum("hiv_incidence_number_attributable_to_chlamydia"),
            pl.sum("hiv_incidence_number_attributable_to_chlamydia_upper"),
            pl.sum("hiv_incidence_number_attributable_to_chlamydia_lower"),
        )
        .with_columns(
            gbd=(
                pl.col("hiv_incidence_number_attributable_to_gc")
                + pl.col("hiv_incidence_number_attributable_to_syphilis")
                + pl.col("hiv_incidence_number_attributable_to_trichomoniasis")
                + pl.col("hiv_incidence_number_attributable_to_chlamydia")
            ),
            gbd_lower=(
                pl.col("hiv_incidence_number_attributable_to_gc_lower")
                + pl.col("hiv_incidence_number_attributable_to_syphilis_lower")
                + pl.col(
                    "hiv_incidence_number_attributable_to_trichomoniasis_lower"
                )
                + pl.col(
                    "hiv_incidence_number_attributable_to_chlamydia_lower"
                )
            ),
            gbd_upper=(
                pl.col("hiv_incidence_number_attributable_to_gc_upper")
                + pl.col("hiv_incidence_number_attributable_to_syphilis_upper")
                + pl.col(
                    "hiv_incidence_number_attributable_to_trichomoniasis_upper"
                )
                + pl.col(
                    "hiv_incidence_number_attributable_to_chlamydia_upper"
                )
            ),
        )
        .sort(by="year")
    )

    # UNAIDS sensitivity: restricted to countries with UNAIDS data
    hiv_unaids_base = hiv.filter(pl.col("cols_unaids_analysis"))
    unaids_locs = set(hiv_unaids_base["location"].unique().to_list())
    assert gbd_locs == unaids_locs, (
        f"Location mismatch — only in GBD: {gbd_locs - unaids_locs}, only in UNAIDS: {unaids_locs - gbd_locs}"
    )

    hiv_unaids = (
        hiv_unaids_base.group_by(["year"])
        .agg(
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc_lower"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis"),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
            ),
            pl.sum("unaids_hiv_incidence_number_attributable_to_chlamydia"),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
            ),
        )
        .with_columns(
            unaids=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia"
                )
            ),
            unaids_lower=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc_lower")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
                )
            ),
            unaids_upper=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc_upper")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
                )
            ),
        )
        .sort(by="year")
    )

    ax.plot(
        hiv_gbd["year"],
        hiv_gbd["gbd"],
        linewidth=3,
        label="GBD",
        color="darkolivegreen",
    )
    ax.fill_between(
        hiv_gbd["year"],
        hiv_gbd["gbd_lower"],
        hiv_gbd["gbd_upper"],
        alpha=0.3,
        color="darkolivegreen",
    )

    ax.plot(
        hiv_unaids["year"],
        hiv_unaids["unaids"],
        linewidth=3,
        label="UNAIDS",
        color="#009EDB",
    )
    ax.fill_between(
        hiv_unaids["year"],
        hiv_unaids["unaids_lower"],
        hiv_unaids["unaids_upper"],
        alpha=0.3,
        color="#009EDB",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Attributable HIV incidence (N)")
    ax.legend(edgecolor="black")
    ax.set_ylim(0)
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_yticks([0, 200000, 400000, 600000, 800000])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    ## add minor ticks on the y axis every 50,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_attributable_hiv_burden_sti(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc_lower"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis"),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
            ),
            pl.sum("unaids_hiv_incidence_number_attributable_to_chlamydia"),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
            ),
        )
        .with_columns(
            gc=pl.col("unaids_hiv_incidence_number_attributable_to_gc"),
            gc_upper=pl.col(
                "unaids_hiv_incidence_number_attributable_to_gc_upper"
            ),
            gc_lower=pl.col(
                "unaids_hiv_incidence_number_attributable_to_gc_lower"
            ),
            syphilis=pl.col(
                "unaids_hiv_incidence_number_attributable_to_syphilis"
            ),
            syphilis_upper=pl.col(
                "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
            ),
            syphilis_lower=pl.col(
                "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
            ),
            trichomoniasis=pl.col(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
            ),
            trichomoniasis_upper=pl.col(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
            ),
            trichomoniasis_lower=pl.col(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
            ),
            chlamydia=pl.col(
                "unaids_hiv_incidence_number_attributable_to_chlamydia"
            ),
            chlamydia_upper=pl.col(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
            ),
            chlamydia_lower=pl.col(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
            ),
        )
    )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv["year"],
        hiv["chlamydia"],
        linewidth=3,
        label="Chlamydia",
        color="forestgreen",
    )

    ax.fill_between(
        hiv["year"],
        hiv["chlamydia_lower"],
        hiv["chlamydia_upper"],
        alpha=0.3,
        color="forestgreen",
    )

    ax.plot(
        hiv["year"],
        hiv["gc"],
        linewidth=3,
        label="Gonorrhea",
        color="firebrick",
    )

    ax.fill_between(
        hiv["year"],
        hiv["gc_lower"],
        hiv["gc_upper"],
        alpha=0.3,
        color="firebrick",
    )

    ax.plot(
        hiv["year"],
        hiv["syphilis"],
        linewidth=3,
        label="Syphilis",
        color="purple",
    )

    ax.fill_between(
        hiv["year"],
        hiv["syphilis_lower"],
        hiv["syphilis_upper"],
        alpha=0.3,
        color="purple",
    )

    ax.plot(
        hiv["year"],
        hiv["trichomoniasis"],
        linewidth=3,
        label="Trichomoniasis",
        color="gold",
    )

    ax.fill_between(
        hiv["year"],
        hiv["trichomoniasis_lower"],
        hiv["trichomoniasis_upper"],
        alpha=0.3,
        color="gold",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Attributable HIV incidence (N)")
    ax.legend(edgecolor="black")
    ax.set_ylim(0)
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    # ticks every 10,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_attributable_hiv_burden_sex(hiv, ax, fig, number):
    hiv = hiv.group_by(["year", "sex"]).agg(
        pl.sum("unaids_hiv_incidence_number_attributable_to_gc"),
        pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper"),
        pl.sum("unaids_hiv_incidence_number_attributable_to_gc_lower"),
        pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis"),
        pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis_upper"),
        pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis_lower"),
        pl.sum("unaids_hiv_incidence_number_attributable_to_trichomoniasis"),
        pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
        ),
        pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
        ),
        pl.sum("unaids_hiv_incidence_number_attributable_to_chlamydia"),
        pl.sum("unaids_hiv_incidence_number_attributable_to_chlamydia_upper"),
        pl.sum("unaids_hiv_incidence_number_attributable_to_chlamydia_lower"),
        pl.sum("un_pop"),
    )
    if number:
        hiv = hiv.with_columns(
            hiv_incidence_rate=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia"
                )
            ),
            hiv_incidence_rate_upper=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc_upper")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
                )
            ),
            hiv_incidence_rate_lower=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc_lower")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
                )
            ),
        )
    else:
        hiv = hiv.with_columns(
            hiv_incidence_rate=100000
            * (
                pl.col("unaids_hiv_incidence_number_attributable_to_gc")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia"
                )
            )
            / pl.col("un_pop"),
            hiv_incidence_rate_upper=100000
            * (
                pl.col("unaids_hiv_incidence_number_attributable_to_gc_upper")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
                )
            )
            / pl.col("un_pop"),
            hiv_incidence_rate_lower=100000
            * (
                pl.col("unaids_hiv_incidence_number_attributable_to_gc_lower")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
                )
            )
            / pl.col("un_pop"),
        )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv.filter(pl.col("sex") == "Male")["year"],
        hiv.filter(pl.col("sex") == "Male")["hiv_incidence_rate"],
        linewidth=3,
        label="Heterosexual male",
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
        label="Heterosexual female",
        color="magenta",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("sex") == "Female")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="magenta",
    )

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
    if number:
        ax.set_ylabel("Attributable HIV incidence (N)")
    else:
        ax.set_ylabel(
            "Attributable HIV incidence rate (per 100,000 population)"
        )
    ax.legend(loc="upper right", edgecolor="black")
    ax.set_ylim(0)
    if number:
        ax.set_yticks([0, 100000, 200000, 300000, 400000])
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
        )
        # ticks every 25,000
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(25000))
    else:
        # ticks every 25
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_attributable_hiv_burden_region(hiv, ax, fig):
    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_gc_lower"),
            pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis"),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
            ),
            pl.sum("unaids_hiv_incidence_number_attributable_to_chlamydia"),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
            ),
            pl.sum(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
            ),
        )
        .with_columns(
            hiv_incidence_rate=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia"
                )
            ),
            hiv_incidence_rate_upper=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc_upper")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
                )
            ),
            hiv_incidence_rate_lower=(
                pl.col("unaids_hiv_incidence_number_attributable_to_gc_lower")
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
                )
                + pl.col(
                    "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
                )
            ),
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
        color="teal",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Central")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="teal",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate"],
        linewidth=3,
        label="Southern",
        color="maroon",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate_lower"],
        hiv.filter(pl.col("region") == "Southern")["hiv_incidence_rate_upper"],
        alpha=0.3,
        color="maroon",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Attributable HIV incidence (N)")
    ax.legend(loc="upper right", edgecolor="black")
    ax.set_xticks([1990, 2000, 2010, 2020])
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    # ticks every 10,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(10000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


def plot_attributable_hiv_burden_sex_pathogen(hiv, ax, fig):
    # for males, the chlamydia value is at 0, but we still want to show a small
    # bar for visualization purposes
    # same with the gonorrhea bar
    hiv = hiv.with_columns(
        pl.when(
            (
                pl.col("unaids_hiv_incidence_number_attributable_to_chlamydia")
                == 0
            )
            & (pl.col("sex") == "Male")
        )
        .then(pl.lit(10))
        .otherwise(
            pl.col("unaids_hiv_incidence_number_attributable_to_chlamydia")
        )
        .alias("unaids_hiv_incidence_number_attributable_to_chlamydia"),
        pl.when(
            (pl.col("unaids_hiv_incidence_number_attributable_to_gc") <= 10)
            & (pl.col("sex") == "Male")
        )
        .then(pl.lit(10))
        .otherwise(pl.col("unaids_hiv_incidence_number_attributable_to_gc"))
        .alias("unaids_hiv_incidence_number_attributable_to_gc"),
    )

    # bar graph of attributable burden in 2023 by sex and pathogen
    # sex on x axis, and each pathogen gets their own color, and the y axis is the percentage of HIV incidence attributable to STIs
    # with error bars for the uncertainty intervals
    hiv = hiv.filter(pl.col("year") == 2023)
    hiv = hiv.group_by(["sex"], maintain_order=True).agg(
        gc=pl.sum("unaids_hiv_incidence_number_attributable_to_gc")
        / pl.sum("unaids_incidence_number")
        * 100,
        gc_upper=pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper")
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        gc_lower=pl.sum("unaids_hiv_incidence_number_attributable_to_gc_lower")
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        syphilis=pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis")
        / pl.sum("unaids_incidence_number")
        * 100,
        syphilis_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        syphilis_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        trichomoniasis=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
        )
        / pl.sum("unaids_incidence_number")
        * 100,
        trichomoniasis_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        trichomoniasis_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        chlamydia=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia"
        )
        / pl.sum("unaids_incidence_number")
        * 100,
        chlamydia_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        chlamydia_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
    )

    stis = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
    sti_names = {
        "gc": "Gonorrhea",
        "syphilis": "Syphilis",
        "trichomoniasis": "Trichomoniasis",
        "chlamydia": "Chlamydia",
    }

    colors = {
        "gc": "firebrick",
        "syphilis": "purple",
        "trichomoniasis": "gold",
        "chlamydia": "forestgreen",
    }

    hiv = hiv.with_columns(
        pl.col("sex").replace(
            {
                "MSM": "MSM",
                "Male": "Heterosexual male",
                "Female": "Heterosexual female",
            }
        )
    )
    # want to get the sex rows in a particular order
    # heterosexual male, heterosexual female, msm
    sexes = ["Heterosexual male", "Heterosexual female", "MSM"]
    hiv = hiv.with_columns(pl.col("sex").cast(pl.Enum(sexes)))
    hiv = hiv.sort("sex")

    x = np.arange(len(sexes))

    width = 0.2
    # Track which labels we've added to avoid duplicates
    added_labels = set()

    for sex_idx, sex in enumerate(sexes):
        sex_data = hiv.filter(pl.col("sex") == sex)

        # Determine which STIs to plot for this sex
        if sex == "Heterosexual female":
            stis_to_plot = stis
            offsets = [width * (i - 1.5) for i in range(4)]  # 4 bars centered
        else:  # Heterosexual male or MSM
            stis_to_plot = [s for s in stis if s != "trichomoniasis"]
            offsets = [width * (i - 1) for i in range(3)]  # 3 bars centered

        for sti_idx, pathogen in enumerate(stis_to_plot):
            offset = offsets[sti_idx]

            val = sex_data[pathogen][0]
            lower = sex_data[f"{pathogen}_lower"][0]
            upper = sex_data[f"{pathogen}_upper"][0]

            # Only add label once per pathogen
            label = (
                sti_names[pathogen] if pathogen not in added_labels else None
            )
            if label:
                added_labels.add(pathogen)

            ax.bar(
                x[sex_idx] + offset,
                val,
                yerr=[[val - lower], [upper - val]],
                capsize=5,
                width=width,
                label=label,
                color=colors[pathogen],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(sexes, rotation=-45, ha="center")
    ax.set_xlabel("Group")
    ax.set_ylabel("Attributable HIV incidence, 2023 (%)")
    ax.legend(edgecolor="black")
    # minor y ticks every 5%
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))


def plot_attributable_hiv_burden_region_pathogen(hiv, ax, fig):
    # bar graph of attributable burden in 2023 by sex and pathogen
    # sex on x axis, and each pathogen gets their own color, and the y axis is the percentage of HIV incidence attributable to STIs
    # with error bars for the uncertainty intervals
    hiv = hiv.filter(pl.col("year") == 2023)

    # enforce ordering of regions
    # we want to get the region rows in a particular order
    # Western, Eastern, Central, Southern
    regions = ["Western", "Eastern", "Central", "Southern"]
    hiv = hiv.with_columns(pl.col("region").cast(pl.Enum(regions)))
    hiv = hiv.sort("region")

    hiv = hiv.group_by(["region"], maintain_order=True).agg(
        gc=pl.sum("unaids_hiv_incidence_number_attributable_to_gc")
        / pl.sum("unaids_incidence_number")
        * 100,
        gc_upper=pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper")
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        gc_lower=pl.sum("unaids_hiv_incidence_number_attributable_to_gc_lower")
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        syphilis=pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis")
        / pl.sum("unaids_incidence_number")
        * 100,
        syphilis_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        syphilis_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        trichomoniasis=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
        )
        / pl.sum("unaids_incidence_number")
        * 100,
        trichomoniasis_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        trichomoniasis_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        chlamydia=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia"
        )
        / pl.sum("unaids_incidence_number")
        * 100,
        chlamydia_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        chlamydia_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
    )

    stis = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
    sti_names = {
        "gc": "Gonorrhea",
        "syphilis": "Syphilis",
        "trichomoniasis": "Trichomoniasis",
        "chlamydia": "Chlamydia",
    }

    colors = {
        "gc": "firebrick",
        "syphilis": "purple",
        "trichomoniasis": "gold",
        "chlamydia": "forestgreen",
    }

    regions = hiv["region"].unique(maintain_order=True).to_list()

    x = np.arange(len(regions))

    width = 0.2
    for i, pathogen in enumerate(stis):
        offset = width * (i - 1.5)  # center the group of bars
        ax.bar(
            x + offset,
            hiv[pathogen],
            yerr=[
                hiv[pathogen] - hiv[f"{pathogen}_lower"],
                hiv[f"{pathogen}_upper"] - hiv[pathogen],
            ],
            capsize=5,
            width=width,
            label=sti_names[pathogen],
            color=colors[pathogen],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=-45, ha="center")
    ax.set_xlabel("Region")
    ax.set_ylabel("Attributable HIV incidence, 2023 (%)")
    ax.legend(loc=(0.01, 0.80), edgecolor="black")
    # add minor y ticks every 1%
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


def compute_country_order(hiv):
    # Sorts countries by descending order by GBD total attributable HIV cases in all years, subset to UNAIDS countries
    # we use GBD for now as reference, but if we switch then swtich which source we order based on
    hiv_sub = hiv.filter(pl.col("cols_unaids_analysis")).with_columns(
        total_gbd=(
            pl.col("hiv_incidence_number_attributable_to_gc")
            + pl.col("hiv_incidence_number_attributable_to_syphilis")
            + pl.col("hiv_incidence_number_attributable_to_chlamydia")
            + pl.col("hiv_incidence_number_attributable_to_trichomoniasis")
        )
    )
    return (
        hiv_sub.group_by("location")
        .agg(pl.sum("total_gbd").alias("total"))
        .sort("total", descending=True)["location"]
        .to_list()
    )


def plot_sensitivity_country_bar(
    hiv, ax, fig, year=None, top_n=10, legend=False
):
    # Bar chart of GBD vs UNAIDS total STI attributable HIV cases by country
    # only includes countries in GBD and UNAIDS analysis (cols_unaids_analysis = True)
    # can be for a specific year or all years, and can include all or subset of countries
    # if year = None, then it is sum all years, if year = int then it is filtered to that year
    hiv = hiv.filter(pl.col("cols_unaids_analysis")).with_columns(
        total_gbd=(
            pl.col("hiv_incidence_number_attributable_to_gc")
            + pl.col("hiv_incidence_number_attributable_to_syphilis")
            + pl.col("hiv_incidence_number_attributable_to_chlamydia")
            + pl.col("hiv_incidence_number_attributable_to_trichomoniasis")
        ),
        total_gbd_upper=(
            pl.col("hiv_incidence_number_attributable_to_gc_upper")
            + pl.col("hiv_incidence_number_attributable_to_syphilis_upper")
            + pl.col("hiv_incidence_number_attributable_to_chlamydia_upper")
            + pl.col(
                "hiv_incidence_number_attributable_to_trichomoniasis_upper"
            )
        ),
        total_gbd_lower=(
            pl.col("hiv_incidence_number_attributable_to_gc_lower")
            + pl.col("hiv_incidence_number_attributable_to_syphilis_lower")
            + pl.col("hiv_incidence_number_attributable_to_chlamydia_lower")
            + pl.col(
                "hiv_incidence_number_attributable_to_trichomoniasis_lower"
            )
        ),
        total_unaids=(
            pl.col("unaids_hiv_incidence_number_attributable_to_gc")
            + pl.col("unaids_hiv_incidence_number_attributable_to_syphilis")
            + pl.col("unaids_hiv_incidence_number_attributable_to_chlamydia")
            + pl.col(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
            )
        ),
        total_unaids_upper=(
            pl.col("unaids_hiv_incidence_number_attributable_to_gc_upper")
            + pl.col(
                "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
            )
            + pl.col(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
            )
            + pl.col(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
            )
        ),
        total_unaids_lower=(
            pl.col("unaids_hiv_incidence_number_attributable_to_gc_lower")
            + pl.col(
                "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
            )
            + pl.col(
                "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
            )
            + pl.col(
                "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
            )
        ),
    )
    if year is not None:
        hiv = hiv.filter(pl.col("year") == year)

    # Compute country order from this panel's data (descending GBD total, all sexes combined)
    country_totals = (
        hiv.group_by("location")
        .agg(pl.sum("total_gbd").alias("total"))
        .sort("total", descending=True)
    )
    plot_countries = country_totals["location"].head(top_n).to_list()

    # sum over years and pathogens to get total attributable cases by country and sex and source
    hiv_agg = hiv.group_by(["location", "sex"]).agg(
        pl.sum("total_gbd"),
        pl.sum("total_gbd_upper"),
        pl.sum("total_gbd_lower"),
        pl.sum("total_unaids"),
        pl.sum("total_unaids_upper"),
        pl.sum("total_unaids_lower"),
    )

    hiv_agg = hiv_agg.filter(pl.col("location").is_in(plot_countries))

    x = np.arange(len(plot_countries))  # make positions for x axis countries

    # Bar formatting: paired by sex, GBD then UNAIDS
    # bw: individual bar width; inner_gap: within a sex pair; outer_gap: between sex groups
    bw = 0.13
    inner_gap = 0.01
    outer_gap = 0.05
    sex_group_width = 2 * bw + inner_gap

    # Centers of the three sex groups
    total_span = 3 * sex_group_width + 2 * outer_gap
    sex_centers = [
        -total_span / 2 + sex_group_width / 2,  # Male (center)
        -total_span / 2
        + sex_group_width / 2
        + sex_group_width
        + outer_gap,  # Female
        -total_span / 2
        + sex_group_width / 2
        + 2 * (sex_group_width + outer_gap),  # MSM
    ]
    # Within each sex group: GBD on left, UNAIDS on right
    src_offset = {
        "GBD": -(bw / 2 + inner_gap / 2),
        "UNAIDS": bw / 2 + inner_gap / 2,
    }

    # Order: Female, Male, MSM — within each pair: GBD left, UNAIDS right
    sexes = ["Female", "Male", "MSM"]
    style = {
        ("GBD", "Female"): dict(facecolor="#4E844F", edgecolor="#388E3C"),
        ("GBD", "Male"): dict(facecolor="#15C306", edgecolor="#30A130"),
        ("GBD", "MSM"): dict(facecolor="#A7F19C", edgecolor="#000000"),
        ("UNAIDS", "Female"): dict(facecolor="#3A78AB", edgecolor="#3355CC"),
        ("UNAIDS", "Male"): dict(facecolor="#2452FA", edgecolor="#579BE8"),
        ("UNAIDS", "MSM"): dict(facecolor="#AFE5F4", edgecolor="#000000"),
    }

    # make bars for each sex-data source combo, looping over sexes and then sources
    for sex_i, sex in enumerate(sexes):
        sex_center = sex_centers[sex_i]
        for src, col, col_upper, col_lower in [
            ("GBD", "total_gbd", "total_gbd_upper", "total_gbd_lower"),
            (
                "UNAIDS",
                "total_unaids",
                "total_unaids_upper",
                "total_unaids_lower",
            ),
        ]:
            vals = []
            lowers = []
            uppers = []
            for loc in plot_countries:
                row = hiv_agg.filter(
                    (pl.col("location") == loc) & (pl.col("sex") == sex)
                )
                vals.append(row[col][0])
                lowers.append(row[col_lower][0])
                uppers.append(row[col_upper][0])

            s = style[(src, sex)]
            yerr_low = [v - lower for v, lower in zip(vals, lowers)]
            yerr_high = [u - v for v, u in zip(vals, uppers)]
            ax.bar(
                x + sex_center + src_offset[src],
                vals,
                yerr=[yerr_low, yerr_high],
                capsize=5,
                width=bw,
                facecolor=s["facecolor"],
                edgecolor=s["edgecolor"],
                label=f"{src} – {sex}",
                linewidth=0.5,
            )

    ax.set_yscale("log")
    rename = {
        "United Republic of Tanzania": "Tanzania",
        "Democratic Republic of the Congo": "DRC",
    }
    display_countries = [rename.get(c, c) for c in plot_countries]
    ax.set_xticks(x)
    ax.set_xticklabels(display_countries, ha="center")
    ax.set_xlim(left=x[0] - total_span / 2 - 0.3)
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    desired_labels = [
        "GBD – Female",
        "GBD – Male",
        "GBD – MSM",
        "UNAIDS – Female",
        "UNAIDS – Male",
        "UNAIDS – MSM",
    ]
    if legend:
        ax.legend(
            [label_to_handle[label] for label in desired_labels],
            desired_labels,
            ncol=2,
            loc=(0.65, 0.82),
            edgecolor="black",
        )
    ax.yaxis.set_major_formatter(  # format log scale with commas and 0s written out
        ticker.FuncFormatter(
            lambda val, pos: f"{int(val):,}" if val >= 1 else ""
        )
    )


def plot_attributable_hiv_burden_age_pathogen(hiv, ax, sex, fig):
    # we need the age groups in a particular order

    age_groups = ["15-24", "25-49", "50+"]
    hiv = hiv.with_columns(pl.col("age_group").cast(pl.Enum(age_groups)))
    hiv = hiv.sort("age_group")

    hiv = hiv.group_by(["age_group"], maintain_order=True).agg(
        gc=pl.sum("unaids_hiv_incidence_number_attributable_to_gc")
        / pl.sum("unaids_incidence_number")
        * 100,
        gc_upper=pl.sum("unaids_hiv_incidence_number_attributable_to_gc_upper")
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        gc_lower=pl.sum("unaids_hiv_incidence_number_attributable_to_gc_lower")
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        syphilis=pl.sum("unaids_hiv_incidence_number_attributable_to_syphilis")
        / pl.sum("unaids_incidence_number")
        * 100,
        syphilis_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_syphilis_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        syphilis_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_syphilis_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        trichomoniasis=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis"
        )
        / pl.sum("unaids_incidence_number")
        * 100,
        trichomoniasis_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        trichomoniasis_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
        chlamydia=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia"
        )
        / pl.sum("unaids_incidence_number")
        * 100,
        chlamydia_upper=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia_upper"
        )
        / pl.sum("unaids_incidence_number_upper")
        * 100,
        chlamydia_lower=pl.sum(
            "unaids_hiv_incidence_number_attributable_to_chlamydia_lower"
        )
        / pl.sum("unaids_incidence_number_lower")
        * 100,
    )

    stis = ["gc", "chlamydia", "syphilis", "trichomoniasis"]
    sti_names = {
        "gc": "Gonorrhea",
        "syphilis": "Syphilis",
        "trichomoniasis": "Trichomoniasis",
        "chlamydia": "Chlamydia",
    }

    colors = {
        "gc": "firebrick",
        "syphilis": "purple",
        "trichomoniasis": "gold",
        "chlamydia": "forestgreen",
    }

    ages = hiv["age_group"].unique(maintain_order=True).to_list()

    x = np.arange(len(ages))

    width = 0.2
    # Determine which STIs to plot for this sex
    if sex == "Female":
        stis = stis
        offsets = [width * (i - 1.5) for i in range(4)]  # 4 bars centered
    else:  # Heterosexual male or MSM
        stis = [s for s in stis if s != "trichomoniasis"]
        offsets = [width * (i - 1) for i in range(3)]  # 3 bars centered

    for i, pathogen in enumerate(stis):
        offset = offsets[i]  # center the group of bars
        ax.bar(
            x + offset,
            hiv[pathogen],
            yerr=[
                hiv[pathogen] - hiv[f"{pathogen}_lower"],
                hiv[f"{pathogen}_upper"] - hiv[pathogen],
            ],
            capsize=5,
            width=width,
            label=sti_names[pathogen],
            color=colors[pathogen],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ages)
    ax.set_xlabel("Age group")
    ax.set_ylabel(f"Attributable HIV incidence among {sex.lower()}s, 2023 (%)")
    ax.legend(loc=(0.63, 0.73), edgecolor="black")
    # add minor y ticks every 1%
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))


if __name__ == "__main__":
    fig, ax = setup_plot(3, 3)  # type: ignore

    # HIDE the original axes - they'll be replaced by GridSpec
    for axis in ax.flatten():  # type: ignore
        axis.set_visible(False)

    # Add two full-width subplots at the top using GridSpec
    gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)

    # First full-width panel
    ax_full_1 = fig.add_subplot(gs[0, :])
    ax_full_1.spines["top"].set_visible(False)
    ax_full_1.spines["right"].set_visible(False)
    ax_full_1.get_xaxis().tick_bottom()
    ax_full_1.get_yaxis().tick_left()
    ax_full_1.tick_params(axis="x", direction="out")
    ax_full_1.tick_params(axis="y", direction="out")
    for spine in ax_full_1.spines.values():
        spine.set_position(("outward", 5))
    ax_full_1.set_axisbelow(True)

    # Second full-width panel
    ax_full_2 = fig.add_subplot(gs[1, :])
    ax_full_2.spines["top"].set_visible(False)
    ax_full_2.spines["right"].set_visible(False)
    ax_full_2.get_xaxis().tick_bottom()
    ax_full_2.get_yaxis().tick_left()
    ax_full_2.tick_params(axis="x", direction="out")
    ax_full_2.tick_params(axis="y", direction="out")
    for spine in ax_full_2.spines.values():
        spine.set_position(("outward", 5))
    ax_full_2.set_axisbelow(True)

    # Create NEW axes for the 3x3 grid in rows 2-4
    ax_new = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            ax_new[i, j] = fig.add_subplot(gs[i + 2, j])
            ax_new[i, j].spines["top"].set_visible(False)
            ax_new[i, j].spines["right"].set_visible(False)
            ax_new[i, j].get_xaxis().tick_bottom()
            ax_new[i, j].get_yaxis().tick_left()
            ax_new[i, j].tick_params(axis="x", direction="out")
            ax_new[i, j].tick_params(axis="y", direction="out")
            for spine in ax_new[i, j].spines.values():
                spine.set_position(("outward", 5))
            ax_new[i, j].set_axisbelow(True)

    # Use ax_new instead of ax for the rest of your code
    ax = ax_new

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))
    hiv = hiv.filter(pl.col("year") <= 2023)

    ax_full_1.text(
        -0.08,
        1.08,
        "a",
        transform=ax_full_1.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # Total attributable HIV cases 1990–2023 by country, GBD vs UNAIDS
    plot_sensitivity_country_bar(hiv, ax_full_1, fig, year=None, legend=True)
    ax_full_1.set_ylabel(
        "Cumulative attributable HIV incidence, 1990-2023 (N)"
    )

    ax_full_2.text(
        -0.08,
        1.08,
        "b",
        transform=ax_full_2.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # Attributable HIV cases in latest year (2023) by country, GBD vs UNAIDS
    latest_year = 2023
    plot_sensitivity_country_bar(
        hiv, ax_full_2, fig, year=latest_year, legend=True
    )
    ax_full_2.set_ylabel("Attributable HIV incidence, 2023 (N)")
    # ax_full_2.get_legend().remove()

    # by source
    ax[0, 0].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[0, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_source(hiv, ax[0, 0], fig)  # type: ignore

    # by sex as number
    ax[0, 1].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[0, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_sex(
        hiv,
        ax[0, 1],  # type: ignore
        fig,
        number=True,
    )

    # by sex as proportion
    ax[0, 2].text(  # type: ignore
        -0.25,
        1.08,
        "e",
        transform=ax[0, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_sex(
        hiv,
        ax[0, 2],  # type: ignore
        fig,
        number=False,
    )

    # by sti
    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "f",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_sti(hiv, ax[1, 0], fig)  # type: ignore

    # by region
    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "g",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_region(hiv, ax[1, 1], fig)  # type: ignore

    # by region-pathogen
    ax[1, 2].text(  # type: ignore
        -0.25,
        1.08,
        "h",
        transform=ax[1, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_region_pathogen(hiv, ax[1, 2], fig)  # type: ignore

    # by sex-pathogen
    ax[2, 2].text(  # type: ignore
        -0.25,
        1.08,
        "k",
        transform=ax[2, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_sex_pathogen(hiv, ax[2, 2], fig)  # type: ignore

    hiv = pl.read_csv(
        os.path.join(output_dir, "hiv_attributable_to_stis_age_stratified.csv")
    )
    hiv = hiv.filter(pl.col("year") <= 2023)

    # by age-pathogen for females
    ax[2, 0].text(  # type: ignore
        -0.25,
        1.08,
        "i",
        transform=ax[2, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_age_pathogen(
        hiv.filter(pl.col("sex") == "Female"),
        ax[2, 0],  # type: ignore
        "Female",
        fig,
    )

    # by age-pathogen for males
    ax[2, 1].text(  # type: ignore
        -0.25,
        1.08,
        "j",
        transform=ax[2, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_age_pathogen(
        hiv.filter(pl.col("sex") != "Female"),
        ax[2, 1],  # type: ignore
        "Male",
        fig,
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_sti_attributable_hiv.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_sti_attributable_hiv.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
