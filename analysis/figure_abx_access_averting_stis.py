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
    fig, ax = plt.subplots(nrows, ncols, figsize=(40, 15))
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


def ax_formatting(ax, forward):
    if forward:
        ax.set_xticks([2025, 2030])
        # minor x ticks every year
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    else:
        ax.set_xticks([2000, 2010, 2020])
        # minor x ticks every two years
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        # set custom y axis minor ticks
        ax.set_ylim(200)
        ax.yaxis.set_minor_locator(
            ticker.FixedLocator(
                np.array(
                    [
                        200,
                        300,
                        400,
                        500,
                        600,
                        700,
                        800,
                        900,
                        2000,
                        3000,
                        4000,
                        5000,
                        6000,
                        7000,
                        8000,
                        9000,
                        20000,
                        30000,
                        40000,
                        50000,
                        60000,
                        70000,
                        80000,
                        90000,
                        200000,
                        300000,
                        400000,
                        500000,
                        600000,
                        700000,
                        800000,
                        900000,
                    ]
                )  # type: ignore
            )
        )


def plot_best_case(hiv, ax, year, fig, forward=False, reference=None):
    if forward:
        scalar = -1.0
        # this may change based on how the future sims are set up?
        stub = "upper_bound_future"
        hiv = hiv.filter(pl.col("year") >= 2025)
    else:
        scalar = 1.0
        stub = "upper_bound"
        hiv = hiv.filter(pl.col("year") <= 2024)

    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum(f"hiv_averted_{stub}"),
            pl.sum(f"hiv_averted_{stub}_lower"),
            pl.sum(f"hiv_averted_{stub}_upper"),
            pl.sum("direct_hiv_averted_upper_bound"),
            pl.sum("direct_hiv_averted_upper_bound_lower"),
            pl.sum("direct_hiv_averted_upper_bound_upper"),
            pl.sum("unaids_incidence_number"),
            pl.sum("unaids_incidence_number_upper"),
            pl.sum("unaids_incidence_number_lower"),
        )
        .with_columns(
            direct=pl.col("direct_hiv_averted_upper_bound"),
            direct_lower=pl.col("direct_hiv_averted_upper_bound_lower"),
            direct_upper=pl.col("direct_hiv_averted_upper_bound_upper"),
            total=pl.col(f"hiv_averted_{stub}"),
            total_lower=pl.col(f"hiv_averted_{stub}_lower"),
            total_upper=pl.col(f"hiv_averted_{stub}_upper"),
            reference=pl.col("unaids_incidence_number"),
            reference_lower=pl.col("unaids_incidence_number_lower"),
            reference_upper=pl.col("unaids_incidence_number_upper"),
        )
    )
    if forward:
        hiv = hiv.with_columns(
            direct=pl.col("reference") + (pl.lit(scalar) * pl.col("direct")),
            direct_lower=pl.col("reference_lower")
            + (pl.lit(scalar) * pl.col("direct_upper")),
            direct_upper=pl.col("reference_upper")
            + (pl.lit(scalar) * pl.col("direct_lower")),
            total=pl.col("reference") + (pl.lit(scalar) * pl.col("total")),
            total_lower=pl.col("reference_lower")
            + (pl.lit(scalar) * pl.col("total_upper")),
            total_upper=pl.col("reference_upper")
            + (pl.lit(scalar) * pl.col("total_lower")),
        )
    else:
        hiv = hiv.with_columns(
            direct=pl.col("reference") + (pl.lit(scalar) * pl.col("direct")),
            direct_lower=pl.col("reference_lower")
            + (pl.lit(scalar) * pl.col("direct_lower")),
            direct_upper=pl.col("reference_upper")
            + (pl.lit(scalar) * pl.col("direct_upper")),
            total=pl.col("reference") + (pl.lit(scalar) * pl.col("total")),
            total_lower=pl.col("reference_lower")
            + (pl.lit(scalar) * pl.col("total_lower")),
            total_upper=pl.col("reference_upper")
            + (pl.lit(scalar) * pl.col("total_upper")),
        )

    hiv = hiv.filter(pl.col("year") >= year)

    hiv = hiv.sort(by="year")
    print("HIV values in 2023")
    print(
        hiv.filter(pl.col("year") == 2023).select(
            [
                "total",
                "total_lower",
                "total_upper",
                "reference",
                "reference_lower",
                "reference_upper",
            ]
        )
    )

    print("Total HIV averted")
    print(hiv["total"].sum() - hiv["reference"].sum())
    print(hiv["total_lower"].sum() - hiv["reference_lower"].sum())
    print(hiv["total_upper"].sum() - hiv["reference_upper"].sum())

    print("Total HIV averted as percentage of reference")
    print(
        (hiv["total"].sum() - hiv["reference"].sum()) / hiv["reference"].sum()
    )
    print(
        (hiv["total_lower"].sum() - hiv["reference_lower"].sum())
        / hiv["reference_lower"].sum()
    )
    print(
        (hiv["total_upper"].sum() - hiv["reference_upper"].sum())
        / hiv["reference_upper"].sum()
    )

    ax.plot(
        hiv["year"],
        hiv["direct"],
        linewidth=3,
        label="Direct",
        color="goldenrod",
    )

    ax.fill_between(
        hiv["year"],
        hiv["direct_lower"],
        hiv["direct_upper"],
        alpha=0.3,
        color="goldenrod",
    )

    ax.plot(
        hiv["year"], hiv["total"], linewidth=3, label="Total", color="brown"
    )

    ax.fill_between(
        hiv["year"],
        hiv["total_lower"],
        hiv["total_upper"],
        alpha=0.3,
        color="brown",
    )

    ax.plot(
        hiv["year"],
        hiv["reference"],
        linewidth=3,
        label="Current" if not forward else "Projected",
        color="black",
    )

    ax.fill_between(
        hiv["year"],
        hiv["reference_lower"],
        hiv["reference_upper"],
        alpha=0.3,
        color="black",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence (N)")
    if forward:
        loc = (0.55, 0.75)
    else:
        loc = None
    ax.legend(loc=loc)
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    if reference is not None:
        ax.axhline(
            0.1 * reference,  # type: ignore
            linestyle="--",
            color="#009CDE",
            label="SDG target",
        )
        ax.set_xticks([2025, 2030])
        # minor x ticks every year
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    else:
        ax.set_xticks([2000, 2010, 2020])
        # minor x ticks every two years
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    # need consistent y axis ticks
    if forward:
        ax.set_yticks([0, 100000, 200000, 300000, 400000, 500000, 600000])
    else:
        ax.set_yticks([0, 1000000, 2000000, 3000000])
    # minor y ticks every 250000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(250000))


def plot_averted_sti(hiv, ax, year, fig, forward=False):
    if forward:
        scalar = -1.0
        stub = "upper_bound_future"
        hiv = hiv.filter(pl.col("year") >= 2024)
    else:
        scalar = 1.0
        stub = "upper_bound"
        hiv = hiv.filter(pl.col("year") <= 2023)

    hiv = (
        hiv.group_by(["year"])
        .agg(
            pl.sum(f"hiv_averted_gc_{stub}"),
            pl.sum(f"hiv_averted_gc_{stub}_lower"),
            pl.sum(f"hiv_averted_gc_{stub}_upper"),
            pl.sum(f"hiv_averted_chlamydia_{stub}"),
            pl.sum(f"hiv_averted_chlamydia_{stub}_lower"),
            pl.sum(f"hiv_averted_chlamydia_{stub}_upper"),
            pl.sum(f"hiv_averted_syphilis_{stub}"),
            pl.sum(f"hiv_averted_syphilis_{stub}_lower"),
            pl.sum(f"hiv_averted_syphilis_{stub}_upper"),
            pl.sum(f"hiv_averted_trichomoniasis_{stub}"),
            pl.sum(f"hiv_averted_trichomoniasis_{stub}_lower"),
            pl.sum(f"hiv_averted_trichomoniasis_{stub}_upper"),
        )
        .with_columns(
            gc=pl.col(f"hiv_averted_gc_{stub}") * pl.lit(scalar),
            gc_lower=pl.col(f"hiv_averted_gc_{stub}_lower") * pl.lit(scalar),
            gc_upper=pl.col(f"hiv_averted_gc_{stub}_upper") * pl.lit(scalar),
            chlamydia=pl.col(f"hiv_averted_chlamydia_{stub}") * pl.lit(scalar),
            chlamydia_lower=pl.col(f"hiv_averted_chlamydia_{stub}_lower")
            * pl.lit(scalar),
            chlamydia_upper=pl.col(f"hiv_averted_chlamydia_{stub}_upper")
            * pl.lit(scalar),
            syphilis=pl.col(f"hiv_averted_syphilis_{stub}") * pl.lit(scalar),
            syphilis_lower=pl.col(f"hiv_averted_syphilis_{stub}_lower")
            * pl.lit(scalar),
            syphilis_upper=pl.col(f"hiv_averted_syphilis_{stub}_upper")
            * pl.lit(scalar),
            trichomoniasis=pl.col(f"hiv_averted_trichomoniasis_{stub}")
            * pl.lit(scalar),
            trichomoniasis_lower=pl.col(
                f"hiv_averted_trichomoniasis_{stub}_lower"
            )
            * pl.lit(scalar),
            trichomoniasis_upper=pl.col(
                f"hiv_averted_trichomoniasis_{stub}_upper"
            )
            * pl.lit(scalar),
        )
    )

    hiv = hiv.filter(pl.col("year") >= year)

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
    ax.set_ylabel("Change in HIV incidence (N)")
    if forward:
        loc = (0.4, 0.7)
    else:
        loc = (0, 0)
    ax.legend(loc=loc)
    ax.set_yscale("symlog")
    # make the numbers appear in non scientific notation
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax_formatting(ax, forward)


def plot_averted_sex(hiv, ax, year, fig, forward=False):
    if forward:
        scalar = -1.0
        stub = "upper_bound_future"
        hiv = hiv.filter(pl.col("year") >= 2024)
    else:
        scalar = 1.0
        stub = "upper_bound"
        hiv = hiv.filter(pl.col("year") <= 2023)

    hiv = (
        hiv.group_by(["year", "sex"])
        .agg(
            pl.sum(f"hiv_averted_{stub}"),
            pl.sum(f"hiv_averted_{stub}_lower"),
            pl.sum(f"hiv_averted_{stub}_upper"),
            pl.sum("unaids_incidence_number"),
            pl.sum("unaids_incidence_number_lower"),
            pl.sum("unaids_incidence_number_upper"),
        )
        .with_columns(
            total=pl.col(f"hiv_averted_{stub}") * pl.lit(scalar),
            total_lower=pl.col(f"hiv_averted_{stub}_lower") * pl.lit(scalar),
            total_upper=pl.col(f"hiv_averted_{stub}_upper") * pl.lit(scalar),
        )
    )

    hiv = hiv.filter(pl.col("year") >= year)

    # sum the values by sex
    print("HIV numbers averted Male:")
    print(hiv.filter(pl.col("sex") == "Male")["total"].sum())
    print(hiv.filter(pl.col("sex") == "Male")["total_upper"].sum())
    print(hiv.filter(pl.col("sex") == "Male")["total_lower"].sum())

    print("HIV numbers averted Female:")
    print(hiv.filter(pl.col("sex") == "Female")["total"].sum())
    print(hiv.filter(pl.col("sex") == "Female")["total_upper"].sum())
    print(hiv.filter(pl.col("sex") == "Female")["total_lower"].sum())

    print("HIV numbers averted MSM:")
    print(hiv.filter(pl.col("sex") == "MSM")["total"].sum())
    print(hiv.filter(pl.col("sex") == "MSM")["total_upper"].sum())
    print(hiv.filter(pl.col("sex") == "MSM")["total_lower"].sum())

    # print these values as percentages
    print("HIV proportion averted Male:")
    print(
        hiv.filter(pl.col("sex") == "Male")["total"].sum()
        / hiv.filter(pl.col("sex") == "Male")["unaids_incidence_number"].sum()
    )
    print(
        hiv.filter(pl.col("sex") == "Male")["total_upper"].sum()
        / hiv.filter(pl.col("sex") == "Male")[
            "unaids_incidence_number_upper"
        ].sum()
    )
    print(
        hiv.filter(pl.col("sex") == "Male")["total_lower"].sum()
        / hiv.filter(pl.col("sex") == "Male")[
            "unaids_incidence_number_lower"
        ].sum()
    )

    print("HIV proportion averted Female:")
    print(
        hiv.filter(pl.col("sex") == "Female")["total"].sum()
        / hiv.filter(pl.col("sex") == "Female")[
            "unaids_incidence_number"
        ].sum()
    )
    print(
        hiv.filter(pl.col("sex") == "Female")["total_upper"].sum()
        / hiv.filter(pl.col("sex") == "Female")[
            "unaids_incidence_number_upper"
        ].sum()
    )
    print(
        hiv.filter(pl.col("sex") == "Female")["total_lower"].sum()
        / hiv.filter(pl.col("sex") == "Female")[
            "unaids_incidence_number_lower"
        ].sum()
    )

    print("HIV proportion averted MSM:")
    print(
        hiv.filter(pl.col("sex") == "MSM")["total"].sum()
        / hiv.filter(pl.col("sex") == "MSM")["unaids_incidence_number"].sum()
    )
    print(
        hiv.filter(pl.col("sex") == "MSM")["total_upper"].sum()
        / hiv.filter(pl.col("sex") == "MSM")[
            "unaids_incidence_number_upper"
        ].sum()
    )
    print(
        hiv.filter(pl.col("sex") == "MSM")["total_lower"].sum()
        / hiv.filter(pl.col("sex") == "MSM")[
            "unaids_incidence_number_lower"
        ].sum()
    )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv.filter(pl.col("sex") == "Male")["year"],
        hiv.filter(pl.col("sex") == "Male")["total"],
        linewidth=3,
        label="Heterosexual male",
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
        label="Heterosexual female",
        color="magenta",
    )

    ax.fill_between(
        hiv.filter(pl.col("sex") == "Female")["year"],
        hiv.filter(pl.col("sex") == "Female")["total_lower"],
        hiv.filter(pl.col("sex") == "Female")["total_upper"],
        alpha=0.3,
        color="magenta",
    )

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
    ax.set_ylabel("Change in HIV incidence (N)")
    if forward:
        loc = (0.4, 0.7)
    else:
        loc = (0, 0)
    ax.legend(loc=loc)
    ax.set_yscale("symlog")
    # make the numbers appear in non scientific notation
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax_formatting(ax, forward)


def plot_averted_region(hiv, ax, year, fig, forward=False):
    if forward:
        scalar = -1.0
        stub = "upper_bound_future"
        hiv = hiv.filter(pl.col("year") >= 2024)
    else:
        scalar = 1.0
        stub = "upper_bound"
        hiv = hiv.filter(pl.col("year") <= 2023)

    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum(f"hiv_averted_{stub}"),
            pl.sum(f"hiv_averted_{stub}_lower"),
            pl.sum(f"hiv_averted_{stub}_upper"),
            pl.sum("unaids_incidence_number"),
            pl.sum("unaids_incidence_number_lower"),
            pl.sum("unaids_incidence_number_upper"),
        )
        .with_columns(
            total=pl.col(f"hiv_averted_{stub}") * pl.lit(scalar),
            total_lower=pl.col(f"hiv_averted_{stub}_lower") * pl.lit(scalar),
            total_upper=pl.col(f"hiv_averted_{stub}_upper") * pl.lit(scalar),
        )
    )

    hiv = hiv.filter(pl.col("year") >= year)

    # print averted percentages by region
    for region in ["Western", "Eastern", "Central", "Southern"]:
        print(region)
        print(
            hiv.filter(pl.col("region") == region)["total"].sum()
            / hiv.filter(pl.col("region") == region)[
                "unaids_incidence_number"
            ].sum()
        )
        print(
            hiv.filter(pl.col("region") == region)["total_lower"].sum()
            / hiv.filter(pl.col("region") == region)[
                "unaids_incidence_number_lower"
            ].sum()
        )
        print(
            hiv.filter(pl.col("region") == region)["total_upper"].sum()
            / hiv.filter(pl.col("region") == region)[
                "unaids_incidence_number_upper"
            ].sum()
        )

    hiv = hiv.sort(by="year")

    ax.plot(
        hiv.filter(pl.col("region") == "Western")["year"],
        hiv.filter(pl.col("region") == "Western")["total"],
        linewidth=3,
        label="Western",
        color="darkorange",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Western")["year"],
        hiv.filter(pl.col("region") == "Western")["total_lower"],
        hiv.filter(pl.col("region") == "Western")["total_upper"],
        alpha=0.3,
        color="darkorange",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Eastern")["year"],
        hiv.filter(pl.col("region") == "Eastern")["total"],
        linewidth=3,
        label="Eastern",
        color="olivedrab",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Eastern")["year"],
        hiv.filter(pl.col("region") == "Eastern")["total_lower"],
        hiv.filter(pl.col("region") == "Eastern")["total_upper"],
        alpha=0.3,
        color="olivedrab",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["total"],
        linewidth=3,
        label="Central",
        color="teal",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Central")["year"],
        hiv.filter(pl.col("region") == "Central")["total_lower"],
        hiv.filter(pl.col("region") == "Central")["total_upper"],
        alpha=0.3,
        color="teal",
    )

    ax.plot(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["total"],
        linewidth=3,
        label="Southern",
        color="maroon",
    )

    ax.fill_between(
        hiv.filter(pl.col("region") == "Southern")["year"],
        hiv.filter(pl.col("region") == "Southern")["total_lower"],
        hiv.filter(pl.col("region") == "Southern")["total_upper"],
        alpha=0.3,
        color="maroon",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Change in HIV incidence (N)")
    if forward:
        loc = (0.6, 0.6)
    else:
        loc = (0, 0)
    ax.legend(loc=loc)
    ax.set_yscale("symlog")
    # make the numbers appear in non scientific notation
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )
    ax_formatting(ax, forward)


if __name__ == "__main__":
    fig, ax = setup_plot(2, 4)  # type: ignore

    hiv = pl.read_csv(
        os.path.join(output_dir, "hiv_averted.csv"),
    )

    first_year_upper_bound = 2001

    year_forward = 2020

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
    plot_best_case(
        hiv,
        ax[0, 0],  # type: ignore
        first_year_upper_bound,
        fig,
    )

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
    plot_averted_sti(
        hiv,
        ax[0, 1],  # type: ignore
        first_year_upper_bound,
        fig,
    )

    ax[0, 2].text(  # type: ignore
        -0.25,
        1.08,
        "c",
        transform=ax[0, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_sex(
        hiv,
        ax[0, 2],  # type: ignore
        first_year_upper_bound,
        fig,
    )

    ax[0, 3].text(  # type: ignore
        -0.25,
        1.08,
        "d",
        transform=ax[0, 3].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_region(
        hiv,
        ax[0, 3],  # type: ignore
        first_year_upper_bound,
        fig,
    )

    ax[1, 0].text(  # type: ignore
        -0.25,
        1.08,
        "e",
        transform=ax[1, 0].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )

    hiv_overall = pl.read_csv(os.path.join(output_dir, "hiv_sti.csv"))

    plot_best_case(
        hiv,
        ax[1, 0],  # type: ignore
        year_forward,
        fig,
        forward=True,
        reference=hiv_overall.filter(pl.col("year") == 2010)[
            "unaids_incidence_number"
        ].sum(),
    )

    ax[1, 1].text(  # type: ignore
        -0.25,
        1.08,
        "f",
        transform=ax[1, 1].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_sti(
        hiv,
        ax[1, 1],  # type: ignore
        year_forward,
        fig,
        forward=True,
    )

    ax[1, 2].text(  # type: ignore
        -0.25,
        1.08,
        "g",
        transform=ax[1, 2].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_sex(
        hiv,
        ax[1, 2],  # type: ignore
        year_forward,
        fig,
        forward=True,
    )

    ax[1, 3].text(  # type: ignore
        -0.25,
        1.08,
        "h",
        transform=ax[1, 3].transAxes,  # type: ignore
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_averted_region(
        hiv,
        ax[1, 3],  # type: ignore
        year_forward,
        fig,
        forward=True,
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(fig_dir, "figure_hiv_averted_via_antibiotic_access.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "figure_hiv_averted_via_antibiotic_access.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
