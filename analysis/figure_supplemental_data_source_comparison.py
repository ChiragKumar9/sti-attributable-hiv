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
    fig, ax = plt.subplots(nrows, ncols, figsize=(30, 30))
    if nrows * ncols == 1:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis="x", direction="out")
        ax.tick_params(axis="y", direction="out")
        ax.tick_params(which="major", length=5.25, width=1.2)
        ax.tick_params(which="minor", length=3.0, width=0.9)
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
        temp.tick_params(which="major", length=5.25, width=1.2)
        temp.tick_params(which="minor", length=3.0, width=0.9)
        # offset the spines
        for spine in temp.spines.values():
            spine.set_position(("outward", 5))
        # put the grid behind
        temp.set_axisbelow(True)
    return fig, ax


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
    ax.set_yticks([0, 200000, 400000, 600000])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    # add minor ticks on the y axis every 50,000
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50000))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))


if __name__ == "__main__":
    fig, ax = setup_plot(3, 3)  # type: ignore

    # HIDE the original axes - they'll be replaced by GridSpec
    for axis in ax.flatten():  # type: ignore
        axis.set_visible(False)

    # Add two full-width subplots at the top using GridSpec
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # First full-width panel (row 0)
    ax_full_1 = fig.add_subplot(gs[0, :])
    ax_full_1.spines["top"].set_visible(False)
    ax_full_1.spines["right"].set_visible(False)
    ax_full_1.get_xaxis().tick_bottom()
    ax_full_1.get_yaxis().tick_left()
    ax_full_1.tick_params(axis="x", direction="out")
    ax_full_1.tick_params(axis="y", direction="out")
    ax_full_1.tick_params(which="major", length=5.25, width=1.2)
    ax_full_1.tick_params(which="minor", length=3.0, width=0.9)
    for spine in ax_full_1.spines.values():
        spine.set_position(("outward", 5))
    ax_full_1.set_axisbelow(True)

    # Second full-width panel (row 1)
    ax_full_2 = fig.add_subplot(gs[1, :])
    ax_full_2.spines["top"].set_visible(False)
    ax_full_2.spines["right"].set_visible(False)
    ax_full_2.get_xaxis().tick_bottom()
    ax_full_2.get_yaxis().tick_left()
    ax_full_2.tick_params(axis="x", direction="out")
    ax_full_2.tick_params(axis="y", direction="out")
    ax_full_2.tick_params(which="major", length=5.25, width=1.2)
    ax_full_2.tick_params(which="minor", length=3.0, width=0.9)
    for spine in ax_full_2.spines.values():
        spine.set_position(("outward", 5))
    ax_full_2.set_axisbelow(True)

    # Third panel (row 2), centered in the middle column
    ax_full_3 = fig.add_subplot(gs[2, 1])
    ax_full_3.spines["top"].set_visible(False)
    ax_full_3.spines["right"].set_visible(False)
    ax_full_3.get_xaxis().tick_bottom()
    ax_full_3.get_yaxis().tick_left()
    ax_full_3.tick_params(axis="x", direction="out")
    ax_full_3.tick_params(axis="y", direction="out")
    ax_full_3.tick_params(which="major", length=5.25, width=1.2)
    ax_full_3.tick_params(which="minor", length=3.0, width=0.9)
    for spine in ax_full_3.spines.values():
        spine.set_position(("outward", 5))
    ax_full_3.set_axisbelow(True)

    hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))
    hiv = hiv.filter(pl.col("year") <= 2023)

    # Panel a - Total attributable HIV cases 1990–2023 by country, GBD vs UNAIDS
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
    plot_sensitivity_country_bar(hiv, ax_full_1, fig, year=None, legend=True)
    ax_full_1.set_ylabel(
        "Cumulative attributable HIV incidence, 1990-2023 (N)"
    )

    # Panel b - Attributable HIV cases in latest year (2023) by country, GBD vs UNAIDS
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
    latest_year = 2023
    plot_sensitivity_country_bar(
        hiv, ax_full_2, fig, year=latest_year, legend=True
    )
    ax_full_2.set_ylabel("Attributable HIV incidence, 2023 (N)")

    # Panel c - by source (GBD vs UNAIDS over time)
    ax_full_3.text(
        -0.25,
        1.08,
        "c",
        transform=ax_full_3.transAxes,
        fontsize=24,
        fontweight="bold",
        va="top",
        ha="right",
    )
    plot_attributable_hiv_burden_source(hiv, ax_full_3, fig)

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_data_source_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir, "figure_supplemental_data_source_comparison.pdf"
        ),
        dpi=300,
        bbox_inches="tight",
    )
