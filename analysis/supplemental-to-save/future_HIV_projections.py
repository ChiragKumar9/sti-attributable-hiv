

# Code for including future HIV projections in figure_overall_hiv_burden_gc_resistance.py
# ALso need to load hiv_proj and hiv_proj_sex CSVs in __main__ and pass them as arguments to plot_hiv_burden_by_sex and plot_hiv_burden_by_region

def plot_hiv_burden_by_sex(hiv, hiv_proj_sex, ax, fig):
    # Aggregate UNAIDS incidence; combine Male + MSM into a single Male total
    import polars as pl

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
        .agg(pl.sum("incidence"), pl.sum("incidence_upper"), pl.sum("incidence_lower"))
        .sort("year")
    )
    female_hist = hiv_agg.filter(pl.col("sex") == "Female").sort("year")

    ax.plot(male_hist["year"], male_hist["incidence"],
            linewidth=3, label="Male", color="midnightblue")
    ax.fill_between(male_hist["year"],
                    male_hist["incidence_lower"], male_hist["incidence_upper"],
                    alpha=0.3, color="midnightblue")

    ax.plot(female_hist["year"], female_hist["incidence"],
            linewidth=3, label="Female", color="magenta")
    ax.fill_between(female_hist["year"],
                    female_hist["incidence_lower"], female_hist["incidence_upper"],
                    alpha=0.3, color="magenta")

    # Projected lines from 2025; Male in proj already encodes total male
    # (non-MSM + MSM) because male_fraction was derived from Male + MSM.
    proj = (
        hiv_proj_sex
        .filter(pl.col("year") >= 2025)
        .group_by(["year", "sex"])
        .agg(pl.sum("value").alias("incidence"))
        .sort("year")
    )
    ax.plot(proj.filter(pl.col("sex") == "Male")["year"],
            proj.filter(pl.col("sex") == "Male")["incidence"],
            linewidth=3, linestyle="--", color="midnightblue")
    ax.plot(proj.filter(pl.col("sex") == "Female")["year"],
            proj.filter(pl.col("sex") == "Female")["incidence"],
            linewidth=3, linestyle="--", color="magenta")

    # Vertical line marking the start of future projections
    ax.axvline(2025, color="grey", linestyle=":", linewidth=1.5)
    ax.text(2025.3, ax.get_ylim()[1] * 0.95, "Projected",
            color="grey", fontsize=16, va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence in\nsub-Saharan Africa (N)")
    ax.legend(loc=(0.5, 0.73))
    ax.set_xticks([1990, 2000, 2010, 2020, 2030])
    ax.set_yticks([0, 500000, 1000000, 1500000])
    ax.yaxis.set_major_formatter(
        __import__("matplotlib").ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    ax.yaxis.set_minor_locator(__import__("matplotlib").ticker.MultipleLocator(100000))
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.MultipleLocator(2))


def plot_hiv_burden_by_region(hiv, hiv_proj, ax, fig):
    import polars as pl

    hiv = (
        hiv.group_by(["year", "region"])
        .agg(
            pl.sum("unaids_incidence_number"),
            pl.sum("unaids_incidence_number_upper"),
            pl.sum("unaids_incidence_number_lower"),
        )
        .rename({
            "unaids_incidence_number": "incidence",
            "unaids_incidence_number_upper": "incidence_upper",
            "unaids_incidence_number_lower": "incidence_lower",
        })
    )
    hiv = hiv.sort(by="year")

    # Aggregate projected data by region and year; start from 2025 (the first
    # true future-projection year)
    proj = (
        hiv_proj
        .filter(pl.col("year") >= 2025)
        .group_by(["year", "region"])
        .agg(pl.sum("value").alias("incidence"))
        .sort("year")
    )

    region_colors = {
        "Western": "darkorange",
        "Eastern": "olivedrab",
        "Central": "teal",
        "Southern": "maroon",
    }

    for region, color in region_colors.items():
        hist = hiv.filter(pl.col("region") == region)
        ax.plot(hist["year"], hist["incidence"], linewidth=3,
                label=region, color=color)
        ax.fill_between(hist["year"], hist["incidence_lower"],
                        hist["incidence_upper"], alpha=0.3, color=color)

        proj_r = proj.filter(pl.col("region") == region)
        ax.plot(proj_r["year"], proj_r["incidence"], linewidth=3,
                linestyle="--", color=color)

    # Vertical dashed line marking the start of future projections
    ax.axvline(2025, color="grey", linestyle=":", linewidth=1.5)
    ax.text(2025.3, ax.get_ylim()[1] * 0.95, "Projected",
            color="grey", fontsize=16, va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("HIV incidence (N)")
    ax.legend(loc=(0.5, 0.68))
    ax.set_xticks([1990, 2000, 2010, 2020, 2030])
    ax.set_yticks([0, 250000, 500000, 750000, 1000000])
    ax.yaxis.set_major_formatter(
        __import__("matplotlib").ticker.FuncFormatter(lambda x, pos: f"{int(x):,}")
    )
    ax.set_ylim(0)
    ax.yaxis.set_minor_locator(__import__("matplotlib").ticker.MultipleLocator(50000))
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.MultipleLocator(2))


# In __main__, load the projection CSVs and pass them to the plot functions:
#
#   hiv_proj = pl.read_csv(os.path.join(output_dir, "unaids_future_hiv_projections_formatted.csv"))
#   hiv_proj_sex = pl.read_csv(os.path.join(output_dir, "unaids_future_hiv_projections_by_sex.csv"))
#
#   plot_hiv_burden_by_sex(hiv, hiv_proj_sex, ax[0, 0], fig)
#   plot_hiv_burden_by_region(hiv, hiv_proj, ax[0, 1], fig)
