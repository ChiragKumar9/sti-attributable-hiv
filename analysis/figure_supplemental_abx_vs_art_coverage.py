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


def plot_abx_treatment_rates(df, ax, label, fig, legend=False):
    df = df.group_by(["year", "sex"]).agg(pl.mean("frac_sought_treatment"))
    df = df.sort(by="year")

    ax.plot(
        df.filter(pl.col("sex") == "Male")["year"],
        df.filter(pl.col("sex") == "Male")["frac_sought_treatment"] * 100,
        linewidth=3,
        label="Male",
        color="midnightblue",
    )

    ax.plot(
        df.filter(pl.col("sex") == "Female")["year"],
        df.filter(pl.col("sex") == "Female")["frac_sought_treatment"] * 100,
        linewidth=3,
        label="Female",
        color="magenta",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    if legend:
        ax.legend()
    ax.set_xticks([2000, 2010, 2020, 2030])
    # minor ticks every two years
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.set_ylim(0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )


def plot_art_coverage(df, ax, label, fig, legend=False):
    df = df.group_by(["year", "sex"]).agg(
        pl.mean("treatment_proportion"),
        pl.mean("treatment_proportion_lower"),
        pl.mean("treatment_proportion_upper"),
    )
    df = df.sort(by="year")

    ax.fill_between(
        df.filter(pl.col("sex") == "Male")["year"],
        df.filter(pl.col("sex") == "Male")["treatment_proportion_lower"] * 100,
        df.filter(pl.col("sex") == "Male")["treatment_proportion_upper"] * 100,
        alpha=0.3,
        color="midnightblue",
    )

    ax.fill_between(
        df.filter(pl.col("sex") == "Female")["year"],
        df.filter(pl.col("sex") == "Female")["treatment_proportion_lower"]
        * 100,
        df.filter(pl.col("sex") == "Female")["treatment_proportion_upper"]
        * 100,
        alpha=0.3,
        color="magenta",
    )

    ax.plot(
        df.filter(pl.col("sex") == "Male")["year"],
        df.filter(pl.col("sex") == "Male")["treatment_proportion"] * 100,
        linewidth=3,
        label="Male",
        color="midnightblue",
    )

    ax.plot(
        df.filter(pl.col("sex") == "Female")["year"],
        df.filter(pl.col("sex") == "Female")["treatment_proportion"] * 100,
        linewidth=3,
        label="Female",
        color="magenta",
    )

    # horizontal line at 95**3%
    ax.axhline((0.95**3) * 100, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    if legend:
        ax.legend()
    ax.set_xticks([2000, 2010, 2020, 2030])
    # minor ticks every two years
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )


if __name__ == "__main__":
    df = pl.read_csv(os.path.join(output_dir, "hiv_sti_with_projections.csv"))
    df = df.filter(pl.col("year") >= 2000)

    fig, ax = setup_plot(1, 2)  # type: ignore

    # art
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
    plot_art_coverage(
        df,
        ax[0],  # type: ignore
        "HIV viral suppression (%)",
        fig,
        legend=True,
    )

    # abx
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
    plot_abx_treatment_rates(
        df,
        ax[1],  # type: ignore
        "STI antibiotic treatment rate (%)",
        fig,
        legend=False,
    )

    fig.tight_layout()

    # save
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_abx_vs_art_coverage.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(
            fig_dir,
            "figure_supplemental_abx_vs_art_coverage.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
