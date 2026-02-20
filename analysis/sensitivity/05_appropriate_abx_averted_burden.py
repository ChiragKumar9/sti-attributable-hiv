import os

import numpy as np
import polars as pl
import yaml
from tqdm.auto import tqdm

from averted_burden import conditional_exposure

data_dir = "data"
output_dir = "outputs_unaids_sensitivity"

# read params
with open("params.yml", "r") as f:
    params = yaml.safe_load(f)

hiv = pl.read_csv(os.path.join(output_dir, "hiv_attributable_to_stis.csv"))

sti_transmission_increase = pl.read_csv(
    os.path.join(output_dir, "meta_estimated_RRs_HIV_detection_given_STI.csv")
)

# rename the bacteria column
bacterial_names = {
    "Gonorrhea": "gc",
    "Chlamydia": "chlamydia",
    "Syphilis": "syphilis",
    "Trichomoniasis": "trichomoniasis",
}
sti_transmission_increase = sti_transmission_increase.with_columns(
    bacteria=pl.col("bacteria").replace(bacterial_names)
)


# we want to calculate the indirect averted cases from the direct averted cases
# not transmitting
# we do this analysis on a country-by-country basis
# we expect a dataframe with columns:
# year, sex, direct_hiv_averted, direct_hiv_averted_lower, direct_hiv_averted_upper
# paf_{sti}_hiv, paf_{sti}_hiv_lower, paf_{sti}_hiv_upper
# hiv_incidence_number, hiv_incidence_number_lower, hiv_incidence_number_upper
# hiv_prevalence, hiv_prevalence_lower, hiv_prevalence_upper
# treatment_proportion, treatment_proportion_lower, treatment_proportion_upper
# we return a dataframe with columns year, sex, indirect_hiv_averted,
# indirect_hiv_averted_lower, indirect_hiv_averted_upper
def calculate_indirect_averted_cases(
    df,
    sti,
    sti_hiv_transmission_increase_male,
    sti_hiv_transmission_increase_lower_male,
    sti_hiv_transmission_increase_upper_male,
    sti_hiv_transmission_increase_female,
    sti_hiv_transmission_increase_lower_female,
    sti_hiv_transmission_increase_upper_female,
    hiv_to_aids=np.array([0.82, 0.72, 0.64, 0.57, 0.26, 0.19, 0.0]),
    hiv_to_aids_upper=np.array([0.95, 0.88, 0.84, 0.89, 0.50, 0.48, 0.0]),
    hiv_to_aids_lower=np.array([0.70, 0.56, 0.44, 0.25, 0.03, 0.09, 0.0]),
) -> pl.DataFrame:
    # calculate how the prevalent infections generated new infections
    # first we need to calculate the adjusted prevalence accounting for incidence
    df = df.with_columns(
        # we ADD this time because we want the full number of infections in the year
        hiv_prevalence=pl.col("hiv_prevalence_number")
        + 0.5 * pl.col("hiv_incidence_number"),
        hiv_prevalence_lower=pl.col("hiv_prevalence_number_lower")
        + 0.5 * pl.col("hiv_incidence_number_lower"),
        hiv_prevalence_upper=pl.col("hiv_prevalence_number_upper")
        + 0.5 * pl.col("hiv_incidence_number_upper"),
    )
    # we want to make this df pivot wider by sex
    df = df.pivot(
        on="sex",
        values=[
            "direct_hiv_averted",
            "direct_hiv_averted_lower",
            "direct_hiv_averted_upper",
            f"paf_{sti}_hiv",
            f"paf_{sti}_hiv_lower",
            f"paf_{sti}_hiv_upper",
            "hiv_incidence_number",
            "hiv_incidence_number_lower",
            "hiv_incidence_number_upper",
            "hiv_prevalence_number",
            "hiv_prevalence_number_lower",
            "hiv_prevalence_number_upper",
            "treatment_proportion",
            "treatment_proportion_lower",
            "treatment_proportion_upper",
        ],
        index="year",
    )
    df = df.sort(by="year")
    # now assess how many incident infections each prevalent infection generated
    # in the next year
    # assume heterosexual transmission
    df = df.with_columns(
        transmission_rate_M_F=pl.col("hiv_incidence_number_Female").shift(-1)
        / (
            pl.col("hiv_prevalence_number_Male")
            * (1 - pl.col("treatment_proportion_Male"))
        ),
        transmission_rate_lower_M_F=pl.col(
            "hiv_incidence_number_lower_Female"
        ).shift(-1)
        / (
            pl.col("hiv_prevalence_number_lower_Male")
            * (1 - pl.col("treatment_proportion_upper_Male"))
        ),
        transmission_rate_upper_M_F=pl.col(
            "hiv_incidence_number_upper_Female"
        ).shift(-1)
        / (
            pl.col("hiv_prevalence_number_upper_Male")
            * (1 - pl.col("treatment_proportion_lower_Male"))
        ),
        transmission_rate_F_M=pl.col("hiv_incidence_number_Male").shift(-1)
        / (
            pl.col("hiv_prevalence_number_Female")
            * (1 - pl.col("treatment_proportion_Female"))
        ),
        transmission_rate_lower_F_M=pl.col(
            "hiv_incidence_number_lower_Male"
        ).shift(-1)
        / (
            pl.col("hiv_prevalence_number_lower_Female")
            * (1 - pl.col("treatment_proportion_upper_Female"))
        ),
        transmission_rate_upper_F_M=pl.col(
            "hiv_incidence_number_upper_Male"
        ).shift(-1)
        / (
            pl.col("hiv_prevalence_number_upper_Female")
            * (1 - pl.col("treatment_proportion_lower_Female"))
        ),
    )

    # now we need to calculate the additional infections averted in future years
    indirect_male = np.array([])
    indirect_lower_male = np.array([])
    indirect_upper_male = np.array([])
    indirect_female = np.array([])
    indirect_lower_female = np.array([])
    indirect_upper_female = np.array([])
    cumulative_male = np.zeros(df.height)
    cumulative_lower_male = np.zeros(df.height)
    cumulative_upper_male = np.zeros(df.height)
    cumulative_female = np.zeros(df.height)
    cumulative_lower_female = np.zeros(df.height)
    cumulative_upper_female = np.zeros(df.height)
    # before propagating, clean up the data for hiv-to-aids progression
    # data from Survival rate of AIDS disease and mortality in HIV-infected patients: a meta-analysis
    # Poorolajal et al., 2016
    # first make these conditional values by dividing by the last value in the sequence
    hiv_to_aids = np.concatenate(
        [[hiv_to_aids[0]], hiv_to_aids[1:] / hiv_to_aids[:-1]]
    )
    hiv_to_aids_upper = np.concatenate(
        [
            [hiv_to_aids_upper[0]],
            hiv_to_aids_upper[1:] / hiv_to_aids_upper[:-1],
        ]
    )
    hiv_to_aids_lower = np.concatenate(
        [
            [hiv_to_aids_lower[0]],
            hiv_to_aids_lower[1:] / hiv_to_aids_lower[:-1],
        ]
    )
    # they report survival values for every two year chunks, so duplicate their values
    hiv_to_aids = np.repeat(hiv_to_aids, 2)
    hiv_to_aids_upper = np.repeat(hiv_to_aids_upper, 2)
    hiv_to_aids_lower = np.repeat(hiv_to_aids_lower, 2)
    for idx, row in enumerate(df.iter_rows(named=True)):
        # pass when we're on the last row because we don't know the forward
        # transmission rate
        if idx == df.height - 1:
            break

        # account for people transitioning from hiv to aids and too sick to transmit
        hiv_to_aids_temp = hiv_to_aids[:idx]
        hiv_to_aids_upper_temp = hiv_to_aids_upper[:idx]
        hiv_to_aids_lower_temp = hiv_to_aids_lower[:idx]
        # reverse
        hiv_to_aids_temp = hiv_to_aids_temp[::-1]
        hiv_to_aids_upper_temp = hiv_to_aids_upper_temp[::-1]
        hiv_to_aids_lower_temp = hiv_to_aids_lower_temp[::-1]
        # pad with zeros at the end to ensure same length as cumulative
        hiv_to_aids_temp = np.pad(
            hiv_to_aids_temp,
            (0, df.height - len(hiv_to_aids_temp)),
            "constant",
            constant_values=(0, 0),
        )
        hiv_to_aids_upper_temp = np.pad(
            hiv_to_aids_upper_temp,
            (0, df.height - len(hiv_to_aids_upper_temp)),
            "constant",
            constant_values=(0, 0),
        )
        hiv_to_aids_lower_temp = np.pad(
            hiv_to_aids_lower_temp,
            (0, df.height - len(hiv_to_aids_lower_temp)),
            "constant",
            constant_values=(0, 0),
        )
        # multiply cumulative
        cumulative_male *= hiv_to_aids_temp
        cumulative_lower_male *= hiv_to_aids_lower_temp
        cumulative_upper_male *= hiv_to_aids_upper_temp
        cumulative_female *= hiv_to_aids_temp
        cumulative_lower_female *= hiv_to_aids_lower_temp
        cumulative_upper_female *= hiv_to_aids_upper_temp

        # of the directly averted cases, we only want to consider those not treated
        transmitting_direct_male = row["direct_hiv_averted_Male"] * (
            1 - row["treatment_proportion_Male"]
        )
        transmitting_direct_lower_male = row[
            "direct_hiv_averted_lower_Male"
        ] * (1 - row["treatment_proportion_upper_Male"])
        transmitting_direct_upper_male = row[
            "direct_hiv_averted_upper_Male"
        ] * (1 - row["treatment_proportion_lower_Male"])
        transmitting_direct_female = row["direct_hiv_averted_Female"] * (
            1 - row["treatment_proportion_Female"]
        )
        transmitting_direct_lower_female = row[
            "direct_hiv_averted_lower_Female"
        ] * (1 - row["treatment_proportion_upper_Female"])
        transmitting_direct_upper_female = row[
            "direct_hiv_averted_upper_Female"
        ] * (1 - row["treatment_proportion_lower_Female"])

        cumulative_male[idx] += transmitting_direct_male
        cumulative_lower_male[idx] += transmitting_direct_lower_male
        cumulative_upper_male[idx] += transmitting_direct_upper_male
        cumulative_female[idx] += transmitting_direct_female
        cumulative_lower_female[idx] += transmitting_direct_lower_female
        cumulative_upper_female[idx] += transmitting_direct_upper_female
        # how many infections do these people generate
        cumulative_male[idx + 1] += np.sum(
            cumulative_female
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_F_M"],
            row[f"paf_{sti}_hiv_Female"],
            sti_hiv_transmission_increase_female,
        )
        cumulative_lower_male[idx + 1] += np.sum(
            cumulative_lower_female
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_lower_F_M"],
            row[f"paf_{sti}_hiv_lower_Female"],
            sti_hiv_transmission_increase_lower_female,
        )
        cumulative_upper_male[idx + 1] += np.sum(
            cumulative_upper_female
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_upper_F_M"],
            row[f"paf_{sti}_hiv_upper_Female"],
            sti_hiv_transmission_increase_upper_female,
        )
        cumulative_female[idx + 1] += np.sum(
            cumulative_male
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_M_F"],
            row[f"paf_{sti}_hiv_Male"],
            sti_hiv_transmission_increase_male,
        )
        cumulative_lower_female[idx + 1] += np.sum(
            cumulative_lower_male
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_lower_M_F"],
            row[f"paf_{sti}_hiv_lower_Male"],
            sti_hiv_transmission_increase_lower_male,
        )
        cumulative_upper_female[idx + 1] += np.sum(
            cumulative_upper_male
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_upper_M_F"],
            row[f"paf_{sti}_hiv_upper_Male"],
            sti_hiv_transmission_increase_upper_male,
        )
        # we also want to store these values in the indirects array
        indirect_male = np.append(indirect_male, cumulative_male[idx + 1])
        indirect_lower_male = np.append(
            indirect_lower_male, cumulative_lower_male[idx + 1]
        )
        indirect_upper_male = np.append(
            indirect_upper_male, cumulative_upper_male[idx + 1]
        )
        indirect_female = np.append(
            indirect_female, cumulative_female[idx + 1]
        )
        indirect_lower_female = np.append(
            indirect_lower_female, cumulative_lower_female[idx + 1]
        )
        indirect_upper_female = np.append(
            indirect_upper_female, cumulative_upper_female[idx + 1]
        )

    # assemble a dataframe to return
    indirects = pl.DataFrame(
        {
            "year": df["year"][:-1],
            "indirect_hiv_averted_Male": indirect_male,
            "indirect_hiv_averted_lower_Male": indirect_lower_male,
            "indirect_hiv_averted_upper_Male": indirect_upper_male,
            "indirect_hiv_averted_Female": indirect_female,
            "indirect_hiv_averted_lower_Female": indirect_lower_female,
            "indirect_hiv_averted_upper_Female": indirect_upper_female,
        }
    )
    # pivot back to long format
    indirects = (
        indirects.unpivot(index="year")
        .with_columns(
            sex=pl.col("variable").str.split("_").list.get(-1),
            variable=pl.col("variable").str.replace(r"_(Male|Female)$", ""),
        )
        .pivot(
            index=["year", "sex"],
            on="variable",
            values="value",
        )
    )

    return indirects


# run the analyses with different set ups of direct averted infections

# calculate direct averted cases by the policy switch from cipro to azithro
hiv = hiv.with_columns(
    gc_treatment_rate=pl.when(pl.col("sex") == "Male")
    .then(pl.lit(params["male_sti_treatment"]))
    .otherwise(pl.lit(params["female_sti_treatment"]))
)

hiv = hiv.with_columns(
    direct_hiv_averted_2016_gc_change=pl.col(
        "hiv_incidence_number_attributable_to_gc"
    )
    * pl.col("gc_treatment_rate"),
    direct_hiv_averted_2016_gc_change_lower=pl.col(
        "hiv_incidence_number_attributable_to_gc_lower"
    )
    * pl.col("gc_treatment_rate"),
    direct_hiv_averted_2016_gc_change_upper=pl.col(
        "hiv_incidence_number_attributable_to_gc_upper"
    )
    * pl.col("gc_treatment_rate"),
)

# for our upper bound scenario, multiply by some assumed fraction of STI cases averted
STIs = ["gc", "chlamydia", "syphilis", "trichomoniasis"]

# multiply by the fraction averted
hiv = hiv.with_columns(
    [
        (
            pl.col(f"hiv_incidence_number_attributable_to_{sti}{estimate}")
            * pl.lit(params["upper_bound_sdg"][sti])
        ).alias(
            f"hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
        )
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)

averted = []
for location in tqdm(hiv["location"].unique()):
    loc_df = hiv.filter(pl.col("location") == location)
    loc_df = loc_df.select(
        [
            *[
                "year",
                "sex",
                "region",
                "location",
                "hiv_prevalence_number",
                "hiv_prevalence_number_lower",
                "hiv_prevalence_number_upper",
                "hiv_incidence_number",
                "hiv_incidence_number_lower",
                "hiv_incidence_number_upper",
                "treatment_proportion",
                "treatment_proportion_lower",
                "treatment_proportion_upper",
                # direct averted cases
                "direct_hiv_averted_2016_gc_change",
                "direct_hiv_averted_2016_gc_change_lower",
                "direct_hiv_averted_2016_gc_change_upper",
            ],
            *[
                f"paf_{sti}_hiv{estimate}"
                for sti in STIs
                for estimate in ["", "_lower", "_upper"]
            ],
            *[
                f"hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
                for sti in STIs
                for estimate in ["", "_lower", "_upper"]
            ],
        ]
    )

    upper_bound = loc_df.filter(pl.col("year") >= 2010)
    upper_bound = upper_bound.drop(
        [
            "direct_hiv_averted_2016_gc_change",
            "direct_hiv_averted_2016_gc_change_lower",
            "direct_hiv_averted_2016_gc_change_upper",
        ]
    )

    # for the upper bound, calculate the number averted from each sti
    upper_bounds = []
    for sti in STIs:
        temp = upper_bound.select(
            [
                *[
                    "year",
                    "sex",
                    "region",
                    "location",
                    "hiv_prevalence_number",
                    "hiv_prevalence_number_lower",
                    "hiv_prevalence_number_upper",
                    "hiv_incidence_number",
                    "hiv_incidence_number_lower",
                    "hiv_incidence_number_upper",
                    "treatment_proportion",
                    "treatment_proportion_lower",
                    "treatment_proportion_upper",
                ],
                *[
                    f"hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
                    for estimate in ["", "_lower", "_upper"]
                ],
                *[
                    f"paf_{sti}_hiv{estimate}"
                    for estimate in ["", "_lower", "_upper"]
                ],
            ]
        )

        temp = temp.rename(
            {
                f"hiv_incidence_number_attributable_to_{sti}_upper_bound": "direct_hiv_averted",
                f"hiv_incidence_number_attributable_to_{sti}_upper_bound_lower": "direct_hiv_averted_lower",
                f"hiv_incidence_number_attributable_to_{sti}_upper_bound_upper": "direct_hiv_averted_upper",
            }
        )

        temp = calculate_indirect_averted_cases(
            temp,
            sti,
            sti_hiv_transmission_increase_male=sti_transmission_increase.filter(
                (pl.col("bacteria") == sti) & (pl.col("sex") == "Both")
            )["rr_mu"].item(),
            sti_hiv_transmission_increase_lower_male=sti_transmission_increase.filter(
                (pl.col("bacteria") == sti) & (pl.col("sex") == "Both")
            )["rr_lower"].item(),
            sti_hiv_transmission_increase_upper_male=sti_transmission_increase.filter(
                (pl.col("bacteria") == sti) & (pl.col("sex") == "Both")
            )["rr_upper"].item(),
            sti_hiv_transmission_increase_female=sti_transmission_increase.filter(
                (pl.col("bacteria") == sti) & (pl.col("sex") == "Women")
            )["rr_mu"].item(),
            sti_hiv_transmission_increase_lower_female=sti_transmission_increase.filter(
                (pl.col("bacteria") == sti) & (pl.col("sex") == "Women")
            )["rr_lower"].item(),
            sti_hiv_transmission_increase_upper_female=sti_transmission_increase.filter(
                (pl.col("bacteria") == sti) & (pl.col("sex") == "Women")
            )["rr_upper"].item(),
        )

        # rename the indirects columns
        temp = temp.rename(
            {
                "indirect_hiv_averted": f"indirect_hiv_averted_{sti}_upper_bound",
                "indirect_hiv_averted_lower": f"indirect_hiv_averted_{sti}_upper_bound_lower",
                "indirect_hiv_averted_upper": f"indirect_hiv_averted_{sti}_upper_bound_upper",
            }
        )

        # sort by year and sex so we can ensure everything is in the same order
        temp = temp.sort(by=["year", "sex"])

        if len(upper_bounds) == 0:
            temp = temp.with_columns(
                location=pl.lit(location),
                region=pl.lit(loc_df[0, "region"]),
            )
        else:
            # drop year and sex to allow for concatenation
            temp = temp.drop(["year", "sex"])
        # append
        upper_bounds.append(temp)

    # concatenate all the upper bounds
    upper_bound = pl.concat(upper_bounds, how="horizontal")

    first_line_treatment_change = loc_df.filter(pl.col("year") >= 2016)
    # drop the columns we don't need
    first_line_treatment_change = first_line_treatment_change.select(
        [
            "year",
            "sex",
            "location",
            "region",
            "hiv_prevalence_number",
            "hiv_prevalence_number_lower",
            "hiv_prevalence_number_upper",
            "hiv_incidence_number",
            "hiv_incidence_number_lower",
            "hiv_incidence_number_upper",
            "direct_hiv_averted_2016_gc_change",
            "direct_hiv_averted_2016_gc_change_lower",
            "direct_hiv_averted_2016_gc_change_upper",
            "treatment_proportion",
            "treatment_proportion_lower",
            "treatment_proportion_upper",
            *[
                f"paf_gc_hiv{estimate}"
                for estimate in ["", "_lower", "_upper"]
            ],
        ]
    )
    # rename the direct averted columns
    first_line_treatment_change = first_line_treatment_change.rename(
        {
            "direct_hiv_averted_2016_gc_change": "direct_hiv_averted",
            "direct_hiv_averted_2016_gc_change_lower": "direct_hiv_averted_lower",
            "direct_hiv_averted_2016_gc_change_upper": "direct_hiv_averted_upper",
        }
    )
    first_line_treatment_change = calculate_indirect_averted_cases(
        first_line_treatment_change,
        "gc",
        sti_hiv_transmission_increase_male=sti_transmission_increase.filter(
            (pl.col("bacteria") == "gc") & (pl.col("sex") == "Both")
        )["rr_mu"].item(),
        sti_hiv_transmission_increase_lower_male=sti_transmission_increase.filter(
            (pl.col("bacteria") == "gc") & (pl.col("sex") == "Both")
        )["rr_lower"].item(),
        sti_hiv_transmission_increase_upper_male=sti_transmission_increase.filter(
            (pl.col("bacteria") == "gc") & (pl.col("sex") == "Both")
        )["rr_upper"].item(),
        sti_hiv_transmission_increase_female=sti_transmission_increase.filter(
            (pl.col("bacteria") == "gc") & (pl.col("sex") == "Women")
        )["rr_mu"].item(),
        sti_hiv_transmission_increase_lower_female=sti_transmission_increase.filter(
            (pl.col("bacteria") == "gc") & (pl.col("sex") == "Women")
        )["rr_lower"].item(),
        sti_hiv_transmission_increase_upper_female=sti_transmission_increase.filter(
            (pl.col("bacteria") == "gc") & (pl.col("sex") == "Women")
        )["rr_upper"].item(),
    )
    first_line_treatment_change = first_line_treatment_change.with_columns(
        location=pl.lit(location),
        region=pl.lit(loc_df[0, "region"]),
    )
    # rename the indirects columns
    first_line_treatment_change = first_line_treatment_change.rename(
        {
            "indirect_hiv_averted": "indirect_hiv_averted_2016_gc_change",
            "indirect_hiv_averted_lower": "indirect_hiv_averted_2016_gc_change_lower",
            "indirect_hiv_averted_upper": "indirect_hiv_averted_2016_gc_change_upper",
        }
    )

    scenarios = upper_bound.join(
        first_line_treatment_change,
        on=["year", "sex", "location", "region"],
        how="left",
    )

    averted.append(scenarios)

averted = pl.concat(averted)
# concatenate with hiv dataframe
hiv = hiv.join(averted, on=["year", "sex", "location", "region"], how="inner")

# now sum direct and indirects
# we need to do two kinds of summing
# sum the direct + indirect for each sti
hiv = hiv.with_columns(
    [
        (
            pl.col(f"indirect_hiv_averted_{sti}_upper_bound{estimate}")
            + pl.col(
                f"hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
            )
        ).alias(f"hiv_averted_{sti}_upper_bound{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)
# we also need to make a total direct and indirect sum across all stis
hiv = hiv.with_columns(
    [
        pl.sum_horizontal(
            [pl.col(f"indirect_hiv_averted_{sti}_upper_bound") for sti in STIs]
        ).alias("indirect_hiv_averted_upper_bound"),
        pl.sum_horizontal(
            [
                pl.col(f"indirect_hiv_averted_{sti}_upper_bound_lower")
                for sti in STIs
            ]
        ).alias("indirect_hiv_averted_upper_bound_lower"),
        pl.sum_horizontal(
            [
                pl.col(f"indirect_hiv_averted_{sti}_upper_bound_upper")
                for sti in STIs
            ]
        ).alias("indirect_hiv_averted_upper_bound_upper"),
        # directs
        pl.sum_horizontal(
            [
                pl.col(
                    f"hiv_incidence_number_attributable_to_{sti}_upper_bound"
                )
                for sti in STIs
            ]
        ).alias("direct_hiv_averted_upper_bound"),
        pl.sum_horizontal(
            [
                pl.col(
                    f"hiv_incidence_number_attributable_to_{sti}_upper_bound_lower"
                )
                for sti in STIs
            ]
        ).alias("direct_hiv_averted_upper_bound_lower"),
        pl.sum_horizontal(
            [
                pl.col(
                    f"hiv_incidence_number_attributable_to_{sti}_upper_bound_upper"
                )
                for sti in STIs
            ]
        ).alias("direct_hiv_averted_upper_bound_upper"),
    ],
)
# and finally, we need to mega sum the direct + indirects across all stis
hiv = hiv.with_columns(
    hiv_averted_upper_bound=pl.col("direct_hiv_averted_upper_bound")
    + pl.col("indirect_hiv_averted_upper_bound"),
    hiv_averted_upper_bound_lower=pl.col(
        "direct_hiv_averted_upper_bound_lower"
    )
    + pl.col("indirect_hiv_averted_upper_bound_lower"),
    hiv_averted_upper_bound_upper=pl.col(
        "direct_hiv_averted_upper_bound_upper"
    )
    + pl.col("indirect_hiv_averted_upper_bound_upper"),
)

# sum the direct + indirect for the 2016 treatment change scenario
hiv = hiv.with_columns(
    hiv_averted_2016_gc_change=pl.col("direct_hiv_averted_2016_gc_change")
    + pl.col("indirect_hiv_averted_2016_gc_change"),
    hiv_averted_2016_gc_change_lower=pl.col(
        "direct_hiv_averted_2016_gc_change_lower"
    )
    + pl.col("indirect_hiv_averted_2016_gc_change_lower"),
    hiv_averted_2016_gc_change_upper=pl.col(
        "direct_hiv_averted_2016_gc_change_upper"
    )
    + pl.col("indirect_hiv_averted_2016_gc_change_upper"),
)

# save csv
hiv.write_csv(os.path.join(output_dir, "hiv_averted.csv"))
