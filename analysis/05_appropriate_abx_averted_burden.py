import os

import numpy as np
import polars as pl
import yaml
from tqdm.auto import tqdm

from averted_burden import conditional_exposure

data_dir = "data"
output_dir = "outputs"

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
        hiv_prevalence=pl.col("unaids_prevalence_year_end_number"),
        hiv_prevalence_lower=pl.col("unaids_prevalence_year_end_number_lower"),
        hiv_prevalence_upper=pl.col("unaids_prevalence_year_end_number_upper"),
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
            "unaids_incidence_number",
            "unaids_incidence_number_lower",
            "unaids_incidence_number_upper",
            "unaids_prevalence_year_end_number",
            "unaids_prevalence_year_end_number_lower",
            "unaids_prevalence_year_end_number_upper",
            "treatment_proportion",
            "treatment_proportion_lower",
            "treatment_proportion_upper",
        ],
        index="year",
    )
    df = df.sort(by="year")

    # now assess how many incident infections each prevalent infection generated
    # in the next year

    df = df.with_columns(
        transmission_rate_M_F=pl.col("unaids_incidence_number_Female").shift(
            -1
        )
        / (
            pl.col("unaids_prevalence_year_end_number_Male")
            * (1 - pl.col("treatment_proportion_Male"))
        ),
        transmission_rate_lower_M_F=pl.col(
            "unaids_incidence_number_lower_Female"
        ).shift(-1)
        / (
            pl.col("unaids_prevalence_year_end_number_lower_Male")
            * (1 - pl.col("treatment_proportion_lower_Male"))
        ),
        transmission_rate_upper_M_F=pl.col(
            "unaids_incidence_number_upper_Female"
        ).shift(-1)
        / (
            pl.col("unaids_prevalence_year_end_number_upper_Male")
            * (1 - pl.col("treatment_proportion_upper_Male"))
        ),
        transmission_rate_F_M=pl.col("unaids_incidence_number_Male").shift(-1)
        / (
            pl.col("unaids_prevalence_year_end_number_Female")
            * (1 - pl.col("treatment_proportion_Female"))
        ),
        transmission_rate_lower_F_M=pl.col(
            "unaids_incidence_number_lower_Male"
        ).shift(-1)
        / (
            pl.col("unaids_prevalence_year_end_number_lower_Female")
            * (1 - pl.col("treatment_proportion_lower_Female"))
        ),
        transmission_rate_upper_F_M=pl.col(
            "unaids_incidence_number_upper_Male"
        ).shift(-1)
        / (
            pl.col("unaids_prevalence_year_end_number_upper_Female")
            * (1 - pl.col("treatment_proportion_upper_Female"))
        ),
        transmission_rate_MSM=pl.col("unaids_incidence_number_MSM").shift(-1)
        / (
            pl.col("unaids_prevalence_year_end_number_MSM")
            * (1 - pl.col("treatment_proportion_MSM"))
        ),
        transmission_rate_lower_MSM=pl.col(
            "unaids_incidence_number_lower_MSM"
        ).shift(-1)
        / (
            pl.col("unaids_prevalence_year_end_number_lower_MSM")
            * (1 - pl.col("treatment_proportion_lower_MSM"))
        ),
        transmission_rate_upper_MSM=pl.col(
            "unaids_incidence_number_upper_MSM"
        ).shift(-1)
        / (
            pl.col("unaids_prevalence_year_end_number_upper_MSM")
            * (1 - pl.col("treatment_proportion_upper_MSM"))
        ),
    )

    # when the treatment rate is 0, we can get inf values for the transmission rate, so replace these with 0s
    # but in reality we know the (effective) transmission rate is 0
    df = df.with_columns(
        [
            pl.col(col).fill_nan(0).replace([float("inf"), float("-inf")], 0)
            for col in df.columns
            if col.startswith("transmission_rate")
        ]
    )

    # now we need to calculate the additional infections averted in future years
    indirect_male = np.array([])
    indirect_lower_male = np.array([])
    indirect_upper_male = np.array([])
    indirect_female = np.array([])
    indirect_lower_female = np.array([])
    indirect_upper_female = np.array([])
    indirect_msm = np.array([])
    indirect_lower_msm = np.array([])
    indirect_upper_msm = np.array([])
    cumulative_male = np.zeros(df.height)
    cumulative_lower_male = np.zeros(df.height)
    cumulative_upper_male = np.zeros(df.height)
    cumulative_female = np.zeros(df.height)
    cumulative_lower_female = np.zeros(df.height)
    cumulative_upper_female = np.zeros(df.height)
    cumulative_msm = np.zeros(df.height)
    cumulative_lower_msm = np.zeros(df.height)
    cumulative_upper_msm = np.zeros(df.height)
    # before propagating, clean up the data for hiv-to-aids progression
    # data from Survival rate of AIDS disease and mortality in HIV-infected patients: a meta-analysis
    # Poorolajal et al., 2016
    # we want to make these conditional values by dividing the last value in the
    # sequence
    # to do this, we need to make sure that there are no increases in the sequence
    # this happens in the uncertainty bounds
    hiv_to_aids = np.minimum.accumulate(hiv_to_aids)
    hiv_to_aids_upper = np.minimum.accumulate(hiv_to_aids_upper)
    hiv_to_aids_lower = np.minimum.accumulate(hiv_to_aids_lower)
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
    # finally, because we multiply each value by the people in that year but these
    # are two year survival numbers, we should take the sqrt of these values to
    # get by year survival numbers
    hiv_to_aids = np.sqrt(hiv_to_aids)
    hiv_to_aids_upper = np.sqrt(hiv_to_aids_upper)
    hiv_to_aids_lower = np.sqrt(hiv_to_aids_lower)
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
        cumulative_msm *= hiv_to_aids_temp
        cumulative_lower_msm *= hiv_to_aids_lower_temp
        cumulative_upper_msm *= hiv_to_aids_upper_temp

        # of the directly averted cases, we only want to consider those not treated
        transmitting_direct_male = (row["direct_hiv_averted_Male"]) * (
            1 - row["treatment_proportion_Male"]
        )
        transmitting_direct_lower_male = (
            row["direct_hiv_averted_lower_Male"]
        ) * (1 - row["treatment_proportion_upper_Male"])
        transmitting_direct_upper_male = (
            row["direct_hiv_averted_upper_Male"]
        ) * (1 - row["treatment_proportion_lower_Male"])
        transmitting_direct_female = row["direct_hiv_averted_Female"] * (
            1 - row["treatment_proportion_Female"]
        )
        transmitting_direct_lower_female = row[
            "direct_hiv_averted_lower_Female"
        ] * (1 - row["treatment_proportion_upper_Female"])
        transmitting_direct_upper_female = row[
            "direct_hiv_averted_upper_Female"
        ] * (1 - row["treatment_proportion_lower_Female"])
        transmitting_direct_msm = row["direct_hiv_averted_MSM"] * (
            1 - row["treatment_proportion_MSM"]
        )
        transmitting_direct_lower_msm = row["direct_hiv_averted_lower_MSM"] * (
            1 - row["treatment_proportion_upper_MSM"]
        )
        transmitting_direct_upper_msm = row["direct_hiv_averted_upper_MSM"] * (
            1 - row["treatment_proportion_lower_MSM"]
        )

        cumulative_male[idx] += transmitting_direct_male
        cumulative_lower_male[idx] += transmitting_direct_lower_male
        cumulative_upper_male[idx] += transmitting_direct_upper_male
        cumulative_female[idx] += transmitting_direct_female
        cumulative_lower_female[idx] += transmitting_direct_lower_female
        cumulative_upper_female[idx] += transmitting_direct_upper_female
        cumulative_msm[idx] += transmitting_direct_msm
        cumulative_lower_msm[idx] += transmitting_direct_lower_msm
        cumulative_upper_msm[idx] += transmitting_direct_upper_msm
        # how many infections do these people generate
        cumulative_male[idx + 1] += np.sum(
            cumulative_female[: (idx + 1)]
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_F_M"],
            row[f"paf_{sti}_hiv_Female"],
            sti_hiv_transmission_increase_female,
            allow_invalid=True,
        )
        cumulative_lower_male[idx + 1] += np.sum(
            cumulative_lower_female[: (idx + 1)]
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_lower_F_M"],
            row[f"paf_{sti}_hiv_lower_Female"],
            sti_hiv_transmission_increase_lower_female,
            allow_invalid=True,
        )
        cumulative_upper_male[idx + 1] += np.sum(
            cumulative_upper_female[: (idx + 1)]
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_upper_F_M"],
            row[f"paf_{sti}_hiv_upper_Female"],
            sti_hiv_transmission_increase_upper_female,
            allow_invalid=True,
        )
        cumulative_msm[idx + 1] += np.sum(
            cumulative_msm[: (idx + 1)]
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_MSM"],
            row[f"paf_{sti}_hiv_MSM"],
            sti_hiv_transmission_increase_male,
            allow_invalid=True,
        )
        cumulative_lower_msm[idx + 1] += np.sum(
            cumulative_lower_msm[: (idx + 1)]
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_lower_MSM"],
            row[f"paf_{sti}_hiv_lower_MSM"],
            sti_hiv_transmission_increase_lower_male,
            allow_invalid=True,
        )
        cumulative_upper_msm[idx + 1] += np.sum(
            cumulative_upper_msm[: (idx + 1)]
        ) * conditional_exposure.p_a_given_b(
            row["transmission_rate_upper_MSM"],
            row[f"paf_{sti}_hiv_upper_MSM"],
            sti_hiv_transmission_increase_upper_male,
            allow_invalid=True,
        )
        cumulative_female[idx + 1] += np.sum(
            cumulative_male[: (idx + 1)]
            * conditional_exposure.p_a_given_b(
                row["transmission_rate_M_F"],
                row[f"paf_{sti}_hiv_Male"],
                sti_hiv_transmission_increase_male,
                allow_invalid=True,
            )
        )
        cumulative_lower_female[idx + 1] += np.sum(
            cumulative_lower_male[: (idx + 1)]
            * conditional_exposure.p_a_given_b(
                row["transmission_rate_lower_M_F"],
                row[f"paf_{sti}_hiv_lower_Male"],
                sti_hiv_transmission_increase_lower_male,
                allow_invalid=True,
            )
        )
        cumulative_upper_female[idx + 1] += np.sum(
            cumulative_upper_male[: (idx + 1)]
            * conditional_exposure.p_a_given_b(
                row["transmission_rate_upper_M_F"],
                row[f"paf_{sti}_hiv_upper_Male"],
                sti_hiv_transmission_increase_upper_male,
                allow_invalid=True,
            )
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
        indirect_msm = np.append(indirect_msm, cumulative_msm[idx + 1])
        indirect_lower_msm = np.append(
            indirect_lower_msm, cumulative_lower_msm[idx + 1]
        )
        indirect_upper_msm = np.append(
            indirect_upper_msm, cumulative_upper_msm[idx + 1]
        )

    # assemble a dataframe to return
    indirects = pl.DataFrame(
        {
            "year": df["year"][1:],
            "indirect_hiv_averted_Male": indirect_male,
            "indirect_hiv_averted_lower_Male": indirect_lower_male,
            "indirect_hiv_averted_upper_Male": indirect_upper_male,
            "indirect_hiv_averted_Female": indirect_female,
            "indirect_hiv_averted_lower_Female": indirect_lower_female,
            "indirect_hiv_averted_upper_Female": indirect_upper_female,
            "indirect_hiv_averted_MSM": indirect_msm,
            "indirect_hiv_averted_lower_MSM": indirect_lower_msm,
            "indirect_hiv_averted_upper_MSM": indirect_upper_msm,
        }
    )
    # pivot back to long format
    indirects = (
        indirects.unpivot(index="year")
        .with_columns(
            sex=pl.col("variable").str.split("_").list.get(-1),
            variable=pl.col("variable").str.replace(
                r"_(Male|Female|MSM)$", ""
            ),
        )
        .pivot(
            index=["year", "sex"],
            on="variable",
            values="value",
        )
    )

    return indirects


# run the analyses with different set ups of direct averted infections

# calculate direct averted cases by the WHO treatment guideline switch from
# cipro to azithro
# What we need to do is calculate the direct HIV cases that there would have
# been had there not been a policy change
# What we observe is the HIV cases attributable to GC
# We know that the cases we observe are either cases that did not seek
# treatment, or if they did seek treatment, the treatment failed.
# Observed HIV attributable to GC = (1-t) * HIV cases + t * (1 - e) * HIV cases
# where t is the treatment seeking rate and e is the effectiveness of the treatment
# This lets us write that the total burden of HIV attributable to GC is
# observed / ((1-t) + t * (1-e))
# we know t (params.yml -- there's a foundation for these numbers in terms of
# fraction of people that have symptoms and then assuming some fraction of
# people with symptoms get treated)

sti_symptom_params = params["sti_symptoms"]
gc_symptom_params = sti_symptom_params["gc"]
hiv = hiv.with_columns(
    gc_treatment_rate=pl.when(pl.col("sex") == "Female")
    .then(gc_symptom_params["female"] * pl.col("frac_sought_treatment"))
    .otherwise(gc_symptom_params["male"] * pl.col("frac_sought_treatment"))
)

# e is a bit harder. We know the resistance rates, but we don't know the overlap
# between the resistance rates
# we can make an upper bound estimate by being the minimum of the resistances
# this assumes that all people who are resistant to the abx with the least
# resistance would have been resistant to all abxs that came before that
hiv = hiv.with_columns(
    gc_treatment_failure=pl.min_horizontal(
        ["Azithromycin", "Cefixime", "Ceftriaxone"]
    ),
    gc_treatment_failure_lower=pl.min_horizontal(
        ["Azithromycin_lower", "Cefixime_lower", "Ceftriaxone_lower"]
    ),
    gc_treatment_failure_upper=pl.min_horizontal(
        ["Azithromycin_upper", "Cefixime_upper", "Ceftriaxone_upper"]
    ),
)

# pre-2016, ciprofloxacin was the first-line treatment for gonorrhea, so
# treatment failure was entirely determined by ciprofloxacin resistance.
# post-2016, the WHO switched first-line treatment so we use the minimum
# resistance across azithromycin, cefixime, and ceftriaxone (as above).
# we stitch these together with a year conditional so that the decomposition
# into untreated vs resistance uses the era-appropriate failure rate.
hiv = hiv.with_columns(
    gc_treatment_failure_era=pl.when(pl.col("year") < 2016)
    .then(pl.col("Ciprofloxacin"))
    .otherwise(pl.col("gc_treatment_failure")),
    gc_treatment_failure_era_lower=pl.when(pl.col("year") < 2016)
    .then(pl.col("Ciprofloxacin_lower"))
    .otherwise(pl.col("gc_treatment_failure_lower")),
    gc_treatment_failure_era_upper=pl.when(pl.col("year") < 2016)
    .then(pl.col("Ciprofloxacin_upper"))
    .otherwise(pl.col("gc_treatment_failure_upper")),
)

# recall our formula -- this allows us to estimate the total number of HIV
# cases that would have been attributable to GC had there not been the treatment
# change
# unobserved = observed / ((1-t) + t * (1-e))
# what we are counting as the direct averted cases is then this unobserved number
# minus the observed number

hiv = hiv.with_columns(
    direct_hiv_averted_2016_gc_change=(
        pl.col("unaids_hiv_incidence_number_attributable_to_gc")
        / (
            (1 - pl.col("gc_treatment_rate"))
            + pl.col("gc_treatment_rate") * pl.col("gc_treatment_failure")
        )
    )
    - pl.col("unaids_hiv_incidence_number_attributable_to_gc"),
    direct_hiv_averted_2016_gc_change_lower=(
        pl.col("unaids_hiv_incidence_number_attributable_to_gc_lower")
        / (
            (1 - pl.col("gc_treatment_rate"))
            + pl.col("gc_treatment_rate")
            * pl.col("gc_treatment_failure_lower")
        )
    )
    - pl.col("unaids_hiv_incidence_number_attributable_to_gc_lower"),
    # upper bound
    direct_hiv_averted_2016_gc_change_upper=(
        pl.col("unaids_hiv_incidence_number_attributable_to_gc_upper")
        / (
            (1 - pl.col("gc_treatment_rate"))
            + pl.col("gc_treatment_rate")
            * pl.col("gc_treatment_failure_upper")
        )
    )
    - pl.col("unaids_hiv_incidence_number_attributable_to_gc_upper"),
)

# decompose the observed gc-attributable HIV burden into the share caused by
# lack of treatment access vs antibiotic resistance, for all years >= 2008.
# total burden = observed / ((1-t) + t*(1-e))  [same formula as above]
# untreated share = (1-t) * total_burden
# resistance share = t * (1-e) * total_burden
# the era-appropriate failure rate e is used so that pre-2016 uses cipro
# resistance only, and post-2016 uses min(azithro, cefixime, ceftriaxone)
hiv = hiv.with_columns(
    # total gc-attributable HIV burden (what would have been observed with
    # perfect treatment coverage and zero resistance)
    gc_hiv_total_burden=pl.col(
        "unaids_hiv_incidence_number_attributable_to_gc"
    )
    / (
        (1 - pl.col("gc_treatment_rate"))
        + pl.col("gc_treatment_rate") * pl.col("gc_treatment_failure_era")
    ),
    gc_hiv_total_burden_lower=pl.col(
        "unaids_hiv_incidence_number_attributable_to_gc_lower"
    )
    / (
        (1 - pl.col("gc_treatment_rate"))
        + pl.col("gc_treatment_rate")
        * pl.col("gc_treatment_failure_era_lower")
    ),
    gc_hiv_total_burden_upper=pl.col(
        "unaids_hiv_incidence_number_attributable_to_gc_upper"
    )
    / (
        (1 - pl.col("gc_treatment_rate"))
        + pl.col("gc_treatment_rate")
        * pl.col("gc_treatment_failure_era_upper")
    ),
)

hiv = hiv.with_columns(
    # direct HIV cases attributable to lack of treatment access:
    # these are the cases among people who never sought care
    direct_hiv_averted_gc_untreated=pl.when(pl.col("year") >= 2008)
    .then((1 - pl.col("gc_treatment_rate")) * pl.col("gc_hiv_total_burden"))
    .otherwise(pl.lit(None)),
    direct_hiv_averted_gc_untreated_lower=pl.when(pl.col("year") >= 2008)
    .then(
        (1 - pl.col("gc_treatment_rate")) * pl.col("gc_hiv_total_burden_lower")
    )
    .otherwise(pl.lit(None)),
    direct_hiv_averted_gc_untreated_upper=pl.when(pl.col("year") >= 2008)
    .then(
        (1 - pl.col("gc_treatment_rate")) * pl.col("gc_hiv_total_burden_upper")
    )
    .otherwise(pl.lit(None)),
    # direct HIV cases attributable to antibiotic resistance:
    # these are the cases among people who sought care but treatment failed
    direct_hiv_averted_gc_resistance=pl.when(pl.col("year") >= 2008)
    .then(
        pl.col("gc_treatment_rate")
        * pl.col("gc_treatment_failure_era")
        * pl.col("gc_hiv_total_burden")
    )
    .otherwise(pl.lit(None)),
    direct_hiv_averted_gc_resistance_lower=pl.when(pl.col("year") >= 2008)
    .then(
        pl.col("gc_treatment_rate")
        * pl.col("gc_treatment_failure_era_lower")
        * pl.col("gc_hiv_total_burden_lower")
    )
    .otherwise(pl.lit(None)),
    direct_hiv_averted_gc_resistance_upper=pl.when(pl.col("year") >= 2008)
    .then(
        pl.col("gc_treatment_rate")
        * pl.col("gc_treatment_failure_era_upper")
        * pl.col("gc_hiv_total_burden_upper")
    )
    .otherwise(pl.lit(None)),
)

# for our abx access scenario, we use a similar formulation, but we now apply it
# to all STIs and we assume that treatment failure for the other STIs is zero?
# recall that the formula is unobserved = observed / (1 - t) if treatment failure is 0

# for our access (upper bound) scenario, calculate averted number because of
# treatment
STIs = ["gc", "chlamydia", "syphilis", "trichomoniasis"]

# First compute all treatment rates
for sti in STIs:
    hiv = hiv.with_columns(
        pl.when(pl.col("sex") == "Female")
        .then(
            sti_symptom_params[sti]["female"] * pl.col("frac_sought_treatment")
        )
        .otherwise(
            sti_symptom_params[sti]["male"] * pl.col("frac_sought_treatment")
        )
        .alias(f"{sti}_treatment_rate")
    )


# Then calculate upper bound averted cases
hiv = hiv.with_columns(
    [
        (
            (
                pl.col(
                    f"unaids_hiv_incidence_number_attributable_to_{sti}{estimate}"
                )
                / (1 - pl.col(f"{sti}_treatment_rate"))
            )
            - (
                pl.col(
                    f"unaids_hiv_incidence_number_attributable_to_{sti}{estimate}"
                )
            )
        ).alias(
            f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
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
                "unaids_prevalence_year_end_number",
                "unaids_prevalence_year_end_number_lower",
                "unaids_prevalence_year_end_number_upper",
                "unaids_incidence_number",
                "unaids_incidence_number_lower",
                "unaids_incidence_number_upper",
                "treatment_proportion",
                "treatment_proportion_lower",
                "treatment_proportion_upper",
                # direct averted cases
                "direct_hiv_averted_2016_gc_change",
                "direct_hiv_averted_2016_gc_change_lower",
                "direct_hiv_averted_2016_gc_change_upper",
                # direct averted cases decomposed into untreated vs resistance
                "direct_hiv_averted_gc_untreated",
                "direct_hiv_averted_gc_untreated_lower",
                "direct_hiv_averted_gc_untreated_upper",
                "direct_hiv_averted_gc_resistance",
                "direct_hiv_averted_gc_resistance_lower",
                "direct_hiv_averted_gc_resistance_upper",
            ],
            *[
                f"paf_{sti}_hiv{estimate}"
                for sti in STIs
                for estimate in ["", "_lower", "_upper"]
            ],
            *[
                f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
                for sti in STIs
                for estimate in ["", "_lower", "_upper"]
            ],
        ]
    )

    # we need to calculate the upper bound twice -- once for
    # >= 2000, and once for >= 2025
    upper_bounds_past_and_future = []
    for year_cutoff in [2000, 2025]:
        upper_bound = loc_df.filter((pl.col("year") >= year_cutoff))
        upper_bound = upper_bound.drop(
            [
                "direct_hiv_averted_2016_gc_change",
                "direct_hiv_averted_2016_gc_change_lower",
                "direct_hiv_averted_2016_gc_change_upper",
                "direct_hiv_averted_gc_untreated",
                "direct_hiv_averted_gc_untreated_lower",
                "direct_hiv_averted_gc_untreated_upper",
                "direct_hiv_averted_gc_resistance",
                "direct_hiv_averted_gc_resistance_lower",
                "direct_hiv_averted_gc_resistance_upper",
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
                        "unaids_prevalence_year_end_number",
                        "unaids_prevalence_year_end_number_lower",
                        "unaids_prevalence_year_end_number_upper",
                        "unaids_incidence_number",
                        "unaids_incidence_number_lower",
                        "unaids_incidence_number_upper",
                        "treatment_proportion",
                        "treatment_proportion_lower",
                        "treatment_proportion_upper",
                    ],
                    *[
                        f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
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
                    f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound": "direct_hiv_averted",
                    f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound_lower": "direct_hiv_averted_lower",
                    f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound_upper": "direct_hiv_averted_upper",
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
        upper_bounds_past_and_future.append(upper_bound)

    first_line_treatment_change = loc_df.filter(
        (pl.col("year") >= 2016) & (pl.col("year") <= 2023)
    )
    # drop the columns we don't need
    first_line_treatment_change = first_line_treatment_change.select(
        [
            "year",
            "sex",
            "location",
            "region",
            "unaids_prevalence_year_end_number",
            "unaids_prevalence_year_end_number_lower",
            "unaids_prevalence_year_end_number_upper",
            "unaids_incidence_number",
            "unaids_incidence_number_lower",
            "unaids_incidence_number_upper",
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

    # calculate indirect averted cases for the untreated pathway (>= 2008),
    # i.e. the HIV cases that propagated forward because the index case never
    # sought care
    gc_untreated = (
        loc_df.filter(pl.col("year") >= 2008)
        .filter(pl.col("year") <= 2023)
        .select(
            [
                "year",
                "sex",
                "unaids_prevalence_year_end_number",
                "unaids_prevalence_year_end_number_lower",
                "unaids_prevalence_year_end_number_upper",
                "unaids_incidence_number",
                "unaids_incidence_number_lower",
                "unaids_incidence_number_upper",
                "treatment_proportion",
                "treatment_proportion_lower",
                "treatment_proportion_upper",
                "direct_hiv_averted_gc_untreated",
                "direct_hiv_averted_gc_untreated_lower",
                "direct_hiv_averted_gc_untreated_upper",
                *[
                    f"paf_gc_hiv{estimate}"
                    for estimate in ["", "_lower", "_upper"]
                ],
            ]
        )
    )
    gc_untreated = gc_untreated.rename(
        {
            "direct_hiv_averted_gc_untreated": "direct_hiv_averted",
            "direct_hiv_averted_gc_untreated_lower": "direct_hiv_averted_lower",
            "direct_hiv_averted_gc_untreated_upper": "direct_hiv_averted_upper",
        }
    )
    gc_untreated = calculate_indirect_averted_cases(
        gc_untreated,
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
    gc_untreated = gc_untreated.rename(
        {
            "indirect_hiv_averted": "indirect_hiv_averted_gc_untreated",
            "indirect_hiv_averted_lower": "indirect_hiv_averted_gc_untreated_lower",
            "indirect_hiv_averted_upper": "indirect_hiv_averted_gc_untreated_upper",
        }
    )
    gc_untreated = gc_untreated.with_columns(
        location=pl.lit(location),
        region=pl.lit(loc_df[0, "region"]),
    )

    # calculate indirect averted cases for the resistance pathway (>= 2008),
    # i.e. the HIV cases that propagated forward because the index case sought
    # care but treatment failed due to antibiotic resistance
    gc_resistance = (
        loc_df.filter(pl.col("year") >= 2008)
        .filter(pl.col("year") <= 2023)
        .select(
            [
                "year",
                "sex",
                "unaids_prevalence_year_end_number",
                "unaids_prevalence_year_end_number_lower",
                "unaids_prevalence_year_end_number_upper",
                "unaids_incidence_number",
                "unaids_incidence_number_lower",
                "unaids_incidence_number_upper",
                "treatment_proportion",
                "treatment_proportion_lower",
                "treatment_proportion_upper",
                "direct_hiv_averted_gc_resistance",
                "direct_hiv_averted_gc_resistance_lower",
                "direct_hiv_averted_gc_resistance_upper",
                *[
                    f"paf_gc_hiv{estimate}"
                    for estimate in ["", "_lower", "_upper"]
                ],
            ]
        )
    )
    gc_resistance = gc_resistance.rename(
        {
            "direct_hiv_averted_gc_resistance": "direct_hiv_averted",
            "direct_hiv_averted_gc_resistance_lower": "direct_hiv_averted_lower",
            "direct_hiv_averted_gc_resistance_upper": "direct_hiv_averted_upper",
        }
    )
    gc_resistance = calculate_indirect_averted_cases(
        gc_resistance,
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
    gc_resistance = gc_resistance.rename(
        {
            "indirect_hiv_averted": "indirect_hiv_averted_gc_resistance",
            "indirect_hiv_averted_lower": "indirect_hiv_averted_gc_resistance_lower",
            "indirect_hiv_averted_upper": "indirect_hiv_averted_gc_resistance_upper",
        }
    )
    gc_resistance = gc_resistance.with_columns(
        location=pl.lit(location),
        region=pl.lit(loc_df[0, "region"]),
    )

    upper_bound = upper_bounds_past_and_future[0].join(
        upper_bounds_past_and_future[1],
        on=["year", "sex", "location", "region"],
        how="left",
        suffix="_future",
    )
    scenarios = upper_bound.join(
        first_line_treatment_change,
        on=["year", "sex", "location", "region"],
        how="left",
    )
    # join the untreated and resistance decomposition indirects
    scenarios = scenarios.join(
        gc_untreated,
        on=["year", "sex", "location", "region"],
        how="left",
    )
    scenarios = scenarios.join(
        gc_resistance,
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
                f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
            )
        ).alias(f"hiv_averted_{sti}_upper_bound{estimate}")
        for sti in STIs
        for estimate in ["", "_lower", "_upper"]
    ]
)


hiv = hiv.with_columns(
    [
        (
            pl.col(f"indirect_hiv_averted_{sti}_upper_bound{estimate}_future")
            + pl.col(
                f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound{estimate}"
            )
        ).alias(f"hiv_averted_{sti}_upper_bound_future{estimate}")
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
                    f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound"
                )
                for sti in STIs
            ]
        ).alias("direct_hiv_averted_upper_bound"),
        pl.sum_horizontal(
            [
                pl.col(
                    f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound_lower"
                )
                for sti in STIs
            ]
        ).alias("direct_hiv_averted_upper_bound_lower"),
        pl.sum_horizontal(
            [
                pl.col(
                    f"unaids_hiv_incidence_number_attributable_to_{sti}_upper_bound_upper"
                )
                for sti in STIs
            ]
        ).alias("direct_hiv_averted_upper_bound_upper"),
    ],
)

hiv = hiv.with_columns(
    [
        pl.sum_horizontal(
            [
                pl.col(f"indirect_hiv_averted_{sti}_upper_bound_future")
                for sti in STIs
            ]
        ).alias("indirect_hiv_averted_upper_bound_future"),
        pl.sum_horizontal(
            [
                pl.col(f"indirect_hiv_averted_{sti}_upper_bound_lower_future")
                for sti in STIs
            ]
        ).alias("indirect_hiv_averted_upper_bound_lower_future"),
        pl.sum_horizontal(
            [
                pl.col(f"indirect_hiv_averted_{sti}_upper_bound_upper_future")
                for sti in STIs
            ]
        ).alias("indirect_hiv_averted_upper_bound_upper_future"),
    ]
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

hiv = hiv.with_columns(
    hiv_averted_upper_bound_future=pl.col("direct_hiv_averted_upper_bound")
    + pl.col("indirect_hiv_averted_upper_bound_future"),
    hiv_averted_upper_bound_future_lower=pl.col(
        "direct_hiv_averted_upper_bound_lower"
    )
    + pl.col("indirect_hiv_averted_upper_bound_lower_future"),
    hiv_averted_upper_bound_future_upper=pl.col(
        "direct_hiv_averted_upper_bound_upper"
    )
    + pl.col("indirect_hiv_averted_upper_bound_upper_future"),
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

# sum direct + indirect for the untreated and resistance decomposition pathways
hiv = hiv.with_columns(
    hiv_averted_gc_untreated=pl.col("direct_hiv_averted_gc_untreated")
    + pl.col("indirect_hiv_averted_gc_untreated"),
    hiv_averted_gc_untreated_lower=pl.col(
        "direct_hiv_averted_gc_untreated_lower"
    )
    + pl.col("indirect_hiv_averted_gc_untreated_lower"),
    hiv_averted_gc_untreated_upper=pl.col(
        "direct_hiv_averted_gc_untreated_upper"
    )
    + pl.col("indirect_hiv_averted_gc_untreated_upper"),
    hiv_averted_gc_resistance=pl.col("direct_hiv_averted_gc_resistance")
    + pl.col("indirect_hiv_averted_gc_resistance"),
    hiv_averted_gc_resistance_lower=pl.col(
        "direct_hiv_averted_gc_resistance_lower"
    )
    + pl.col("indirect_hiv_averted_gc_resistance_lower"),
    hiv_averted_gc_resistance_upper=pl.col(
        "direct_hiv_averted_gc_resistance_upper"
    )
    + pl.col("indirect_hiv_averted_gc_resistance_upper"),
)

# save csv
hiv.write_csv(os.path.join(output_dir, "hiv_averted.csv"))

# we need an output table for the paper that summarizes everything
hiv = hiv.sort(by=["location", "sex", "year"]).select(
    [
        "region",
        "location",
        "country_code",
        "sex",
        "year",
        "paf_gc_hiv",
        "paf_gc_hiv_lower",
        "paf_gc_hiv_upper",
        "paf_chlamydia_hiv",
        "paf_chlamydia_hiv_lower",
        "paf_chlamydia_hiv_upper",
        "paf_syphilis_hiv",
        "paf_syphilis_hiv_lower",
        "paf_syphilis_hiv_upper",
        "paf_trichomoniasis_hiv",
        "paf_trichomoniasis_hiv_lower",
        "paf_trichomoniasis_hiv_upper",
        "unaids_hiv_incidence_number_attributable_to_gc",
        "unaids_hiv_incidence_number_attributable_to_gc_lower",
        "unaids_hiv_incidence_number_attributable_to_gc_upper",
        "unaids_hiv_incidence_number_attributable_to_chlamydia",
        "unaids_hiv_incidence_number_attributable_to_chlamydia_lower",
        "unaids_hiv_incidence_number_attributable_to_chlamydia_upper",
        "unaids_hiv_incidence_number_attributable_to_syphilis",
        "unaids_hiv_incidence_number_attributable_to_syphilis_lower",
        "unaids_hiv_incidence_number_attributable_to_syphilis_upper",
        "unaids_hiv_incidence_number_attributable_to_trichomoniasis",
        "unaids_hiv_incidence_number_attributable_to_trichomoniasis_lower",
        "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper",
        "Ciprofloxacin_resistant_number",
        "Ciprofloxacin_resistant_number_lower",
        "Ciprofloxacin_resistant_number_upper",
        "Azithromycin_resistant_number",
        "Azithromycin_resistant_number_lower",
        "Azithromycin_resistant_number_upper",
        "Cefixime_resistant_number",
        "Cefixime_resistant_number_lower",
        "Cefixime_resistant_number_upper",
        "Ceftriaxone_resistant_number",
        "Ceftriaxone_resistant_number_lower",
        "Ceftriaxone_resistant_number_upper",
        "unaids_hiv_incidence_number_attributable_to_gc_upper_bound",
        "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_lower",
        "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_upper",
        "unaids_hiv_incidence_number_attributable_to_chlamydia_upper_bound",
        "unaids_hiv_incidence_number_attributable_to_chlamydia_upper_bound_lower",
        "unaids_hiv_incidence_number_attributable_to_chlamydia_upper_bound_upper",
        "unaids_hiv_incidence_number_attributable_to_syphilis_upper_bound",
        "unaids_hiv_incidence_number_attributable_to_syphilis_upper_bound_lower",
        "unaids_hiv_incidence_number_attributable_to_syphilis_upper_bound_upper",
        "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper_bound",
        "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper_bound_lower",
        "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper_bound_upper",
        "indirect_hiv_averted_gc_upper_bound",
        "indirect_hiv_averted_gc_upper_bound_lower",
        "indirect_hiv_averted_gc_upper_bound_upper",
        "indirect_hiv_averted_chlamydia_upper_bound",
        "indirect_hiv_averted_chlamydia_upper_bound_lower",
        "indirect_hiv_averted_chlamydia_upper_bound_upper",
        "indirect_hiv_averted_syphilis_upper_bound",
        "indirect_hiv_averted_syphilis_upper_bound_lower",
        "indirect_hiv_averted_syphilis_upper_bound_upper",
        "indirect_hiv_averted_trichomoniasis_upper_bound",
        "indirect_hiv_averted_trichomoniasis_upper_bound_lower",
        "indirect_hiv_averted_trichomoniasis_upper_bound_upper",
        "direct_hiv_averted_2016_gc_change",
        "direct_hiv_averted_2016_gc_change_lower",
        "direct_hiv_averted_2016_gc_change_upper",
        "indirect_hiv_averted_2016_gc_change",
        "indirect_hiv_averted_2016_gc_change_lower",
        "indirect_hiv_averted_2016_gc_change_upper",
        # untreated vs resistance decomposition (direct + indirect + total)
        "direct_hiv_averted_gc_untreated",
        "direct_hiv_averted_gc_untreated_lower",
        "direct_hiv_averted_gc_untreated_upper",
        "indirect_hiv_averted_gc_untreated",
        "indirect_hiv_averted_gc_untreated_lower",
        "indirect_hiv_averted_gc_untreated_upper",
        "hiv_averted_gc_untreated",
        "hiv_averted_gc_untreated_lower",
        "hiv_averted_gc_untreated_upper",
        "direct_hiv_averted_gc_resistance",
        "direct_hiv_averted_gc_resistance_lower",
        "direct_hiv_averted_gc_resistance_upper",
        "indirect_hiv_averted_gc_resistance",
        "indirect_hiv_averted_gc_resistance_lower",
        "indirect_hiv_averted_gc_resistance_upper",
        "hiv_averted_gc_resistance",
        "hiv_averted_gc_resistance_lower",
        "hiv_averted_gc_resistance_upper",
    ]
)

# rename columns to be more informative
mapping = {
    "Ciprofloxacin_resistant_number": "hiv_gc_cases_with_ciprofloxacin_resistance",
    "Azithromycin_resistant_number": "hiv_gc_cases_with_azithromycin_resistance",
    "Cefixime_resistant_number": "hiv_gc_cases_with_cefixime_resistance",
    "Ceftriaxone_resistant_number": "hiv_gc_cases_with_ceftriaxone_resistance",
    "Ciprofloxacin_resistant_number_lower": "hiv_gc_cases_with_ciprofloxacin_resistance_lower",
    "Azithromycin_resistant_number_lower": "hiv_gc_cases_with_azithromycin_resistance_lower",
    "Cefixime_resistant_number_lower": "hiv_gc_cases_with_cefixime_resistance_lower",
    "Ceftriaxone_resistant_number_lower": "hiv_gc_cases_with_ceftriaxone_resistance_lower",
    "Ciprofloxacin_resistant_number_upper": "hiv_gc_cases_with_ciprofloxacin_resistance_upper",
    "Azithromycin_resistant_number_upper": "hiv_gc_cases_with_azithromycin_resistance_upper",
    "Cefixime_resistant_number_upper": "hiv_gc_cases_with_cefixime_resistance_upper",
    "Ceftriaxone_resistant_number_upper": "hiv_gc_cases_with_ceftriaxone_resistance_upper",
    "unaids_hiv_incidence_number_attributable_to_gc_upper_bound": "direct_hiv_incidence_number_averted_by_treatment_of_gc",
    "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_lower": "direct_hiv_incidence_number_averted_by_treatment_of_gc_lower",
    "unaids_hiv_incidence_number_attributable_to_gc_upper_bound_upper": "direct_hiv_incidence_number_averted_by_treatment_of_gc_upper",
    "unaids_hiv_incidence_number_attributable_to_chlamydia_upper_bound": "direct_hiv_incidence_number_averted_by_treatment_of_chlamydia",
    "unaids_hiv_incidence_number_attributable_to_chlamydia_upper_bound_lower": "direct_hiv_incidence_number_averted_by_treatment_of_chlamydia_lower",
    "unaids_hiv_incidence_number_attributable_to_chlamydia_upper_bound_upper": "direct_hiv_incidence_number_averted_by_treatment_of_chlamydia_upper",
    "unaids_hiv_incidence_number_attributable_to_syphilis_upper_bound": "direct_hiv_incidence_number_averted_by_treatment_of_syphilis",
    "unaids_hiv_incidence_number_attributable_to_syphilis_upper_bound_lower": "direct_hiv_incidence_number_averted_by_treatment_of_syphilis_lower",
    "unaids_hiv_incidence_number_attributable_to_syphilis_upper_bound_upper": "direct_hiv_incidence_number_averted_by_treatment_of_syphilis_upper",
    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper_bound": "direct_hiv_incidence_number_averted_by_treatment_of_trichomoniasis",
    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper_bound_lower": "direct_hiv_incidence_number_averted_by_treatment_of_trichomoniasis_lower",
    "unaids_hiv_incidence_number_attributable_to_trichomoniasis_upper_bound_upper": "direct_hiv_incidence_number_averted_by_treatment_of_trichomoniasis_upper",
    "indirect_hiv_averted_gc_upper_bound": "indirect_hiv_averted_by_treatment_of_gc",
    "indirect_hiv_averted_gc_upper_bound_lower": "indirect_hiv_averted_by_treatment_of_gc_lower",
    "indirect_hiv_averted_gc_upper_bound_upper": "indirect_hiv_averted_by_treatment_of_gc_upper",
    "indirect_hiv_averted_chlamydia_upper_bound": "indirect_hiv_averted_by_treatment_of_chlamydia",
    "indirect_hiv_averted_chlamydia_upper_bound_lower": "indirect_hiv_averted_by_treatment_of_chlamydia_lower",
    "indirect_hiv_averted_chlamydia_upper_bound_upper": "indirect_hiv_averted_by_treatment_of_chlamydia_upper",
    "indirect_hiv_averted_syphilis_upper_bound": "indirect_hiv_averted_by_treatment_of_syphilis",
    "indirect_hiv_averted_syphilis_upper_bound_lower": "indirect_hiv_averted_by_treatment_of_syphilis_lower",
    "indirect_hiv_averted_syphilis_upper_bound_upper": "indirect_hiv_averted_by_treatment_of_syphilis_upper",
    "indirect_hiv_averted_trichomoniasis_upper_bound": "indirect_hiv_averted_by_treatment_of_trichomoniasis",
    "indirect_hiv_averted_trichomoniasis_upper_bound_lower": "indirect_hiv_averted_by_treatment_of_trichomoniasis_lower",
    "indirect_hiv_averted_trichomoniasis_upper_bound_upper": "indirect_hiv_averted_by_treatment_of_trichomoniasis_upper",
}

hiv = hiv.rename(mapping)

hiv.write_csv(os.path.join(output_dir, "table_s2.csv"))
