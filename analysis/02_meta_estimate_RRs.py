import os

import numpy as np
import polars as pl

from averted_burden import rr_meta_estimate

np.random.seed(108)

data_dir = "data"
output_dir = "outputs"


def workhorse(filename, output_filename):
    rrs = pl.read_csv(os.path.join(data_dir, f"{filename}.csv"))

    # convert rrs to log space
    rrs = rrs.with_columns(
        log_val=pl.col("val").log(),
        log_lower=pl.col("lower").log(),
        log_upper=pl.col("upper").log(),
    )

    # fit normal distributions to each of these values
    # we can do this because we are making no assumptions about the bias of the RR
    # CIs, just that the represent upper and lower bounds (not that more mass is on
    # one side or the other)
    # we know the mean is the log_val
    # we need to calculate the sd from the 95% CI
    rrs = rrs.with_columns(
        sigma=(pl.col("log_upper") - pl.col("log_lower")) / (2 * 1.96)
    )
    rrs = rrs.with_columns(group=pl.col("sex") + pl.col("bacteria"))

    rr_mu, rr_lower, rr_upper = rr_meta_estimate.meta_estimate_rrs(
        means=rrs["log_val"].to_list(),
        sigmas=rrs["sigma"].to_list(),
        group_assignments=rrs["group"].to_list(),
        group_assignments_unique=rrs["group"]
        .unique(maintain_order=True)
        .to_list(),
    )

    # we want to condense by bacteria and sex
    rrs_striped = (
        rrs.select(["bacteria", "sex"])
        .group_by(["bacteria", "sex"], maintain_order=True)
        .len()
    )

    results = pl.DataFrame(
        {
            "bacteria": rrs_striped["bacteria"].to_list(),
            "sex": rrs_striped["sex"].to_list(),
            "rr_mu": rr_mu,
            "rr_lower": rr_lower,
            "rr_upper": rr_upper,
        }
    )

    # save the results
    results.write_csv(os.path.join(output_dir, f"{output_filename}.csv"))


workhorse(
    "RRs_causal_STI_given_HIV", "meta_estimated_RRs_causal_STI_given_HIV"
)
workhorse("RRs_STI_HIV_coinfection", "meta_estimated_RRs_STI_HIV_coinfection")
workhorse(
    "RRs_HIV_detection_given_STI", "meta_estimated_RRs_HIV_detection_given_STI"
)
