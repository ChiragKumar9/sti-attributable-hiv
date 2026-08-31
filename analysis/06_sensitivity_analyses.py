"""Sensitivity analyses for table_s2: vary msm_fraction (0.05, 0.30) and the
UNAIDS future HIV scenario, then merge all variants into table_s2.csv as a
long-format table tagged by sensitivity_axis/sensitivity_value (baseline gets
its own single "baseline" tag rather than being duplicated across axes).

Each variant reruns 01 -> 03 -> 04 -> 05 with the relevant env var override,
writing to its own outputs/sensitivity_*/ directory. 02's RR outputs are
parameter-invariant, so they are copied into each variant's directory rather
than rerunning 02.
"""

import os
import shutil
import subprocess

import polars as pl

output_dir = "outputs"
table_s2_final_path = os.path.join(output_dir, "table_s2.csv")

MSM_VARIANTS = [0.05, 0.30]

SCENARIO_VARIANTS = [
    "98-98-99 treatment, 2024 prevention",
    "95-95-95 treatment and prevention reaches 2030 target",
    "2024 level all HIV services",
    "Historical trend",
    # "2024 level treatment, reduced prevention" excluded: source projections only
    # cover 1 of 42 countries beyond 2025, making it degenerate as a sensitivity
    # variant (see outputs/sensitivity_scenario_2024_level_treatment__reduced_prevention/).
]

RR_FILES = [
    "meta_estimated_RRs_causal_STI_given_HIV.csv",
    "meta_estimated_RRs_STI_HIV_coinfection.csv",
    "meta_estimated_RRs_HIV_detection_given_STI.csv",
]

PIPELINE_SCRIPTS = [
    "analysis/01_assemble_data.py",
    "analysis/03_assemble_resistance_data.py",
    "analysis/04_sti_attributable_hiv.py",
    "analysis/05_appropriate_abx_averted_burden.py",
]


def slugify(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def run_variant(variant_dir, env_overrides):
    os.makedirs(variant_dir, exist_ok=True)
    for f in RR_FILES:
        src = os.path.join(output_dir, f)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"{src} missing -- run analysis/02_meta_estimate_RRs.py first"
            )
        shutil.copy(src, os.path.join(variant_dir, f))
    env = {**os.environ, "OUTPUT_DIR": variant_dir, **env_overrides}
    for script in PIPELINE_SCRIPTS:
        subprocess.run(["uv", "run", script], env=env, check=True)


def tag(df, axis, value):
    return df.with_columns(
        pl.lit(axis).alias("sensitivity_axis"),
        pl.lit(value).alias("sensitivity_value"),
    )


# ---- baseline: single set of rows, from 05's intermediate output ----
frames = [
    tag(
        pl.read_csv(os.path.join(output_dir, "table_s2_intermediate.csv")),
        "baseline",
        "baseline",
    )
]

# ---- msm_fraction variants: affect all years ----
for f in MSM_VARIANTS:
    variant_dir = os.path.join(output_dir, f"sensitivity_msm_{f}")
    run_variant(variant_dir, {"MSM_FRACTION_OVERRIDE": str(f)})
    df = pl.read_csv(os.path.join(variant_dir, "table_s2_intermediate.csv"))
    frames.append(tag(df, "msm_fraction", str(f)))

# ---- unaids_scenario variants: restricted to 2025-2030, the window every scenario has data for ----
for scenario in SCENARIO_VARIANTS:
    variant_dir = os.path.join(
        output_dir, f"sensitivity_scenario_{slugify(scenario)}"
    )
    run_variant(variant_dir, {"UNAIDS_SCENARIO_COL": scenario})
    df = pl.read_csv(os.path.join(variant_dir, "table_s2_intermediate.csv"))
    frames.append(
        tag(df.filter(pl.col("year") > 2024), "unaids_scenario", scenario)
    )

merged = pl.concat(frames, how="vertical").sort(
    ["sensitivity_axis", "sensitivity_value", "location", "sex", "year"]
)
merged.write_csv(table_s2_final_path)
