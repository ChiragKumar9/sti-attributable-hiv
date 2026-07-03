from __future__ import annotations

import math

import uncertainties.umath as umath
from uncertainties import ufloat

Z95 = 1.959963984540054  # exact 97.5th percentile of standard normal

# Inverted bounds within this relative/absolute tolerance are treated as a
# zero-width CI (float-noise on a point estimate); anything wider is a genuine
# inversion and is raised rather than silently reinterpreted.
_INVERT_REL_TOL = 1e-9
_INVERT_ABS_TOL = 1e-12


# ----------------------------------------------------------------------
# Logit-normal leaves for [0,1] values
# ----------------------------------------------------------------------


def _logit(p):
    """Logit transformation: log(p / (1-p))"""
    return math.log(p / (1 - p))


def _expit(x):
    """Inverse logit (expit): 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + math.exp(-x))


def logit_normal_sigma_from_ci(mean, lower, upper):
    """sigma of logit(X) implied by a symmetric-in-logit-space 95% CI.
    Assumes neither `lower` nor `upper` is at the 0/1 boundary -- callers
    with a degenerate bound should use the one-sided derivation in
    `logit_normal_leaf` instead, since flooring a degenerate bound to a
    small epsilon before calling this inflates sigma arbitrarily."""
    return (_logit(upper) - _logit(lower)) / (2 * Z95)


def logit_normal_leaf(mean, lower, upper, tag):
    """
    Logit-normal leaf for [0,1] proportions: expit(Normal(logit(mean), sigma)).

    This mirrors lognormal_leaf but for bounded [0,1] quantities. All
    uncertainty propagation happens on the unbounded logit scale. Nominal
    value is exactly `mean`.

    Handles degenerate bounds (lower == 0 and/or upper == 1, which do
    occur here -- e.g. the coinfection-deduction block's clip(upper_bound=
    1.0) calls can legitimately drive a prevalence's lower bound to exactly
    0) without an epsilon-flooring artifact: rather than clipping the
    degenerate bound to ~1e-10 and feeding that into the symmetric sigma
    formula (which can inflate sigma by 5-10x for an otherwise unremarkable
    CI), we derive sigma from the distance between `mean` and whichever
    bound is NOT degenerate.
    """
    if mean is None or lower is None or upper is None:
        return None

    if lower > upper:
        # Zero-width CI (point estimate) whose endpoints were flipped by
        # floating-point rounding: no interval to model, so emit a point mass.
        # A genuinely inverted CI is a data problem we refuse to silently
        # reinterpret -- surface it (diagnose_ci_bounds.py locates these).
        if math.isclose(
            lower, upper, rel_tol=_INVERT_REL_TOL, abs_tol=_INVERT_ABS_TOL
        ):
            return ufloat(mean, 0.0, tag)
        raise ValueError(
            f"{tag}: inverted CI lower={lower} > upper={upper} (not float noise)"
        )

    if lower == upper == mean:
        return ufloat(mean, 0.0, tag)

    lower_degenerate = lower <= 0
    upper_degenerate = upper >= 1

    if lower_degenerate and upper_degenerate:
        # both bounds collapsed -- the CI carries no usable width
        # information; fall back to the widest representable logit-normal
        # span as a conservative default. Rare; worth a second look if it
        # ever actually fires on real data.
        eps = 1e-10
        mean_safe = min(max(mean, eps), 1 - eps)
        sigma = (_logit(1 - eps) - _logit(eps)) / (2 * Z95)
    elif lower_degenerate:
        # only the lower bound is degenerate: derive sigma purely from
        # mean -> upper (one tail, Z95 not 2*Z95). The floor used to keep
        # mean off of exactly 0 MUST be scaled to `upper`'s own magnitude,
        # not a fixed absolute epsilon -- this data legitimately contains
        # probabilities far smaller than 1e-10 (e.g. ~1e-12 MSM
        # trichomoniasis prevalence), and a fixed floor larger than such a
        # value would push mean_safe past upper and produce a negative
        # sigma, which is exactly the bug this replaces.
        floor = max(upper * 1e-9, 1e-300)
        mean_safe = max(mean, floor)
        if mean_safe >= upper:
            # the gap between mean and upper is too small to resolve at
            # this scale -- treat as a tight point mass rather than divide
            # by a vanishing or negative span
            return ufloat(mean, 0.0, tag)
        sigma = (_logit(upper) - _logit(mean_safe)) / Z95
    elif upper_degenerate:
        # mirror image of the lower_degenerate case, scaled to (1 - lower)
        floor = max((1 - lower) * 1e-9, 1e-300)
        mean_safe = min(mean, 1 - floor)
        if mean_safe <= lower:
            return ufloat(mean, 0.0, tag)
        sigma = (_logit(mean_safe) - _logit(lower)) / Z95
    else:
        # both bounds are informative (away from 0/1) -- no epsilon
        # flooring needed at all, just defensively clamp mean into
        # [lower, upper] in case of upstream data anomalies
        mean_safe = min(max(mean, lower), upper)
        sigma = logit_normal_sigma_from_ci(mean_safe, lower, upper)

    logit_x = ufloat(_logit(mean_safe), sigma, f"logit_{tag}")
    exp_x = umath.exp(logit_x)  # type: ignore
    return exp_x / (1 + exp_x)


# ----------------------------------------------------------------------
# Independent lognormal leaves (RRs)
# ----------------------------------------------------------------------


def lognormal_sigma_from_ci(mean, lower, upper):
    """sigma of ln(X) implied by a symmetric-in-log-space 95% CI."""
    return (math.log(upper) - math.log(lower)) / (2 * Z95)


def lognormal_leaf(mean, lower, upper, tag):
    """Independent lognormal leaf: exp(Normal(ln(mean), sigma)). Nominal
    value is exactly `mean`."""
    if mean is None or lower is None or upper is None or mean <= 0:
        return None
    if lower > upper:
        # see logit_normal_leaf: float-noise zero-width CI -> point mass;
        # genuine inversion -> raise rather than silently swap endpoints.
        if math.isclose(
            lower, upper, rel_tol=_INVERT_REL_TOL, abs_tol=_INVERT_ABS_TOL
        ):
            return ufloat(mean, 0.0, tag)
        raise ValueError(
            f"{tag}: inverted CI lower={lower} > upper={upper} (not float noise)"
        )
    if lower == upper == mean:
        return ufloat(mean, 0.0, tag)
    sigma = lognormal_sigma_from_ci(mean, lower, upper)
    log_x = ufloat(math.log(mean), sigma, f"log_{tag}")
    return umath.exp(log_x)  # type: ignore


# ----------------------------------------------------------------------
# Perfectly-correlated (rho=1) lognormal clusters
# ----------------------------------------------------------------------


def new_cluster_latent(tag):
    """One shared standard-normal latent for a correlated cluster. Create
    exactly one of these per row per cluster, and reuse it across every
    member of that cluster for that row."""
    return ufloat(0.0, 1.0, tag)


def lognormal_cluster_member(mean, lower, upper, z):
    """One member of a rho=1 lognormal cluster: mean * exp(sigma * z) for
    the cluster's shared latent z. Any two members built off the same `z`
    are automatically perfectly correlated through ordinary chain rule."""
    if mean is None or lower is None or upper is None or mean <= 0:
        return None
    if lower > upper:
        # see logit_normal_leaf: float-noise zero-width CI -> zero-variance
        # member (kept UFloat-typed via `+ 0.0 * z`); genuine inversion ->
        # raise rather than silently swap endpoints. (No `tag` param here.)
        if math.isclose(
            lower, upper, rel_tol=_INVERT_REL_TOL, abs_tol=_INVERT_ABS_TOL
        ):
            return mean + 0.0 * z
        raise ValueError(
            f"cluster member: inverted CI lower={lower} > upper={upper} (not float noise)"
        )
    if lower == upper == mean:
        return mean + 0.0 * z  # keep it a UFloat-typed expression, var=0
    sigma = lognormal_sigma_from_ci(mean, lower, upper)
    return mean * umath.exp(sigma * z)  # type: ignore


# ----------------------------------------------------------------------
# Fixed (zero-variance) leaf, for fallback/missing-data cases
# ----------------------------------------------------------------------


def point_mass(value, tag):
    """A fixed, zero-variance leaf -- for filling in a known constant
    (e.g. 'no resistance data available, assume 0') rather than an
    estimated quantity with its own uncertainty."""
    return ufloat(value, 0.0, tag)


# ----------------------------------------------------------------------
# Leaf caching for values shared across many rows
# ----------------------------------------------------------------------


class LeafCache:
    """
    Cache of leaves keyed by whatever grain they're actually estimated at
    (e.g. one rr_causal_gc leaf per `sex`, one resistance leaf per
    (location, year)), so the same upstream estimate is represented by the
    same ufloat object everywhere it's used. This is required for
    correlation to come out right at any later sum/aggregation step.
    """

    def __init__(self, builder):
        self._builder = builder
        self._cache = {}

    def get(self, key, *args):
        if key not in self._cache:
            self._cache[key] = self._builder(*args)
        return self._cache[key]


# ----------------------------------------------------------------------
# Final interval extraction: transform to the right scale before
# building +/- 1.96*SE, so the result respects the quantity's domain and
# preserves asymmetry instead of discarding it.
# ----------------------------------------------------------------------


def ci_via_logit(u):
    """For quantities bounded in (0,1): propagate on the logit scale using
    the library's own autodiff, then map back. Returns (mean, lower, upper).
    """
    if u is None:
        return None, None, None
    p = u.nominal_value
    if u.std_dev == 0 or p <= 0.0 or p >= 1.0:
        p_clipped = min(max(p, 0.0), 1.0)
        return p_clipped, p_clipped, p_clipped
    logit_u = umath.log(u / (1 - u))  # type: ignore
    lower = _expit(logit_u.nominal_value - Z95 * logit_u.std_dev)
    upper = _expit(logit_u.nominal_value + Z95 * logit_u.std_dev)
    return p, lower, upper


def ci_via_log(u):
    """For strictly-positive, unbounded-above quantities (counts):
    propagate on the log scale, then map back. Returns (mean, lower, upper).
    """
    if u is None:
        return None, None, None
    y = u.nominal_value
    if u.std_dev == 0 or y <= 0.0:
        return y, y, y
    log_u = umath.log(u)  # type: ignore
    lower = math.exp(log_u.nominal_value - Z95 * log_u.std_dev)
    upper = math.exp(log_u.nominal_value + Z95 * log_u.std_dev)
    return y, lower, upper
