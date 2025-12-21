import numpy as np
from scipy import stats


def meta_estimate_rrs(
    means, sigmas, group_assignments, group_assignments_unique, n=10000
):
    n_unique = len(group_assignments_unique)
    mu = np.repeat(0.0, n_unique)
    nu = np.repeat(n, n_unique)
    beta = np.repeat(0.0, n_unique)
    alpha = np.repeat(n / 2, n_unique)

    for mean, sigma, group in zip(means, sigmas, group_assignments):
        idx = group_assignments_unique.index(group)
        if mu[idx] == 0.0:
            mu[idx] = mean
            beta[idx] = n * (sigma**2) / 2
        else:
            # now we can calculate the posterior parameters
            # recall the formulas
            # posterior mu = (nu * mu_0 + sum(x_i)) / (nu + n)
            # posterior nu = nu + n
            # posterior alpha = alpha + n/2
            # posterior beta = beta + 0.5 * sum((x_i - x_bar)^2) + (nu * n * (x_bar - mu_0)^2) / (2 * (nu + n))

            alpha[idx] += n / 2
            beta[idx] += 0.5 * n * (sigma**2) + (
                nu[idx] * n * (mean - mu[idx]) ** 2
            ) / (2 * (nu[idx] + n))
            # have to update nu and mu last because we want the old value when updating beta
            mu[idx] = (nu[idx] * mu[idx] + n * mean) / (nu[idx] + n)
            nu[idx] += n

    # posterior distribution is t with 2*alpha' degrees of freedom with mean
    # mu' and scale (beta' * (nu' + 1)) / (alpha' * nu')

    dof = alpha * 2
    t_scale = (beta * (nu + 1)) / (alpha * nu)

    rr_mu = np.array(
        [
            float(
                np.exp(
                    stats.t.mean(
                        df=dof[i], loc=mu[i], scale=np.sqrt(t_scale[i])
                    )
                )
            )
            if t_scale[i] != 0.0
            else float(np.exp(mu[i]))
            for i in range(n_unique)
        ]
    )
    rr_lower = np.array(
        [
            float(
                np.exp(
                    stats.t.ppf(
                        q=0.025,
                        df=dof[i],
                        loc=mu[i],
                        scale=np.sqrt(t_scale[i]),
                    )
                )
            )
            if t_scale[i] != 0.0
            else float(np.exp(mu[i]))
            for i in range(n_unique)
        ]
    )
    rr_upper = np.array(
        [
            float(
                np.exp(
                    stats.t.ppf(
                        q=0.975,
                        df=dof[i],
                        loc=mu[i],
                        scale=np.sqrt(t_scale[i]),
                    )
                )
            )
            if t_scale[i] != 0.0
            else float(np.exp(mu[i]))
            for i in range(n_unique)
        ]
    )

    return rr_mu, rr_lower, rr_upper
