"""Reproduce the numerical oracle-gate comparisons after Theorem 3."""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.isotonic import isotonic_regression


def posterior(y, theta, dtheta, slab, epsilon):
    if slab is None:
        atoms = np.array([0, -4, 4])
        mass = np.array([1 - epsilon, epsilon / 2, epsilon / 2])
        log_joint = stats.norm.logpdf(y[:, None] - atoms) + np.log(mass)
        log_marginal = logsumexp(log_joint, axis=1)
        probabilities = np.exp(log_joint - log_marginal[:, None])
        return np.exp(log_marginal), probabilities @ atoms

    densities, means = [], []
    for block in np.array_split(y, 60):
        joint = stats.norm.pdf(block[:, None] - theta) * slab * dtheta
        marginal = (1 - epsilon) * stats.norm.pdf(block) + epsilon * joint.sum(axis=1)
        densities.append(marginal)
        means.append(epsilon * (joint * theta).sum(axis=1) / marginal)
    return np.concatenate(densities), np.concatenate(means)


def main():
    dtheta = 0.01
    theta = np.arange(-300, 300.0001, dtheta)
    epsilon, threshold = 0.05, 2.0
    y_grid = np.linspace(threshold + 1e-6, 60, 6000)
    slabs = {
        "Laplace": 0.25 * np.exp(-0.5 * np.abs(theta)),
        "t3": stats.t.pdf(theta / 2, 3) / 2,
        "N(0,9)": stats.norm.pdf(theta, 0, 3),
        "two-point": None,
    }
    for name, slab in slabs.items():
        density, mean = posterior(y_grid, theta, dtheta, slab, epsilon)
        # Ignore tails with effectively zero weight in double precision.
        keep = (density > 1e-300) & np.isfinite(mean)
        y = y_grid[keep]
        gate = isotonic_regression(
            mean[keep] / y, sample_weight=2 * y**2 * density[keep], y_min=0, y_max=1
        )
        u = 1 - threshold / y
        diff = gate - u
        below = diff < 0
        edges = np.flatnonzero(np.diff(np.r_[False, below, False]))
        intervals = [f"({u[a]:.3f}, {u[b - 1]:.3f})" for a, b in edges.reshape(-1, 2)]
        index = np.argmin(diff)
        excess = np.max(y * (1 - gate)) / threshold
        print(
            f"{name:9s} G(0+)={gate[0]:.3f}; G at last grid point={gate[-1]:.3f}; "
            f"min(G-u)={diff[index]:+.3f} at u={u[index]:.3f}\n"
            f"          below on {', '.join(intervals) or 'no intervals'}; "
            f"max shrinkage / threshold = {excess:.2f}"
        )


if __name__ == "__main__":
    main()
