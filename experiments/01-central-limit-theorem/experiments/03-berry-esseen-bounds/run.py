"""
Berry-Esseen Bounds Analysis
============================
Explores the Berry-Esseen theorem which provides explicit bounds on
CLT convergence rates. Inspired by Banerjee & Kuchibhotla (2023)
"Central Limit Theorems and Approximation Theory".

The Berry-Esseen theorem states:
    sup_x |F_n(x) - Φ(x)| ≤ C * ρ / (σ³ * √n)

where:
    - F_n: CDF of standardized sample mean
    - Φ: Standard normal CDF
    - ρ: Third absolute central moment E[|X - μ|³]
    - σ: Standard deviation
    - C ≈ 0.4748 (best known constant, Shevtsova 2011)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


BERRY_ESSEEN_CONSTANT = 0.4748  # Best known constant (Shevtsova, 2011)


def generate_population(distribution: str, size: int, seed: int = 42) -> np.ndarray:
    """Generate a population with specified distribution."""
    rng = np.random.default_rng(seed)

    distributions = {
        'uniform': lambda: rng.uniform(0, 10, size),
        'exponential': lambda: rng.exponential(2, size),
        'bimodal': lambda: np.concatenate([
            rng.normal(2, 0.5, size // 2),
            rng.normal(8, 0.5, size - size // 2)
        ]),
        'chi_squared': lambda: rng.chisquare(df=3, size=size),
        'bernoulli': lambda: rng.choice([0, 1], size=size, p=[0.3, 0.7]),
    }

    if distribution not in distributions:
        raise ValueError(f"Unknown distribution: {distribution}")

    return distributions[distribution]()


def compute_third_absolute_moment(data: np.ndarray) -> float:
    """Compute third absolute central moment E[|X - μ|³]."""
    centered = data - np.mean(data)
    return np.mean(np.abs(centered) ** 3)


def berry_esseen_bound(data: np.ndarray, n: int) -> float:
    """
    Compute the Berry-Esseen bound for sample size n.

    Bound = C * ρ / (σ³ * √n)
    """
    sigma = np.std(data)
    rho = compute_third_absolute_moment(data)

    if sigma == 0:
        return np.inf

    bound = BERRY_ESSEEN_CONSTANT * rho / (sigma ** 3 * np.sqrt(n))
    return bound


def empirical_sup_deviation(sample_means: np.ndarray) -> float:
    """
    Compute empirical supremum deviation from normal CDF.

    sup_x |F_n(x) - Φ(x)|
    """
    # Standardize sample means
    standardized = (sample_means - sample_means.mean()) / sample_means.std()

    # Compute empirical CDF
    sorted_means = np.sort(standardized)
    n = len(sorted_means)
    ecdf = np.arange(1, n + 1) / n

    # Normal CDF at same points
    normal_cdf = stats.norm.cdf(sorted_means)

    # Supremum deviation (considering both sides)
    ecdf_minus = np.concatenate([[0], ecdf[:-1]])
    deviations = np.maximum(np.abs(ecdf - normal_cdf),
                           np.abs(ecdf_minus - normal_cdf))

    return np.max(deviations)


def analyze_berry_esseen(population: np.ndarray,
                         sample_sizes: list,
                         num_samples: int = 5000) -> dict:
    """Compare empirical deviations with Berry-Esseen bounds."""
    results = {
        'sample_sizes': sample_sizes,
        'be_bounds': [],
        'empirical_deviations': [],
        'tightness_ratios': [],
    }

    for n in sample_sizes:
        # Compute theoretical Berry-Esseen bound
        be_bound = berry_esseen_bound(population, n)
        results['be_bounds'].append(be_bound)

        # Compute empirical deviation
        rng = np.random.default_rng(42)
        sample_means = np.array([
            rng.choice(population, size=n, replace=False).mean()
            for _ in range(num_samples)
        ])
        emp_dev = empirical_sup_deviation(sample_means)
        results['empirical_deviations'].append(emp_dev)

        # Tightness ratio (how close empirical is to bound)
        ratio = emp_dev / be_bound if be_bound > 0 else 0
        results['tightness_ratios'].append(ratio)

    return results


def plot_berry_esseen_analysis(results: dict, distribution: str,
                                output_dir: Path) -> Path:
    """Visualize Berry-Esseen bounds vs empirical deviations."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sample_sizes = np.array(results['sample_sizes'])

    # Plot 1: Bounds vs Empirical
    axes[0].loglog(sample_sizes, results['be_bounds'], 'r-o',
                   linewidth=2, markersize=6, label='Berry-Esseen Bound')
    axes[0].loglog(sample_sizes, results['empirical_deviations'], 'b-s',
                   linewidth=2, markersize=6, label='Empirical Deviation')

    # Reference line
    ref = results['be_bounds'][0] * np.sqrt(sample_sizes[0] / sample_sizes)
    axes[0].loglog(sample_sizes, ref, 'k--', alpha=0.5,
                   linewidth=1.5, label=r'$O(1/\sqrt{n})$')

    axes[0].set_xlabel('Sample Size (n)')
    axes[0].set_ylabel('Supremum Deviation')
    axes[0].set_title('Berry-Esseen Bound vs Empirical')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Tightness ratio
    axes[1].semilogx(sample_sizes, results['tightness_ratios'], 'g-o',
                     linewidth=2, markersize=6)
    axes[1].axhline(y=1.0, color='r', linestyle='--', linewidth=1.5,
                    label='Bound = Empirical')
    axes[1].set_xlabel('Sample Size (n)')
    axes[1].set_ylabel('Empirical / Bound')
    axes[1].set_title('Bound Tightness (lower = looser bound)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, min(1.2, max(results['tightness_ratios']) * 1.1))

    # Plot 3: Gap between bound and empirical
    gaps = np.array(results['be_bounds']) - np.array(results['empirical_deviations'])
    axes[2].semilogx(sample_sizes, gaps, 'm-o', linewidth=2, markersize=6)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('Sample Size (n)')
    axes[2].set_ylabel('Bound - Empirical')
    axes[2].set_title('Gap (positive = bound holds)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Berry-Esseen Analysis: {distribution.upper()}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / f'berry_esseen_{distribution}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_distribution_moments(populations: dict, output_dir: Path) -> Path:
    """Compare third moments across distributions (affects BE bound)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dists = list(populations.keys())
    third_moments = []
    sigmas = []
    rho_over_sigma3 = []

    for dist, pop in populations.items():
        sigma = np.std(pop)
        rho = compute_third_absolute_moment(pop)
        third_moments.append(rho)
        sigmas.append(sigma)
        rho_over_sigma3.append(rho / sigma ** 3)

    x = np.arange(len(dists))
    width = 0.35

    axes[0].bar(x - width/2, third_moments, width, label=r'$\rho = E[|X-\mu|^3]$',
                color='steelblue')
    axes[0].bar(x + width/2, sigmas, width, label=r'$\sigma$', color='coral')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dists, rotation=45, ha='right')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Distribution Moments')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, rho_over_sigma3, color='forestgreen')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dists, rotation=45, ha='right')
    axes[1].set_ylabel(r'$\rho / \sigma^3$')
    axes[1].set_title('Berry-Esseen Coefficient (higher = slower convergence)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'distribution_moments.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def run_experiment(population_size: int = 50000,
                   sample_sizes: list = None,
                   num_samples: int = 5000,
                   distributions: list = None,
                   output_dir: Path = None) -> dict:
    """Run the Berry-Esseen bounds experiment."""

    if sample_sizes is None:
        sample_sizes = [10, 20, 30, 50, 100, 200, 500, 1000]

    if distributions is None:
        distributions = ['uniform', 'exponential', 'bimodal', 'chi_squared', 'bernoulli']

    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Berry-Esseen Bounds Analysis")
    print("=" * 60)
    print(f"Berry-Esseen constant C = {BERRY_ESSEEN_CONSTANT}")
    print(f"Sample sizes: {sample_sizes}")

    all_results = {}
    populations = {}

    for dist in distributions:
        print(f"\n--- {dist.upper()} Distribution ---")

        population = generate_population(dist, population_size)
        populations[dist] = population

        # Compute moments
        sigma = np.std(population)
        rho = compute_third_absolute_moment(population)
        print(f"  σ = {sigma:.4f}, ρ = {rho:.4f}, ρ/σ³ = {rho/sigma**3:.4f}")

        # Analyze
        results = analyze_berry_esseen(population, sample_sizes, num_samples)
        all_results[dist] = results

        # Plot
        plot_path = plot_berry_esseen_analysis(results, dist, output_dir)
        print(f"  Plot saved: {plot_path}")

        # Summary
        print(f"  n=30: bound={results['be_bounds'][2]:.4f}, "
              f"empirical={results['empirical_deviations'][2]:.4f}, "
              f"ratio={results['tightness_ratios'][2]:.4f}")

    # Distribution comparison
    moments_path = plot_distribution_moments(populations, output_dir)
    print(f"\nMoments comparison saved: {moments_path}")

    print("\n" + "=" * 60)

    return all_results


if __name__ == "__main__":
    run_experiment()
