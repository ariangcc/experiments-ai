"""
Basic CLT Demonstration
=======================
Demonstrates the Central Limit Theorem by showing how sample means
from various non-normal distributions converge to a normal distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


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
        'beta': lambda: rng.beta(a=2, b=5, size=size) * 10,
    }

    if distribution not in distributions:
        raise ValueError(f"Unknown distribution: {distribution}. "
                        f"Available: {list(distributions.keys())}")

    return distributions[distribution]()


def compute_sample_means(population: np.ndarray, sample_size: int,
                         num_samples: int, seed: int = 42) -> np.ndarray:
    """Compute means of multiple random samples from population."""
    rng = np.random.default_rng(seed)

    sample_means = np.array([
        rng.choice(population, size=sample_size, replace=False).mean()
        for _ in range(num_samples)
    ])

    return sample_means


def plot_clt_demonstration(population: np.ndarray, sample_means: np.ndarray,
                           distribution: str, sample_size: int,
                           output_dir: Path) -> None:
    """Create visualization comparing population and sample means distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Population distribution
    axes[0].hist(population, bins=50, density=True, alpha=0.7,
                 edgecolor='black', color='steelblue')
    axes[0].set_title(f'Original Population ({distribution})',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    pop_mean = np.mean(population)
    axes[0].axvline(pop_mean, color='red', linestyle='--', linewidth=2,
                    label=f'μ = {pop_mean:.2f}')
    axes[0].legend()

    # Sample means distribution with normal fit
    axes[1].hist(sample_means, bins=50, density=True, alpha=0.7,
                 edgecolor='black', color='steelblue', label='Sample Means')

    mu, sigma = np.mean(sample_means), np.std(sample_means)
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
                 label='Normal Fit')

    axes[1].set_title(f'Distribution of Sample Means (n={sample_size})',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sample Mean')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    output_path = output_dir / f'clt_{distribution}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def compute_statistics(population: np.ndarray, sample_means: np.ndarray,
                       sample_size: int) -> dict:
    """Compute and return CLT-related statistics."""
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    sample_means_mean = np.mean(sample_means)
    sample_means_std = np.std(sample_means)
    theoretical_se = pop_std / np.sqrt(sample_size)

    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(sample_means[:5000])  # Shapiro limited to 5000

    return {
        'population_mean': pop_mean,
        'population_std': pop_std,
        'sample_means_mean': sample_means_mean,
        'sample_means_std': sample_means_std,
        'theoretical_se': theoretical_se,
        'se_ratio': sample_means_std / theoretical_se,
        'shapiro_statistic': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'is_normal': shapiro_p > 0.05,
    }


def run_experiment(population_size: int = 10000,
                   sample_size: int = 30,
                   num_samples: int = 1000,
                   distributions: list = None,
                   output_dir: Path = None) -> dict:
    """Run the basic CLT demonstration experiment."""

    if distributions is None:
        distributions = ['uniform', 'exponential', 'bimodal', 'chi_squared', 'beta']

    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    results = {}

    print("=" * 60)
    print("Basic CLT Demonstration")
    print("=" * 60)

    for dist in distributions:
        print(f"\n--- {dist.upper()} Distribution ---")

        population = generate_population(dist, population_size)
        sample_means = compute_sample_means(population, sample_size, num_samples)

        plot_path = plot_clt_demonstration(
            population, sample_means, dist, sample_size, output_dir
        )

        stats_dict = compute_statistics(population, sample_means, sample_size)
        results[dist] = stats_dict

        print(f"  Population: μ={stats_dict['population_mean']:.4f}, "
              f"σ={stats_dict['population_std']:.4f}")
        print(f"  Sample Means: μ={stats_dict['sample_means_mean']:.4f}, "
              f"σ={stats_dict['sample_means_std']:.4f}")
        print(f"  Theoretical SE: {stats_dict['theoretical_se']:.4f} "
              f"(ratio: {stats_dict['se_ratio']:.4f})")
        print(f"  Shapiro-Wilk: p={stats_dict['shapiro_p_value']:.4f} "
              f"({'Normal' if stats_dict['is_normal'] else 'Non-normal'})")
        print(f"  Plot saved: {plot_path}")

    print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    run_experiment()
