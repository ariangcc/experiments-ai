"""
CLT Convergence Rate Analysis
=============================
Analyzes how quickly sample means converge to normal distribution
as sample size increases. Inspired by quantitative CLT bounds from
Banerjee & Kuchibhotla (2023).
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
    }

    if distribution not in distributions:
        raise ValueError(f"Unknown distribution: {distribution}")

    return distributions[distribution]()


def compute_sample_means(population: np.ndarray, sample_size: int,
                         num_samples: int, seed: int = 42) -> np.ndarray:
    """Compute means of multiple random samples."""
    rng = np.random.default_rng(seed)
    return np.array([
        rng.choice(population, size=sample_size, replace=False).mean()
        for _ in range(num_samples)
    ])


def kolmogorov_smirnov_distance(sample_means: np.ndarray) -> float:
    """Compute KS distance between sample means and fitted normal."""
    standardized = (sample_means - sample_means.mean()) / sample_means.std()
    ks_stat, _ = stats.kstest(standardized, 'norm')
    return ks_stat


def anderson_darling_statistic(sample_means: np.ndarray) -> float:
    """Compute Anderson-Darling statistic for normality."""
    standardized = (sample_means - sample_means.mean()) / sample_means.std()
    result = stats.anderson(standardized, dist='norm')
    return result.statistic


def wasserstein_distance(sample_means: np.ndarray) -> float:
    """Compute Wasserstein-1 distance to fitted normal."""
    standardized = (sample_means - sample_means.mean()) / sample_means.std()
    normal_samples = np.random.default_rng(0).standard_normal(len(standardized))
    return stats.wasserstein_distance(standardized, normal_samples)


def analyze_convergence(population: np.ndarray,
                        sample_sizes: list,
                        num_samples: int = 2000) -> dict:
    """Analyze convergence metrics across different sample sizes."""
    results = {
        'sample_sizes': sample_sizes,
        'ks_distances': [],
        'ad_statistics': [],
        'wasserstein_distances': [],
        'std_errors': [],
        'theoretical_se': [],
    }

    pop_std = np.std(population)

    for n in sample_sizes:
        sample_means = compute_sample_means(population, n, num_samples)

        results['ks_distances'].append(kolmogorov_smirnov_distance(sample_means))
        results['ad_statistics'].append(anderson_darling_statistic(sample_means))
        results['wasserstein_distances'].append(wasserstein_distance(sample_means))
        results['std_errors'].append(np.std(sample_means))
        results['theoretical_se'].append(pop_std / np.sqrt(n))

    return results


def plot_convergence_rates(results: dict, distribution: str,
                           output_dir: Path) -> Path:
    """Create comprehensive convergence rate visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sample_sizes = np.array(results['sample_sizes'])

    # KS Distance
    axes[0, 0].loglog(sample_sizes, results['ks_distances'], 'bo-',
                      linewidth=2, markersize=6, label='Empirical KS')
    # Theoretical O(1/sqrt(n)) reference
    ks_ref = results['ks_distances'][0] * np.sqrt(sample_sizes[0] / sample_sizes)
    axes[0, 0].loglog(sample_sizes, ks_ref, 'r--', linewidth=1.5,
                      label=r'$O(1/\sqrt{n})$ reference')
    axes[0, 0].set_xlabel('Sample Size (n)')
    axes[0, 0].set_ylabel('KS Distance')
    axes[0, 0].set_title('Kolmogorov-Smirnov Distance vs Sample Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Anderson-Darling
    axes[0, 1].semilogx(sample_sizes, results['ad_statistics'], 'go-',
                        linewidth=2, markersize=6)
    axes[0, 1].axhline(y=0.787, color='r', linestyle='--', linewidth=1.5,
                       label='Critical value (α=0.05)')
    axes[0, 1].set_xlabel('Sample Size (n)')
    axes[0, 1].set_ylabel('AD Statistic')
    axes[0, 1].set_title('Anderson-Darling Statistic vs Sample Size')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Wasserstein Distance
    axes[1, 0].loglog(sample_sizes, results['wasserstein_distances'], 'mo-',
                      linewidth=2, markersize=6, label='Empirical W1')
    w_ref = results['wasserstein_distances'][0] * np.sqrt(sample_sizes[0] / sample_sizes)
    axes[1, 0].loglog(sample_sizes, w_ref, 'r--', linewidth=1.5,
                      label=r'$O(1/\sqrt{n})$ reference')
    axes[1, 0].set_xlabel('Sample Size (n)')
    axes[1, 0].set_ylabel('Wasserstein-1 Distance')
    axes[1, 0].set_title('Wasserstein Distance vs Sample Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Standard Error Comparison
    axes[1, 1].loglog(sample_sizes, results['std_errors'], 'co-',
                      linewidth=2, markersize=6, label='Empirical SE')
    axes[1, 1].loglog(sample_sizes, results['theoretical_se'], 'r--',
                      linewidth=2, label=r'Theoretical $\sigma/\sqrt{n}$')
    axes[1, 1].set_xlabel('Sample Size (n)')
    axes[1, 1].set_ylabel('Standard Error')
    axes[1, 1].set_title('Standard Error: Empirical vs Theoretical')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'CLT Convergence Analysis: {distribution.upper()} Distribution',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / f'convergence_{distribution}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_convergence_comparison(all_results: dict, output_dir: Path) -> Path:
    """Compare convergence rates across distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for dist, results in all_results.items():
        sample_sizes = results['sample_sizes']
        axes[0].loglog(sample_sizes, results['ks_distances'], 'o-',
                       linewidth=2, markersize=5, label=dist)
        axes[1].loglog(sample_sizes, results['wasserstein_distances'], 'o-',
                       linewidth=2, markersize=5, label=dist)

    # Reference line
    sample_sizes = np.array(list(all_results.values())[0]['sample_sizes'])
    ref_line = 0.1 * np.sqrt(sample_sizes[0] / sample_sizes)
    axes[0].loglog(sample_sizes, ref_line, 'k--', linewidth=1.5,
                   label=r'$O(1/\sqrt{n})$', alpha=0.5)
    axes[1].loglog(sample_sizes, ref_line * 0.5, 'k--', linewidth=1.5,
                   label=r'$O(1/\sqrt{n})$', alpha=0.5)

    axes[0].set_xlabel('Sample Size (n)')
    axes[0].set_ylabel('KS Distance')
    axes[0].set_title('KS Distance Convergence Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Sample Size (n)')
    axes[1].set_ylabel('Wasserstein Distance')
    axes[1].set_title('Wasserstein Distance Convergence Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'convergence_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def run_experiment(population_size: int = 50000,
                   sample_sizes: list = None,
                   num_samples: int = 2000,
                   distributions: list = None,
                   output_dir: Path = None) -> dict:
    """Run the convergence rate analysis experiment."""

    if sample_sizes is None:
        sample_sizes = [5, 10, 20, 30, 50, 100, 200, 500]

    if distributions is None:
        distributions = ['uniform', 'exponential', 'bimodal', 'chi_squared']

    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CLT Convergence Rate Analysis")
    print("=" * 60)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Distributions: {distributions}")

    all_results = {}

    for dist in distributions:
        print(f"\n--- Analyzing {dist.upper()} ---")

        population = generate_population(dist, population_size)
        results = analyze_convergence(population, sample_sizes, num_samples)
        all_results[dist] = results

        plot_path = plot_convergence_rates(results, dist, output_dir)
        print(f"  Individual plot saved: {plot_path}")

        # Print convergence summary
        print(f"  KS distance: {results['ks_distances'][0]:.4f} (n={sample_sizes[0]}) "
              f"-> {results['ks_distances'][-1]:.4f} (n={sample_sizes[-1]})")

    # Comparison plot
    comparison_path = plot_convergence_comparison(all_results, output_dir)
    print(f"\nComparison plot saved: {comparison_path}")

    print("\n" + "=" * 60)

    return all_results


if __name__ == "__main__":
    run_experiment()
