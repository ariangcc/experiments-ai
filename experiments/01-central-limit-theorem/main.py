import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def demonstrate_central_limit_theorem(
    population_size=10000,
    sample_size=30,
    num_samples=1000,
    distribution='uniform'
):
    np.random.seed(42)
    
    if distribution == 'uniform':
        population = np.random.uniform(0, 10, population_size)
    elif distribution == 'exponential':
        population = np.random.exponential(2, population_size)
    elif distribution == 'bimodal':
        population = np.concatenate([
            np.random.normal(2, 0.5, population_size // 2),
            np.random.normal(8, 0.5, population_size // 2)
        ])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(population, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Original Population Distribution ({distribution})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].axvline(np.mean(population), color='red', linestyle='--', linewidth=2, label=f'μ = {np.mean(population):.2f}')
    axes[0].legend()
    
    axes[1].hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black', label='Sample Means')
    
    mu = np.mean(sample_means)
    sigma = np.std(sample_means)
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
    
    axes[1].set_title(f'Distribution of Sample Means (n={sample_size})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sample Mean')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('clt_demonstration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print(f"Central Limit Theorem Demonstration")
    print(f"{'='*60}")
    print(f"Population Distribution: {distribution}")
    print(f"Population Mean: {np.mean(population):.4f}")
    print(f"Population Std Dev: {np.std(population):.4f}")
    print(f"\nSample Size: {sample_size}")
    print(f"Number of Samples: {num_samples}")
    print(f"\nSample Means Mean: {mu:.4f}")
    print(f"Sample Means Std Dev: {sigma:.4f}")
    print(f"Theoretical Std Error: {np.std(population) / np.sqrt(sample_size):.4f}")
    print(f"\nShapiro-Wilk Test for Normality:")
    statistic, p_value = stats.shapiro(sample_means)
    print(f"  Test Statistic: {statistic:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Result: {'Normally distributed' if p_value > 0.05 else 'Not normally distributed'} (α=0.05)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("Demonstrating Central Limit Theorem with different distributions...\n")
    
    for dist in ['uniform', 'exponential', 'bimodal']:
        print(f"\n--- Testing with {dist.upper()} distribution ---")
        demonstrate_central_limit_theorem(distribution=dist)
        input("Press Enter to continue to next distribution...")
