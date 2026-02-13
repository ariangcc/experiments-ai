"""
Central Limit Theorem Experiments
=================================
Orchestrates all CLT sub-experiments exploring theoretical and empirical
aspects of the Central Limit Theorem.

Based on insights from:
- Banerjee & Kuchibhotla (2023) "Central Limit Theorems and Approximation Theory"
  https://arxiv.org/abs/2306.05947

Sub-experiments:
1. Basic Demonstration - Visual proof of CLT across distributions
2. Convergence Rates - How quickly sample means converge to normal
3. Berry-Esseen Bounds - Theoretical bounds on convergence error
"""

import sys
from pathlib import Path

# Add experiments directory to path
EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENTS_DIR))


def run_basic_demonstration():
    """Run the basic CLT demonstration."""
    from importlib import import_module

    module_path = EXPERIMENTS_DIR / '01-basic-demonstration'
    sys.path.insert(0, str(module_path))

    try:
        import run as basic_demo
        return basic_demo.run_experiment()
    finally:
        sys.path.remove(str(module_path))


def run_convergence_rates():
    """Run the convergence rate analysis."""
    module_path = EXPERIMENTS_DIR / '02-convergence-rates'
    sys.path.insert(0, str(module_path))

    try:
        import importlib
        import run as convergence
        importlib.reload(convergence)
        return convergence.run_experiment()
    finally:
        sys.path.remove(str(module_path))


def run_berry_esseen():
    """Run the Berry-Esseen bounds analysis."""
    module_path = EXPERIMENTS_DIR / '03-berry-esseen-bounds'
    sys.path.insert(0, str(module_path))

    try:
        import importlib
        import run as berry_esseen
        importlib.reload(berry_esseen)
        return berry_esseen.run_experiment()
    finally:
        sys.path.remove(str(module_path))


def main():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("CENTRAL LIMIT THEOREM: COMPREHENSIVE EXPERIMENT SUITE")
    print("=" * 70)

    experiments = [
        ("01 - Basic CLT Demonstration", run_basic_demonstration),
        ("02 - Convergence Rate Analysis", run_convergence_rates),
        ("03 - Berry-Esseen Bounds", run_berry_esseen),
    ]

    results = {}

    for name, runner in experiments:
        print(f"\n{'#' * 70}")
        print(f"# {name}")
        print(f"{'#' * 70}")

        try:
            results[name] = runner()
            print(f"\n✓ {name} completed successfully")
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            results[name] = None

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print("\nOutput directories:")
    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('0'):
            output_dir = exp_dir / 'output'
            if output_dir.exists():
                print(f"  {exp_dir.name}/output/")

    return results


if __name__ == "__main__":
    main()
