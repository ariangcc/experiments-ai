# Central Limit Theorem Demonstration

## Objective

Demonstrate the Central Limit Theorem (CLT) by showing that the distribution of sample means approaches a normal distribution, regardless of the shape of the original population distribution.

## Methodology

1. **Generate Population**: Create a population following different distributions:
   - Uniform distribution
   - Exponential distribution
   - Bimodal distribution

2. **Sample Collection**: Draw multiple random samples of size `n` from the population

3. **Calculate Sample Means**: Compute the mean of each sample

4. **Analyze Distribution**: Examine the distribution of sample means and compare to normal distribution

5. **Statistical Testing**: Use Shapiro-Wilk test to verify normality of sample means

## Key Parameters

- **Population Size**: 10,000
- **Sample Size (n)**: 30
- **Number of Samples**: 1,000

## Expected Results

According to the Central Limit Theorem:
- The distribution of sample means should be approximately normal
- The mean of sample means should equal the population mean
- The standard deviation of sample means (standard error) should equal σ/√n

## Running the Experiment

```bash
cd experiments/01-central-limit-theorem
python main.py
```

## References

### Online Documentation
- [Central Limit Theorem - Wikipedia](https://en.wikipedia.org/wiki/Central_limit_theorem)
- [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [NumPy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html)

### Academic Resources
- **Book**: "Mathematical Statistics and Data Analysis" by John A. Rice (Chapter 5)
- **Paper**: "The Central Limit Theorem: From Laplace to Lyapunov" - Statistical Science
- **Tutorial**: [Khan Academy - Central Limit Theorem](https://www.khanacademy.org/math/statistics-probability/sampling-distributions-library/sample-means/v/central-limit-theorem)

### Additional Reading
- [Understanding the Central Limit Theorem](https://towardsdatascience.com/understanding-the-central-limit-theorem-642473c63ad8)
- [Visual Explanation of CLT](https://seeing-theory.brown.edu/probability-distributions/index.html)

## Notes

- The CLT works best with larger sample sizes (typically n ≥ 30)
- More skewed distributions may require larger sample sizes to achieve normality
- The theorem is fundamental to inferential statistics and hypothesis testing

## Output

The script generates:
- Visualization comparing original population vs. distribution of sample means
- Statistical summary including means, standard deviations, and normality test results
- Saved plot: `clt_demonstration.png`
