# Stochastic Derivatives Pricing Engine

A quantitative derivatives pricing framework implementing analytical Black–Scholes valuation, Monte Carlo simulation, variance reduction techniques, Greeks estimation, convergence diagnostics, and exotic option pricing under the risk-neutral measure.

This project evaluates both mathematical correctness and statistical efficiency of stochastic pricing estimators.

## Mathematical Model

The underlying asset is modeled as a Geometric Brownian Motion (GBM):

dS_t = μ S_t dt + σ S_t dW_t


Under the risk-neutral measure, drift is replaced by the risk-free rate:

dS_t = r S_t dt + σ S_t dW_t


The European call price is given by:

C = e^{-rT} E[max(S_T - K, 0)]


Closed-form benchmark (Black–Scholes):

C = S0 Φ(d1) − K e^{-rT} Φ(d2)


## Features
### 1. Black–Scholes Analytical Pricing
- European call price
- Delta
- Gamma
- Vega
- Theta (daily)

Used as:
- Benchmark for Monte Carlo validation
- Control variate reference
- Convergence ground truth

### 2. Monte Carlo Simulation
Terminal simulation:
S_T = S0 * exp((r − 0.5σ²)T + σ√T Z)

Outputs:
- Estimated price
- Sample variance
- Standard error
- 95% confidence interval

Empirical convergence validated:

Error ∝ O(N^(-1/2))


### 3. Variance Reduction Techniques
Antithetic Variates

Pairs each normal sample with its negative to reduce estimator variance without increasing computational cost.

Control Variates

Uses terminal stock price as control:

X_adj = X − β(Y − E[Y])


Optimal coefficient:

β = Cov(X,Y) / Var(Y)


Measured:
- Variance reduction percentage
- Efficiency improvement

### 4. Greeks Estimation
Analytical Greeks

Closed-form Delta, Gamma, Vega, Theta.

Pathwise Monte Carlo Delta
Delta = e^{-rT} * 1{S_T > K} * (S_T / S0)


Compared against:
- Analytical Delta
- Finite difference estimator
- Bootstrap confidence intervals

### 5. Exotic Option Pricing
Arithmetic Asian Call
Payoff = max(mean(S_t) − K, 0)

Simulates full GBM paths using time discretization.

Demonstrates pricing when no closed-form solution exists.

### 6. Convergence Analysis

Controlled experiments across increasing path counts:

- Mean absolute pricing error
- Standard error decay
- Confidence interval width
- Log-log regression to estimate convergence slope

Validated empirical slope ≈ -0.5.

### 7. Performance Benchmarking

Measured:
- Runtime vs number of simulated paths
- Time per million paths
- Scaling behavior
- Accuracy vs computational cost trade-off

Example Outcomes
- Monte Carlo price converges to Black–Scholes benchmark
- Variance reduction significantly lowers estimator variance
- Pathwise Delta closely matches analytical Delta
- Asian option priced via full path simulation
- Empirical convergence rate matches theoretical expectation

<br/>

## Tech Stack
- Python
- NumPy
- SciPy
- Matplotlib
- Dataclasses
- Statistical bootstrapping
- Vectorized numerical computation

<br/>

## Project Structure
- OptionParams
- BlackScholesAnalytical
- MonteCarloPricing
- GreeksEstimation
- ExoticOptionPricing
- ConvergenceAnalysis
- PerformanceBenchmark
- Visualization utilities

<br/>

## Future Work
- Quasi-Monte Carlo (Sobol sequences)
- Barrier option pricing
- Implied volatility solver (Newton–Raphson)
- Multi-asset basket options
- Parallelized Monte Carlo
- C++ performance comparison

<br/>

## What This Demonstrates
- Risk-neutral valuation framework
- Statistical consistency of Monte Carlo estimators
- Bias–variance tradeoffs in Greeks estimation
- Efficiency gains from variance reduction
- Numerical stability and convergence diagnostics
- Path-dependent derivative valuation

Computational performance awareness

This project treats derivatives pricing as structured stochastic computation under uncertainty, validated empirically and theoretically.
