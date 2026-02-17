import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

@dataclass
class OptionParams:
    """Option pricing parameters"""
    S0: float = 100.0      # Initial stock price
    K: float = 105.0       # Strike price
    T: float = 1.0         # Time to maturity (years)
    r: float = 0.05        # Risk-free rate
    sigma: float = 0.2     # Volatility
    mu: float = 0.05       # Drift (for simulation, replaced by r in risk-neutral)

class BlackScholesAnalytical:
    """Black-Scholes analytical pricing formulas"""

    @staticmethod
    def calculate_d1_d2(params: OptionParams) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters"""
        d1 = (np.log(params.S0 / params.K) +
              (params.r + 0.5 * params.sigma**2) * params.T) / (params.sigma * np.sqrt(params.T))
        d2 = d1 - params.sigma * np.sqrt(params.T)
        return d1, d2

    @staticmethod
    def european_call(params: OptionParams) -> float:
        """Calculate European call option price"""
        d1, d2 = BlackScholesAnalytical.calculate_d1_d2(params)
        call_price = (params.S0 * norm.cdf(d1) -
                     params.K * np.exp(-params.r * params.T) * norm.cdf(d2))
        return call_price

    @staticmethod
    def delta(params: OptionParams) -> float:
        """Calculate analytical delta"""
        d1, _ = BlackScholesAnalytical.calculate_d1_d2(params)
        return norm.cdf(d1)

    @staticmethod
    def gamma(params: OptionParams) -> float:
        """Calculate analytical gamma"""
        d1, _ = BlackScholesAnalytical.calculate_d1_d2(params)
        gamma = norm.pdf(d1) / (params.S0 * params.sigma * np.sqrt(params.T))
        return gamma

    @staticmethod
    def vega(params: OptionParams) -> float:
        """Calculate analytical vega"""
        d1, _ = BlackScholesAnalytical.calculate_d1_d2(params)
        vega = params.S0 * norm.pdf(d1) * np.sqrt(params.T)
        return vega

    @staticmethod
    def theta(params: OptionParams) -> float:
        """Calculate analytical theta (per day)"""
        d1, d2 = BlackScholesAnalytical.calculate_d1_d2(params)
        theta = (- (params.S0 * norm.pdf(d1) * params.sigma) / (2 * np.sqrt(params.T))
                 - params.r * params.K * np.exp(-params.r * params.T) * norm.cdf(d2))
        return theta / 365  # Daily theta

class MonteCarloPricing:
    """Monte Carlo simulation for option pricing"""

    def __init__(self, params: OptionParams):
        self.params = params

    def simulate_terminal_prices(self, n_paths: int, antithetic: bool = False) -> np.ndarray:
        """
        Simulate terminal asset prices using GBM

        S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        """
        if antithetic:
            # Generate half the paths and their antithetic pairs
            n_half = n_paths // 2
            Z = np.random.standard_normal(n_half)
            Z = np.concatenate([Z, -Z])
        else:
            Z = np.random.standard_normal(n_paths)

        drift = (self.params.r - 0.5 * self.params.sigma**2) * self.params.T
        diffusion = self.params.sigma * np.sqrt(self.params.T) * Z

        S_T = self.params.S0 * np.exp(drift + diffusion)
        return S_T

    def price_european_call(self, n_paths: int, use_antithetic: bool = False) -> dict:
        """
        Price European call option using Monte Carlo
        Returns price, variance, standard error, and confidence interval
        """
        S_T = self.simulate_terminal_prices(n_paths, use_antithetic)
        payoffs = np.maximum(S_T - self.params.K, 0)
        discounted_payoffs = np.exp(-self.params.r * self.params.T) * payoffs

        price = np.mean(discounted_payoffs)
        variance = np.var(discounted_payoffs, ddof=1)
        std_error = np.sqrt(variance / n_paths)

        # 95% confidence interval
        ci_margin = 1.96 * std_error
        ci_lower = price - ci_margin
        ci_upper = price + ci_margin

        return {
            'price': price,
            'variance': variance,
            'std_error': std_error,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_paths': n_paths
        }

    def control_variate_pricing(self, n_paths: int) -> dict:
        """
        Price using control variate technique with terminal stock price as control
        """
        # Generate random numbers
        Z = np.random.standard_normal(n_paths)

        # Simulate terminal prices
        drift = (self.params.r - 0.5 * self.params.sigma**2) * self.params.T
        diffusion = self.params.sigma * np.sqrt(self.params.T) * Z
        S_T = self.params.S0 * np.exp(drift + diffusion)

        # Option payoffs
        option_payoffs = np.maximum(S_T - self.params.K, 0)
        discounted_option = np.exp(-self.params.r * self.params.T) * option_payoffs

        # Control variate: stock price (known expectation)
        control = S_T
        expected_control = self.params.S0 * np.exp(self.params.r * self.params.T)

        # Calculate optimal coefficient
        covariance = np.cov(discounted_option, control)[0, 1]
        variance_control = np.var(control, ddof=1)
        beta = covariance / variance_control

        # Adjust option payoffs
        adjusted_payoffs = discounted_option - beta * (control - expected_control)

        price = np.mean(adjusted_payoffs)
        variance = np.var(adjusted_payoffs, ddof=1)
        std_error = np.sqrt(variance / n_paths)

        # Variance reduction ratio
        original_variance = np.var(discounted_option, ddof=1)
        reduction_ratio = 1 - variance / original_variance

        return {
            'price': price,
            'variance': variance,
            'std_error': std_error,
            'beta': beta,
            'reduction_ratio': reduction_ratio,
            'n_paths': n_paths
        }

class GreeksEstimation:
    """Estimate Greeks using various methods"""

    def __init__(self, params: OptionParams):
        self.params = params
        self.mc = MonteCarloPricing(params)

    def pathwise_delta(self, n_paths: int) -> Tuple[float, float]:
        """
        Estimate delta using pathwise derivative method
        Delta = e^{-rT} * 1{S_T > K} * (S_T / S0)
        """
        S_T = self.mc.simulate_terminal_prices(n_paths)
        indicator = (S_T > self.params.K).astype(float)
        delta_estimates = np.exp(-self.params.r * self.params.T) * indicator * (S_T / self.params.S0)

        delta = np.mean(delta_estimates)
        std_error = np.std(delta_estimates, ddof=1) / np.sqrt(n_paths)

        return delta, std_error

    def finite_difference_delta(self, n_paths: int, epsilon: float = 0.01) -> Tuple[float, float]:
        """
        Estimate delta using finite difference
        """
        # Prices with bumped S0
        params_up = OptionParams(
            S0=self.params.S0 * (1 + epsilon),
            K=self.params.K,
            T=self.params.T,
            r=self.params.r,
            sigma=self.params.sigma
        )
        params_down = OptionParams(
            S0=self.params.S0 * (1 - epsilon),
            K=self.params.K,
            T=self.params.T,
            r=self.params.r,
            sigma=self.params.sigma
        )

        # Use same random numbers for both simulations
        Z = np.random.standard_normal(n_paths)

        def price_with_S0(params, Z):
            drift = (params.r - 0.5 * params.sigma**2) * params.T
            diffusion = params.sigma * np.sqrt(params.T) * Z
            S_T = params.S0 * np.exp(drift + diffusion)
            payoffs = np.maximum(S_T - params.K, 0)
            return np.exp(-params.r * params.T) * np.mean(payoffs)

        price_up = price_with_S0(params_up, Z)
        price_down = price_with_S0(params_down, Z)

        delta = (price_up - price_down) / (2 * self.params.S0 * epsilon)

        # Bootstrap for standard error
        n_bootstrap = 100
        bootstrap_deltas = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_paths, n_paths, replace=True)
            price_up_boot = price_with_S0(params_up, Z[indices])
            price_down_boot = price_with_S0(params_down, Z[indices])
            delta_boot = (price_up_boot - price_down_boot) / (2 * self.params.S0 * epsilon)
            bootstrap_deltas.append(delta_boot)

        std_error = np.std(bootstrap_deltas)

        return delta, std_error

class ExoticOptionPricing:
    """Pricing exotic options via path simulation"""

    def __init__(self, params: OptionParams):
        self.params = params

    def simulate_paths(self, n_paths: int, n_steps: int) -> np.ndarray:
        """
        Simulate full price paths using discretized GBM
        Returns array of shape (n_paths, n_steps + 1)
        """
        dt = self.params.T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize price paths
        prices = np.zeros((n_paths, n_steps + 1))
        prices[:, 0] = self.params.S0

        # Generate random numbers
        Z = np.random.standard_normal((n_paths, n_steps))

        # Simulate paths
        for t in range(n_steps):
            drift = (self.params.r - 0.5 * self.params.sigma**2) * dt
            diffusion = self.params.sigma * sqrt_dt * Z[:, t]
            prices[:, t + 1] = prices[:, t] * np.exp(drift + diffusion)

        return prices

    def price_asian_call(self, n_paths: int, n_steps: int = 252) -> dict:
        """
        Price Asian call option using arithmetic average
        """
        prices = self.simulate_paths(n_paths, n_steps)

        # Calculate arithmetic average (excluding initial price)
        avg_prices = np.mean(prices[:, 1:], axis=1)

        # Asian option payoff
        payoffs = np.maximum(avg_prices - self.params.K, 0)
        discounted_payoffs = np.exp(-self.params.r * self.params.T) * payoffs

        price = np.mean(discounted_payoffs)
        variance = np.var(discounted_payoffs, ddof=1)
        std_error = np.sqrt(variance / n_paths)

        return {
            'price': price,
            'variance': variance,
            'std_error': std_error,
            'n_paths': n_paths,
            'n_steps': n_steps
        }

class ConvergenceAnalysis:
    """Analyze convergence of Monte Carlo estimators"""

    def __init__(self, params: OptionParams):
        self.params = params
        self.bs = BlackScholesAnalytical()
        self.mc = MonteCarloPricing(params)
        self.true_price = self.bs.european_call(params)

    def convergence_study(self, path_counts: list, n_trials: int = 10) -> dict:
        """
        Study convergence of Monte Carlo estimator
        """
        results = {
            'n_paths': [],
            'mean_error': [],
            'std_error': [],
            'mean_std': [],
            'mean_ci_width': []
        }

        for n in path_counts:
            errors = []
            std_errors = []
            ci_widths = []

            for _ in range(n_trials):
                result = self.mc.price_european_call(n)
                error = abs(result['price'] - self.true_price)
                errors.append(error)
                std_errors.append(result['std_error'])
                ci_widths.append(result['ci_upper'] - result['ci_lower'])

            results['n_paths'].append(n)
            results['mean_error'].append(np.mean(errors))
            results['std_error'].append(np.std(errors))
            results['mean_std'].append(np.mean(std_errors))
            results['mean_ci_width'].append(np.mean(ci_widths))

        return results

    def variance_comparison(self, n_paths: int = 100000) -> dict:
        """
        Compare variance of different estimators
        """
        # Standard MC
        std_result = self.mc.price_european_call(n_paths)

        # Antithetic variates
        antithetic_result = self.mc.price_european_call(n_paths, use_antithetic=True)

        # Control variates
        cv_result = self.mc.control_variate_pricing(n_paths)

        return {
            'standard': {
                'variance': std_result['variance'],
                'std_error': std_result['std_error'],
                'price': std_result['price']
            },
            'antithetic': {
                'variance': antithetic_result['variance'],
                'std_error': antithetic_result['std_error'],
                'price': antithetic_result['price']
            },
            'control_variate': {
                'variance': cv_result['variance'],
                'std_error': cv_result['std_error'],
                'price': cv_result['price'],
                'reduction_ratio': cv_result['reduction_ratio']
            }
        }

class PerformanceBenchmark:
    """Benchmark computational performance"""

    @staticmethod
    def benchmark_monte_carlo(params: OptionParams, path_counts: list) -> dict:
        """
        Benchmark runtime for different numbers of paths
        """
        mc = MonteCarloPricing(params)
        results = {'n_paths': [], 'time': [], 'price': [], 'std_error': []}

        for n in path_counts:
            start_time = time.time()
            result = mc.price_european_call(n)
            end_time = time.time()

            results['n_paths'].append(n)
            results['time'].append(end_time - start_time)
            results['price'].append(result['price'])
            results['std_error'].append(result['std_error'])

        # Calculate time per million paths
        for i in range(len(results['n_paths'])):
            million_paths = results['n_paths'][i] / 1e6
            if million_paths > 0:
                results[f'time_per_million'] = results['time'][i] / million_paths

        return results

def plot_convergence(convergence_results: dict):
    """Plot convergence analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Log-log plot of error vs paths
    axes[0, 0].loglog(convergence_results['n_paths'], convergence_results['mean_error'],
                     'bo-', label='Actual Error')

    # Fit line to estimate convergence rate
    log_n = np.log(convergence_results['n_paths'])
    log_error = np.log(convergence_results['mean_error'])
    slope, intercept = np.polyfit(log_n, log_error, 1)

    axes[0, 0].loglog(convergence_results['n_paths'],
                     np.exp(intercept) * np.array(convergence_results['n_paths'])**slope,
                     'r--', label=f'Fit: slope = {slope:.3f}')
    axes[0, 0].loglog(convergence_results['n_paths'],
                     1/np.sqrt(convergence_results['n_paths']),
                     'g:', label='Theoretical: -0.5')
    axes[0, 0].set_xlabel('Number of Paths')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('Convergence Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Standard error vs paths
    axes[0, 1].loglog(convergence_results['n_paths'], convergence_results['mean_std'],
                     'bo-', label='Standard Error')
    axes[0, 1].loglog(convergence_results['n_paths'],
                     1/np.sqrt(convergence_results['n_paths']),
                     'r--', label='1/√N')
    axes[0, 1].set_xlabel('Number of Paths')
    axes[0, 1].set_ylabel('Standard Error')
    axes[0, 1].set_title('Standard Error Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Confidence interval width
    axes[1, 0].loglog(convergence_results['n_paths'], convergence_results['mean_ci_width'],
                     'bo-')
    axes[1, 0].set_xlabel('Number of Paths')
    axes[1, 0].set_ylabel('95% CI Width')
    axes[1, 0].set_title('Confidence Interval Width')
    axes[1, 0].grid(True)

    # Error distribution
    axes[1, 1].hist(convergence_results['mean_error'], bins=20, edgecolor='black')
    axes[1, 1].set_xlabel('Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_variance_comparison(variance_results: dict):
    """Plot variance comparison"""
    methods = list(variance_results.keys())
    variances = [variance_results[m]['variance'] for m in methods]
    std_errors = [variance_results[m]['std_error'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Variance comparison
    x = np.arange(len(methods))
    ax1.bar(x, variances)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel('Variance')
    ax1.set_title('Estimator Variance Comparison')
    ax1.grid(True, alpha=0.3)

    # Standard error comparison
    ax2.bar(x, std_errors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('Standard Error')
    ax2.set_title('Standard Error Comparison')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_performance(perf_results: dict):
    """Plot performance benchmarks"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Runtime vs paths
    ax1.loglog(perf_results['n_paths'], perf_results['time'], 'bo-')
    ax1.set_xlabel('Number of Paths')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Computational Performance')
    ax1.grid(True)

    # Price convergence with confidence intervals
    ax2.semilogx(perf_results['n_paths'], perf_results['price'], 'bo-', label='Estimated Price')
    ax2.axhline(y=perf_results['price'][-1], color='r', linestyle='--',
                label='Final Estimate')
    ax2.fill_between(perf_results['n_paths'],
                     np.array(perf_results['price']) - np.array(perf_results['std_error']),
                     np.array(perf_results['price']) + np.array(perf_results['std_error']),
                     alpha=0.3, label='±1 Std Error')
    ax2.set_xlabel('Number of Paths')
    ax2.set_ylabel('Option Price')
    ax2.set_title('Price Convergence')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("=" * 60)
    print("STOCHASTIC DERIVATIVES PRICING ENGINE")
    print("=" * 60)

    # Initialize parameters
    params = OptionParams(S0=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2)

    # 1. Black-Scholes Analytical Pricing
    print("\n1. BLACK-SCHOLES ANALYTICAL PRICING")
    print("-" * 40)
    bs = BlackScholesAnalytical()
    bs_price = bs.european_call(params)
    bs_delta = bs.delta(params)
    bs_gamma = bs.gamma(params)
    bs_vega = bs.vega(params)
    bs_theta = bs.theta(params)

    print(f"European Call Option Price: ${bs_price:.4f}")
    print(f"Delta: {bs_delta:.4f}")
    print(f"Gamma: {bs_gamma:.6f}")
    print(f"Vega: {bs_vega:.4f}")
    print(f"Theta (daily): {bs_theta:.6f}")

    # 2. Monte Carlo Pricing
    print("\n2. MONTE CARLO PRICING")
    print("-" * 40)
    mc = MonteCarloPricing(params)

    n_paths = 100000
    mc_result = mc.price_european_call(n_paths)
    print(f"Standard MC (n={n_paths:,}):")
    print(f"  Price: ${mc_result['price']:.4f}")
    print(f"  Variance: {mc_result['variance']:.6f}")
    print(f"  Std Error: {mc_result['std_error']:.6f}")
    print(f"  95% CI: [${mc_result['ci_lower']:.4f}, ${mc_result['ci_upper']:.4f}]")
    print(f"  Error vs BS: ${abs(mc_result['price'] - bs_price):.6f}")

    # 3. Variance Reduction Techniques
    print("\n3. VARIANCE REDUCTION TECHNIQUES")
    print("-" * 40)

    # Antithetic variates
    antithetic_result = mc.price_european_call(n_paths, use_antithetic=True)
    print(f"Antithetic Variates:")
    print(f"  Price: ${antithetic_result['price']:.4f}")
    print(f"  Variance: {antithetic_result['variance']:.6f}")
    print(f"  Variance Reduction: {(1 - antithetic_result['variance']/mc_result['variance'])*100:.2f}%")

    # Control variates
    cv_result = mc.control_variate_pricing(n_paths)
    print(f"Control Variates:")
    print(f"  Price: ${cv_result['price']:.4f}")
    print(f"  Variance: {cv_result['variance']:.6f}")
    print(f"  Optimal Beta: {cv_result['beta']:.4f}")
    print(f"  Variance Reduction: {cv_result['reduction_ratio']*100:.2f}%")

    # 4. Greeks Estimation
    print("\n4. GREEKS ESTIMATION")
    print("-" * 40)
    greeks = GreeksEstimation(params)

    # Pathwise delta
    pathwise_delta, pathwise_se = greeks.pathwise_delta(n_paths)
    print(f"Pathwise Delta: {pathwise_delta:.4f} ± {2*pathwise_se:.4f} (95% CI)")
    print(f"  Error vs Analytical: {abs(pathwise_delta - bs_delta):.6f}")

    # Finite difference delta
    fd_delta, fd_se = greeks.finite_difference_delta(n_paths)
    print(f"Finite Difference Delta: {fd_delta:.4f} ± {2*fd_se:.4f} (95% CI)")
    print(f"  Error vs Analytical: {abs(fd_delta - bs_delta):.6f}")

    # 5. Exotic Option Pricing
    print("\n5. EXOTIC OPTION PRICING (ASIAN OPTION)")
    print("-" * 40)
    exotic = ExoticOptionPricing(params)
    asian_result = exotic.price_asian_call(n_paths // 10, n_steps=252)  # Fewer paths for path-dependent
    print(f"Asian Call Option (arithmetic average):")
    print(f"  Price: ${asian_result['price']:.4f}")
    print(f"  Std Error: {asian_result['std_error']:.6f}")
    print(f"  Note: No closed-form solution available")

    # 6. Convergence Analysis
    print("\n6. CONVERGENCE ANALYSIS")
    print("-" * 40)
    convergence = ConvergenceAnalysis(params)
    path_counts = [1000, 5000, 10000, 50000, 100000, 500000]

    print("Running convergence study...")
    conv_results = convergence.convergence_study(path_counts, n_trials=5)

    print("\nConvergence Results:")
    print(f"{'Paths':>10} {'Mean Error':>12} {'Std Error':>12} {'CI Width':>12}")
    print("-" * 50)
    for i, n in enumerate(conv_results['n_paths']):
        print(f"{n:10,d} {conv_results['mean_error'][i]:12.6f} "
              f"{conv_results['std_error'][i]:12.6f} {conv_results['mean_ci_width'][i]:12.6f}")

    # Estimate convergence rate
    log_n = np.log(conv_results['n_paths'])
    log_error = np.log(conv_results['mean_error'])
    slope, intercept = np.polyfit(log_n, log_error, 1)
    print(f"\nEstimated convergence rate: O(N^{slope:.3f})")
    print(f"Theoretical rate: O(N^-0.5)")

    # 7. Variance Comparison
    print("\n7. VARIANCE COMPARISON")
    print("-" * 40)
    var_results = convergence.variance_comparison(n_paths=100000)
    print(f"Standard MC Variance: {var_results['standard']['variance']:.8f}")
    print(f"Antithetic Variance: {var_results['antithetic']['variance']:.8f}")
    print(f"Control Variate Variance: {var_results['control_variate']['variance']:.8f}")
    print(f"Control Variate Reduction: {var_results['control_variate']['reduction_ratio']*100:.2f}%")

    # 8. Performance Benchmarking
    print("\n8. PERFORMANCE BENCHMARKING")
    print("-" * 40)
    perf = PerformanceBenchmark()
    perf_results = perf.benchmark_monte_carlo(params, [10000, 50000, 100000, 500000, 1000000])

    print(f"\nRuntime for 1M paths: {perf_results['time'][-1]/perf_results['n_paths'][-1]*1e6:.3f} seconds")
    print("\nDetailed Performance:")
    print(f"{'Paths':>10} {'Time (s)':>12} {'Price':>10} {'Std Error':>12}")
    print("-" * 50)
    for i in range(len(perf_results['n_paths'])):
        print(f"{perf_results['n_paths'][i]:10,d} {perf_results['time'][i]:12.3f} "
              f"{perf_results['price'][i]:10.4f} {perf_results['std_error'][i]:12.6f}")

    # Generate plots
    print("\nGenerating plots...")

    # Plot convergence
    plot_convergence(conv_results)

    # Plot variance comparison
    plot_variance_comparison(var_results)

    # Plot performance
    plot_performance(perf_results)

    # Additional validation plot: Monte Carlo vs Analytical
    fig, ax = plt.subplots(figsize=(10, 6))

    # Vary strike prices
    strikes = np.arange(80, 131, 5)
    mc_prices = []
    bs_prices = []
    mc_errors = []

    for K in strikes:
        params.K = K
        bs = BlackScholesAnalytical()
        bs_price = bs.european_call(params)
        bs_prices.append(bs_price)

        mc = MonteCarloPricing(params)
        mc_result = mc.price_european_call(100000)
        mc_prices.append(mc_result['price'])
        mc_errors.append(mc_result['std_error'])

    ax.errorbar(strikes, mc_prices, yerr=2*np.array(mc_errors),
                fmt='bo', capsize=3, label='Monte Carlo (95% CI)')
    ax.plot(strikes, bs_prices, 'r-', linewidth=2, label='Black-Scholes')
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Option Price')
    ax.set_title('Monte Carlo vs Black-Scholes: Varying Strike Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()