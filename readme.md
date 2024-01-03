# Gaussian copula

A copula is a mathematical function used to describe the dependence between random variables. The Gaussian copula, in particular, is derived from the multivariate normal distribution. It captures the correlations between the variables but, unlike a joint distribution, does not require specifying their marginal distributions.

The Gaussian copula is defined as:

$$
C_\Sigma(u_1, \ldots, u_n) = \Phi_\Sigma(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_n))
$$

Where:

- \( C\_\Sigma \) is the Gaussian copula.
- \( \Phi\_\Sigma \) is the cumulative distribution function (CDF) of the multivariate normal distribution with correlation matrix \( \Sigma \).
- \( \Phi^{-1} \) is the inverse CDF (quantile function) of the standard normal distribution.
- \( u_1, \ldots, u_n \) are the marginal uniform distributions.

Brief explanation of how it works:

1. **Marginal Distributions**: First, you need to have the marginal distributions of the variables (entities) in question. These are the distributions of each variable considered in isolation.

2. **Transform to Uniform**: Using the cumulative distribution function (CDF) of these marginals, you transform the data to a uniform scale. This process is often referred to as "uniformization" or "copula transformation."

3. **Apply Gaussian Copula**: Once transformed, you apply the Gaussian copula to model the dependence structure. The Gaussian copula uses the correlation matrix derived from the original data, but applied to the transformed, uniformly distributed data.

4. **Inverse Transform**: To use this model for prediction or simulation, you typically generate correlated uniform variables using the copula and then transform them back to the original scale using the inverse CDFs of the marginals.

This method gained notoriety in the financial world, especially post-2008 financial crisis, as it was used extensively for pricing and risk management of complex financial instruments like collateralized debt obligations (CDOs). The Gaussian copula allowed for the modeling of dependencies between various assets, but its misuse and the underestimation of tail dependencies were criticized for contributing to the crisis.

## Install

Used data:

https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=365

https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=365

Uses Numpy, scipy, matplotlib, pandas. ta-lib.

### Apple silicon

```
brew install ta-lib
arch -arm64 brew install ta-lib
pip install TA-Lib
pip install numpy
pip install scipy matplotlib pandas
```

This is just a playground project.

### LICENCE

DO WHATEVER YOU WANT WITH IT.
This software is provided AS IS.
