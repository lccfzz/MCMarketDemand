# %%
# MSD Hometask
# Monte carlo that replicates the stochastic behaviour of market demand

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import logistic, gamma, norm
from scipy.stats import ks_2samp
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from fitter import Fitter

df = pd.read_csv('data.csv', delimiter=';')

#Plot DEMAND/FORECAST vs MONTH for all MARKET_ID

def plot_demand_over_time(df,market_id):
    market_filter = df[df['MARKET_ID'] == market_id]
    plt.figure(figsize=(8, 5))
    plt.plot(market_filter["MONTH"], market_filter["FORECAST"], label="Forecast", marker="o")
    plt.plot(market_filter["MONTH"], market_filter["DEMAND"], label="Demand", marker="o")

    plt.xlabel("Month")
    plt.ylabel("Values")
    plt.title(f"Product {market_id}: Demand and Forecast vs Month")
    plt.legend()
    plt.grid(True)
    plt.show()

for id in df['MARKET_ID'].unique():
    plot_demand_over_time(df,id)

market_1 = df[df['MARKET_ID'] == 1]
market_2 = df[df['MARKET_ID'] == 2]
market_3 = df[df['MARKET_ID'] == 3]

# Plot demand/forecast trends
plt.figure(figsize=(15, 8))

# Market 1
plt.subplot(3, 1, 1)
plt.plot(market_1['MONTH'], market_1['DEMAND'], marker='x', label='Market 1 Demand', color='black')
plt.plot(market_1['MONTH'], market_1['FORECAST'], marker='o', label='Market 1 Forecast')
plt.title('Market 1 Forecast/Demand')
plt.xlabel('Month')
plt.ylabel('Forecast & Demand')
plt.legend()
plt.grid()

# Market 2
plt.subplot(3, 1, 2)
plt.plot(market_2['MONTH'], market_2['DEMAND'], marker='x', label='Market 2 Demand', color='black')
plt.plot(market_2['MONTH'], market_2['FORECAST'], marker='o', label='Market 2 Forecast', color='orange')
plt.title('Market 2 Forecast/Demand')
plt.xlabel('Month')
plt.ylabel('Forecast & Demand')
plt.legend()
plt.grid()

# Market 3
plt.subplot(3, 1, 3)
plt.plot(market_3['MONTH'], market_3['DEMAND'], marker='x', label='Market 3 Demand', color='black')
plt.plot(market_3['MONTH'], market_3['FORECAST'], marker='o', label='Market 3 Forecast', color='green')
plt.title('Market 3 Forecast/Demand')
plt.xlabel('Month')
plt.ylabel('Forecast & Demand')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
# %%
# Plot histograms
plt.figure(figsize=(15, 8))

# Market 1
plt.subplot(3, 1, 1)
plt.hist(market_1['DEMAND'], bins=10, color='blue', alpha=0.7, edgecolor='black')
plt.title('Market 1 Demand Distribution')
plt.xlabel('Demand')
plt.ylabel('Frequency')

# Market 2
plt.subplot(3, 1, 2)
plt.hist(market_2['DEMAND'], bins=10, color='orange', alpha=0.7, edgecolor='black')
plt.title('Market 2 Demand Distribution')
plt.xlabel('Demand')
plt.ylabel('Frequency')

# Market 3
plt.subplot(3, 1, 3)
plt.hist(market_3['DEMAND'], bins=10, color='green', alpha=0.7, edgecolor='black')
plt.title('Market 3 Demand Distribution')
plt.xlabel('Demand')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# %%
months = np.linspace(1,24,24)
demand = market_3['DEMAND']

# Fit a trend using linear regression
model = LinearRegression()
model.fit(months.reshape(-1, 1), demand)
trend = model.predict(months.reshape(-1, 1))
residuals = demand - trend

f = Fitter(residuals, distributions=["norm", "expon", "gamma", "lognorm", "beta", "weibull_min", "logistic", "pareto"])
f.fit()

print("Best Fit Distribution:", f.get_best(method="sumsquare_error"))

num_simulations = 10000
a = 2.2339651120296966
loc = -366.1522346832018
scale = 163.90238128753919

simulated_residuals = gamma.rvs(a, loc, scale, size=(num_simulations, len(months)))
simulated_trend = np.tile(trend, (num_simulations, 1))
simulated_demand = simulated_trend + simulated_residuals

# Step 4: Optional - Add seasonality
decomposition = seasonal_decompose(demand, period=12, model='additive')
seasonality = decomposition.seasonal
simulated_demand += np.tile(seasonality, (num_simulations, 1))

# Step 5: Analyze results
simulated_mean_3 = simulated_demand.mean(axis=0)

plt.plot(months, simulated_mean_3, color='blue', alpha=0.7, label='Monte Carlo')
plt.plot(months, market_3['DEMAND'], color='red', alpha=0.7, label='Data')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")
plt.grid(True)
plt.show()

plt.hist(simulated_mean_3, bins=10, density=True, alpha=0.7, color='gray', edgecolor='black')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("MC Demand")
plt.show()

plt.hist(market_3['DEMAND'], bins=10, density=True, alpha=0.7, color='green', edgecolor='black')
plt.title("Empirical Data of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")
plt.show()

print("Historical Mean:", np.mean(demand))
print("Simulated Mean:", np.mean(simulated_mean_3))
print("Historical Variance:", np.var(demand))
print("Simulated Variance:", np.var(simulated_mean_3))

historical_residuals = demand - trend
simulated_residuals_flat = simulated_residuals.flatten()
ks_stat, p_value = ks_2samp(historical_residuals, simulated_residuals_flat)
print("KS Test Statistic:", ks_stat, "P-value:", p_value)

# %%
demand = market_2['DEMAND']

# Fit a trend using linear regression
model = LinearRegression()
model.fit(months.reshape(-1, 1), demand)
trend = model.predict(months.reshape(-1, 1))
residuals = demand - trend
f = Fitter(residuals, distributions=["norm", "expon", "gamma", "lognorm", "beta", "weibull_min", "logistic", "pareto"])
f.fit()

print("Best Fit Distribution:", f.get_best(method="sumsquare_error"))

num_simulations = 10000
loc = -5.210646728907401e-14
scale = 304.9036388426163

# Step 3: Simulate Monte Carlo values with trend + randomness
num_simulations = 10000
simulated_residuals = norm.rvs(loc, scale, size=(num_simulations, len(months)))
simulated_trend = np.tile(trend, (num_simulations, 1))
simulated_demand =simulated_trend+ simulated_residuals 

# Step 4: Optional - Add seasonality
decomposition = seasonal_decompose(demand, period=12, model='additive')
seasonality = decomposition.seasonal
simulated_demand += np.tile(seasonality, (num_simulations, 1))

# Step 5: Analyze results
simulated_mean_2 = simulated_demand.mean(axis=0)

plt.plot(months, simulated_mean_2, color='blue', alpha=0.7,label='Monte Carlo')
plt.plot(months, market_2['DEMAND'], color='red', alpha=0.7,label='Data')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")
plt.grid(True)
plt.show()

plt.hist(simulated_mean_2, bins=10, density=True, alpha=0.7, color='green', edgecolor='black')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")
plt.show()

plt.hist(market_2['DEMAND'], bins=10, density=True, alpha=0.7, color='orange', edgecolor='black')
plt.title("Empirical Data of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("MC Demand")
plt.show()

print("Historical Mean:", np.mean(demand))
print("Simulated Mean:", np.mean(simulated_mean_2))
print("Historical Variance:", np.var(demand))
print("Simulated Variance:", np.var(simulated_mean_2))

historical_residuals = demand - trend
simulated_residuals_flat = simulated_residuals.flatten()
ks_stat, p_value = ks_2samp(historical_residuals, simulated_residuals_flat)
print("KS Test Statistic:", ks_stat, "P-value:", p_value)

# %%
demand = market_1['DEMAND']

# Fit a trend using linear regression
model = LinearRegression()
model.fit(months.reshape(-1, 1), demand)
trend = model.predict(months.reshape(-1, 1))
residuals = demand - trend
f = Fitter(residuals, distributions=["norm", "expon", "gamma", "lognorm", "beta", "weibull_min", "logistic", "pareto"])
f.fit()

print("Best Fit Distribution:", f.get_best(method="sumsquare_error"))

num_simulations = 10000
loc = -2.800646122957684
scale = 25.06390684688853

num_simulations = 10000
simulated_residuals = logistic.rvs(loc, scale, size=(num_simulations, len(months)))
simulated_trend = np.tile(trend, (num_simulations, 1))
simulated_demand = simulated_trend + simulated_residuals

# Step 4: Optional - Add seasonality
decomposition = seasonal_decompose(demand, period=12, model='additive')
seasonality = decomposition.seasonal
simulated_demand += np.tile(seasonality, (num_simulations, 1))

# Step 5: Analyze results
simulated_mean_1 = simulated_demand.mean(axis=0)

plt.plot(months, simulated_mean_1, color='blue', alpha=0.7,label='Monte Carlo')
plt.plot(months, market_1['DEMAND'], color='red', alpha=0.7, label='Data')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")
plt.grid(True)
plt.show()

plt.hist(simulated_mean_1, bins=10, density=True, alpha=0.7, color='green', edgecolor='black')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("MC Results")
plt.show()

plt.hist(market_1['DEMAND'], bins=10, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title("Empirical Data of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")
plt.show()

print("Historical Mean:", np.mean(demand))
print("Simulated Mean:", np.mean(simulated_mean_1))
print("Historical Variance:", np.var(demand))
print("Simulated Variance:", np.var(simulated_mean_1))

historical_residuals = demand - trend
simulated_residuals_flat = simulated_residuals.flatten()
ks_stat, p_value = ks_2samp(historical_residuals, simulated_residuals_flat)
print("KS Test Statistic:", ks_stat, "P-value:", p_value)
# %%
plt.figure(figsize=(15, 8))

# Market 1
plt.subplot(3, 1, 1)
plt.plot(months, simulated_mean_1, marker='x', color='black', alpha=0.7,label='Monte Carlo')
plt.plot(months, market_1['DEMAND'], marker='o',color='blue', alpha=0.7, label='Data')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("MC Results")
plt.grid()

# Market 2
plt.subplot(3, 1, 2)
plt.plot(months, simulated_mean_2, marker='x', color='black', alpha=0.7,label='Monte Carlo')
plt.plot(months, market_2['DEMAND'], marker='o', color='orange', alpha=0.7,label='Data')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")
plt.grid()

# Market 3
plt.subplot(3, 1, 3)
plt.plot(months, simulated_mean_3, marker='x', color='black', alpha=0.7,label='Monte Carlo')
plt.plot(months, market_3['DEMAND'], marker='o',color='green', alpha=0.7,label='Data')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")
plt.grid()

plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.hist(simulated_mean_1, bins=10, density=True, alpha=0.7, color='gray', edgecolor='black')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("MC Results")

plt.subplot(2, 1, 2)
plt.hist(market_1['DEMAND'], bins=10, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.title("Empirical Data of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.hist(simulated_mean_2, bins=10, density=True, alpha=0.7, color='gray', edgecolor='black')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("MC Results")

plt.subplot(2, 1, 2)
plt.hist(market_2['DEMAND'], bins=10, density=True, alpha=0.7, color='orange', edgecolor='black')
plt.title("Empirical Data of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.hist(simulated_mean_3, bins=10, density=True, alpha=0.7, color='gray', edgecolor='black')
plt.title("Monte Carlo Simulation of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("MC Results")

plt.subplot(2, 1, 2)
plt.hist(market_3['DEMAND'], bins=10, density=True, alpha=0.7, color='green', edgecolor='black')
plt.title("Empirical Data of Market Demand")
plt.legend()
plt.xlabel("Months")
plt.ylabel("Demand")

plt.tight_layout()
plt.show()
# %%
