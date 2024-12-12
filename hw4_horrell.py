# Import libraries
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

# Load the dataset
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv")

# I still don't know if this is right. I wanted to limit the data to 
# ONLY users who could have seen a relevant gate. Any of the users who 
# had not seen a relevant gate could skew the data or add noise.
data = data[data['sum_gamerounds'] >= 30]
# data.head(), data.describe()

# Extract retention data by version
retention1_A = data[data['version'] == 'gate_30']['retention_1'].astype(int).values
retention1_B = data[data['version'] == 'gate_40']['retention_1'].astype(int).values

retention7_A = data[data['version'] == 'gate_30']['retention_7'].astype(int).values
retention7_B = data[data['version'] == 'gate_40']['retention_7'].astype(int).values


# 1-day retention model
with pm.Model() as model_retention1:
    # Priors
    # I don't have any info to suggest what these might be, so used uniform. 
    # Another contender might be a beta distribution with alpha = beta = 2
    p_A = pm.Uniform("p_A", 0, 1)
    p_B = pm.Uniform("p_B", 0, 1)
    
    # Difference in retention rates
    delta = pm.Deterministic("delta", p_A - p_B)
    
    # Likelihoods
    obs_A = pm.Bernoulli("obs_A", p_A, observed=retention1_A)
    obs_B = pm.Bernoulli("obs_B", p_B, observed=retention1_B)
    
    # Use Metropolis sampler
    step = pm.Metropolis()
    # Originslly set to 2 chains, but got the following during model run:
    # "We recommend running at least 4 chains for robust computation of convergence diagnostics"
    trace_retention1 = pm.sample(40000, step=step, chains=4)

# Visualize 1-day model
az.plot_posterior(trace_retention1, var_names=["p_A","p_B","delta"], hdi_prob=0.95, figsize=(12, 6))
plt.show()

# 7-day retention model
with pm.Model() as model_retention7:
    # Priors
    p_A = pm.Uniform("p_A", 0, 1)
    p_B = pm.Uniform("p_B", 0, 1)
    
    # Difference in retention rates
    delta = pm.Deterministic("delta", p_A - p_B)
    
    # Likelihoods
    obs_A = pm.Bernoulli("obs_A", p_A, observed=retention7_A)
    obs_B = pm.Bernoulli("obs_B", p_B, observed=retention7_B)
    
    # Use Metropolis sampler
    step = pm.Metropolis()
    trace_retention7 = pm.sample(40000, step=step, chains=4)

# Visualize 7-day model
az.plot_posterior(trace_retention7, var_names=["p_A","p_B","delta"], hdi_prob=0.95, figsize=(12, 6))
plt.show()

# Plot posterior distributions of delta for 1-day and 7-day retention models
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# 1-Day Retention
az.plot_posterior(trace_retention1, var_names=["delta"], hdi_prob=0.95, ax=ax[0])
ax[0].set_title("1-Day Retention (delta)")
# 7-Day Retention
az.plot_posterior(trace_retention7, var_names=["delta"], hdi_prob=0.95, ax=ax[1])
ax[1].set_title("7-Day Retention (delta)")
plt.show()

# Extract posterior samples for 1-day retention
p_A_samples_1 = np.concatenate(trace_retention1.posterior.p_A.values[:, 1000:])
p_B_samples_1 = np.concatenate(trace_retention1.posterior.p_B.values[:, 1000:])
delta_samples_1 = np.concatenate(trace_retention1.posterior.delta.values[:, 1000:])

# Extract posterior samples for 7-day retention
p_A_samples_7 = np.concatenate(trace_retention7.posterior.p_A.values[:, 1000:])
p_B_samples_7 = np.concatenate(trace_retention7.posterior.p_B.values[:, 1000:])
delta_samples_7 = np.concatenate(trace_retention7.posterior.delta.values[:, 1000:])

# Calculate probabilities for 1-day retention
print("1-Day Retention:")
print("Probability gate_30 is WORSE than gate_40: %.3f" % np.mean(delta_samples_1 < 0))
print("Probability gate_30 is BETTER than gate_40: %.3f" % np.mean(delta_samples_1 > 0))

# Calculate probabilities for 7-day retention
print("\n7-Day Retention:")
print("Probability gate_30 is WORSE than gate_40: %.3f" % np.mean(delta_samples_7 < 0))
print("Probability gate_30 is BETTER than gate_40: %.3f" % np.mean(delta_samples_7 > 0))



