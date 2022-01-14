# Title: Simulate Unit Roots

# Notes:
    #* Description: script for simulating unit roots and examining their consequences
    #* Updated: 2022-01-14
    #* Updated by: DCR

# Setup
    #* load libraries
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
    #* set seed
np.random.seed(23923)

# Simulate a unit root
ar1 = np.array([1, 1])
ma1 = np.array([1])

ur = ArmaProcess(ar1, ma1).generate_sample(nsample=100)

# Simulate unit root with drift
urd = np.random.normal(0, 1, 100)
b0 = 0.6
for i in range(1, 99):
    urd[i] = b0 + 1*urd[i-1] + np.random.normal(0, 1, 1)

# Simulate unit root with deterministic trend
urdt = np.random.normal(0, 1, 100)
delta = 0.05
for i in range(1, 99):
    urdt[i] = delta*i + 1*urdt[i-1] + np.random.normal(0, 1, 1)

# Simulate unit root with deterministic trend and drift
urddt = np.random.normal(0, 1, 100)
delta = 0.05
b0 = 0.9
for i in range(1, 99):
    urddt[i] = delta*i + b0 + 1*urddt[i-1] + np.random.normal(0,1,1)

# Plots
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(ur)
axs[0, 0].set_title('Unit Root')
axs[0, 1].plot(urd)
axs[0, 1].set_title('Unit Root w/ drift')
axs[1, 0].plot(urdt)
axs[1, 0].set_title('Unit Root w/ determinisitic trend')
axs[1, 1].plot(urddt)
axs[1, 1].set_title('Unit Root w/ drift and deterministic trend')

fig.tight_layout(h_pad=3.0, w_pad = 6.0)
plt.savefig('figures/1_introduction/unit_roots.png')