# Title: Simulating an autoregressive process

# Notes:
    #* Description: Script for simulating an AR process
    #* Updated: 2022-01-14
    #* Updated by: DCR 

# Setup
    #* Import modules
from traceback import clear_frames
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess 
    #* set seed - uses Mersenne Twister algorithm like R so setting the same seed should be good
np.random.seed(23923)

# Simulating an AR(1) process where phi = 0.9
ar1 = np.array([1, 0.9])
ma1 = np.array([1])

arProcessPhi09 = ArmaProcess(ar1, ma1).generate_sample(nsample = 100)

# Simulating an AR(1) process where phi = 0
ar1 = np.array([1,0])
ma1 = np.array([1])

arProcessPhi00 = ArmaProcess(ar1,ma1).generate_sample(nsample = 100)

# Simulating an AR(1) process where phi = 0.2
ar1 = np.array([1, 0.2])
ma1 = np.array([1])

arProcessPhi02 = ArmaProcess(ar1, ma1).generate_sample(nsample = 100)

# Simulating an Ar(1) process where phi = -0.9
ar1 = np.array([1, -0.9])
ma1 = np.array([1])

arProcessPhiNeg09 = ArmaProcess(ar1, ma1).generate_sample(nsample = 100)

# Simulating an AR(3) process where phi = 0.9, -0.5, 0.1
ar3 = np.array([1, 0.9, -0.5, 0.1])
ma3 = np.array([1])

arProcess3 = ArmaProcess(ar3, ma3).generate_sample(nsample = 100)

# Plots
fig, axs = plt.subplots(2,2)
axs[0,0].plot(arProcessPhi00)
axs[0,0].set_title('Phi = 0')
axs[0,1].plot(arProcessPhi02)
axs[0,1].set_title('Phi = 0.2')
axs[1,0].plot(arProcessPhi09)
axs[1,0].set_title('Phi = 0.9')
axs[1,1].plot(arProcessPhiNeg09)
axs[1,1].set_title('Phi = -0.9')

fig.tight_layout(pad = 3)
plt.savefig('figures/1_introduction/ar1_phi.png')
plt.clf()

plt.plot(arProcess3)
plt.title('AR(3) w/ phi = 0.9, -0.5, 0.1')
plt.savefig('figures/1_introduction/ar3.png')
plt.clf()

# Q's
# 1. what happens as phi becomes positive and large?
ar1 = np.array([1, 0.99])
ma1 = np.array([1])

arProcessPhi099 = ArmaProcess(ar1, ma1).generate_sample(nsample=100)

fig, axs = plt.subplots(2)
fig.suptitle('AR(1) w/ phi = 0.99')
axs[0].plot(arProcessPhi099)
axs[1].hist(arProcessPhi099, bins = 5)

fig.tight_layout(pad = 3.0)
plt.savefig('figures/1_introduction/large_phi.png')
plt.clf()

    #* We have positive autocorrelation (a value of error at a point in time is positively correlated with other time points). We also have a stronger relationship the larger it is. Here, my phi is greater than 1 meaning that the autocorrelation is positive, and as it is larger than 1 we do not have stationarity. 
# 2. negative and large?
ar1 = np.array([1, -0.99])
ma1 = np.array([1])

arProcessPhiNeg099 = ArmaProcess(ar1, ma1).generate_sample(nsample=100)

fig, axs = plt.subplots(2)
fig.suptitle('AR(1) w/ phi = 0.99')
axs[0].plot(arProcessPhiNeg099)
axs[1].hist(arProcessPhiNeg099, bins = 5)

fig.tight_layout(pad = 3.0)
plt.savefig('figures/1_introduction/large_neg_phi.png')
plt.clf()
    #* We have negative autocorrelation (a value of error at a point in time is negatively correlated with other time points). We also have a stronger relationship the larger it is. Here, the absolute value of my phi is less than -1 which means that the autocorrelation is negative, and as it's absolute value is larger than 1 (less than -1), we do not have stationarity
# 3. Near 0? how does a histogram looks? Does it look normal?
ar1 = np.array([1, 0])
ma1 = np.array([1])

arProcessPhi0 = ArmaProcess(ar1, ma1).generate_sample(nsample=100)

fig, axs = plt.subplots(2)
fig.suptitle('AR(1) w/ phi = 0.99')
axs[0].plot(arProcessPhi0)
axs[1].hist(arProcessPhi0, bins = 5)

fig.tight_layout(pad = 3.0)
plt.savefig('figures/1_introduction/large_zero_phi.png')
plt.clf()
    #* It looks more like a normal distribution than what you get with the negative phi. It does look less normal than the positive phi. Though I think this is just an artifact of the seed. An AR(1) where phi = 0 means we have less correlation between values over time. 