# Title: Time Series Correlations

# Notes: 
    #* Description: Script for examining correlations over time
    #* Updated: 2022-01-14
    #* Updated by: DCR

# Setup
    #* Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    #* Set seed
np.random.seed(20395)

# create 8 uncorrelated series of length 60 from a normal dist.
uncorr = [np.random.normal(0, 1, 60) for i in range(8)]

for i in range(8):
    plt.plot(uncorr[i])
plt.show()
plt.clf()

uncorrDf = pd.DataFrame(uncorr).T
corrs = uncorrDf.corr(method = 'pearson')