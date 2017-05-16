from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from sklearn import linear_model
from energy_hw3_q1 import *


def scatter(F, adhub):
    plt.figure()
    plt.scatter(F, adhub)
    plt.xlabel("forward power price at PJMWH")
    plt.ylabel("forward power price at ADHUB")
    plt.title("scatter plot of forward prices")
    plt.show()

def regress(F, adhub):
    F = F.reshape(F.size, 1)
    adhub = adhub.reshape(adhub.size, 1)
    
    X = sm.add_constant(F)
    
    ols_model = sm.OLS(adhub, X)
    ols_results = ols_model.fit()
    
    regr = linear_model.LinearRegression()
    regr.fit(F, adhub)
    residuals = adhub - regr.predict(F).reshape(adhub.size, 1)
    
    return ols_results, regr, residuals
    
def simulate_adhub(regr, residuals, rho=0.7, F0=38.05, G0=2.956, sigF=0.4, sigG=0.35, T=168/252):
    F, G = simulate_forward(F0, G0, rho, sigF, sigG, T)
    residual_vol = np.std(residuals)
    simulated_adhub = regr.predict(F).reshape(F.size, 1) + residual_vol * np.random.randn(F.size, 1)
    return simulated_adhub, G
    
def adhub_revenue_put(F, adhub, H=7.2, K=5, capacity=800, L=1.5*1e6):
    N = capacity * 16 * 21
    _, regr, residuals = regress(F, adhub)
    simulated_adhub, G = simulate_adhub(regr, residuals)
    V = np.maximum(simulated_adhub - H * G - K, np.zeros((simulated_adhub.size, 1))) * N