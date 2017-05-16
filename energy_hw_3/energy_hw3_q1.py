from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

def make_correlated_samples(rho, n=1000):
    '''
    Generate n correlated samples for 2 features
    '''
    rho_column = np.array([rho, math.sqrt(1-rho**2)]).reshape(2,1)
    Z = np.random.randn(n, 2)
    return Z[:,0], np.dot(Z, rho_column)
    
def make_lognormal(sig, T, Z, mu=0):
    '''
    From the normal Z, generate log normal 
    '''
    exponent = (mu - 0.5 * sig**2) * T + sig * math.sqrt(T) * Z
    return np.exp(exponent)
    
def simulate_forward(F0, G0, rho, sigF, sigG, T):
    Y1, Y2 = make_correlated_samples(rho)    
    F = F0 * make_lognormal(sigF, T, Y1)
    G = G0 * make_lognormal(sigG, T, Y2)
    return F.reshape(1000,1), G.reshape(1000,1)

def value_revenue_put(rho, F0=38.05, G0=2.956, sigF=0.4, sigG=0.35, T=168/252,
                        H=7.2, K=5, capacity=800, L=1.5*1e6):
    N = capacity * 16 * 21
    F, G = simulate_forward(F0, G0, rho, sigF, sigG, T)
    V = np.maximum(F - H * G - K, np.zeros((F.size, 1))) * N
    payoffs = np.maximum(-V + L, np.zeros((V.size, 1)))
    return np.average(V), np.average(payoffs)

def question1():
    rhos = np.linspace(0.5, 1.0, 10)
    V, revenue_puts = [], []
    for rho in rhos:
        v, revenue_put = value_revenue_put(rho)
        V.append(v)
        revenue_puts.append(revenue_put)
    
    plt.figure()
    plt.plot(rhos, revenue_puts)
    plt.xlabel("correlation coefficient")
    plt.ylabel("value of the revenue put (undiscounted)")
    plt.title("plot revenue put value against correlations")
    
    plt.figure()
    plt.plot(rhos, V)
    plt.xlabel("correlation coefficient")
    plt.ylabel("value of the generation call (undiscounted)")
    plt.title("plot generation value against correlations")
    plt.show()
    
def main():
    question1()

if __name__ == "__main__":
    main()