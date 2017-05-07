from __future__ import division
import numpy as np
import scipy.stats as ss
from math import log, exp, sqrt

def sabr_sgma(K, F, T, alpha, beta, nu, rho=0):

    z = (nu/alpha) * (F*K)**((1-beta)/2) * log(F/K) 

    x = log(((1 - 2*rho**z + z**2)**0.5 + z - rho) / (1 - rho))

    first_denom = (F*K)**((1-beta)/2) * (1 + ((1-beta)**2)/24 * log(F/K)**2 + ((1-beta)**4)/1920 * log(F/K)**4)

    third_term = (1 + ((1-beta)**2 / 24 * alpha**2 / (F*K)**(1-beta) + 0.25 * (rho*beta*nu*alpha) / (F*K)**((1-beta)/2) + (2 - 3 * rho**2) / 24 * nu**2) * T)

    sigma = alpha / first_denom * (z/x) * third_term
    
    return sigma

def find_K_dns_atm(S0, r, q, T, sigma):    
    F = S0 * exp(r - q) * T
    return F * exp(0.5 * sigma**2 * T)

def black_scholes(S0, K, r, q, sigma, T, option_type):    
    d1 = (log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - q - sigma**2 / 2) * T) / (sigma * sqrt(T))
    F = S0 * exp((r - q) * T)
    if option_type == 'call':
        value = exp(-r * T) * (F * ss.norm.cdf(d1) - K * ss.norm.cdf(d2))
    elif option_type == 'put':
        value = exp(-r * T) * (-F * ss.norm.cdf(-d1) + K * ss.norm.cdf(-d2))
    else:
        raise ValueError('option_type input is bad; input either "call" or "put"')
    return value