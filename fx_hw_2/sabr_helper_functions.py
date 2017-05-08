from __future__ import division
import numpy as np
import scipy.stats as ss
from math import log, exp, sqrt

def black_scholes(S0, K, r, q, sigma, T, option_type):
    d1 = (log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type == 'call':
        return exp(-q * T) * S0 * ss.norm.cdf(d1) - K * exp(-r * T) * ss.norm.cdf(d2)        
    elif option_type == 'put':
        return - exp(-q * T) * S0 * ss.norm.cdf(-d1) + K * exp(-r * T) * ss.norm.cdf(-d2)
    else:
        raise ValueError('use either "call" or "put" or option_type')    
    
def delta_call(S0, r, q, T, sigma, K):
    '''
    Find delta of a call for a given strike
    '''
    d1 = (log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
    return ss.norm.cdf(d1) * exp(-q * T)

def delta_put(S0, r, q, T, sigma, K):
    '''
    Find delta of a put for a given strike
    '''
    d1 = (log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * sqrt(T))
    return -ss.norm.cdf(-d1) * exp(-q * T)

def call_strike(delta, sigma, r, q, S0, T):
    '''
    Calculate the strike so that the delta of a call is same as target, for a given sigma
    '''
    if delta < 0:
        delta = 0
    if delta > 1:
        delta = 1
    d1 = ss.norm.ppf(delta / exp(-q * T))
    return S0 / exp(d1 * (sigma * sqrt(T)) - (r - q + 0.5 * sigma**2) * T)

def put_strike(delta, sigma, r, q, S0, T):
    '''
    Calculate the strike so that the delta of a put is same as target, for a given sigma
    '''
    if delta < -1:
        delta = -1
    if delta > 0:
        delta = 0
    d1 = - ss.norm.ppf(-delta / exp(-q * T))
    return S0 / exp(d1 * (sigma * sqrt(T)) - (r - q + 0.5 * sigma**2) * T)

def find_atm_strike(S0, r, q, T, sigma):
    '''
    Calculate the ATM strike for a delta neutral straddle
    '''
    return S0 * exp((r - q + 0.5 * sigma**2) * T)
    
def find_sabr_vol(alpha, beta, nu, K, S0, r, q, T, rho=0):
    '''
    Given strike and expiry, find SABR implied vol     
    '''
    F = S0 * exp((r - q) * T)
    z = (nu/alpha) * (F*K)**((1-beta)/2) * log(F/K) 
    x = log(((1 - 2*rho*z + z**2)**0.5 + z - rho) / (1 - rho))
    factor = (1 + ((1-beta)**2 / 24 * alpha**2 / (F*K)**(1-beta) + 0.25 * (rho*beta*nu*alpha) / (F*K)**((1-beta)/2) + (2 - 3 * rho**2) / 24 * nu**2) * T)
    return alpha / ((F*K)**((1-beta)/2) * (1 + ((1-beta)**2)/24 * log(F/K)**2 + ((1-beta)**4)/1920 * log(F/K)**4)) * (z/x) * factor