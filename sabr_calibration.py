from __future__ import division
import numpy as np
import scipy.stats as ss
import math
from sabr_helper_functions import *

# define global variables for use everywhere
S0 = 1.1 # rate of EURUSD
r = 0.4 / 100 # USD interest rate
q = -0.6 / 100 # EUR interest rate

def sum_squared_error(alpha, beta, nu, sigma_rr_put, sigmas, T):
    '''
    Four variables to calibrate: alpha, beta, nu, sigma_rr_put
    Args:
        sigmas(tuple): pack of three vols ATM, RR, BF
        T(float): time to expiry
    Returns:
        float: sum of squared errors
    '''
    # unpacking market observed vols 
    atm, rr, bf = sigmas
    
    # first condition
    K_atm = find_atm_strike(S0, r, q, T, atm)
    vol_atm = find_sabr_vol(alpha, beta, nu, K_atm, S0, r, q, T)
    error_vol_atm = (vol_atm - atm)**2
    
    # second condition
    K_bf_call = call_strike(0.25, atm + bf, r, q, S0, T)
    K_bf_put = put_strike(-0.25, atm + bf, r, q, S0, T)
    
    BS_bf_call_price = black_scholes(S0, K_bf_call, r, q, atm + bf, T, 'call')
    BS_bf_put_price = black_scholes(S0, K_bf_put, r, q, atm + bf, T, 'put')
    
    vol_bf_call = find_sabr_vol(alpha, beta, nu, K_bf_call, S0, r, q, T)
    vol_bf_put = find_sabr_vol(alpha, beta, nu, K_bf_put, S0, r, q, T)
    
    sabr_bf_call_price = black_scholes(S0, K_bf_call, r, q, vol_bf_call, T, 'call')
    sabr_bf_put_price = black_scholes(S0, K_bf_put, r, q, vol_bf_put, T, 'put')
    
    error_bf = (BS_bf_call_price + BS_bf_put_price - sabr_bf_call_price - sabr_bf_put_price)**2
    
    # third and fourth condition
    K_rr_call = call_strike(0.25, sigma_rr_put + rr, r, q, S0, T)
    K_rr_put = put_strike(-0.25, sigma_rr_put, r, q, S0, T)
    
    vol_rr_call = find_sabr_vol(alpha, beta, nu, K_rr_call, S0, r, q, T)
    vol_rr_put = find_sabr_vol(alpha, beta, nu, K_rr_put, S0, r, q, T)
    
    error_rr = (vol_rr_call - (sigma_rr_put + rr))**2 + (vol_rr_put - sigma_rr_put)**2
    
    # fifth condition
    delta_rr_call = delta_call(S0, r, q, T, vol_rr_call, K_rr_call)
    delta_rr_put = delta_put(S0, r, q, T, vol_rr_put, K_rr_put)
    
    error_delta = (delta_rr_call - 0.25)**2 + (delta_rr_put - -0.25)**2
    
    return error_vol_atm + error_bf + error_rr + error_delta