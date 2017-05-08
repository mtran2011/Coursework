from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
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
        float: squared errors
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
    
    errors = (error_vol_atm, error_bf, error_rr, error_delta)
    strikes = (K_atm, K_bf_put, K_bf_call, K_rr_put, K_rr_call)
    vols = (vol_atm, vol_bf_put, vol_bf_call, vol_rr_put, vol_rr_call)
    
    return errors, strikes, vols
    
def calibrate():
    triple_sigmas_list = [
    [7.00, 0.50, 0.15],
    [11.75, 0.75, 0.15],
    [11.15, 0.60, 0.175],
    [9.95, 0.35, 0.225],
    [10.35, 0.05, 0.225],
    [10.2, -0.2, 0.25],
    [10.25, -0.75, 0.275],
    [10.3, -1.15, 0.275],
    ]
    for row in triple_sigmas_list:
        for i in range(len(row)):
            row[i] = row[i] / 100
    
    tenor = ['ON','1W','2W','1M','2M','3M','6M','1Y']
    expiry = [3, 7, 14, 33, 62, 91, 182, 368]
    num_data_rows = len(expiry)
    for i in range(num_data_rows):
        expiry[i] = expiry[i] / 365.0
    
    # prepare the output table 
    table_df = pd.DataFrame(data=[], index=tenor)
    table_df['T'] = expiry
    
    table_df['K_atm'] = np.zeros(num_data_rows)
    table_df['K_bf_put'] = np.zeros(num_data_rows)
    table_df['K_bf_call'] = np.zeros(num_data_rows)
    table_df['K_rr_put'] = np.zeros(num_data_rows)
    table_df['K_rr_call'] = np.zeros(num_data_rows)
    
    table_df['vol_atm'] = np.zeros(num_data_rows)
    table_df['vol_bf_put'] = np.zeros(num_data_rows)
    table_df['vol_bf_call'] = np.zeros(num_data_rows)
    table_df['vol_rr_put'] = np.zeros(num_data_rows)
    table_df['vol_rr_call'] = np.zeros(num_data_rows)
    
    table_df['alpha'] = np.zeros(num_data_rows)
    table_df['beta'] = np.zeros(num_data_rows)
    table_df['nu'] = np.zeros(num_data_rows)
    
    # report true if the optimizer was success
    table_df['success'] = np.zeros(num_data_rows, dtype=bool)
    # sum of the squared errors
    table_df['SSE'] = np.zeros(num_data_rows)
    
    # for each row of data
    for i, sigmas in enumerate(triple_sigmas_list):    
        T = expiry[i]
        
        # define the objective function 
        def objective_func(args):
            alpha, beta, nu, sigma_rr_put = args[:]
            four_errors = sum_squared_error(alpha, beta, nu, sigma_rr_put, sigmas, T)[0]            
            return four_errors
        
        # now optimize
        guess = np.array([0.1, 1.0, 1.2, 0.1])
        sol = optimize.root(objective_func, guess)
        a, b, v, vol_rr_put = sol.x
        
        # get the output by unpacking
        errors, strikes, vols = sum_squared_error(a, b, v, vol_rr_put, sigmas, T)
        sse = sum(errors)
        K_atm, K_bf_put, K_bf_call, K_rr_put, K_rr_call = strikes
        vol_atm, vol_bf_put, vol_bf_call, _, vol_rr_call = vols
        
        # put results into table
        table_df['K_atm'].iloc[i] = K_atm
        table_df['K_bf_put'].iloc[i] = K_bf_put
        table_df['K_bf_call'].iloc[i] = K_bf_call
        table_df['K_rr_put'].iloc[i] = K_rr_put
        table_df['K_rr_call'].iloc[i] = K_rr_call
        
        table_df['vol_atm'].iloc[i] = vol_atm
        table_df['vol_bf_put'].iloc[i] = vol_bf_put
        table_df['vol_bf_call'].iloc[i] = vol_bf_call
        table_df['vol_rr_put'].iloc[i] = vol_rr_put
        table_df['vol_rr_call'].iloc[i] = vol_rr_call
        
        table_df['alpha'].iloc[i] = a
        table_df['beta'].iloc[i] = b
        table_df['nu'].iloc[i] = v
        
        table_df['success'].iloc[i] = sol.success
        table_df['SSE'].iloc[i] = sse
    
    return table_df
    
def calibrate_strikes(series1Y):
    # solve for the strikes for question 2
    T = series1Y.loc['T']
    alpha = series1Y.loc['alpha']
    beta = series1Y.loc['beta']
    nu = series1Y.loc['nu']
    delta = 0.1
    
    def objective_func(args):
        K_put, K_call = args[:]
        
        vol_call = find_sabr_vol(alpha, beta, nu, K_call, S0, r, q, T)
        vol_put = find_sabr_vol(alpha, beta, nu, K_put, S0, r, q, T) 
        
        diff_call = K_call - call_strike(delta, vol_call, r, q, S0, T)
        diff_put = K_put - put_strike(delta, vol_put, r, q, S0, T)
        
        return diff_call, diff_put
    
    sol = optimize.root(objective_func, [1.0, 1.0])
    K_10_put, K_10_call = sol.x
    
    return K_10_put, K_10_call
    
    
    
    
    
    
    
    
    
    
    
    