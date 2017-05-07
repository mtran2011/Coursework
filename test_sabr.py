from __future__ import division
import unittest
from sabr_calibration import *

class SabrTestCase(unittest.TestCase):
    def test_2m_expiry(self):
        S0 = 1.1
        r = 0.004
        q = -0.006
        T = 0.169863
        sigma_atm = 0.1035
        sigma_rr = 0.0005
        sigma_bf = 0.00225
        alpha = 0.10055
        beta = 1.0667
        nu = 1.26488        
        
        K_atm = 1.10087 
        K_bf_put = 1.07093 
        K_bf_call = 1.13586
        K_rr_put = 1.071 
        K_rr_call = 1.13595 
        vol_atm = 0.1035
        vol_bf_put = 0.10551
        vol_bf_call = 0.10599
        vol_rr_put = 0.105501
        vol_rr_call = 0.106001
        
        F = S0 * exp((r - q) * T)
        
        # first condition 
        guess_K_atm = atm_dns_K(S0, r, q, T, sigma_atm)        
        delta_atm_call = call_delta(S0, r, q, T, sigma_atm, guess_K_atm)
        delta_atm_put = put_delta(S0, r, q, T, sigma_atm, guess_K_atm)
        self.assertAlmostEqual(delta_atm_call + delta_atm_put, 0, 6)
        self.assertAlmostEqual(guess_K_atm, K_atm, 2)
        
        # second condition        
        sabr_vol_bf_call = find_sabr_vol(K_bf_call, F, T, alpha, beta, nu)
        sabr_vol_bf_put = find_sabr_vol(K_bf_put, F, T, alpha, beta, nu)
        call_bf_sabr = black_scholes(S0, K_bf_call, r, q, sabr_vol_bf_call, T, 'call')
        put_bf_sabr = black_scholes(S0, K_bf_put, r, q, sabr_vol_bf_put, T, 'put')        
        call_bf_bs = black_scholes(S0, K_bf_call, r, q, sigma_atm + sigma_bf, T, 'call')
        put_bf_bs = black_scholes(S0, K_bf_put, r, q, sigma_atm + sigma_bf, T, 'put')         
        self.assertAlmostEqual(call_bf_bs + put_bf_bs, call_bf_sabr + put_bf_sabr, 6)
        
        # third condition
        sabr_vol_rr_call = find_sabr_vol(K_rr_call, F, T, alpha, beta, nu)
        sabr_vol_rr_put = find_sabr_vol(K_rr_put, F, T, alpha, beta, nu)
        self.assertAlmostEqual(sabr_vol_rr_call, vol_rr_call, 5)
        self.assertAlmostEqual(sabr_vol_rr_put, vol_rr_put, 5)
        self.assertAlmostEqual(sabr_vol_rr_call - sabr_vol_rr_put, sigma_rr, 4)

if __name__ == '__main__':
    unittest.main()