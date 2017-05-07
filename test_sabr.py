from __future__ import division
import unittest
from sabr_calibration import *

class SabrTestCase(unittest.TestCase):
    S0 = 1.1
    r = 0.4 / 100
    q = -0.6 / 100
    T = 0.169863
    sigma_atm = 0.1035
    sigma_rr = 0.0005
    sigma_bf = 0.00225
    alpha = 0.10055
    beta = 1.0667
    nu = 1.26488
    
    K_atm = 1.09944
    self.assertAlmostEqual(find_K_dns_atm(S0, r, q, T, sigma_atm), K_atm)
    
    

if __name__ == '__main__':
    unittest.main()