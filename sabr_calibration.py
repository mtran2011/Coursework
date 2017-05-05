from __future__ import division
import numpy as np 
from math import log

def sabr_sgma(K, F, T, alpha, beta, nu, rho=0):
	'''
	Calculate BS implied vol from SABR model 	
	'''
	z = (nu/alpha) * (F*K)**((1-beta)/2) * log(F/K)	
	x = log(((1 - 2*rho**z + z**2)**0.5 + z - rho) / (1 - rho))
	
	first_denom = (F*K)**((1-beta)/2) * (1 + ((1-beta)**2)/24 * log(F/K)**2 
										   + ((1-beta)**4)/1920 * log(F/K)**4)
	
	third_term = (1 + (
					  (1-beta)**2 / 24 * alpha**2 / (F*K)**(1-beta) + 
					  0.25 * (rho*beta*nu*alpha) / (F*K)**((1-beta)/2) +
					  (2 - 3 * rho**2) / 24 * nu**2
					  ) * T
				 )
	
	sigma = alpha / first_denom * (z/x) * third_term
	return sigma