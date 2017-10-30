import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
%matplotlib inline

############################################################
# Simple data input
############################################################
strikes = np.array(range(1000,3000,100))
iv = [0.05, 0.07, 0.08, 0.1, 0.12, 0.13, 0.15, 0.16, 0.18, 0.19, 
      0.21, 0.23, 0.24, 0.26, 0.28, 0.29, 0.31, 0.33, 0.34, 0.35]
iv = np.array(list(reversed(iv)))
df_data = pd.DataFrame(list(zip(strikes,iv)), columns=['Strikes', 'IV'])
print('Observed market data in implied volatilities (IV)')
print(df_data)

# helper function to calculate European option price
def option_price(sig, k, is_call):
    ''' Calculate option price under given assumptions. 
    Assume zero risk free rate and dividend. F = 2000, T = 1 year 
    Args:
        sig (float): the implied volatility corresponding to strike k
        k (float): the strike price
        is_call (boolean): for simplicity, True for call and False for put
    Returns:
        float: the European option price with expiry of 1 year
    '''
    f, T = 2000, 1
    d1 = (math.log(f/k) + 0.5 * sig**2 * T) / (sig * T**0.5)
    d2 = d1 - sig * T**0.5
    if is_call:
        price = f * norm.cdf(d1) - k * norm.cdf(d2)
    else:
        price = k * norm.cdf(-d2) - f * norm.cdf(-d1)
    return price

############################################################
# Interpolate implied volatility with piecewise linear func 
############################################################
k_range = np.arange(1000,2901,1)
iv_linear_interpolator = interp1d(strikes, iv, kind='linear')
linear_interp_iv = iv_linear_interpolator(k_range)
call_prices = [option_price(vol, k, is_call=True) 
               for vol, k in zip(linear_interp_iv, k_range)]
linear_implied_pdf = [call_prices[i+1] - 2 * call_prices[i] + call_prices[i-1] 
                      for i in range(1, len(call_prices)-1)]
# double check the integrated area under the pdf curve
curve_integral = np.trapz(linear_implied_pdf, x=k_range[1:-1])
label = 'area under curve: {:.2f}'.format(curve_integral)
plt.figure()
plt.plot(k_range[1:-1], linear_implied_pdf, label=label)
plt.xlabel('stock price at expiry')
plt.ylabel('pdf implied by call premium')
plt.xticks(range(1000,3200,200))
plt.title('Implied pdf of S(T) using linear-interpolated IV')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('call implied pdf with linear-interpolated iv')

############################################################
# Interpolate implied volatility with spline
############################################################
iv_spline_interpolator = CubicSpline(strikes, iv, bc_type='natural')
k_range = np.arange(999,2902,1)
interp_iv = iv_spline_interpolator(k_range)
plt.figure()
plt.plot(strikes, iv, 'o', label='given data of IV')
plt.plot(k_range[1:-1], interp_iv[1:-1], label='cubic spline interpolation')
plt.xlabel('strike')
plt.ylabel('implied volatility')
plt.xticks(range(1000,3200,200))
plt.title('Interpolate implied volatility with cubic spline')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('implied vol cubic spline')



############################################################
# first method: interpolate implied vol then calculate C
# then differentiate C twice to obtain implied density
############################################################
call_implied_pdf = []
for i in range(1, len(k_range)-1):
    # numerical estimate of the second derivative with dx = 1
    # use the cubic spline interpolated value of IV to price
    left = option_price(interp_iv[i-1], k_range[i-1], is_call=True)
    right = option_price(interp_iv[i+1], k_range[i+1], is_call=True)
    c = option_price(interp_iv[i], k_range[i], is_call=True)    
    pdf = right - 2 * c + left
    call_implied_pdf.append(pdf)
# double check that the implied density should integrate to 1
curve_integral = np.trapz(call_implied_pdf, x=k_range[1:-1])
label = 'area under curve: {:.2f}'.format(curve_integral)
plt.figure()
plt.plot(k_range[1:-1], call_implied_pdf, label=label)
plt.xlabel('stock price at expiry')
plt.ylabel('pdf implied by call premium')
plt.xticks(range(1000,3200,200))
plt.title('Implied pdf of S(T) using interpolated IV')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('call implied pdf with interpolated iv')

############################################################
# second method: use interpolated IV. 
# but for K < F, take derivative of P. For K > F, use C 
############################################################
call_put_implied_pdf = []
F = 2000
for i in range(1, len(k_range)-1):    
    if k_range[i] < F:
        is_call = False
    else:
        is_call = True
    left = option_price(interp_iv[i-1], k_range[i-1], is_call)
    right = option_price(interp_iv[i+1], k_range[i+1], is_call)
    mid = option_price(interp_iv[i], k_range[i], is_call)
    call_put_implied_pdf.append(right - 2 * mid + left)
# double check that the implied density should integrate to 1
curve_integral = np.trapz(call_put_implied_pdf, x=k_range[1:-1])
label = 'area under curve: {:.2f}'.format(curve_integral)
plt.figure()
plt.plot(k_range[1:-1], call_put_implied_pdf, label=label)
plt.xlabel('stock price at expiry')
plt.ylabel('pdf implied at OTM strikes')
plt.xticks(range(1000,3200,200))
plt.title('Implied pdf of S(T) using interpolated IV and OTM strikes')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('OTM implied pdf with interpolated iv')

############################################################
# third method: do not interpolate implied vol (IV)
# directly use given IV to differentiate C with large step h
# the error in this numerical differentiation is high
# but we include only for completeness
############################################################
sparse_call_implied_pdf = []
for i in range(1, strikes.size-1):
    left = option_price(iv[i-1], strikes[i-1], is_call=True)
    right = option_price(iv[i+1], strikes[i+1], is_call=True)
    c = option_price(iv[i], strikes[i], is_call=True)
    pdf = (right - 2 * c + left) / (strikes[i+1] - strikes[i])**2
    sparse_call_implied_pdf.append(pdf)
curve_integral = np.trapz(sparse_call_implied_pdf, x=strikes[1:-1])
label = 'area under curve: {:.2f}'.format(curve_integral)
plt.figure()
plt.plot(strikes[1:-1], sparse_call_implied_pdf, label=label)
plt.xlabel('stock price at expiry')
plt.ylabel('pdf implied by call premium')
plt.xticks(range(1000,3200,200))
plt.title('Implied pdf of S(T) using only given IV')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('call implied pdf with only given iv')

############################################################
# fourth method: use given IV to calculate C 
# then interpolate C with cubic spline and take derivative
############################################################
call_prices = [option_price(iv, k, is_call=True) for iv, k in zip(iv, strikes)]
c_spline = CubicSpline(strikes, call_prices)
interp_call_prices = c_spline(k_range)
# plot the cubic spline interpolation of call premiums
plt.figure()
plt.plot(strikes, call_prices, 'o', label='based on given IV data')
plt.plot(k_range[1:-1], interp_call_prices[1:-1], label='cubic spline interpolation')
plt.xlabel('strike')
plt.ylabel('calculated call premiums')
plt.xticks(range(1000,3200,200))
plt.title('Interpolate call premiums with cubic spline')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('call premium cubic spline')
# now numerically differentiate with FDM step dx=1
interp_call_implied_pdf = [interp_call_prices[i-1] - 2*interp_call_prices[i] + interp_call_prices[i+1] 
                           for i in range(1, len(k_range)-1)]
curve_integral = np.trapz(interp_call_implied_pdf, x=k_range[1:-1])
label = 'area under curve: {:.2f}'.format(curve_integral)
plt.figure()
plt.plot(k_range[1:-1], interp_call_implied_pdf, label=label)
plt.xlabel('stock price at expiry')
plt.ylabel('pdf implied by spline-interpolated call')
plt.xticks(range(1000,3200,200))
plt.title('Implied pdf of S(T) using interpolated call premiums')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('call implied pdf with interpolated call premiums')

