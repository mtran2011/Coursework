import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline
import matplotlib.pyplot as plt

############################################################
# Simple data input
############################################################
strikes = np.array(range(1000,3000,100))
iv = [0.05, 0.07, 0.08, 0.1, 0.12, 0.13, 0.15, 0.16, 0.18, 0.19, 
      0.21, 0.23, 0.24, 0.26, 0.28, 0.29, 0.31, 0.33, 0.34, 0.35]
iv = np.array(list(reversed(iv)))
df_data = pd.DataFrame(list(zip(strikes,iv)), columns=['Strikes', 'IV'])
print('Observed market data of implied volatilities (IV)')
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
# Look for any butterfly arbitrage
############################################################
call_prices = [option_price(vol, k, is_call=True)
               for vol, k in zip(iv, strikes)]
butterfly_prices = [call_prices[i+1] - 2 * call_prices[i] + call_prices[i-1]
                    for i in range(1, len(call_prices)-1)]
plt.figure()
plt.plot(strikes[1:-1], butterfly_prices, 'o')
plt.vlines(strikes[1:-1], ymin=np.zeros(len(butterfly_prices)), ymax=butterfly_prices, linestyles='dotted')
plt.plot((1000,3000), (0,0), 'r--')
plt.xlabel('strikes serving as butterfly centers with')
plt.ylabel('butterfly prices based on given IV')
plt.xticks(range(1000,3200,200))
plt.title('Prices of butterflies centered at various strikes')
plt.tight_layout()
plt.savefig('butterfly given')
print(pd.DataFrame({'Strikes': strikes[1:-1], 'Butterfly prices': butterfly_prices}))

############################################################
# Interpolate implied volatility with cubic spline
# Compare this with smoothing spline
############################################################
k_range = np.arange(1000,2901,1)

iv_spline_interpolator = CubicSpline(strikes, iv)
cs_interpolated_iv = iv_spline_interpolator(k_range)

iv_smoothing_interpolator = UnivariateSpline(strikes, iv)
smooth_interpolated_iv = iv_smoothing_interpolator(k_range)

plt.figure()
plt.plot(strikes, iv, 'o', label='market data of IV')
plt.plot(k_range, cs_interpolated_iv, label='cubic spline interpolation')
plt.plot(k_range, smooth_interpolated_iv, 'r--', label='smoothing spline interpolation')
plt.xlabel('strike')
plt.ylabel('implied volatility')
plt.xticks(range(1000,3200,200))
plt.title('Interpolate implied volatility (IV) with splines')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('smoothing spline implied vol')

############################################################
# Use cubic-spline-interpolated IV to calculate call price C
# then differentiate C twice for risk neutral pdf
############################################################
call_prices = [option_price(vol, k, is_call=True)
               for vol, k in zip(cs_interpolated_iv, k_range)]
cs_iv_implied_pdf = [call_prices[i+1] - 2 * call_prices[i] + call_prices[i-1] 
                     for i in range(1, len(call_prices)-1)]
curve_integral = np.trapz(cs_iv_implied_pdf, x=k_range[1:-1])
label = 'area under curve: {:.2f}'.format(curve_integral)
plt.figure()
plt.plot(k_range[1:-1], cs_iv_implied_pdf, label=label)
plt.xlabel('stock price at expiry')
plt.ylabel('pdf implied by call premium')
plt.xticks(range(1000,3200,200))
plt.title('Implied pdf of S(T) using cubic-spline-interpolated IV')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('pdf with spline-interpolated iv')

############################################################
# Use smoothing-spline-interpolated IV to calculate C
# then differentiate C twice for risk neutral pdf
############################################################
call_prices = [option_price(vol, k, is_call=True)
               for vol, k in zip(smooth_interpolated_iv, k_range)]
smooth_iv_implied_pdf = [call_prices[i+1] - 2 * call_prices[i] + call_prices[i-1] 
                         for i in range(1, len(call_prices)-1)]
curve_integral = np.trapz(smooth_iv_implied_pdf, x=k_range[1:-1])
label = 'area under curve: {:.2f}'.format(curve_integral)
plt.figure()
plt.plot(k_range[1:-1], smooth_iv_implied_pdf, 'r--', label=label)
plt.plot((1000,3000), (0,0), 'b:')
plt.xlabel('stock price at expiry')
plt.ylabel('pdf implied by call premium')
plt.xticks(range(1000,3200,200))
plt.title('Implied pdf of S(T) using smoothing-spline-interpolated IV')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('pdf with smooth-interpolated iv')

############################################################
# Use smoothing-spline-interpolated IV to obtain pdf of S(T)
# Estimate the skewness of this pdf
############################################################
mu = np.trapz([f * s for f, s in zip(smooth_iv_implied_pdf, k_range[1:-1])], 
              x=k_range[1:-1])
var = -mu**2 + np.trapz([f * s**2 for f, s in zip(smooth_iv_implied_pdf, k_range[1:-1])], 
                        x=k_range[1:-1])
std = var**0.5
skewness = np.trapz([f * ((s-mu)/std)**3 for f, s in zip(smooth_iv_implied_pdf, k_range[1:-1])], 
                    x=k_range[1:-1])

# use price of OTM digital to show skewness
def digital_price(sig, k, is_call):
    f, T = 2000, 1
    d2 = (math.log(f/k) - 0.5 * sig**2 * T) / (sig * T**0.5)    
    if is_call:
        price = norm.cdf(d2)
    else:
        price = norm.cdf(-d2)
    return price

digitals = [digital_price(vol, k, is_call=True) if k > 2000 else digital_price(vol, k, is_call=False)
            for vol, k in zip(smooth_interpolated_iv, k_range)]
plt.plot(k_range, digitals, 'b-')
plt.plot((1000,3000), (0,0), 'k:')
plt.xlabel('strike of OTM digital')
plt.ylabel('price of OTM cash-or-nothing digital')
plt.xticks(range(1000,3200,200))
plt.title('OTM cash-or-nothing digital prices using smoothing-spline-interpolated IV')
plt.tight_layout()
plt.savefig('digital OTM prices')

############################################################
# use given IV to calculate C 
# then interpolate C with cubic spline and take derivative
############################################################
given_calls = [option_price(vol, k, is_call=True)
               for vol, k in zip(iv, strikes)]
call_spline_interpolator = CubicSpline(strikes, given_calls)
call_prices = call_spline_interpolator(k_range)
# plot the cubic spline interpolation of call premiums
plt.figure()
plt.plot(strikes, given_calls, 'o', label='based on given IV data')
plt.plot(k_range, call_prices, label='cubic spline interpolation')
plt.xlabel('strike')
plt.ylabel('calculated call premiums')
plt.xticks(range(1000,3200,200))
plt.title('Interpolate call premiums with cubic spline')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('cubic spline call premium')
# now numerically differentiate with FDM step h=1
cs_call_implied_pdf = [call_prices[i+1] - 2 * call_prices[i] + call_prices[i-1] 
                       for i in range(1, len(call_prices)-1)]
curve_integral = np.trapz(cs_call_implied_pdf, x=k_range[1:-1])
label = 'area under curve: {:.2f}'.format(curve_integral)
plt.figure()
plt.plot(k_range[1:-1], cs_call_implied_pdf, label=label)
plt.xlabel('stock price at expiry')
plt.ylabel('pdf implied by spline-interpolated call')
plt.xticks(range(1000,3200,200))
plt.title('Implied pdf of S(T) using interpolated call premiums')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('pdf with interpolated call premiums')

############################################################
# example 2Y vol curve
############################################################
# 1 year expiry
linear = interp1d([1000,2900], [0.35, 0.05])
iv_1y = linear(k_range)
calls = [option_price(v, k, is_call=True)
         for v, k in zip(iv_1y, k_range)]
pdf_1y = [calls[i+1] - 2 * calls[i] + calls[i-1]
          for i in range(1, len(k_range)-1)]

# 2 year expiry
linear = interp1d([1000, 2900], [0.37, 0.15])
iv_2y = linear(k_range)
calls = [option_price(v, k, is_call=True)
         for v, k in zip(iv_2y, k_range)]
pdf_2y = [calls[i+1] - 2 * calls[i] + calls[i-1]
          for i in range(1, len(k_range)-1)]

fig, ax1 = plt.subplots()
ax1.plot(k_range[1:-1], pdf_1y, 'r--', label='1Y density')
ax1.plot(k_range[1:-1], pdf_2y, 'b-', label='2Y density')
ax1.set_xlabel('strike')
ax1.set_ylabel('probability density function')
ax2 = ax1.twinx()
ax2.plot(k_range, iv_1y, label='1Y implied vol')
ax2.plot(k_range, iv_2y, label='2Y implied vol')
ax2.set_ylabel('implied vol')
ax1.legend(loc=1)
ax2.legend(loc=2)
fig.tight_layout()
fig.savefig('example 2y density')