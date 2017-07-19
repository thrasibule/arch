import feather
import numpy as np

from arch.univariate.volatility import GARCH
from arch.univariate.mean import ConstantMean
from arch.univariate.distribution import Normal
from arch.univariate.recursions import garch_recursion
from arch.univariate.recursions import dgarch_recursion_vp, dgarch_recursion_mp

from arch import arch_model

from statsmodels.tools.numdiff import approx_fprime

from scipy.optimize import minimize

df = feather.read_dataframe("index_returns.fth")
y = df.ig.dropna()
v = GARCH()
d = Normal()
am = ConstantMean(y)
am.volatility = v
am.distribution = d

am._adjust_sample(None, None)
resids = am.resids(am.starting_values())
sigma2 = np.zeros_like(resids)
sv_volatility = v.starting_values(resids)
backcast = v.backcast(resids)
var_bounds = v.variance_bounds(resids)
v.compute_variance(sv_volatility, resids, sigma2, backcast, var_bounds)
std_resids = resids / np.sqrt(sigma2)
fresids = np.square(resids)
sresids = np.sign(resids)

params = np.hstack((am.starting_values(),
                    v.starting_values(resids),
                    d.starting_values(std_resids)))

def loglik(params, am):
    mu = params[0]
    sv_volatility = params[1:]
    resids = am.resids(mu)
    backcast = am.volatility.backcast(resids)
    sigma2 = np.zeros_like(resids)
    var_bounds = am.volatility.variance_bounds(resids)
    am.volatility.compute_variance(sv_volatility, resids, sigma2, backcast, var_bounds)
    return 0.5 * np.sum(np.log(2 * np.pi * sigma2) + resids ** 2 / sigma2)

def dloglik(params, am):
    mu = params[0]
    sv_volatility = params[1:]
    resids = am.resids(mu)
    backcast = am.volatility.backcast(resids)
    fresids = abs(resids) ** am.volatility.power
    sresids = np.sign(resids)
    sigma2 = np.zeros_like(resids)
    v.compute_variance(sv_volatility, resids, sigma2, backcast, var_bounds)
    J_analytical = np.zeros((fresids.shape[0], sv_volatility.shape[0]))
    dgarch_recursion_vp(sv_volatility, fresids, sresids, J_analytical, sigma2,
                        am.volatility.p, am.volatility.o, am.volatility.q,
                        fresids.shape[0], backcast)
    temp = fresids / np.square(sigma2)
    dll = J_analytical / sigma2[:,None] - J_analytical * temp[:,None]
    dloglik_vol = 0.5 * dll.sum(axis=0)
    dresids = am.dresids(mu)
    if am.volatility.power == 2.:
        dfresids = 2 * resids * dresids
    else:
        dfresids = am.volatility.power * abs(resids) ** (am.volatility.power - 1) * sresids * dresids
    dbackcast = am.volatility.dbackcast(resids)
    dsigma2 = np.zeros_like(resids)
    dgarch_recursion_mp(sv_volatility, dfresids, dsigma2, am.volatility.p, am.volatility.o,
                        am.volatility.q, fresids.shape[0], dbackcast)
    dll = dsigma2 / sigma2 + dfresids / sigma2 - temp * dsigma2
    return np.hstack([0.5 * dll.sum(axis=0), dloglik_vol])

J = approx_fprime(params, loglik, args=(am,))

opt = minimize(loglik, params, jac = dloglik, args=(am,))
am2 = arch_model(10*y)
am2.fit()

# df = index_returns(index="IG", series=[27, 28], tenor='5yr')
# df = df.groupby('date').nth(-1)
# df = df['spread_return'].dropna()
