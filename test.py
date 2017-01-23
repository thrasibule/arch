import feather
import numpy as np

from arch.univariate.volatility import GARCH
from arch.univariate.mean import ConstantMean
from arch.univariate.distribution import Normal
from arch.univariate.recursions import garch_recursion, dgarch_recursion_vp

from statsmodels.tools.numdiff import approx_fprime

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
                    d.starting_values(resids)))

def f(params, *args):
    sigma2 = np.empty_like(fresids)
    x = (params,) + args[:2] + (sigma2,) + args[2:]
    garch_recursion(*x)
    return sigma2

J = approx_fprime(sv_volatility,
                  f,
                  args = (fresids, sresids, 1, 0, 1, fresids.shape[0], backcast, var_bounds))

J_analytical = np.zeros((fresids.shape[0], 3))
dgarch_recursion_vp(sv_volatility, fresids, sresids, J_analytical, sigma2, 1, 0, 1,
                    fresids.shape[0], backcast)
