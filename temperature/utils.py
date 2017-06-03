import emcee
import numpy as np
import pandas as pd

from scipy import stats
from scipy import optimize


def parameter_bounds():
  return {
    # Air is about 700 J/(kg K), water 4100
    'c_v': [100, 10000],
    # litres, it'd probably be hard to control 1/10th litre and
    # hard to heat 1000 litres of something evenly.
    'volume': [0.1, 1000],
    'h': [0.1, 10000],
    # lag times of a 1/10 secon to 20 minutes
    't_lag': [0.01, 1200],
    # assume the object starts between freezing and boiling water
    'u_0' : [273., 373.],
    # the ambient temperature between 10 and 40 C
    'u_env': [283., 313.],
    's_0': [0., 1.],
   }


def pretty_params(p):
  return '\n'.join(['%s: %f' % (k, v) for k, v in p.iteritems()])


def heat_source_function(t, t_lag=30., u_0=0., u_inf=1.):
  """
  """
  exponent = 2 / t_lag * np.log(1. / 0.995 - 1) * (t - 1. - t_lag / 2)
  return u_0 + (u_inf - u_0) * (1. / (1 + np.exp(exponent)))


def compute_heat_source(ts, heat_on, t_lag, s_0=0.):
  """
  Computes a normalized heat source function for a heat source with times,
  ts and indicator of heat being on
  """
  was_on = not heat_on[0]
  s = s_0
  
  for is_on, t in zip(heat_on, ts):
    if is_on:
      s_inf = 1.
      if not was_on:
        t_0 = t
        s_0 = s
      was_on = True
    else:
      s_inf = 0.
      if was_on:
        t_0 = t
        s_0 = s
      was_on = False

    s = heat_source_function(t - t_0, t_lag=t_lag, u_0=s_0, u_inf=s_inf)
    yield s


def temperature_given_heat(ts, q_ext, volume, c_v, h, u_0, u_env):
  """
  
  """
  prev_t = ts[0]
  u = u_0
  
  for t, qq in zip(ts, q_ext):
    du_dt = (qq - h * (u - u_env)) / (c_v * volume)
    u = u + du_dt * (t - prev_t)
    prev_t = t
    yield u


def compute_temperature(ts, heat_on, wattage, t_lag, s_0, volume, c_v, h, u_0, u_env):
  s = np.array(list(compute_heat_source(ts, heat_on, t_lag, s_0)))
  q_ext = s * wattage
  return np.array(list(temperature_given_heat(ts, q_ext, volume, c_v, h, u_0, u_env)))


def log_likelihood(ts, heat_on, obs, wattage, t_lag, s_0, volume, c_v, h, u_0, u_env, sig):
  
  u = compute_temperature(ts, heat_on, wattage, t_lag, s_0, volume, c_v, h, u_0, u_env)

  lls = stats.norm.logpdf((u - obs) / sig)
  if np.any(np.isnan(lls)):
    lls[np.isnan(lls)] = -0.5 * np.finfo(np.float).max

  return np.mean(lls)


def optimal_parameters(ts, heat_on, obs, params=None, *args, **kwdargs):

  param_names = sorted(params.keys())
  x0 = np.array([params[k] for k in param_names])

  def nll(x):
    p = {k: v for k, v in zip(param_names, x)}
    p.update(kwdargs)
    return -log_likelihood(ts, heat_on, obs, **p)

  all_bounds = parameter_bounds()
  bounds = [all_bounds.get(k, (None, None)) for k in param_names]
  lbfgs = optimize.fmin_l_bfgs_b(nll, x0=x0, approx_grad=True, bounds=bounds)
  optimal = lbfgs[0]

#   ret = optimize.fmin(nll, x0=x0, ftol=1e-6, xtol=1e-6, maxiter=10000, disp=1)
#   optimal = ret

  return {k: v for k, v in zip(param_names, optimal)}


def sample_parameters(ts, heat_on, obs, n, params=None, prior=None, iterations=1000, **kwdargs):
  
  np.random.seed(1982)
  param_names = sorted(params.keys())

  if params is None:
    all_bounds = parameter_bounds()
    bounds = [all_bounds[k] for k in param_names]
    p0 = [np.random.uniform(low, high, size=n) for low, high in bounds]
  
  def ll(x):
    p = {k: v for k, v in zip(param_names, x)}
    p.update(kwdargs)
    return log_likelihood(ts, heat_on, obs, **p)

  sampler = emcee.EnsembleSampler(n, len(param_names), ll, args=[ts, heat_on, obs])

  def iter_positions():
    for i, (pos, prob, _) in enumerate(sampler.sample(p0, iterations=iterations)):
      print i, np.mean(prob), np.max(prob), pos[np.argmax(prob)]
      yield pos, prob
  
  positions, probs = zip(*iter_positions())

  df = pd.DataFrame(positions[-1], columns=param_names)
  df['t_lag'] = np.exp(df['t_lag'])
  df['h'] = np.exp(df['h'])
  df['r'] = np.exp(df['r'])
  df['log_prob'] = probs[-1]

  return df
