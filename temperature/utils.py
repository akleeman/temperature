import os
import emcee
import logging
import simplejson
import filelock
import numpy as np
import pandas as pd

from cStringIO import StringIO
from contextlib import contextmanager

from scipy import stats
from scipy import optimize

logger = logging.getLogger(os.path.basename(__file__))


def parameter_bounds():
    return {
      # Air is about 700 J/(kg K), water 4100
      'c_v': [100, 10000],
      # litres, it'd probably be hard to control 1/10th litre and
      # hard to heat 1000 litres of something evenly.
      'volume': [0.1, 1000],
      'h': [0.1, 10000],
      # lag times of a 1/10 secon to 20 minutes
      'k': [0.01, 10.],
      # assume the object starts between freezing and boiling water
      'u_0' : [273., 373.],
      # the ambient temperature between 10 and 40 C
      'u_env': [283., 313.],
      's_0': [0., 1.],
     }


def pretty_params(p):
    return '\n'.join(['"%s": %f' % (k, v) for k, v in p.iteritems()])


def heat_source_function(x, t, k):
    return x * np.exp(-x ** 2 / (4 * k * t)) / np.sqrt(4 * np.pi * k * np.power(t, 3.))


def compute_heat_source(ts, heat_on, k, init_time_off=None):
    x = 1.
    ts_post = ts * k
    diffs = np.diff(ts_post)
    
    if init_time_off is not None:
        epsilon = 1e-3
        time_scale = np.power(4 * np.pi * k * epsilon ** 2, -1. / 3.)
        init_heat_duration = np.linspace(0., time_scale, 101)
        init_heat_duration_dt = np.diff(init_heat_duration)[0]

    def temperature(t):
        s = ts_post[ts_post < t]
        h = heat_on[ts_post < t]
        if not ts.size:
            return 0.
        dt = diffs[ts_post[:-1] < t]
        values = heat_source_function(x, t - s, k) * h
        heat_source = np.sum(dt * values)

        if init_time_off is not None:
            residual_values = heat_source_function(x, t + init_time_off + init_heat_duration, k)
            residual_before_start = init_heat_duration_dt * np.sum(residual_values)
            heat_source += residual_before_start

        return heat_source

    return np.array([temperature(t) for t in ts_post])



def temperature_given_heat(ts, q_ext, volume, c_v, h, u_0, u_env, **kwdargs):
  """
  
  """
  prev_t = ts[0]
  u = u_0
  
  for t, qq in zip(ts, q_ext):
    du_dt = (qq - h * (u - u_env)) / (c_v * volume)
    u = u + du_dt * (t - prev_t)
    prev_t = t
    yield u


def compute_temperature(ts, heat_on, wattage, k, volume, c_v, h, u_0, u_env, init_time_off):
  s = np.array(list(compute_heat_source(ts, heat_on, k, init_time_off)))
  q_ext = s * wattage
  return np.array(list(temperature_given_heat(ts, q_ext, volume, c_v, h, u_0, u_env)))


def log_likelihood(ts, heat_on, obs, wattage, k, volume, c_v, h, u_0, u_env, init_time_off, sig):
  
  u = compute_temperature(ts, heat_on, wattage, k, volume, c_v, h, u_0, u_env, init_time_off)

  lls = stats.norm.logpdf((u - obs) / sig)
  if np.any(np.isnan(lls)):
    lls[np.isnan(lls)] = -1e-4 * np.finfo(np.float).max

  return np.mean(lls)


def optimal_parameters(ts, heat_on, obs, params=None, *args, **kwdargs):

  param_names = sorted(params.keys())
  x0 = np.array([params[k] for k in param_names])

  def nll(x):
    p = {k: v for k, v in zip(param_names, x)}
    p.update(kwdargs)
    ll = log_likelihood(ts, heat_on, obs, **p)
    print pretty_params(p), ll
    return -ll

  all_bounds = parameter_bounds()
  bounds = [all_bounds.get(k, (None, None)) for k in param_names]
  lbfgs = optimize.fmin_l_bfgs_b(nll, x0=x0, approx_grad=True, bounds=bounds)
  optimal = lbfgs[0]

#   ret = optimize.fmin(nll, x0=x0, ftol=1e-4, xtol=1e-4, maxiter=10000, disp=1)
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
    df['k'] = np.exp(df['k'])
    df['h'] = np.exp(df['h'])
    df['r'] = np.exp(df['r'])
    df['log_prob'] = probs[-1]

    return df


@contextmanager
def locked_file(path, timeout=-1):
    lock_path = '%s.lock' % path
    lock = filelock.FileLock(lock_path, timeout=timeout)
    with lock.acquire():
        with open(path, 'r') as f:
            yield StringIO(f.read())


def read_log(log_file, timeout=-1):

    with locked_file(log_file, timeout) as f:
        df = pd.read_csv(f)
        df.columns = ['time', 'temperature', 'heat_on']
        # remove rows where the temperature was 0. which is
        # an indication of an invalid temperature
        df = df.iloc[df['temperature'].values > 0.]
        # Convert to kelvin.
        df['temperature'] += 273.
        return df


def read_params(param_file, timeout=-1):
    with locked_file(param_file, timeout) as f:
        return simplejson.load(f)
