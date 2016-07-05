import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from timeit import itertools


def process_model(x, dt, t, t_0=0., s_0=0., s_max=1.0, m=0.15, deg_inf=25, *args, **kwdargs):
  x_new = x.copy()
  # s(t) = s_0 + (s_max - s_0) / (1 + e^(-m(t-t_0)))
  #      = s_0 + (s_max - s_0) * f(m(t - t_0))
  f = 1 / (1 + np.exp(-x[1] * (t - t_0)))
  s = s_0 + (s_max - s_0) * f
  # dT/dt = exp(H) s(t) - exp(k) (T - T_inf)
  x_new[0] += dt * (np.exp(x[2]) * s - np.exp(x[3]) * (x[0] - 25));
  # dm/dt = 0
  x_new[1] += 0
  # dH(t)/dt = 0
  x_new[2] += 0.
  # dk(t)/dt = 0
  x_new[2] += 0.

  F = np.eye(x_new.size)
  # T^{n+1} = T^{n} + dt * [exp(H^{n}) s^{n} - exp(k^{n})(T^{n} - T_inf)]
  # dT^{n+1}/dT^{n} = 1 - dt * exp(k^{n})
  F[0][0] = 1.0 - dt * np.exp(x[3]);
  # dT^{n+1}/dm^{n} = dt * exp(H)^{n} ds/dm.
  #                 = dt * exp(H)^{n} (s_max - s_0) * (t - t_0) * f * (1 - f)
  F[0][1] = dt * np.exp(x[2]) * (s_max - s_0) * (t - t_0) * f * (1 - f);
  # dT^{n+1}/dH_max^{n} = dt * s^{n} exp(H^{n})
  F[0][2] = dt * s * np.exp(x[2]);
  # dT^{n+1}/k^{n} = -dt * (T - T_inf) * exp(k)
  F[0][3] = dt * (deg_inf - x[0]) * np.exp(x[3]);

  # s^{n+1} = s^n + m * (s_max - s_0) * l(m * (t - t_0)) * l(m * (t - t_0))
  # ds^{n+1} / dT^n = 0
  F[1][0] = 0.0
  # ds^{n+1} / dm^n = 1
  F[1][1] = 1.0
  # ds^{n+1} / dH_max^n = 0.
  F[1][2] = 0.0
  # ds^{n+1} / dk = 0.
  F[1][2] = 0.0

  Q = np.square(dt) * np.diag([0.1, 0.01, 0.5, 0.1])

  return x_new, F, Q


def observation_model(x, t, t_0, s_0, s_max, deg_inf, sig_T):
  # We observe T and dT/dt
  T = x[0]
  f = 1 / (1 + np.exp(-x[1] * (t - t_0)))
  s = s_0 + (s_max - s_0) * f

  delta_T = (np.exp(x[2]) * s - np.exp(x[3]) * (x[0] - deg_inf))

  z = np.array([T, delta_T])
  H = np.zeros((z.size, x.size))
  # T = T
  H[0, 0] = 1.0
  # z[1] = dT/dt = exp(H) * s(t) - k (T - T_inf)
  # dz[1]/dT = -k
  H[1, 0] = -x[3]
  # dz[1]/dm = np.exp(x[2]) * (s_max - s_0) * (t - t_0) * f * (1 - f);
  H[1, 1] = np.exp(x[2]) * (s_max - s_0) * (t - t_0) * f * (1 - f);
  # dz[1]/dH = s
  H[1, 2] = s
  # dz[1]/dk = T_inf - T
  H[1, 3] = deg_inf - x[0]

  R = np.diag([np.square(sig_T), 2 * np.square(sig_T)])

  return z, H, R



def run_filter(x, obs, **kwdargs):
  P0 = np.diag([np.square(30), 1., np.square(30), np.square(30)])
  P = P0.copy()
  prev_t = None

  for t, z in obs:
    if prev_t is None:
      dt = 0.
    else:
      dt = t - prev_t
    prev_t = t

    x_new, F, Q = process_model(x, dt, t, **kwdargs)
    x, P = kalman_predict(x, P, (lambda xx: x_new, F), Q)
    x[2] = min(x[2], np.log(2))
    x[3] = min(x[3], np.log(0.01))
    y, H, R = observation_model(x, t, **kwdargs)
    x, P = kalman_update(x, P, z, (lambda yy: y, H), R)
    x[2] = min(x[2], np.log(2))
    x[3] = min(x[3], np.log(0.01))
    yield t, x, P


def derivatives(obs):
  ts, xs = map(np.array, zip(*list(obs)))

  last_time = ts[-1]
  ts = ts - last_time
  A = np.array([np.ones(len(ts)),
                ts]).T

  ret = np.linalg.lstsq(A, xs)
  return last_time, ret[0]


def split_every(iterable, n):
  """
  Breaks an iterable into chunks of size n.

  Parameters
  ----------
  iterable : iterable
    The iterable that will be broken into size n chunks.  If
    the end of the iterable is reached and the remainder is not
    size n (ie, if the overall length is not a multiple of the
    chunk size) an exception is raised.
  n : int
    The size of each chunk.

  Returns
  -------
  chunks : iterable
    A generator that yields size tuples of length n, containing
    the next n elements in iterable.
  """
  n = int(n)
  i = iter(iterable)
  piece = list(itertools.islice(i, n))
  while len(piece) > 0:
    if len(piece) == n:
      yield piece
    piece = list(itertools.islice(i, n))


def real():
  import pandas as pd
  df = pd.read_csv('one_gallon_on_burner.log')
  df.columns = ['time', 'T']
  df['time'] /= 1000.

  df = df.iloc[df['time'].values >= 50.]
  df = df.iloc[df['time'].values <= 100.]

  deg_inf = 25
  sig_T = 0.02
  t_0 = 30
  s_0 = 0
  s_max = 1

  observations = zip(df['time'].values, df['T'].values)


  x0 = np.array([30., 0.15, np.log(0.2), np.log(0.05)])
  ret = list(run_filter(x0, observations, t_0=t_0, s_0=s_0, s_max=s_max,
                        deg_inf=deg_inf, sig_T=sig_T))

  ts = df['time'].values


  def extend(cnt):
    t, x, P = ret[-1]
    dt = 1.
    for i in range(cnt):
      t += dt
      x_new, F, Q = process_model(x, dt, t, t_0=t_0, s_0=s_0, s_max=s_max,
                                  deg_inf=deg_inf, sig_T=sig_T)
      x, P = kalman_predict(x, P, (lambda xx: x_new, F), Q)
      x[2] = min(x[2], np.log(2))
      x[3] = min(x[3], np.log(0.01))
      yield t, x, P

  ret = itertools.chain(ret, list(extend(10)))

  ts, mus, Ps = zip(*list(ret))
  mus = np.array(mus)
  vars = np.array([np.diag(P) for P in Ps])

  fig, axes = plt.subplots(2, 2, sharex=True)

  for ax, mu, var, vname in zip(axes.reshape(-1), mus.T,
                                vars.T, ['T', 'm', 'H_max', 'k']):
    ax.set_title(vname)
    ax.plot(ts, mu, color='steelblue', lw=2)
    ax.fill_between(ts, mu + np.sqrt(var), mu - np.sqrt(var), alpha=0.8, color='steelblue')
    delta = np.max(mu) - np.min(mu)
    lim = [np.min(mu) - 0.3 * delta, np.max(mu) + 0.3 * delta]
    ax.set_ylim(lim)

  plt.show()

if __name__ == "__main__":
  synthetic()


