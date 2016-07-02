import numpy as np
import seaborn as sns


def synthetic(T_inf=25, sig_T=0.01, T_0=30, s_0=0, s_max=1):

  np.random.seed(1982)

  def draw_sample(t, x):
    z, H, R = observation_model(x, t, t_0=t_0, s_0=s_0, s_max=s_max,
                                  deg_inf=deg_inf, sig_T=sig_T)
    noise = np.dot(np.linalg.cholesky(R), np.random.normal(size=z.size))
    return z + noise

  x0 = np.array([30., 0.15, np.log(0.2), np.log(0.01)])
  ts = np.linspace(0., 120., 121.)
  dts = np.diff(ts)

  def iter_x(x):
    yield ts[0], x.copy()
    for dt, t in zip(dts, ts[1:]):
      x = process_model(x, dt, t, t_0=30, deg_inf=deg_inf)[0]
      yield t, x.copy()

  xs = list(iter_x(x0))
  observations = [(t, draw_sample(t, x)[0]) for t, x in xs]

  observations = list(split_every(observations, 5))

  observations = [derivatives(o) for o in observations]

  x0[2] = np.log(1.)
  x0[3] = np.log(0.1)

  ts, mus, Ps = zip(*list(run_filter(x0, observations, t_0=t_0, s_0=s_0, s_max=s_max,
                                  deg_inf=deg_inf, sig_T=sig_T)))
  mus = np.array(mus)
  vars = np.array([np.diag(P) for P in Ps])

  subset_xs = xs[::5][1:]
  truths = np.array([x[1] for x in subset_xs])
  ts = np.array([x[0] for x in subset_xs])
  fig, axes = plt.subplots(2, 2, sharex=True)

  for ax, one_x, mu, var, vname in zip(axes.reshape(-1), truths.T, mus.T,
                                       vars.T, ['T', 'm', 'H_max', 'k']):
    ax.set_title(vname)
    ax.plot(ts, mu, color='steelblue', lw=2)
    ax.fill_between(ts, mu + np.sqrt(var), mu - np.sqrt(var), alpha=0.8, color='steelblue')
    ax.plot(ts, one_x, color='black')
    delta = np.max(mu) - np.min(mu)
    lim = [np.min(mu) - 0.3 * delta, np.max(mu) + 0.3 * delta]
    ax.set_ylim(lim)

  plt.show()
