import os
import sys
import logging
import argparse
import functools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import models
import filters
import synthetic


def estimate(args):

  times = np.linspace(0., 120., 121.)

  deg_inf = 25
  sig_T = 0.01
  t_0 = 30
  s_0 = 0
  s_max = 1
  m = 0.15

  process = functools.partial(models.process_model,
                              t_0=t_0, s_0=s_0, s_max=s_max, m=m,
                              deg_inf=deg_inf)
  observation = functools.partial(models.observation_model,
                                  t_0=t_0, s_0=s_0, s_max=s_max,
                                  deg_inf=deg_inf, sig_T=sig_T)

  x0 = np.array([30., 0.15, np.log(0.2), np.log(0.01)])
  synth = list(synthetic.synthetic(times, x0, process, observation))

  truth_times, truths, zs = zip(*synth)
  truths = np.array(truths)
  zs = np.array(zs)

  P0 = np.diag([np.square(30), 1., np.square(30), np.square(30)])

  preds = list(filters.run_filter(zip(truth_times, zs),
                                  x0, P0, process, observation))

  ts, xs, Ps = zip(*preds)

  xs = np.array(xs)

  vars = np.array([np.diag(P) for P in Ps])

  fig, axes = plt.subplots(2, 2, sharex=True)

  sns.set_style('darkgrid')
  for ax, mu, var, truth, vname in zip(axes.reshape(-1), xs.T, vars.T,
                                truths.T, ['T', 'm', 'H_max', 'k']):
    ax.set_title(vname)
    ax.plot(ts, mu, color='steelblue', lw=2)
    ax.plot(truth_times, truth, color='black', lw=1)
    ax.fill_between(ts, mu + np.sqrt(var), mu - np.sqrt(var), alpha=0.8, color='steelblue')
    delta = np.max(mu) - np.min(mu)
    lim = [np.min(mu) - 0.3 * delta, np.max(mu) + 0.3 * delta]
    ax.set_ylim(lim)

  plt.show()

  import ipdb; ipdb.set_trace()



if __name__ == "__main__":
  script_name = os.path.basename(sys.argv[0])
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('input', nargs="*")
  parser.add_argument('--profile', action="store_true", default=False)
  args = parser.parse_args()

  logging.basicConfig(stream=sys.stderr, level=logging.INFO)
  logging.getLogger().setLevel(logging.INFO)

  if args.profile:
    import cProfile
    cProfile.runctx('estimate(args)', globals(), locals(),
                    '%s.prof' % script_name)
  else:
    estimate(args)
