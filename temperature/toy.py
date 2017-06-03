import os
import sys
import copy
import logging
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from temperature import utils

sns.set_style('darkgrid')

import utils


def pot_of_water(radius_m, height_m):
  
  area = np.pi * radius_m ** 2
  volume = height_m * area * 1000  # volume in litres
  
  return {'c_v': 4185.5,  # J/(kg K)
          # litres
          'volume': volume,
          # low speed flow of air over surface then less heat loss
          # over surface of the pot
          'h': 100 * area + 2 * (2 * np.pi * radius_m + area),
          # It takes a minute and a half for the stove to heat up.
          't_lag': 90.,
          # The pot starts at 30 C
          'u_0' : 303.,
          # the ambient temperature is 25 C
          'u_env': 298.,
          'wattage': 400,
   }


def main(args):

  np.random.seed(1982)

  ts = np.linspace(0., 1600, 201)

  heat_on = np.mod((ts / 200).astype('int'), 2) != 1
  heat_on[ts > 800.] = False
  heat_on = heat_on.astype('bool')

  p = pot_of_water(0.10, 0.10)
  p['s_0'] = 0.

  print "==============TRUE PARAMETERS====================="
  print utils.pretty_params(p)
  print "=================================================="

  sig = 0.1

  utils.compute_heat_source(ts, heat_on, p['t_lag'], s_0=0.)
  utils.compute_temperature(ts, heat_on, **p)

  s = np.array(list(utils.compute_heat_source(ts, heat_on, p['t_lag'], s_0=0.)))
  u = np.array(list(utils.compute_temperature(ts, heat_on, **p)))
  # Add observation noise
  obs = u + sig * np.random.normal(size=u.size)
  
  fig, ax = plt.subplots(1, 2)
  n = 1
  best = -np.inf
  best_params = None
  for i in range(n):

    x0 = {'t_lag': 180.,
          'c_v': 3000.,
          'volume': 1.,
          'u_0': 300., 
          'h': 3.}

    fixed = {'wattage': 400,
             'u_env': 298.,
             's_0': 0.}

    optimal = utils.optimal_parameters(ts, heat_on, obs, params=x0, sig=sig, **fixed)

    optimal.update(fixed)

    print utils.pretty_params(optimal)

    ll = utils.log_likelihood(ts, heat_on, obs, sig=sig, **optimal)
    print "optimal_likelihood", ll
    if ll > best:
      best = ll
      best_params = optimal

    u_samp = np.array(list(utils.compute_temperature(ts, heat_on, **optimal)))

    ax[0].plot(ts, u, color='black')
    ax[0].plot(ts, obs, alpha=0.5, color='green')
    ax[0].plot(ts, u_samp, color='steelblue')

  plt.show()

  np.random.seed(1982)
  sample = utils.sample_parameters(ts, heat_on, obs, n=100, params=optimal, iterations=10)

  fig, ax = plt.subplots(1, 2)

  ax[0].plot(ts, u, color='black')
  a0_ylim = ax[0].get_ylim()

  ax[1].plot(ts, s, color='black')
  a1_ylim = ax[1].get_ylim()

  for t_lag, u_0, u_max, h, r in sample[['t_lag', 'u_0', 'u_max', 'h', 'r']].values:
    s_samp = np.array(list(utils.compute_heat_source(ts, heat_on, t_lag, u_0, u_max, u_env=25.)))
    u_samp = np.array(list(utils.compute_temperature(ts, s_samp, h, r, u_0, u_env=25.)))
    ax[0].plot(ts, u_samp, alpha=0.5, color='steelblue')
    ax[1].plot(ts, s_samp, alpha=0.5, color='steelblue')

  ax[0].set_ylim(a0_ylim)
  ax[1].set_ylim(a1_ylim)

  plt.show()

  import ipdb; ipdb.set_trace()




def create_parser(parser):
  parser.add_argument('--profile', default=False, action="store_true")
  parser.add_argument('-n', type=int, default=None)
  return parser


if __name__ == "__main__":
  script_name = os.path.basename(sys.argv[0])
  parser = argparse.ArgumentParser(description=__doc__)
  parser = create_parser(parser)
  args = parser.parse_args()

  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
  logging.getLogger().setLevel(logging.INFO)

  if args.profile:
    import cProfile
    cProfile.runctx('main(args)', globals(), locals(),
                    '%s.prof' % script_name)
  else:
    main(args)
