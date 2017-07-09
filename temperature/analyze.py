import os
import sys
import simplejson
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils, parameters, control

def main(args):
    df = utils.read_log(args.input)

    #   df = df.iloc[:1100]
    #   df['heat_on'] = (df['time'] < 700.) * df['heat_on']
    #   df['temperature'] += 273.

    x0 = {'k': 0.17,
          'c_v': 3000.,
          'volume': 1.,
          'u_0': 300., 
          'h': 2.2,
          'wattage': 800,
          'u_env': 298.,
          'init_time_off': 80.}

    fixed = {}

    output_path = '%s.params' % args.input
    if os.path.exists(output_path):
        print "reading params from: %s" % output_path
        with open(output_path, 'r') as f:
            optimal = simplejson.load(f)
    else:
        optimal = utils.optimal_parameters(df['time'].values,
                                         df['heat_on'].values,
                                         df['temperature'].values,
                                         params=x0,
                                         sig=0.1,
                                         **fixed)
        optimal.update(fixed)

        with open(output_path, 'w') as f:
            simplejson.dump(optimal, f, sort_keys=True, indent=2)

    print utils.pretty_params(optimal)

    short = df.iloc[:250]
    heat = control.control_params(320, short['time'].values,
                                  short['heat_on'].values,
                                  short['temperature'].values,
                                  optimal)

    u_samp = np.array(list(utils.compute_temperature(df['time'].values,
                                                     df['heat_on'].values,
                                                     **optimal)))

    plt.plot(df['time'], u_samp)
    plt.plot(df['time'], df['temperature'])
    ax = plt.twinx()
    ax.fill_between(df['time'], 0., df['heat_on'], color='r', alpha=0.5)
    ax.set_ylim([-0.1, 1.1])
    plt.show()

    import ipdb; ipdb.set_trace()


def create_parser(parser):
  parser.add_argument('input')
  parser.add_argument('--profile', default=False, action="store_true")
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
