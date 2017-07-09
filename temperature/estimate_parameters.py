import os
import sys
import copy
import logging
import filelock
import argparse
import simplejson
import numpy as np
import matplotlib.pyplot as plt

import utils, parameters, control


logger = logging.getLogger(os.path.basename(__file__))


def main(args):
  x0 = {'k': 0.17,
        'c_v': 3000.,
        'volume': 1.,
        'u_0': 300.,
        'h': 2.2,
        'wattage': 800,
        'u_env': 298.,
        'init_time_off': 80.}

  fixed = {}
  
  output_path = args.output or '%s.params' % args.input

  while True:
    df = utils.read_log()

    df = df.iloc[:1100]
    df['heat_on'] = (df['time'] < 700.) * df['heat_on']
    df['temperature'] += 273.

    if os.path.exists(output_path):
      logger.info("Reading params from: %s" % output_path)
      with open(output_path, 'r') as f:
        prev_optimal = simplejson.load(f)
        initial = {k: prev_optimal[k] for k in x0.keys()}
    else:
      initial = x0
    
    optimal = utils.optimal_parameters(df['time'].values,
                                       df['heat_on'].values,
                                       df['temperature'].values,
                                       params=initial,
                                       sig=0.1,
                                       **fixed)
    optimal.update(fixed)
  
    with open(output_path, 'w') as f:
      simplejson.dump(optimal, f, sort_keys=True, indent=2)

    logging.info(utils.pretty_params(optimal))



def create_parser(parser):
  parser.add_argument('input')
  parser.add_argument('--output', default=None)
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
