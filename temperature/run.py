import os
import sys
import time
import logging
import argparse
import filelock
import threading
import numpy as np
import RPi.GPIO as GPIO


logger = logging.getLogger(os.path.basename(__file__))

import learn
import control
import measure


def main(args):
  
    output_path = args.output or 'output.log'

    measure_thread = threading.Thread(name='measure', target=measure.measurement_daemon,
                                      args=(output_path,
                                            args.sample_rate,
                                            args.heat_pin))
    measure_thread.setDaemon(True)
    

    control_thread = threading.Thread(name='control', target=control.control_daemon,
                                      args=(output_path,
                                            args.heat_pin))
    control_thread.setDaemon(True)


    learn_thread = threading.Thread(name='learn', target=learn.learn_daemon,
                                      args=(output_path,
                                            args.timeout))
    

    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(args.heat_pin, GPIO.OUT, initial=GPIO.LOW)

        measure_thread.start()
        control_thread.start()

    finally:
        GPIO.output(25, GPIO.LOW)


def create_parser(parser):
    parser.add_argument('--output', default=None)
    parser.add_argument('--rate', type=int, default=1)
    parser.add_argument('--profile', default=False, action="store_true")
    parser.add_argument('--heat-pin', default=25, type=int)
    parser.add_argument('--sample-rate', default=1., type=float)
    parser.add_argument('--timeout', default=-1, type=float)
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
