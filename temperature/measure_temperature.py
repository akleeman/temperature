import os
import sys
import time
import logging
import argparse
import filelock
import threading
import numpy as np
import RPi.GPIO as GPIO

from w1thermsensor import W1ThermSensor
from w1thermsensor.core import SensorNotReadyError

GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT, initial=GPIO.LOW)

sensor = W1ThermSensor()

logger = logging.getLogger(os.path.basename(__file__))


def write_measurement(output_line, output_path, lock_path, time_out):
  # Write the new measurement to file
  lock = filelock.FileLock(lock_path)
  try:
    with lock.acquire(timeout=time_out):
      with open(output_path, 'a') as f:
        f.write(output_line.strip('\n') + "\n")
        logger.info(output_line)
    return True
  except filelock.Timeout:
    logger.info("Failed to acquire lock.")
    return False


def main(args):
  output_path = args.output or 'output.log'

  lock_path = '%s.lock' % output_path

  st = time.time()
  try:

      while True:
          time.sleep(1.)
          try:
              temperature_in_celsius = sensor.get_temperature()
          except SensorNotReadyError:
              temperature_in_celsius = np.nan

          elapsed = time.time() - st
          heat_on = np.mod(np.floor(elapsed / 100.), 2) == 1

          if heat_on:
              GPIO.output(25, GPIO.HIGH)
          else:
              GPIO.output(25, GPIO.LOW)

          output = ",".join(map(str, [elapsed, temperature_in_celsius, heat_on]))

          multiprocessing.


  except KeyboardInterrupt:
      f.close()
      GPIO.output(25, GPIO.LOW)


def create_parser(parser):
  parser.add_argument('--output', default=None)
  parser.add_argument('--rate', type=int, default=1)
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
