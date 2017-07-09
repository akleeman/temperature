import os
import time
import logging
import datetime
import filelock
import numpy as np
import RPi.GPIO as GPIO

from w1thermsensor import W1ThermSensor
from w1thermsensor.core import SensorNotReadyError

sensor = W1ThermSensor()

logger = logging.getLogger(os.path.basename(__file__))


def write_measurement(output_line, output_path, lock_path, timeout=-1):
    """
    Takes a pre-formatted output line and writes it to file, first
    making sure to optain a lock.
    """
    lock = filelock.FileLock(lock_path, timeout=timeout)
    try:
        with lock.acquire():
            with open(output_path, 'a') as f:
                f.write(output_line.strip('\n') + "\n")
                logger.info(output_line)
        return True
    except filelock.Timeout:
        logger.info("Failed to acquire lock.")
        return False


def measure_temperature():
    """
    Uses the OneWire interface to read the temperature.
    """
    try:
        # Sometimes the sensor returns zero or sub zero temperatures
        # which are invalid.  For now we simply never allow sub zero
        # readings.
        temperature_in_celsius = sensor.get_temperature()
        if temperature_in_celsius > 0.:
          return temperature_in_celsius
        else:
          logger.warn("Encountered a sub-zero temperature, skipping it.")
          return np.nan

    except SensorNotReadyError:
        logger.warn("Temperature sensor is not ready.")
        return np.nan


def measurement_daemon(output_path, sample_rate=1., heat_pin=25):

    lock_path = '%s.lock' % output_path
    st = time.time()

    while True:
        time.sleep(sample_rate)
        temperature_celsius = measure_temperature()
        # Read the output pin to see if the heat is on.
        heat_on = GPIO.input(heat_pin)
        # Include an absolute date time
        time_str = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')
        # as well as an elapsed number of seconds
        elapsed = time.time() - st
        output = ",".join(map(str, [time_str, elapsed, temperature_celsius, heat_on]))
        # attempt to write to file.
        success = write_measurement(output, output_path,
                                    lock_path, time_out=0.9 * sample_rate)
        if not success:
          logger.warn("Failed to write measurement (%s) to file." % output)
