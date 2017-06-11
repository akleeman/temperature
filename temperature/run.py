import time
import numpy as np
import RPi.GPIO as GPIO

from w1thermsensor import W1ThermSensor
from w1thermsensor.core import SensorNotReadyError

GPIO.setmode(GPIO.BCM)
GPIO.setup(25, GPIO.OUT, initial=GPIO.LOW)

sensor = W1ThermSensor()

st = time.time()
try:
    f = open('output.log', 'w')
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
        f.write(output + "\n")
        f.flush()
        print output
except KeyboardInterrupt:
    f.close()
    GPIO.output(25, GPIO.LOW)
