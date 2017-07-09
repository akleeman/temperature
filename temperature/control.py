import copy
import numpy as np
import RPi.GPIO as GPIO

from scipy import optimize

import utils


def control_params(target, ts, heat_on, us, params):
  
    params = copy.copy(params)
    k = params.get('k')

    n = 40
    new_ts = 1. + np.linspace(0., 2 * np.power(4 * np.pi * k * 1e-3 ** 2, -1. / 3.), n)
    t_aug = np.concatenate([ts, ts[-1] + new_ts])

    def temperature(h):
      heat_on_aug = np.concatenate([heat_on, h])

      s_aug = np.array(list(utils.compute_heat_source(t_aug, heat_on_aug, k)))

      q_ext_pred = s_aug[-n:] * params.get('wattage')
      params['u_0'] = us[-1]

      u_pred = np.array(list(utils.temperature_given_heat(new_ts, q_ext_pred, **params)))
      return u_pred

    u_start = us[-1]
    if u_start < target:
      u_when_on = temperature(np.ones(n))
      if u_when_on.max() < target:
        # Leaving the heat on all the time still doesn't heat us up
        # enough to hit the desired temperature, so we leave the heat
        # on.
        return True
      # at some point the temperature must have crossed the target
      # value.  Since leaving the heat on all the time should yield
      # the fastest path to target, we use the time at which it
      # crossed as the beginning point for our error metric.
      fastest_to_target = np.nonzero(u_when_on < target)[0][-1]
    if u_start > target:
      u_when_off = temperature(np.zeros(n))
      if u_when_off.min() > target:
        # Leaving the heat off the whole time still doesn't
        # cool us off enough to hit the target, so we leave it off
        return False
      fastest_to_target = np.nonzero(u_when_off > target)[0][-1]
      
    def error(h):
      errors = temperature(h) - target
      err = np.linalg.norm(errors[fastest_to_target:])
      return err

    x0 = 0.5 * np.ones(n)

    ret = optimize.fmin_l_bfgs_b(error, x0,
                                 approx_grad=True,
                                 bounds=[(0, 1) for i in range(n)])

    h_star = ret[0]
    return h_star[0]


def control_daemon(log_path, heat_pin, timeout):
    param_path = '%s.params' % log_path

    while True:
        df = utils.read_log(log_path, timeout=timeout)

        params = utils.read_params(param_path, timeout=timeout)
        heat_on = control_params(320, df['time'].values,
                                 df['heat_on'].values,
                                 df['temperature'].values,
                                 params)

        import ipdb; ipdb.set_trace()

        if heat_on:
            GPIO.output(heat_pin, GPIO.HIGH)
        else:
            GPIO.output(heat_pin, GPIO.LOW)

