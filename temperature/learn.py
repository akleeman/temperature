import os
import logging
import filelock
import simplejson

import utils
import parameters

logger = logging.getLogger(os.path.basename(__file__))

      
def learn_daemon(log_path, timeout):
    param_path = '%s.params' % log_path

    params = parameters.pot_of_water(0.1, 0.05)
    while True:
        df = utils.read_log(log_path)

        params = utils.optimal_parameters(df['time'].values,
                                          df['heat_on'].values,
                                          df['temperature'].values,
                                          params=params,
                                          sig=0.1)

        lock_path = '%s.lock' % param_path
        lock = filelock.FileLock(lock_path, timeout=timeout)
        with lock.acquire():
            with open(param_path, 'w') as f:
                simplejson.dump(params, f, sort_keys=True, indent=2)
