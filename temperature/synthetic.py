import numpy as np


def synthetic(iter_time, x0, process_model, observation_model):
  """
  Produces a stream of truth and observations using a process model
  and observation model initialized at x0 for each of the times in
  iter_time.

  Parameters
  ----------
  iter_time : iterable
    A iterable which generates times to be used for the simulation.
    The first time should correspond to the state x0
  x0 : np.ndarray
    The filter state at time 0.
  process_model : callable
    A function with signature f(x, dt, t) which should advance the
    filter state, x, from time (t - dt) to time t.  Return value
    should be a tuple (x, F, Q) where x is the new filter state,
    F is the linearization of the process model jacobian in the vicinity
    of x and Q is the covariance of the process model uncertainty.
  observation_model : callable
    A function with signature f(x, t) which should return a tuple
    (y, H, R) which correspond to the observable, y, the linearization
    of the observation model jacobian in the vicinity of x and R
    the covariance of the observation.

  Yields
  -------
  truth : tuple
    A tuple (t, x) which is a single time (t) and filter state, x
    that represent the actual filter state at that time.
  observed : tuple
    A tuple (t, z) which is a single time (t) and observation z
    that represents an observation of the actual state x.
  """

  def observe(t, x):
    y, H, R = observation_model(t, x)
    noise = np.dot(np.linalg.cholesky(R), np.random.normal(size=y.size))
    return y + noise

  iter_time = iter(iter_time)
  prev_time = iter_time.next()
  x = x0

  yield (prev_time, x), (prev_time, observe(x, prev_time))
  for t in iter_time:
    dt = t - prev_time
    x, F, Q = process_model(x, dt, t)
    z = observe(x, t)
    yield (t, x), (t, z)
