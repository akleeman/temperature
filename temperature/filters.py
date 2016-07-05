import numpy as np


def kalman_predict(x, P, F, Q, B=None, u=None):
  """
  Takes a pair of state and covariance arrays (x, P) corresponding
  to the state at some iteration of a kalman filter, and advances
  the state to the next time step using the transition F, process
  noise Q and optional forcings B *u.

  Parameters
  ----------
  x : np.ndarray
    A 1-d array representing the mean of the state estimate at
    some iteration, k-1, of a kalman filter.
  P : np.ndarray
    A 2-d array representing the covariance of the state estimate at
    some iteration, k-1, of a kalman filter.
  F : operator (see expand_operator)
    The state transition model.
  Q : np.ndarray
    A 2-d array representing the noise introduced in the process
    model during one time step.
  B : np.ndarray
    A 2-d array that projects the describes the impact on forcings
    (u) on the state (x)
  u : np.ndarray
    A 1-d array that holds forcings, aka control variables.

  Returns
  ---------
  x : np.ndarray
    A 1-d array representing the mean of the a priori state estimate at
    some iteration, k, of a kalman filter.  This is the best estimate
    of x_k given x_{k-1} but without accounting for observations at step k.
  P : np.ndarray
    A 2-d array representing the covariance of the a priori state estimate at
    some iteration, k, of a kalman filter. This is the best estimate
    of P_k given x_{k-1} but without accounting for observations at step k.
  """
  # F should be convertable to an operator pair (function, jacobian) that
  # applies the transition process model, ie x_k = f(x_{k-1}).
  f, F = F
  # apply the state transition function f
  x = f(x)
  # optionally apply and forcings
  if B is not None and u is not None:
    x += np.dot(B, u)
  # update the covariance of x, this comes in two parts
  # F P F^T, which is the covariance of x_k due to the
  # transition function, and Q which is the noise
  # of the process model.
  P = np.dot(np.dot(F, P), F.T) + Q
  return x, P


def kalman_update(x, P, y, H, R):
  """
  Takes a pair of state and covariance arrays (x, P) corresponding
  to the state at some iteration of a kalman filter, and computes
  and update to the state conditional on a new set of observations
  y.

  Parameters
  ----------
  x : np.ndarray
    A 1-d array representing the mean of the state estimate at
    some iteration, k-1, of a kalman filter.
  P : np.ndarray
    A 2-d array representing the covariance of the state estimate at
    some iteration, k-1, of a kalman filter.
  y : np.ndarray
    A 1-d array representing a set of new observations.
  H : operator (see expand_operator)
    The observation model. x' = H(y) + R
  R : np.ndarray
    A 2-d array representing the noise introduced in the measurement process

  Returns
  ---------
  x : np.ndarray
    A 1-d array representing the mean of the posterior state estimate at
    some iteration, k, of a kalman filter.  This is the best estimate
    of x_k given x_{k-1} and all observations up to step k.
  P : np.ndarray
    A 2-d array representing the covariance of the posterior state estimate at
    some iteration, k, of a kalman filter. This is the best estimate
    of P_k given x_{k-1} and all observations up to step k.
  """
  h, H = H
  # actual observation minus the observation model
  innov = y - h(x)
  PHT = np.dot(P, H.T)
  S = np.dot(H, PHT) + R
  dx = PHT.dot(np.linalg.solve(S, innov))
  x = x + dx
  P = np.dot(np.eye(P.shape[0]) - PHT.dot(np.linalg.solve(S, H)), P)
  return x, P


def run_filter(obs, x0, P0, process_model, observation_model):

  x = x0.copy()
  P = P0.copy()
  prev_t = None

  for t, z in obs:
    if prev_t is None:
      dt = 0.
    else:
      dt = t - prev_t
    prev_t = t

    x_new, F, Q = process_model(x, dt, t)
    x, P = kalman_predict(x, P, (lambda xx: x_new, F), Q)
    x[2] = min(x[2], np.log(2))
    x[3] = min(x[3], np.log(0.01))
    y, H, R = observation_model(x, t)
    x, P = kalman_update(x, P, z, (lambda yy: y, H), R)
    x[2] = min(x[2], np.log(2))
    x[3] = min(x[3], np.log(0.01))
    yield t, x, P

