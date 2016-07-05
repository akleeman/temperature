import numpy as np


def process_model(x, dt, t, t_0=0., s_0=0., s_max=1.0, m=0.15, deg_inf=25):
  # x = [T m H_max k]
  x_new = x.copy()
  # s(t) = s_0 + (s_max - s_0) / (1 + e^(-m(t-t_0)))
  #      = s_0 + (s_max - s_0) * f(m(t - t_0))
  f = 1 / (1 + np.exp(-x[1] * (t - t_0)))
  s = s_0 + (s_max - s_0) * f
  # dT/dt = exp(H) s(t) - exp(k) (T - T_inf)
  x_new[0] += dt * (np.exp(x[2]) * s - np.exp(x[3]) * (x[0] - 25));
  # dm/dt = 0
  x_new[1] += 0
  # dH(t)/dt = 0
  x_new[2] += 0.
  # dk(t)/dt = 0
  x_new[2] += 0.

  F = np.eye(x_new.size)
  # T^{n+1} = T^{n} + dt * [exp(H^{n}) s^{n} - exp(k^{n})(T^{n} - T_inf)]
  # dT^{n+1}/dT^{n} = 1 - dt * exp(k^{n})
  F[0][0] = 1.0 - dt * np.exp(x[3])
  # dT^{n+1}/dm^{n} = dt * exp(H)^{n} ds/dm.
  #                 = dt * exp(H)^{n} (s_max - s_0) * (t - t_0) * f * (1 - f)
  F[0][1] = dt * np.exp(x[2]) * (s_max - s_0) * (t - t_0) * f * (1 - f)
  # dT^{n+1}/dH_max^{n} = dt * s^{n} exp(H^{n})
  F[0][2] = dt * s * np.exp(x[2])
  # dT^{n+1}/k^{n} = -dt * (T - T_inf) * exp(k)
  F[0][3] = dt * (deg_inf - x[0]) * np.exp(x[3])

  # s^{n+1} = s^n + m * (s_max - s_0) * l(m * (t - t_0)) * l(m * (t - t_0))
  # ds^{n+1} / dT^n = 0
  F[1][0] = 0.0
  # ds^{n+1} / dm^n = 1
  F[1][1] = 1.0
  # ds^{n+1} / dH_max^n = 0.
  F[1][2] = 0.0
  # ds^{n+1} / dk = 0.
  F[1][2] = 0.0

  Q = np.square(dt) * np.diag([0.1, 0.01, 0.5, 0.1])

  return x_new, F, Q


def observation_model(x, t, t_0, s_0, s_max, deg_inf, sig_T):
  # We observe T and dT/dt
  T = x[0]
  f = 1 / (1 + np.exp(-x[1] * (t - t_0)))
  s = s_0 + (s_max - s_0) * f

  delta_T = (np.exp(x[2]) * s - np.exp(x[3]) * (x[0] - deg_inf))

  z = np.array([T, delta_T])
  H = np.zeros((z.size, x.size))
  # T = T
  H[0, 0] = 1.0
  # z[1] = dT/dt = exp(H) * s(t) - k (T - T_inf)
  # dz[1]/dT = -k
  H[1, 0] = -x[3]
  # dz[1]/dm = np.exp(x[2]) * (s_max - s_0) * (t - t_0) * f * (1 - f);
  H[1, 1] = np.exp(x[2]) * (s_max - s_0) * (t - t_0) * f * (1 - f)
  # dz[1]/dH = s
  H[1, 2] = s
  # dz[1]/dk = T_inf - T
  H[1, 3] = deg_inf - x[0]

  R = np.diag([np.square(sig_T), 2 * np.square(sig_T)])

  return z, H, R
