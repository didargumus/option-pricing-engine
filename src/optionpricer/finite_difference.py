r"""Crank–Nicolson finite-difference solver for the Black–Scholes PDE.

The PDE is solved in log-price :math:`x = \ln S` and time-to-maturity
:math:`\tau = T - t`:

.. math::
    \frac{\partial V}{\partial \tau} = \tfrac12\sigma^2 V_{xx}
        + (r - q - \tfrac12\sigma^2) V_x - rV .

* **Rannacher smoothing** – the first few time steps are fully implicit, which
  removes the spurious oscillations Crank–Nicolson produces around the
  non-smooth payoff (Rannacher 1984; Giles & Carter 2006).
* **American options** are handled with the Brennan–Schwartz (1977)
  algorithm: a Thomas-type tridiagonal solve in which the early-exercise
  constraint :math:`V \ge \text{payoff}` is applied during substitution. For
  puts (calls) it is exact for the linear complementarity problem because the
  exercise region is a single interval at low (high) prices.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

__all__ = ["cn_price"]


def _brennan_schwartz(a, b, c, d, payoff, is_call):
    """Solve a tridiagonal system with an obstacle constraint.

    ``a`` is the sub-diagonal, ``b`` the diagonal, ``c`` the super-diagonal
    (``a[0]`` and ``c[-1]`` unused).
    """
    n = len(b)
    bp, dp = b.astype(float).copy(), d.astype(float).copy()
    V = np.empty(n)
    if is_call:      # exercise region at high prices: eliminate upwards, substitute downwards
        for i in range(1, n):
            m = a[i] / bp[i - 1]
            bp[i] -= m * c[i - 1]
            dp[i] -= m * dp[i - 1]
        V[-1] = max(dp[-1] / bp[-1], payoff[-1])
        for i in range(n - 2, -1, -1):
            V[i] = max((dp[i] - c[i] * V[i + 1]) / bp[i], payoff[i])
    else:            # put: eliminate downwards, substitute upwards
        for i in range(n - 2, -1, -1):
            m = c[i] / bp[i + 1]
            bp[i] -= m * a[i + 1]
            dp[i] -= m * dp[i + 1]
        V[0] = max(dp[0] / bp[0], payoff[0])
        for i in range(1, n):
            V[i] = max((dp[i] - a[i] * V[i - 1]) / bp[i], payoff[i])
    return V


def cn_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
             option: str = "put", american: bool = False, n_space: int = 400,
             n_time: int = 400, width: float = 6.0, rannacher_steps: int = 4) -> float:
    """Price a vanilla option with Crank–Nicolson (plus Rannacher start-up).

    Parameters
    ----------
    n_space, n_time : grid sizes in log-price and time.
    width : half-width of the log-price grid in units of ``sigma*sqrt(T)``.
    rannacher_steps : number of initial fully-implicit half steps.
    """
    is_call = option.lower() == "call"
    half = width * sigma * np.sqrt(T)
    x0 = np.log(S)
    x = np.linspace(x0 - half, x0 + half, n_space + 1)
    dx = x[1] - x[0]
    S_grid = np.exp(x)
    payoff = np.maximum(S_grid - K, 0.0) if is_call else np.maximum(K - S_grid, 0.0)

    alpha = 0.5 * sigma**2 / dx**2
    beta = (r - q - 0.5 * sigma**2) / (2.0 * dx)
    lo, di, up = alpha - beta, -2.0 * alpha - r, alpha + beta   # L V_i = lo V_{i-1} + di V_i + up V_{i+1}

    def boundary(tau):
        if is_call:
            low = 0.0
            high = S_grid[-1] - K if american else S_grid[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
        else:
            high = 0.0
            low = K - S_grid[0] if american else K * np.exp(-r * tau) - S_grid[0] * np.exp(-q * tau)
        return low, high

    # schedule: Rannacher steps use dt/2 fully implicit, rest Crank-Nicolson
    dt = T / n_time
    schedule = [(dt / 2, 1.0)] * (2 * rannacher_steps) + [(dt, 0.5)] * (n_time - rannacher_steps)

    V = payoff.copy()
    m = n_space - 1                                   # interior points
    tau = 0.0
    for h, theta in schedule:
        tau_new = tau + h
        lo_new, hi_new = boundary(tau_new)
        # RHS: (I + (1-theta) h L) V^n on interior
        rhs = V[1:-1] + (1 - theta) * h * (lo * V[:-2] + di * V[1:-1] + up * V[2:])
        rhs[0] += theta * h * lo * lo_new
        rhs[-1] += theta * h * up * hi_new
        sub = np.full(m, -theta * h * lo)
        diag = np.full(m, 1.0 - theta * h * di)
        sup = np.full(m, -theta * h * up)
        if american:
            interior = _brennan_schwartz(sub, diag, sup, rhs, payoff[1:-1], is_call)
        else:
            ab = np.vstack([np.r_[0.0, sup[:-1]], diag, np.r_[sub[1:], 0.0]])
            interior = solve_banded((1, 1), ab, rhs)
        V = np.r_[lo_new, interior, hi_new]
        tau = tau_new

    return float(np.interp(x0, x, V))
