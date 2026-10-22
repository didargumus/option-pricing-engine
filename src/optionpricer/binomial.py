"""Cox–Ross–Rubinstein (1979) binomial tree for European and American options."""

from __future__ import annotations

import numpy as np

__all__ = ["crr_price", "crr_exercise_boundary"]


def _payoff(ST: np.ndarray, K: float, is_call: bool) -> np.ndarray:
    return np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)


def _tree_params(T: float, r: float, sigma: float, q: float, steps: int):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0,1): increase `steps`")
    return dt, u, d, p, np.exp(-r * dt)


def crr_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
              option: str = "call", american: bool = False, steps: int = 500) -> float:
    """Price a vanilla option on a recombining CRR tree.

    Backward induction is vectorised over the nodes of each time slice, so the
    cost is O(steps^2) floating point operations but only O(steps) Python
    iterations.
    """
    is_call = option.lower() == "call"
    dt, u, d, p, disc = _tree_params(T, r, sigma, q, steps)

    j = np.arange(steps + 1)
    ST = S * u ** (steps - j) * d**j          # node j = number of down moves
    V = _payoff(ST, K, is_call)

    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if american:
            jj = np.arange(i + 1)
            V = np.maximum(V, _payoff(S * u ** (i - jj) * d**jj, K, is_call))
    return float(V[0])


def crr_exercise_boundary(S: float, K: float, T: float, r: float, sigma: float,
                          q: float = 0.0, steps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Early-exercise boundary of an American **put**.

    Returns ``(t, S_star)`` where ``S_star[k]`` is the highest node price at
    which immediate exercise is optimal at time ``t[k]`` (NaN if none).
    """
    dt, u, d, p, disc = _tree_params(T, r, sigma, q, steps)
    j = np.arange(steps + 1)
    V = np.maximum(K - S * u ** (steps - j) * d**j, 0.0)
    times, boundary = [T], [K if r >= q else np.nan]

    for i in range(steps - 1, -1, -1):
        jj = np.arange(i + 1)
        Si = S * u ** (i - jj) * d**jj
        cont = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        exer = np.maximum(K - Si, 0.0)
        V = np.maximum(cont, exer)
        mask = (exer > cont) & (exer > 0)
        times.append(i * dt)
        boundary.append(Si[mask].max() if mask.any() else np.nan)
    return np.array(times[::-1]), np.array(boundary[::-1])
