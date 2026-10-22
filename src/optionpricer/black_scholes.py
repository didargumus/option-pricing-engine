"""Black–Scholes–Merton closed-form prices and Greeks.

All functions are fully vectorised: every argument may be a scalar or a NumPy
array, and standard broadcasting rules apply. A continuous dividend yield ``q``
is supported throughout (Merton 1973).

Notation
--------
S      spot price
K      strike
T      time to maturity in years
r      continuously-compounded risk-free rate
sigma  volatility (annualised)
q      continuous dividend yield
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

ArrayLike = float | np.ndarray

__all__ = ["d1_d2", "bs_price", "bs_greeks"]


def _check_option(option: str) -> bool:
    option = option.lower()
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    return option == "call"


def d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
          sigma: ArrayLike, q: ArrayLike = 0.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the standard :math:`d_1, d_2` terms.

    .. math::
        d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt T},
        \qquad d_2 = d_1 - \sigma\sqrt T
    """
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
             sigma: ArrayLike, q: ArrayLike = 0.0, option: str = "call") -> np.ndarray:
    """Black–Scholes–Merton price of a European call or put."""
    is_call = _check_option(option)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S, K, T, r, q = map(np.asarray, (S, K, T, r, q))
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if is_call:
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def bs_greeks(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
              sigma: ArrayLike, q: ArrayLike = 0.0, option: str = "call") -> dict[str, np.ndarray]:
    """Analytic first- and second-order Greeks.

    Returns a dict with keys ``delta, gamma, vega, theta, rho, vanna, volga``.

    Conventions: ``vega`` and ``rho`` are per 1.00 change (multiply by 0.01 for
    "per vol point"); ``theta`` is the derivative with respect to calendar
    time *t* (i.e. ``-dV/dT``) per year.
    """
    is_call = _check_option(option)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    sqrt_t = np.sqrt(T)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * df_q * pdf_d1 * sqrt_t
    vanna = -df_q * pdf_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    common_theta = -S * df_q * pdf_d1 * sigma / (2 * sqrt_t)

    if is_call:
        delta = df_q * norm.cdf(d1)
        theta = common_theta - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
        rho = K * T * df_r * norm.cdf(d2)
    else:
        delta = -df_q * norm.cdf(-d1)
        theta = common_theta + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
        rho = -K * T * df_r * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta,
            "rho": rho, "vanna": vanna, "volga": volga}
