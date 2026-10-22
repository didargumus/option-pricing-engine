"""Robust, vectorised Black–Scholes implied volatility.

Algorithm
---------
1. In-the-money quotes are converted to the equivalent out-of-the-money option
   through put–call parity (identical implied vol, but far better conditioned
   because the price is pure time value).
2. Prices violating no-arbitrage bounds return ``NaN``.
3. Safeguarded Newton–Raphson from the Brenner–Subrahmanyam (1988) start.
4. Any option where Newton did not converge (tiny vega in the wings) is solved
   by a *vectorised* bisection on [1e-4, 5], which always converges because
   the price is monotone in σ.
"""

from __future__ import annotations

import numpy as np

from .black_scholes import bs_greeks, bs_price

__all__ = ["implied_vol"]

_VOL_LO, _VOL_HI = 1e-4, 5.0


def _otm_price(sig, S, K, T, r, q, is_call):
    return np.where(is_call, bs_price(S, K, T, r, sig, q, "call"), bs_price(S, K, T, r, sig, q, "put"))


def _bisect(price, S, K, T, r, q, is_call, n_iter=100):
    lo = np.full_like(price, _VOL_LO)
    hi = np.full_like(price, _VOL_HI)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_high = _otm_price(mid, S, K, T, r, q, is_call) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    return 0.5 * (lo + hi)


def implied_vol(price, S, K, T, r, q=0.0, option: str = "call",
                tol: float = 1e-12, max_iter: int = 30) -> np.ndarray:
    """Implied volatility of European option quotes (vectorised; NaN when undefined)."""
    price, S, K, T, r, q = np.broadcast_arrays(*(np.asarray(a, float) for a in (price, S, K, T, r, q)))
    shape = price.shape
    price, S, K, T, r, q = (a.ravel().copy() for a in (price, S, K, T, r, q))

    fwd_disc = S * np.exp(-q * T)
    k_disc = K * np.exp(-r * T)
    is_call = np.full(price.shape, option == "call")

    # 1. ITM -> OTM via parity  C - P = S e^{-qT} - K e^{-rT}
    parity = fwd_disc - k_disc
    itm = np.where(is_call, parity > 0, parity < 0)
    price = np.where(itm, np.where(is_call, price - parity, price + parity), price)
    is_call = np.where(itm, ~is_call, is_call)

    # 2. no-arbitrage bounds for the OTM option: 0 < price < upper
    upper = np.where(is_call, fwd_disc, k_disc)
    valid = (price > 1e-14 * S) & (price < upper) & (T > 0)

    # 3. Newton–Raphson
    sigma = np.clip(np.sqrt(2 * np.pi / np.maximum(T, 1e-12)) * price / np.maximum(fwd_disc, 1e-12), 0.05, 1.5)
    converged = np.zeros_like(valid)
    failed = np.zeros_like(valid)
    for _ in range(max_iter):
        idx = valid & ~converged & ~failed
        if not idx.any():
            break
        m = np.flatnonzero(idx)
        diff = _otm_price(sigma[m], S[m], K[m], T[m], r[m], q[m], is_call[m]) - price[m]
        vega = bs_greeks(S[m], K[m], T[m], r[m], sigma[m], q[m])["vega"]
        ok = vega > 1e-10 * S[m]
        step = np.where(ok, diff / np.where(ok, vega, 1.0), 0.0)
        new = sigma[m] - step
        left_bracket = ~ok | (new <= _VOL_LO) | (new >= _VOL_HI)
        sigma[m] = np.clip(new, _VOL_LO, _VOL_HI)
        converged[m] = ~left_bracket & ((np.abs(diff) < tol * np.maximum(price[m], 1.0)) | (np.abs(step) < 1e-12))
        failed[m] = left_bracket

    out = np.full_like(price, np.nan)
    good = valid & converged
    out[good] = sigma[good]

    # 4. vectorised bisection fallback
    rest = valid & ~converged
    if rest.any():
        out[rest] = _bisect(price[rest], S[rest], K[rest], T[rest], r[rest], q[rest], is_call[rest])
    return out.reshape(shape) if shape else out[0]
