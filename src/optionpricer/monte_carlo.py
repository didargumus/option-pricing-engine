r"""Monte Carlo pricing under geometric Brownian motion with variance reduction.

Implemented techniques
----------------------
* **Antithetic variates** – each normal draw :math:`Z` is paired with
  :math:`-Z`.
* **Control variates** – for vanilla options the discounted terminal stock
  price (known mean :math:`S_0e^{-qT}`); for arithmetic Asian options the
  geometric Asian option, whose price is known in closed form
  (Kemna & Vorst 1990). The optimal coefficient :math:`\beta^*=
  \mathrm{Cov}(Y,X)/\mathrm{Var}(X)` is estimated from the sample.
* **Barrier options** with discrete monitoring, compared against the
  continuous-monitoring formula with the Broadie–Glasserman–Kou (1997)
  continuity correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black_scholes import bs_price

__all__ = [
    "MCResult", "simulate_gbm", "mc_european", "mc_asian_arithmetic",
    "geometric_asian_price", "mc_down_and_out_call", "down_and_out_call_price",
    "bgk_shifted_barrier",
]


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate with its standard error and 95% confidence interval."""
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lo, hi = self.ci95
        return f"MCResult(price={self.price:.5f}, se={self.std_error:.5f}, 95% CI=[{lo:.5f}, {hi:.5f}])"


def _normals(rng: np.random.Generator, n_paths: int, n_steps: int, antithetic: bool) -> np.ndarray:
    if antithetic:
        half = rng.standard_normal(((n_paths + 1) // 2, n_steps))
        return np.vstack([half, -half])[:n_paths]
    return rng.standard_normal((n_paths, n_steps))


def simulate_gbm(S0: float, T: float, r: float, sigma: float, q: float = 0.0,
                 n_steps: int = 252, n_paths: int = 10_000, antithetic: bool = False,
                 seed: int | None = None) -> np.ndarray:
    """Simulate GBM paths exactly on a uniform grid. Returns array (n_paths, n_steps+1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = _normals(rng, n_paths, n_steps, antithetic)
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def _pair_means(x: np.ndarray, antithetic: bool) -> np.ndarray:
    """With antithetic sampling the i.i.d. units are the *pairs* of paths."""
    if antithetic and len(x) % 2 == 0:
        h = len(x) // 2
        return 0.5 * (x[:h] + x[h:])
    return x


def _control_variate(y: np.ndarray, x: np.ndarray, ex: float) -> np.ndarray:
    cov = np.cov(y, x)
    beta = cov[0, 1] / cov[1, 1]
    return y - beta * (x - ex)


def _summarise(samples: np.ndarray, n_paths: int) -> MCResult:
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples))), n_paths)


def mc_european(S0: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
                option: str = "call", n_paths: int = 100_000, antithetic: bool = True,
                control_variate: bool = True, seed: int | None = None) -> MCResult:
    """European option by exact terminal sampling of GBM."""
    rng = np.random.default_rng(seed)
    Z = _normals(rng, n_paths, 1, antithetic)[:, 0]
    ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    disc = np.exp(-r * T)
    payoff = np.maximum(ST - K, 0) if option == "call" else np.maximum(K - ST, 0)
    y = _pair_means(disc * payoff, antithetic)
    if control_variate:
        x = _pair_means(disc * ST, antithetic)
        y = _control_variate(y, x, S0 * np.exp(-q * T))
    return _summarise(y, n_paths)


def geometric_asian_price(S0: float, K: float, T: float, r: float, sigma: float,
                          q: float = 0.0, n_fixings: int = 12, option: str = "call") -> float:
    r"""Closed-form price of a discretely monitored geometric-average Asian option.

    With fixings at :math:`t_i = iT/n`, :math:`\ln G` is normal with

    .. math::
        \mu = \ln S_0 + (r-q-\tfrac12\sigma^2)\,T\frac{n+1}{2n},\qquad
        v = \sigma^2 T\frac{(n+1)(2n+1)}{6n^2}.
    """
    n = n_fixings
    mu = np.log(S0) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d2 = (mu - np.log(K)) / np.sqrt(v)
    d1 = d2 + np.sqrt(v)
    fwd = np.exp(mu + 0.5 * v)
    if option == "call":
        return float(np.exp(-r * T) * (fwd * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(np.exp(-r * T) * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1)))


def mc_asian_arithmetic(S0: float, K: float, T: float, r: float, sigma: float,
                        q: float = 0.0, n_fixings: int = 12, option: str = "call",
                        n_paths: int = 100_000, antithetic: bool = True,
                        control_variate: bool = True, seed: int | None = None) -> MCResult:
    """Arithmetic-average Asian option; geometric Asian used as control variate."""
    paths = simulate_gbm(S0, T, r, sigma, q, n_fixings, n_paths, antithetic, seed)[:, 1:]
    disc = np.exp(-r * T)
    A = paths.mean(axis=1)
    G = np.exp(np.log(paths).mean(axis=1))
    sign = 1.0 if option == "call" else -1.0
    y = _pair_means(disc * np.maximum(sign * (A - K), 0), antithetic)
    if control_variate:
        x = _pair_means(disc * np.maximum(sign * (G - K), 0), antithetic)
        y = _control_variate(y, x, geometric_asian_price(S0, K, T, r, sigma, q, n_fixings, option))
    return _summarise(y, n_paths)


def down_and_out_call_price(S0: float, K: float, H: float, T: float, r: float,
                            sigma: float, q: float = 0.0) -> float:
    r"""Continuously monitored down-and-out call, :math:`H \le K` (Merton 1973;
    Reiner & Rubinstein 1991): :math:`C_{do} = C_{BS} - C_{di}`."""
    if S0 <= H:
        return 0.0
    if H > K:
        raise NotImplementedError("formula implemented for H <= K")
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    vst = sigma * np.sqrt(T)
    y = np.log(H**2 / (S0 * K)) / vst + lam * vst
    c_di = (S0 * np.exp(-q * T) * (H / S0) ** (2 * lam) * norm.cdf(y)
            - K * np.exp(-r * T) * (H / S0) ** (2 * lam - 2) * norm.cdf(y - vst))
    return float(bs_price(S0, K, T, r, sigma, q, "call") - c_di)


def mc_down_and_out_call(S0: float, K: float, H: float, T: float, r: float, sigma: float,
                         q: float = 0.0, n_monitor: int = 252, n_paths: int = 100_000,
                         antithetic: bool = True, seed: int | None = None) -> MCResult:
    """Discretely monitored down-and-out call (barrier checked on ``n_monitor`` dates)."""
    paths = simulate_gbm(S0, T, r, sigma, q, n_monitor, n_paths, antithetic, seed)
    alive = paths[:, 1:].min(axis=1) > H
    y = _pair_means(np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0) * alive, antithetic)
    return _summarise(y, n_paths)


def bgk_shifted_barrier(H: float, sigma: float, T: float, n_monitor: int, down: bool = True) -> float:
    """Broadie–Glasserman–Kou continuity correction: discrete barrier H is
    equivalent to a continuous barrier ``H*exp(∓0.5826 σ sqrt(Δt))``."""
    beta = 0.5826  # -zeta(1/2)/sqrt(2*pi)
    shift = np.exp(beta * sigma * np.sqrt(T / n_monitor))
    return H / shift if down else H * shift
