r"""Heston (1993) stochastic-volatility model.

.. math::
    dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW^S_t, \qquad
    dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^v_t, \qquad
    d\langle W^S, W^v\rangle_t = \rho\,dt

Contents
--------
* :func:`heston_cf` – characteristic function of :math:`\ln S_T` in the
  numerically stable "little Heston trap" form (Albrecher et al. 2007), which
  avoids the branch-cut discontinuity of the original formula.
* :func:`heston_price` – semi-analytic European prices via Gil-Pelaez
  inversion with Gauss–Legendre quadrature, vectorised over strikes.
* :func:`heston_mc` – full-truncation Euler Monte Carlo (Lord et al. 2010),
  used to validate the semi-analytic formula.
* :func:`calibrate_heston` – vega-weighted least-squares calibration to an
  implied-volatility surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from .black_scholes import bs_greeks
from .implied_vol import implied_vol
from .monte_carlo import MCResult

__all__ = ["HestonParams", "heston_cf", "heston_price", "heston_mc", "calibrate_heston"]


@dataclass(frozen=True)
class HestonParams:
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # spot/vol correlation
    v0: float      # initial variance

    @property
    def feller(self) -> float:
        """Feller ratio 2κθ/ξ². ≥ 1 means the variance stays strictly positive."""
        return 2 * self.kappa * self.theta / self.xi**2

    def as_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.xi, self.rho, self.v0])

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return ("HestonParams(" + ", ".join(f"{k}={v:.4f}" for k, v in self.to_dict().items())
                + f", feller={self.feller:.2f})")


def heston_cf(u: np.ndarray, S: float, T: float, r: float, p: HestonParams, q: float = 0.0) -> np.ndarray:
    """E[exp(i u ln S_T)] under the risk-neutral measure (little-trap form)."""
    u = np.asarray(u, dtype=complex)
    k, th, xi, rho, v0 = p.kappa, p.theta, p.xi, p.rho, p.v0
    iu = 1j * u
    beta = k - rho * xi * iu
    d = np.sqrt(beta**2 + xi**2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    C = (k * th / xi**2) * ((beta - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
    D = ((beta - d) / xi**2) * (1 - e) / (1 - g * e)
    return np.exp(iu * (np.log(S) + (r - q) * T) + C + D * v0)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(256)


def heston_price(S: float, K, T: float, r: float, p: HestonParams, q: float = 0.0,
                 option: str = "call", u_max: float | None = None) -> np.ndarray:
    r"""Semi-analytic European price (vectorised over ``K``).

    .. math::
        C = S e^{-qT} P_1 - K e^{-rT} P_2,\quad
        P_j = \tfrac12 + \tfrac1\pi\int_0^\infty
        \operatorname{Re}\!\left[\frac{e^{-iu\ln K}\,f_j(u)}{iu}\right]du

    with :math:`f_2(u)=\varphi(u)` and :math:`f_1(u)=\varphi(u-i)/\varphi(-i)`.
    """
    K = np.atleast_1d(np.asarray(K, float))
    if u_max is None:
        u_max = max(100.0, 60.0 / np.sqrt(max(T, 1e-4)) / max(np.sqrt(p.v0), 0.05))
    u = 0.5 * u_max * (_GL_X + 1.0)          # map [-1,1] -> [0,u_max]
    w = 0.5 * u_max * _GL_W

    phi = heston_cf(u, S, T, r, p, q)
    phi_shift = heston_cf(u - 1j, S, T, r, p, q)
    fwd = S * np.exp((r - q) * T)            # = phi(-i)
    lnK = np.log(K)[:, None]
    kern = np.exp(-1j * u[None, :] * lnK) / (1j * u[None, :])
    P1 = 0.5 + (w * np.real(kern * phi_shift / fwd)).sum(axis=1) / np.pi
    P2 = 0.5 + (w * np.real(kern * phi)).sum(axis=1) / np.pi

    call = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    if option == "call":
        out = call
    else:
        out = call - S * np.exp(-q * T) + K * np.exp(-r * T)   # put-call parity
    return out if out.size > 1 else out[0]


def heston_mc(S: float, K: float, T: float, r: float, p: HestonParams, q: float = 0.0,
              option: str = "call", n_paths: int = 100_000, n_steps: int = 200,
              seed: int | None = None) -> MCResult:
    """Full-truncation Euler scheme with exact log-Euler for the stock."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    x = np.full(n_paths, np.log(S))
    v = np.full(n_paths, p.v0)
    sq = np.sqrt(1 - p.rho**2)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + sq * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        x += (r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
    ST = np.exp(x)
    pay = np.maximum(ST - K, 0) if option == "call" else np.maximum(K - ST, 0)
    y = np.exp(-r * T) * pay
    return MCResult(float(y.mean()), float(y.std(ddof=1) / np.sqrt(n_paths)), n_paths)


_BOUNDS = (np.array([0.05, 1e-3, 0.05, -0.99, 1e-3]), np.array([15.0, 1.0, 3.0, 0.99, 1.0]))


def calibrate_heston(S: float, r: float, strikes: np.ndarray, maturities: np.ndarray,
                     market_iv: np.ndarray, q: float = 0.0,
                     x0: HestonParams | None = None) -> tuple[HestonParams, dict]:
    """Calibrate to a surface of implied vols.

    Prices are fitted using residuals ``(model - market)/vega``, a first-order
    approximation of implied-vol error that avoids inverting Black–Scholes at
    every iteration. OTM options are used (puts below the forward, calls
    above), as is standard because they are the most liquid.

    ``strikes``, ``maturities`` and ``market_iv`` are 1-D arrays of equal length.
    """
    strikes, maturities, market_iv = map(np.asarray, (strikes, maturities, market_iv))
    fwd = S * np.exp((r - q) * maturities)
    is_call = strikes >= fwd
    from .black_scholes import bs_price
    mkt = np.where(is_call, bs_price(S, strikes, maturities, r, market_iv, q, "call"),
                   bs_price(S, strikes, maturities, r, market_iv, q, "put"))
    vega = bs_greeks(S, strikes, maturities, r, market_iv, q)["vega"]
    vega = np.maximum(vega, 1e-3 * S)

    uniq_T = np.unique(maturities)

    def model_prices(theta_vec):
        hp = HestonParams(*theta_vec)
        out = np.empty_like(strikes, dtype=float)
        for T in uniq_T:
            m = maturities == T
            c = heston_price(S, strikes[m], T, r, hp, q, "call")
            c = np.atleast_1d(c)
            put = c - S * np.exp(-q * T) + strikes[m] * np.exp(-r * T)
            out[m] = np.where(is_call[m], c, put)
        return out

    def resid(theta_vec):
        return (model_prices(theta_vec) - mkt) / vega

    start = (x0 or HestonParams(2.0, float(np.mean(market_iv) ** 2), 0.5, -0.5,
                                float(np.mean(market_iv) ** 2))).as_array()
    res = least_squares(resid, start, bounds=_BOUNDS, method="trf", x_scale="jac",
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
    fitted = HestonParams(*map(float, res.x))

    model_px = model_prices(res.x)
    model_iv = np.where(is_call, implied_vol(model_px, S, strikes, maturities, r, q, "call"),
                        implied_vol(model_px, S, strikes, maturities, r, q, "put"))
    err = model_iv - market_iv
    info = {"rmse_iv_bp": float(np.sqrt(np.nanmean(err**2)) * 1e4),
            "max_abs_iv_bp": float(np.nanmax(np.abs(err)) * 1e4),
            "n_options": int(len(strikes)), "nfev": int(res.nfev),
            "model_iv": model_iv, "success": bool(res.success)}
    return fitted, info
