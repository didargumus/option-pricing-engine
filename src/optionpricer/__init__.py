"""optionpricer – a from-scratch derivatives pricing library.

Models: Black–Scholes–Merton, Cox–Ross–Rubinstein trees, Crank–Nicolson PDE,
Monte Carlo with variance reduction, and the Heston stochastic-volatility
model with calibration.
"""

from .binomial import crr_exercise_boundary, crr_price
from .black_scholes import bs_greeks, bs_price, d1_d2
from .finite_difference import cn_price
from .heston import HestonParams, calibrate_heston, heston_cf, heston_mc, heston_price
from .implied_vol import implied_vol
from .monte_carlo import (
    MCResult,
    bgk_shifted_barrier,
    down_and_out_call_price,
    geometric_asian_price,
    mc_asian_arithmetic,
    mc_down_and_out_call,
    mc_european,
    simulate_gbm,
)

__version__ = "1.0.0"

__all__ = [
    "bs_price", "bs_greeks", "d1_d2",
    "crr_price", "crr_exercise_boundary",
    "cn_price",
    "MCResult", "simulate_gbm", "mc_european", "mc_asian_arithmetic", "geometric_asian_price",
    "mc_down_and_out_call", "down_and_out_call_price", "bgk_shifted_barrier",
    "implied_vol",
    "HestonParams", "heston_cf", "heston_price", "heston_mc", "calibrate_heston",
]
