"""Five-minute tour of the library.  Run:  python examples/quickstart.py"""

import numpy as np

from optionpricer import (
    HestonParams,
    bs_greeks,
    bs_price,
    calibrate_heston,
    cn_price,
    crr_price,
    heston_price,
    implied_vol,
    mc_asian_arithmetic,
    mc_european,
)

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02

print("1) Black–Scholes")
print(f"   call = {bs_price(S, K, T, r, sigma, q):.4f}   put = {bs_price(S, K, T, r, sigma, q, 'put'):.4f}")
g = bs_greeks(S, K, T, r, sigma, q)
print("   Greeks:", {k: round(float(v), 4) for k, v in g.items()})

print("\n2) American put — two independent numerical methods")
print(f"   CRR tree (2000 steps)      : {crr_price(S, K, T, r, sigma, q, 'put', american=True, steps=2000):.4f}")
print(f"   Crank–Nicolson PDE (400²)  : {cn_price(S, K, T, r, sigma, q, 'put', american=True):.4f}")
print(f"   European put (lower bound) : {bs_price(S, K, T, r, sigma, q, 'put'):.4f}")

print("\n3) Monte Carlo with variance reduction")
print("   European call:", mc_european(S, K, T, r, sigma, q, n_paths=100_000, seed=0))
print("   Asian call   :", mc_asian_arithmetic(S, K, T, r, sigma, q, n_fixings=12, seed=0))

print("\n4) Heston stochastic volatility")
p = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.04)
strikes = np.array([80.0, 90, 100, 110, 120])
iv = implied_vol(heston_price(S, strikes, T, r, p, q), S, strikes, T, r, q)
for k, v in zip(strikes, iv):
    print(f"   K = {k:5.0f}   implied vol = {v:.2%}")

print("\n5) Calibrate Heston back from that smile")
Ts = np.repeat([0.5, 1.0], 5)
Ks = np.tile(strikes, 2)
ivs = np.concatenate([implied_vol(heston_price(S, strikes, t, r, p, q), S, strikes, t, r, q) for t in (0.5, 1.0)])
fit, info = calibrate_heston(S, r, Ks, Ts, ivs, q)
print(f"   fitted: {fit}")
print(f"   IV RMSE: {info['rmse_iv_bp']:.3f} bp")
