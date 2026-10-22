"""Reproduce every figure and the benchmark table in the README.

    python scripts/make_figures.py          # writes docs/figures/*.png and docs/benchmarks.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optionpricer import (  # noqa: E402
    HestonParams,
    bs_greeks,
    bs_price,
    calibrate_heston,
    cn_price,
    crr_exercise_boundary,
    crr_price,
    geometric_asian_price,
    heston_mc,
    heston_price,
    implied_vol,
    mc_asian_arithmetic,
    mc_european,
)

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Categorical palette in fixed order (validated for CVD separation)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 130,
})

S, K, T, R, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.02


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. Greeks
def fig_greeks():
    spot = np.linspace(60, 140, 300)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for i, tau in enumerate([0.1, 0.5, 1.0]):
        g = bs_greeks(spot, K, tau, R, SIG, Q)
        for ax, key in zip(axes, ["delta", "gamma", "vega"]):
            ax.plot(spot, g[key], color=C[i], label=f"T = {tau}y")
    for ax, key in zip(axes, ["Delta  ∂V/∂S", "Gamma  ∂²V/∂S²", "Vega  ∂V/∂σ"]):
        ax.set_title(key, loc="left")
        ax.set_xlabel("Spot")
        ax.axvline(K, color=INK2, lw=0.8, ls=":")
    axes[0].legend(loc="upper left")
    fig.suptitle("Black–Scholes call Greeks (K = 100, σ = 20%)", x=0.01, ha="left", color=INK, weight="bold")
    save(fig, "greeks.png")


# ---------------------------------------------------------------- 2. Convergence of lattice / PDE methods
def fig_convergence():
    exact = bs_price(S, K, T, R, SIG, Q, "put")
    steps = np.unique(np.logspace(1, 3.3, 40).astype(int))
    crr_err = [abs(crr_price(S, K, T, R, SIG, Q, "put", steps=n) - exact) for n in steps]
    grid = np.array([25, 50, 100, 200, 400, 800])
    cn_err = [abs(cn_price(S, K, T, R, SIG, Q, "put", n_space=n, n_time=n) - exact) for n in grid]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(steps, crr_err, color=C[0], marker="o", ms=3, lw=1.2, label="CRR binomial tree")
    ax.loglog(grid, cn_err, color=C[1], marker="s", ms=6, label="Crank–Nicolson + Rannacher")
    ref = crr_err[0] * steps[0] / steps
    ax.loglog(steps, ref, color=INK2, ls="--", lw=1, label="O(1/N) reference")
    ref2 = cn_err[0] * (grid[0] / grid) ** 2
    ax.loglog(grid, ref2, color=INK2, ls=":", lw=1, label="O(1/N²) reference")
    ax.set_xlabel("Time steps / grid points N")
    ax.set_ylabel("|price − Black–Scholes|")
    ax.set_title("Convergence to the closed form (European put)", loc="left")
    ax.legend()
    save(fig, "convergence.png")
    return dict(crr_1000=abs(crr_price(S, K, T, R, SIG, Q, "put", steps=1000) - exact),
                cn_400=abs(cn_price(S, K, T, R, SIG, Q, "put") - exact))


# ---------------------------------------------------------------- 3. Variance reduction
def fig_variance_reduction():
    ns = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000])
    configs = [("Plain Monte Carlo", False, False), ("Antithetic", True, False),
               ("Antithetic + control variate", True, True)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    table = {}
    for i, (name, anti, cv) in enumerate(configs):
        se = [mc_european(S, K, T, R, SIG, Q, n_paths=n, antithetic=anti, control_variate=cv, seed=42).std_error
              for n in ns]
        axes[0].loglog(ns, se, color=C[i], marker="o", ms=5, label=name)
        table[f"European call — {name}"] = se[-2]
    se_plain_asian = [mc_asian_arithmetic(S, K, T, R, SIG, Q, n_paths=n, antithetic=False,
                                          control_variate=False, seed=42).std_error for n in ns]
    se_cv_asian = [mc_asian_arithmetic(S, K, T, R, SIG, Q, n_paths=n, seed=42).std_error for n in ns]
    axes[1].loglog(ns, se_plain_asian, color=C[0], marker="o", ms=5, label="Plain Monte Carlo")
    axes[1].loglog(ns, se_cv_asian, color=C[2], marker="o", ms=5, label="Antithetic + geometric-Asian CV")
    table["Asian call — Plain Monte Carlo"] = se_plain_asian[-2]
    table["Asian call — Antithetic + geometric-Asian CV"] = se_cv_asian[-2]
    for ax, t in zip(axes, ["European call", "Arithmetic Asian call (12 fixings)"]):
        ax.set_title(f"{t}: standard error vs paths", loc="left")
        ax.set_xlabel("Number of paths")
        ax.set_ylabel("Standard error")
        ax.legend()
    save(fig, "variance_reduction.png")
    return table


# ---------------------------------------------------------------- 4. Heston smiles
BASE = HestonParams(kappa=2.0, theta=0.04, xi=0.6, rho=-0.7, v0=0.04)


def fig_heston_smiles():
    strikes = np.linspace(70, 130, 61)
    tau = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, rho in enumerate([-0.8, -0.4, 0.0, 0.4]):
        p = HestonParams(BASE.kappa, BASE.theta, BASE.xi, rho, BASE.v0)
        iv = implied_vol(heston_price(S, strikes, tau, R, p, Q), S, strikes, tau, R, Q)
        axes[0].plot(strikes, iv * 100, color=C[i], label=f"ρ = {rho:+.1f}")
    for i, xi in enumerate([0.2, 0.5, 0.8, 1.1]):
        p = HestonParams(BASE.kappa, BASE.theta, xi, 0.0, BASE.v0)
        iv = implied_vol(heston_price(S, strikes, tau, R, p, Q), S, strikes, tau, R, Q)
        axes[1].plot(strikes, iv * 100, color=C[i], label=f"ξ = {xi:.1f}")
    axes[0].set_title("Correlation ρ controls the skew", loc="left")
    axes[1].set_title("Vol-of-vol ξ controls the curvature (ρ = 0)", loc="left")
    for ax in axes:
        ax.set_xlabel("Strike")
        ax.set_ylabel("Implied volatility (%)")
        ax.legend()
    save(fig, "heston_smiles.png")


# ---------------------------------------------------------------- 5. Calibration
def fig_calibration():
    true = HestonParams(kappa=1.8, theta=0.045, xi=0.75, rho=-0.75, v0=0.03)
    rng = np.random.default_rng(2026)
    mats = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0])
    Ts, Ks, ivs = [], [], []
    for tau in mats:
        width = 0.12 + 0.18 * np.sqrt(tau)
        k = S * np.exp(np.linspace(-1.6 * width, 0.8 * width, 13))
        iv = implied_vol(heston_price(S, k, tau, R, true, Q), S, k, tau, R, Q)
        iv = iv + rng.normal(0, 0.0025, iv.shape)      # bid/ask-like noise: 25 bp
        Ts.append(np.full_like(k, tau)), Ks.append(k), ivs.append(iv)
    Ts, Ks, ivs = map(np.concatenate, (Ts, Ks, ivs))

    t0 = time.perf_counter()
    fit, info = calibrate_heston(S, R, Ks, Ts, ivs, Q)
    elapsed = time.perf_counter() - t0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, tau in enumerate(mats):
        m = Ts == tau
        kk = np.linspace(Ks[m].min(), Ks[m].max(), 100)
        model = implied_vol(heston_price(S, kk, tau, R, fit, Q), S, kk, tau, R, Q)
        ax.plot(kk / S, model * 100, color=C[i], lw=1.8)
        ax.scatter(Ks[m] / S, ivs[m] * 100, color=C[i], s=22, edgecolor=SURF, linewidth=1.5, zorder=3,
                   label=f"T = {tau:.2f}y")
    ax.set_xlabel("Moneyness K / S")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Heston calibration: quotes (dots) vs fitted model (lines)", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, "calibration.png")
    return true, fit, info, elapsed


# ---------------------------------------------------------------- 6. American put exercise boundary
def fig_exercise_boundary():
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, vol in enumerate([0.15, 0.25, 0.40]):
        t, b = crr_exercise_boundary(S, K, T, R, vol, 0.0, steps=3000)
        ok = ~np.isnan(b)
        t, b = t[ok], b[ok]
        # lattice nodes are discrete, so the raw boundary is a staircase that alternates
        # between odd/even node levels; a short moving average shows the continuous curve
        w = 41
        b_s = np.convolve(b, np.ones(w) / w, mode="valid")
        t_s = t[w // 2: len(t) - w // 2]
        ax.plot(t_s, b_s, color=C[i], label=f"σ = {vol:.0%}")
    ax.axhline(K, color=INK2, lw=0.8, ls=":")
    ax.text(0.02, K + 0.6, "strike", color=INK2, fontsize=8)
    ax.set_xlabel("Calendar time t (years)")
    ax.set_ylabel("Critical stock price S*(t)")
    ax.set_title("American put: optimal early-exercise boundary", loc="left")
    ax.legend(loc="lower right")
    save(fig, "exercise_boundary.png")


# ---------------------------------------------------------------- benchmark table
def timing(fn, n=3):
    best = np.inf
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    fig_greeks()
    conv = fig_convergence()
    vr = fig_variance_reduction()
    fig_heston_smiles()
    true, fit, info, cal_time = fig_calibration()
    fig_exercise_boundary()

    bs = bs_price(S, K, T, R, SIG, Q)
    am_crr = crr_price(S, K, T, R, SIG, Q, "put", True, 2000)
    am_cn = cn_price(S, K, T, R, SIG, Q, "put", True)
    hp = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
    h_an = heston_price(S, K, T, R, hp, Q)
    h_mc = heston_mc(S, K, T, R, hp, Q, n_paths=200_000, n_steps=400, seed=1)
    asian = mc_asian_arithmetic(S, K, T, R, SIG, Q, n_paths=200_000, seed=1)

    lines = ["# Benchmarks", "",
             f"Generated by `scripts/make_figures.py`. Parameters: S = K = {S:g}, T = {T:g}y, r = {R:.0%}, "
             f"q = {Q:.0%}, σ = {SIG:.0%}.", "",
             "## Accuracy", "",
             "| Instrument | Method | Price | Benchmark | Abs. error |", "|---|---|---:|---:|---:|",
             f"| European put | CRR, 1000 steps | {bs_price(S,K,T,R,SIG,Q,'put')+conv['crr_1000']:.5f} | "
             f"{bs_price(S,K,T,R,SIG,Q,'put'):.5f} (BS) | {conv['crr_1000']:.1e} |",
             f"| European put | Crank–Nicolson 400×400 | {cn_price(S,K,T,R,SIG,Q,'put'):.5f} | "
             f"{bs_price(S,K,T,R,SIG,Q,'put'):.5f} (BS) | {conv['cn_400']:.1e} |",
             f"| European call | MC 100k, antithetic + CV | "
             f"{mc_european(S,K,T,R,SIG,Q,seed=1).price:.5f} | {bs:.5f} (BS) | "
             f"{abs(mc_european(S,K,T,R,SIG,Q,seed=1).price-bs):.1e} |",
             f"| American put | CRR 2000 vs CN 400×400 (Brennan–Schwartz) | {am_crr:.5f} | {am_cn:.5f} | "
             f"{abs(am_crr-am_cn):.1e} |",
             f"| Heston call | Gil-Pelaez (256-pt Gauss–Legendre) | {h_an:.6f} | 8.752596 (QuantLib) | "
             f"{abs(h_an-8.752596280):.1e} |",
             f"| Heston call | MC full-truncation Euler, 200k × 400 | {h_mc.price:.4f} ± {1.96*h_mc.std_error:.4f} | "
             f"{h_an:.4f} | {abs(h_mc.price-h_an):.1e} |",
             f"| Arithmetic Asian call | MC 200k + geometric CV | {asian.price:.5f} ± {1.96*asian.std_error:.5f} | "
             f"{geometric_asian_price(S,K,T,R,SIG,Q):.5f} (geometric, lower bound) | — |",
             "", "## Variance reduction (standard error at 100,000 paths)", "",
             "| Estimator | Std. error | Variance reduction factor |", "|---|---:|---:|"]
    base_eu = vr["European call — Plain Monte Carlo"]
    base_as = vr["Asian call — Plain Monte Carlo"]
    for k, v in vr.items():
        base = base_eu if k.startswith("European") else base_as
        lines.append(f"| {k} | {v:.5f} | {(base / v) ** 2:,.0f}× |")

    lines += ["", "## Heston calibration (synthetic surface, 65 quotes, 25 bp noise)", "",
              "| Parameter | True | Calibrated |", "|---|---:|---:|"]
    for name in ("kappa", "theta", "xi", "rho", "v0"):
        lines.append(f"| {name} | {getattr(true, name):.4f} | {getattr(fit, name):.4f} |")
    lines += ["", f"IV RMSE: **{info['rmse_iv_bp']:.1f} bp** (noise level 25 bp) · "
              f"calibration time: **{cal_time:.2f} s** · function evaluations: {info['nfev']}", "",
              "## Speed (best of 3, single core)", "", "| Task | Time |", "|---|---:|"]
    strikes = np.linspace(50, 150, 1000)
    lines += [
        f"| 1,000,000 Black–Scholes prices (vectorised) | {timing(lambda: bs_price(S, np.linspace(50,150,1_000_000), T, R, SIG, Q)):.0f} ms |",
        f"| 1,000 Heston prices, one maturity | {timing(lambda: heston_price(S, strikes, T, R, hp, Q)):.1f} ms |",
        f"| American put, CRR 1000 steps | {timing(lambda: crr_price(S, K, T, R, SIG, Q, 'put', True, 1000)):.0f} ms |",
        f"| American put, Crank–Nicolson 400×400 | {timing(lambda: cn_price(S, K, T, R, SIG, Q, 'put', True)):.0f} ms |",
        f"| 1,000 implied vols (Newton, vectorised) | "
        f"{timing(lambda: implied_vol(bs_price(S, strikes, T, R, SIG, Q), S, strikes, T, R, Q)):.1f} ms |",
    ]
    (ROOT / "docs" / "benchmarks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
