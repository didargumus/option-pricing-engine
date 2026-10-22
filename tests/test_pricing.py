"""Correctness tests: every numerical method is checked against an independent
benchmark (closed form, another method, or QuantLib when installed)."""

import numpy as np
import pytest

from optionpricer import (
    HestonParams,
    bgk_shifted_barrier,
    bs_greeks,
    bs_price,
    calibrate_heston,
    cn_price,
    crr_price,
    down_and_out_call_price,
    geometric_asian_price,
    heston_mc,
    heston_price,
    implied_vol,
    mc_asian_arithmetic,
    mc_down_and_out_call,
    mc_european,
)

S, K, T, R, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.02


# --------------------------------------------------------------------------- Black–Scholes
def test_bs_reference_value():
    # Hull, Options Futures & Other Derivatives, Example 15.6: S=42, K=40, r=10%, σ=20%, T=0.5
    assert bs_price(42, 40, 0.5, 0.1, 0.2) == pytest.approx(4.76, abs=5e-3)
    assert bs_price(42, 40, 0.5, 0.1, 0.2, option="put") == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("k", [70, 90, 100, 110, 140])
def test_put_call_parity(k):
    c = bs_price(S, k, T, R, SIG, Q, "call")
    p = bs_price(S, k, T, R, SIG, Q, "put")
    assert c - p == pytest.approx(S * np.exp(-Q * T) - k * np.exp(-R * T), abs=1e-12)


@pytest.mark.parametrize("option", ["call", "put"])
def test_greeks_match_finite_differences(option):
    g = bs_greeks(S, 105, T, R, SIG, Q, option)
    h = 1e-4
    f = lambda **kw: bs_price(**{**dict(S=S, K=105, T=T, r=R, sigma=SIG, q=Q, option=option), **kw})  # noqa: E731
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sigma=SIG + h) - f(sigma=SIG - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=R + h) - f(r=R - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-6)


def test_vectorisation():
    strikes = np.linspace(50, 150, 11)
    out = bs_price(S, strikes, T, R, SIG, Q)
    assert out.shape == (11,)
    assert np.all(np.diff(out) < 0)          # call price decreasing in strike


# --------------------------------------------------------------------------- Trees & PDE
@pytest.mark.parametrize("option", ["call", "put"])
def test_crr_converges_to_bs(option):
    exact = bs_price(S, K, T, R, SIG, Q, option)
    assert crr_price(S, K, T, R, SIG, Q, option, steps=2000) == pytest.approx(exact, abs=5e-3)


def test_american_put_premium():
    eu = bs_price(S, K, T, R, SIG, Q, "put")
    am = crr_price(S, K, T, R, SIG, Q, "put", american=True, steps=2000)
    assert am > eu
    assert am == pytest.approx(cn_price(S, K, T, R, SIG, Q, "put", american=True), abs=2e-3)


def test_american_call_no_dividend_equals_european():
    eu = bs_price(S, K, T, R, SIG, 0.0, "call")
    assert crr_price(S, K, T, R, SIG, 0.0, "call", american=True, steps=1000) == pytest.approx(eu, abs=5e-3)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("k", [80, 100, 120])
def test_crank_nicolson_european(option, k):
    assert cn_price(S, k, T, R, SIG, Q, option) == pytest.approx(bs_price(S, k, T, R, SIG, Q, option), abs=2e-3)


# --------------------------------------------------------------------------- Monte Carlo
@pytest.mark.parametrize("anti,cv", [(False, False), (True, False), (True, True)])
def test_mc_european_within_confidence(anti, cv):
    res = mc_european(S, K, T, R, SIG, Q, n_paths=200_000, antithetic=anti, control_variate=cv, seed=7)
    assert abs(res.price - bs_price(S, K, T, R, SIG, Q)) < 4 * res.std_error


def test_variance_reduction_effective():
    plain = mc_european(S, K, T, R, SIG, Q, n_paths=100_000, antithetic=False, control_variate=False, seed=1)
    best = mc_european(S, K, T, R, SIG, Q, n_paths=100_000, seed=1)
    assert best.std_error < plain.std_error / 3


def test_geometric_asian_closed_form_vs_mc():
    from optionpricer import simulate_gbm
    paths = simulate_gbm(S, T, R, SIG, Q, 12, 400_000, antithetic=True, seed=3)[:, 1:]
    G = np.exp(np.log(paths).mean(axis=1))
    y = np.exp(-R * T) * np.maximum(G - K, 0)
    assert y.mean() == pytest.approx(geometric_asian_price(S, K, T, R, SIG, Q, 12), abs=3 * y.std() / np.sqrt(len(y)))


def test_arithmetic_asian_bounds_and_cv():
    geo = geometric_asian_price(S, K, T, R, SIG, Q, 12)
    cv = mc_asian_arithmetic(S, K, T, R, SIG, Q, 12, n_paths=100_000, seed=4)
    plain = mc_asian_arithmetic(S, K, T, R, SIG, Q, 12, n_paths=100_000, control_variate=False, seed=4)
    assert geo < cv.price < bs_price(S, K, T, R, SIG, Q)   # AM-GM and averaging reduce value
    assert abs(cv.price - plain.price) < 4 * plain.std_error
    assert cv.std_error < plain.std_error / 10


def test_barrier_bgk_correction():
    H, n = 90.0, 52
    mc = mc_down_and_out_call(S, K, H, T, R, SIG, Q, n_monitor=n, n_paths=400_000, seed=5)
    corrected = down_and_out_call_price(S, K, bgk_shifted_barrier(H, SIG, T, n), T, R, SIG, Q)
    assert abs(mc.price - corrected) < 4 * mc.std_error + 0.02
    assert down_and_out_call_price(S, K, H, T, R, SIG, Q) < mc.price      # continuous knocks out more


# --------------------------------------------------------------------------- Implied vol
def test_implied_vol_round_trip_grid():
    strikes = np.linspace(40, 200, 33)
    for T_ in (0.05, 0.5, 3.0):
        for vol in (0.08, 0.25, 0.9):
            for opt in ("call", "put"):
                px = bs_price(S, strikes, T_, R, vol, Q, opt)
                iv = implied_vol(px, S, strikes, T_, R, Q, opt)
                fwd, kd = S * np.exp(-Q * T_), strikes * np.exp(-R * T_)
                intrinsic = np.maximum(fwd - kd, 0) if opt == "call" else np.maximum(kd - fwd, 0)
                identifiable = px - intrinsic > 1e-6       # enough time value to pin down σ
                assert not np.isnan(iv[identifiable]).any()
                np.testing.assert_allclose(iv[identifiable], vol, atol=1e-5)


def test_implied_vol_rejects_arbitrage():
    assert np.isnan(implied_vol(0.0, S, K, T, R, Q))
    assert np.isnan(implied_vol(S * 2, S, K, T, R, Q))


# --------------------------------------------------------------------------- Heston
P = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)


def test_heston_reduces_to_bs():
    p = HestonParams(2.0, 0.04, 1e-4, 0.0, 0.04)
    assert heston_price(S, K, T, R, p, Q) == pytest.approx(bs_price(S, K, T, R, 0.2, Q), abs=1e-5)


def test_heston_reference_values():
    # Reference values computed with QuantLib's AnalyticHestonEngine (v1.3x)
    expected = [23.400518486, 8.752596280, 1.267894945]
    np.testing.assert_allclose(heston_price(S, [80, 100, 120], T, R, P, Q), expected, atol=1e-6)


def test_heston_against_quantlib():
    ql = pytest.importorskip("QuantLib")
    d = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = d
    dc = ql.Actual365Fixed()
    proc = ql.HestonProcess(ql.YieldTermStructureHandle(ql.FlatForward(d, R, dc)),
                            ql.YieldTermStructureHandle(ql.FlatForward(d, Q, dc)),
                            ql.QuoteHandle(ql.SimpleQuote(S)), P.v0, P.kappa, P.theta, P.xi, P.rho)
    eng = ql.AnalyticHestonEngine(ql.HestonModel(proc))
    for k in (70, 90, 100, 115, 140):
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, k), ql.EuropeanExercise(d + 365))
        opt.setPricingEngine(eng)
        assert heston_price(S, k, T, R, P, Q) == pytest.approx(opt.NPV(), abs=1e-6)


def test_heston_mc_agrees_with_semi_analytic():
    mc = heston_mc(S, K, T, R, P, Q, n_paths=200_000, n_steps=400, seed=11)
    assert abs(mc.price - heston_price(S, K, T, R, P, Q)) < 4 * mc.std_error + 0.03


def test_heston_put_call_parity():
    c = heston_price(S, 110, 0.5, R, P, Q, "call")
    p = heston_price(S, 110, 0.5, R, P, Q, "put")
    assert c - p == pytest.approx(S * np.exp(-Q * 0.5) - 110 * np.exp(-R * 0.5), abs=1e-10)


def test_negative_rho_produces_skew():
    strikes = np.array([80.0, 100.0, 120.0])
    iv = implied_vol(heston_price(S, strikes, T, R, P, Q), S, strikes, T, R, Q)
    assert iv[0] > iv[1] > iv[2]


def test_calibration_recovers_parameters():
    true = HestonParams(1.5, 0.05, 0.6, -0.65, 0.03)
    Ts = np.repeat([0.25, 0.5, 1.0, 2.0], 9)
    Ks = np.tile(np.linspace(75, 125, 9), 4)
    ivs = np.empty_like(Ks)
    for t in np.unique(Ts):
        m = Ts == t
        ivs[m] = implied_vol(heston_price(S, Ks[m], t, R, true, Q), S, Ks[m], t, R, Q)
    fit, info = calibrate_heston(S, R, Ks, Ts, ivs, Q)
    assert info["rmse_iv_bp"] < 1.0
    np.testing.assert_allclose(fit.as_array(), true.as_array(), rtol=0.05, atol=0.01)
