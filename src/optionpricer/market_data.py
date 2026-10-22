"""Optional: download a live option chain and build a cleaned implied-vol surface.

Requires the optional dependency ``yfinance`` (``pip install .[data]``).
Data from Yahoo Finance is delayed and occasionally stale; the cleaning
filters below (minimum volume/open interest, positive bid, bid–ask width,
OTM only, moneyness window) remove most bad quotes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .implied_vol import implied_vol

__all__ = ["fetch_iv_surface"]


def fetch_iv_surface(ticker: str = "SPY", r: float = 0.04, q: float = 0.013,
                     min_days: int = 14, max_days: int = 400, moneyness: tuple[float, float] = (0.7, 1.3),
                     max_spread: float = 0.25, min_open_interest: int = 100) -> tuple[float, pd.DataFrame]:
    """Return ``(spot, surface)`` where ``surface`` has columns
    ``expiry, T, strike, type, mid, iv``. Out-of-the-money options only."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install the optional dependency: pip install yfinance") from exc

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    now = datetime.now(timezone.utc)
    rows = []
    for exp in tk.options:
        T = (pd.Timestamp(exp, tz="UTC") + pd.Timedelta(hours=20) - now).total_seconds() / (365.25 * 86400)
        if not (min_days / 365 <= T <= max_days / 365):
            continue
        chain = tk.option_chain(exp)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            df = df[(df["bid"] > 0) & (df["ask"] > df["bid"]) & (df["openInterest"] >= min_open_interest)].copy()
            df["mid"] = 0.5 * (df["bid"] + df["ask"])
            df = df[(df["ask"] - df["bid"]) / df["mid"] <= max_spread]
            fwd = spot * np.exp((r - q) * T)
            df = df[df["strike"] >= fwd] if kind == "call" else df[df["strike"] < fwd]
            df = df[(df["strike"] / spot).between(*moneyness)]
            for _, row in df.iterrows():
                rows.append({"expiry": exp, "T": T, "strike": row["strike"], "type": kind, "mid": row["mid"]})
    surf = pd.DataFrame(rows)
    if surf.empty:
        return spot, surf
    surf["iv"] = [float(implied_vol(m, spot, k, t, r, q, ty))
                  for m, k, t, ty in zip(surf["mid"], surf["strike"], surf["T"], surf["type"])]
    return spot, surf.dropna(subset=["iv"]).reset_index(drop=True)
