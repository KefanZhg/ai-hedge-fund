import math
import requests
import pandas as pd
import numpy as np

TODAY = "2026-04-19"
FROM = "2025-01-01"

TICKERS = {
    "SPY": "US Large Cap",
    "QQQ": "US Tech/Growth",
    "IWM": "US Small Cap",
    "DIA": "US Blue Chips",
    "XLF": "Financials",
    "XLK": "Technology Sector",
    "XLE": "Energy Sector",
    "XLU": "Utilities (Defensive)",
    "TLT": "Long Treasuries",
    "GLD": "Gold",
}


def fetch_nasdaq_hist(symbol, assetclass="etf", fromdate=FROM, todate=TODAY, limit=400):
    url = f"https://api.nasdaq.com/api/quote/{symbol}/historical"
    params = {
        "assetclass": assetclass,
        "fromdate": fromdate,
        "todate": todate,
        "limit": limit,
    }
    headers = {"user-agent": "Mozilla/5.0", "accept": "application/json"}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    rows = (((data or {}).get("data") or {}).get("tradesTable") or {}).get("rows") or []
    if not rows:
        return pd.DataFrame()

    output = []
    for row in rows:
        try:
            output.append(
                {
                    "Date": pd.to_datetime(row["date"], format="%m/%d/%Y"),
                    "open": float(str(row["open"]).replace("$", "").replace(",", "")),
                    "high": float(str(row["high"]).replace("$", "").replace(",", "")),
                    "low": float(str(row["low"]).replace("$", "").replace(",", "")),
                    "close": float(str(row["close"]).replace("$", "").replace(",", "")),
                    "volume": float(str(row["volume"]).replace(",", "")),
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(output)
    if df.empty:
        return df
    return df.sort_values("Date").set_index("Date")


# Same formulas as src/agents/technicals.py

def calculate_ema(df, window):
    return df["close"].ewm(span=window, adjust=False).mean()


def calculate_adx(df, period=14):
    d = df.copy()
    d["high_low"] = d["high"] - d["low"]
    d["high_close"] = (d["high"] - d["close"].shift()).abs()
    d["low_close"] = (d["low"] - d["close"].shift()).abs()
    d["tr"] = d[["high_low", "high_close", "low_close"]].max(axis=1)
    d["up_move"] = d["high"] - d["high"].shift()
    d["down_move"] = d["low"].shift() - d["low"]
    d["plus_dm"] = np.where((d["up_move"] > d["down_move"]) & (d["up_move"] > 0), d["up_move"], 0)
    d["minus_dm"] = np.where((d["down_move"] > d["up_move"]) & (d["down_move"] > 0), d["down_move"], 0)
    d["+di"] = 100 * (pd.Series(d["plus_dm"], index=d.index).ewm(span=period).mean() / d["tr"].ewm(span=period).mean())
    d["-di"] = 100 * (pd.Series(d["minus_dm"], index=d.index).ewm(span=period).mean() / d["tr"].ewm(span=period).mean())
    d["dx"] = 100 * (d["+di"] - d["-di"]).abs() / (d["+di"] + d["-di"])
    d["adx"] = d["dx"].ewm(span=period).mean()
    return d[["adx", "+di", "-di"]]


def trend_signal(df):
    ema_8 = calculate_ema(df, 8)
    ema_21 = calculate_ema(df, 21)
    ema_55 = calculate_ema(df, 55)
    adx = calculate_adx(df, 14)

    short_trend = ema_8.iloc[-1] > ema_21.iloc[-1]
    medium_trend = ema_21.iloc[-1] > ema_55.iloc[-1]
    adx_last = adx["adx"].iloc[-1]
    strength = float(adx_last) / 100 if pd.notna(adx_last) else 0.5

    if short_trend and medium_trend:
        sig = "bullish"
    elif (not short_trend) and (not medium_trend):
        sig = "bearish"
    else:
        sig = "neutral"

    return sig, max(0.0, min(1.0, strength)), float(adx_last) if pd.notna(adx_last) else np.nan


def momentum_signal(df):
    returns = df["close"].pct_change()
    mom_1m = returns.rolling(21).sum().iloc[-1]
    mom_3m = returns.rolling(63).sum().iloc[-1]
    mom_6m = returns.rolling(126).sum().iloc[-1]
    volume_ma = df["volume"].rolling(21).mean()
    volume_momentum = (df["volume"] / volume_ma).iloc[-1]
    score = 0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m

    if score > 0.05 and volume_momentum > 1.0:
        sig = "bullish"
    elif score < -0.05 and volume_momentum > 1.0:
        sig = "bearish"
    else:
        sig = "neutral"

    return sig, float(score), float(volume_momentum)


def vol_regime(df):
    returns = df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)
    vol_ma = hist_vol.rolling(63).mean()
    regime = (hist_vol / vol_ma).iloc[-1]
    return float(regime) if pd.notna(regime) else np.nan


def pct(df, n):
    if len(df) <= n:
        return np.nan
    return (df["close"].iloc[-1] / df["close"].iloc[-n - 1] - 1.0) * 100


def main():
    results = []
    for ticker, name in TICKERS.items():
        try:
            df = fetch_nasdaq_hist(ticker)
        except Exception as exc:
            print(f"WARN {ticker}: fetch failed ({exc})")
            continue

        if df.empty or len(df) < 70:
            print(f"WARN {ticker}: insufficient rows ({len(df)})")
            continue

        tsig, _tconf, adx = trend_signal(df)
        msig, mscore, _vm = momentum_signal(df)
        vr = vol_regime(df)

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "last_close": round(float(df["close"].iloc[-1]), 2),
                "ret_5d": round(pct(df, 5), 2),
                "ret_20d": round(pct(df, 20), 2),
                "ret_60d": round(pct(df, 60), 2),
                "trend": tsig,
                "adx": round(adx, 2) if adx == adx else None,
                "momentum": msig,
                "momentum_score": round(mscore, 4),
                "vol_regime": round(vr, 2) if vr == vr else None,
            }
        )

    if not results:
        print("No valid market rows collected for selected symbols.")
        return

    out = pd.DataFrame(results).sort_values("ticker")
    print(out.to_string(index=False))

    risk_on = ["QQQ", "IWM", "XLK", "XLF"]
    risk_off = ["TLT", "XLU", "GLD"]
    subset = {row["ticker"]: row for row in results}

    ro_20 = np.nanmean([subset[t]["ret_20d"] for t in risk_on if t in subset])
    rf_20 = np.nanmean([subset[t]["ret_20d"] for t in risk_off if t in subset])
    ro_trend_bull = sum(1 for t in risk_on if t in subset and subset[t]["trend"] == "bullish")
    rf_trend_bull = sum(1 for t in risk_off if t in subset and subset[t]["trend"] == "bullish")

    print("\nREGIME")
    print(f"risk_on_avg_20d={ro_20:.2f}%")
    print(f"risk_off_avg_20d={rf_20:.2f}%")
    print(f"risk_on_bullish_trend={ro_trend_bull}/{len([t for t in risk_on if t in subset])}")
    print(f"risk_off_bullish_trend={rf_trend_bull}/{len([t for t in risk_off if t in subset])}")


if __name__ == "__main__":
    main()
