"""
Find stocks that surged >100% within one month over the past 2 years,
and extract their pre-surge technical characteristics.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import (
    SURGE_THRESHOLD, SURGE_WINDOW_DAYS, PRE_SURGE_WINDOW,
    ANALYSIS_RESULT, FEATURES_CACHE, DAILY_DATA_DIR
)

# Technical indicator computation
def compute_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def compute_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def compute_macd(close: pd.Series):
    ema12 = compute_ema(close, 12)
    ema26 = compute_ema(close, 26)
    dif = ema12 - ema26
    dea = compute_ema(dif, 9)
    hist = 2 * (dif - dea)
    return dif, dea, hist

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    ma = compute_ma(close, window)
    std = close.rolling(window=window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    bandwidth = (upper - lower) / ma  # relative width
    pct_b = (close - lower) / (upper - lower)  # position within bands
    return upper, lower, bandwidth, pct_b

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()

def compute_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3):
    """Compute KDJ indicator. Returns K, D, J series."""
    low_n = df["low"].rolling(window=n).min()
    high_n = df["high"].rolling(window=n).max()
    rsv = ((df["close"] - low_n) / (high_n - low_n + 1e-10)) * 100
    k = rsv.ewm(alpha=1/m1, adjust=False).mean()
    d = k.ewm(alpha=1/m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def extract_features_for_window(
    df: pd.DataFrame, end_idx: int, window: int = PRE_SURGE_WINDOW
) -> dict:
    """
    Extract technical features from a window ending at end_idx (exclusive).
    end_idx is the position just before the surge starts.
    """
    if end_idx < window + 30:
        return None

    start_idx = max(0, end_idx - window)
    segment = df.iloc[start_idx:end_idx].copy()
    if len(segment) < 30:
        return None

    close = segment["close"]
    volume = segment["volume"]
    high = segment["high"]
    low = segment["low"]

    # Basic price features
    price_start = close.iloc[0]
    price_end = close.iloc[-1]
    price_change = (price_end - price_start) / price_start

    # Recent trend (last 5, 10, 20 days)
    recent_5d = close.iloc[-5:].pct_change().mean() if len(close) >= 5 else 0
    recent_10d = close.iloc[-10:].pct_change().mean() if len(close) >= 10 else 0
    recent_20d = close.iloc[-20:].pct_change().mean() if len(close) >= 20 else 0

    # MA positions
    ma5 = compute_ma(close, 5).iloc[-1]
    ma10 = compute_ma(close, 10).iloc[-1]
    ma20 = compute_ma(close, 20).iloc[-1]
    ma60 = compute_ma(close, 60).iloc[-1] if len(close) >= 60 else close.mean()

    last_close = close.iloc[-1]
    # Price relative to MAs (%)
    pct_ma5 = (last_close - ma5) / ma5 if pd.notna(ma5) else 0
    pct_ma10 = (last_close - ma10) / ma10 if pd.notna(ma10) else 0
    pct_ma20 = (last_close - ma20) / ma20 if pd.notna(ma20) else 0
    pct_ma60 = (last_close - ma60) / ma60 if pd.notna(ma60) else 0

    # MA alignment (多头排列 or 空头排列)
    ma_bullish = 1 if (ma5 > ma10 > ma20) else (0 if (ma5 < ma10 < ma20) else 0.5)

    # MACD
    dif, dea, hist = compute_macd(close)
    macd_dif_last = dif.iloc[-1] if pd.notna(dif.iloc[-1]) else 0
    macd_dea_last = dea.iloc[-1] if pd.notna(dea.iloc[-1]) else 0
    macd_hist_last = hist.iloc[-1] if pd.notna(hist.iloc[-1]) else 0
    macd_golden_cross = 1 if dif.iloc[-1] > dea.iloc[-1] else 0

    # RSI
    rsi6 = compute_rsi(close, 6).iloc[-1] if len(close) >= 6 else 50
    rsi14 = compute_rsi(close, 14).iloc[-1] if len(close) >= 14 else 50
    rsi24 = compute_rsi(close, 24).iloc[-1] if len(close) >= 24 else 50

    # Bollinger Bands
    bb_upper, bb_lower, bb_width, bb_pct_b = compute_bollinger(close)
    bb_pct_b_last = bb_pct_b.iloc[-1] if pd.notna(bb_pct_b.iloc[-1]) else 0.5
    bb_width_last = bb_width.iloc[-1] if pd.notna(bb_width.iloc[-1]) else 0

    # Volume features
    vol_ma5 = compute_ma(volume, 5).iloc[-1]
    vol_ma20 = compute_ma(volume, 20).iloc[-1]
    vol_ratio = vol_ma5 / vol_ma20 if vol_ma20 and vol_ma20 > 0 else 1.0
    vol_trend = (volume.iloc[-5:].mean() / volume.iloc[-20:].mean()
                 if len(volume) >= 20 and volume.iloc[-20:].mean() > 0 else 1.0)

    # Volatility
    daily_returns = close.pct_change().dropna()
    volatility_5d = daily_returns.iloc[-5:].std() if len(daily_returns) >= 5 else 0
    volatility_20d = daily_returns.iloc[-20:].std() if len(daily_returns) >= 20 else 0

    # Amplitude (avg daily range %)
    amplitude = ((high - low) / close.shift(1)).dropna()
    avg_amplitude_5d = amplitude.iloc[-5:].mean() if len(amplitude) >= 5 else 0
    avg_amplitude_20d = amplitude.iloc[-20:].mean() if len(amplitude) >= 20 else 0

    # ATR ratio
    atr = compute_atr(segment)
    atr_pct = (atr.iloc[-1] / last_close) if last_close > 0 else 0

    # KDJ
    k, d, j = compute_kdj(segment)
    k_last = k.iloc[-1] if pd.notna(k.iloc[-1]) else 50
    d_last = d.iloc[-1] if pd.notna(d.iloc[-1]) else 50
    j_last = j.iloc[-1] if pd.notna(j.iloc[-1]) else 50

    # Turnover features
    if "turnover" in segment.columns:
        turnover = segment["turnover"].dropna()
        avg_turnover_5d = turnover.iloc[-5:].mean() if len(turnover) >= 5 else 0
        avg_turnover_20d = turnover.iloc[-20:].mean() if len(turnover) >= 20 else 0
    else:
        avg_turnover_5d = 0
        avg_turnover_20d = 0

    # Price position in recent range
    price_position = (
        (last_close - low.iloc[-window:].min()) /
        (high.iloc[-window:].max() - low.iloc[-window:].min() + 1e-10)
    )

    # Max drawdown in window
    cumulative_max = close.expanding().max()
    drawdown = (close - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    # Consecutive positive/negative days near end
    recent_signs = np.sign(daily_returns.iloc[-10:].dropna()) if len(daily_returns) >= 10 else np.array([0])
    consecutive_positive = 0
    for s in recent_signs[::-1]:
        if s > 0:
            consecutive_positive += 1
        else:
            break
    consecutive_negative = 0
    for s in recent_signs[::-1]:
        if s < 0:
            consecutive_negative += 1
        else:
            break

    return {
        "price_change_60d": price_change,
        "recent_5d_return": recent_5d,
        "recent_10d_return": recent_10d,
        "recent_20d_return": recent_20d,
        "pct_from_ma5": pct_ma5,
        "pct_from_ma10": pct_ma10,
        "pct_from_ma20": pct_ma20,
        "pct_from_ma60": pct_ma60,
        "ma_bullish": ma_bullish,
        "macd_dif": macd_dif_last,
        "macd_hist": macd_hist_last,
        "macd_golden_cross": macd_golden_cross,
        "rsi_6": rsi6,
        "rsi_14": rsi14,
        "rsi_24": rsi24,
        "bb_position": bb_pct_b_last,
        "bb_width": bb_width_last,
        "vol_ratio_5_20": vol_ratio,
        "vol_trend": vol_trend,
        "volatility_5d": volatility_5d,
        "volatility_20d": volatility_20d,
        "avg_amplitude_5d": avg_amplitude_5d,
        "avg_amplitude_20d": avg_amplitude_20d,
        "atr_pct": atr_pct,
        "kdj_k": k_last,
        "kdj_d": d_last,
        "kdj_j": j_last,
        "avg_turnover_5d": avg_turnover_5d,
        "avg_turnover_20d": avg_turnover_20d,
        "price_position": price_position,
        "max_drawdown_60d": max_drawdown,
        "consecutive_positive": consecutive_positive,
        "consecutive_negative": consecutive_negative,
    }


def find_surges(df: pd.DataFrame, code: str, name: str = "") -> list[dict]:
    """
    Find all periods where the stock surged >SURGE_THRESHOLD within SURGE_WINDOW_DAYS.
    Returns list of dicts with surge details and pre-surge features.
    """
    if df is None or len(df) < SURGE_WINDOW_DAYS + PRE_SURGE_WINDOW:
        return []

    close = df["close"]
    surges = []

    # Rolling maximum gain over SURGE_WINDOW_DAYS
    for i in range(SURGE_WINDOW_DAYS + PRE_SURGE_WINDOW, len(close)):
        window_start = i - SURGE_WINDOW_DAYS
        window_end = i

        price_start = close.iloc[window_start]
        price_end = close.iloc[window_end]
        if pd.isna(price_start) or pd.isna(price_end) or price_start <= 0:
            continue

        gain = (price_end - price_start) / price_start
        if gain < SURGE_THRESHOLD:
            continue

        # Found a surge - extract pre-surge features
        pre_surge_end = window_start  # just before surge
        features = extract_features_for_window(df, pre_surge_end)

        if features is None:
            continue

        surge_date = df.index[window_start]
        surge_end_date = df.index[window_end]

        # Skip if we already captured an overlapping surge (keep the larger one)
        if surges:
            last = surges[-1]
            if (surge_date - last["surge_end"]).days < 10:
                if gain > last["gain"]:
                    surges[-1] = None  # mark for replacement
                else:
                    continue

        surges.append({
            "code": code,
            "name": name,
            "surge_start": surge_date,
            "surge_end": surge_end_date,
            "gain": gain,
            "gain_pct": gain * 100,
            "surge_start_price": price_start,
            "surge_end_price": price_end,
            "pre_surge_features": features,
        })

    # Filter out None entries from overlapping replacement
    return [s for s in surges if s is not None]


def analyze_all_stocks(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze all stocks and return DataFrame of surge events with features."""
    code_to_name = dict(zip(stock_info["code"], stock_info["name"]))
    all_surges = []

    total = len(stock_data)
    for i, (code, df) in enumerate(stock_data.items()):
        if (i + 1) % 500 == 0:
            print(f"    Analyzing: {i+1}/{total} stocks, found {len(all_surges)} surges so far")

        name = code_to_name.get(code, "")
        surges = find_surges(df, code, name)
        all_surges.extend(surges)

    print(f"[+] Found {len(all_surges)} surge events across {total} stocks")

    if not all_surges:
        return pd.DataFrame()

    # Flatten features into columns
    rows = []
    for s in all_surges:
        row = {
            "code": s["code"],
            "name": s["name"],
            "surge_start": s["surge_start"],
            "surge_end": s["surge_end"],
            "gain_pct": s["gain_pct"],
            "surge_start_price": s["surge_start_price"],
            "surge_end_price": s["surge_end_price"],
        }
        row.update(s["pre_surge_features"])
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(ANALYSIS_RESULT, index=False)
    print(f"[+] Saved {len(result)} surge events to {ANALYSIS_RESULT}")
    return result


def compute_common_features(result_df: pd.DataFrame) -> dict:
    """Compute aggregate statistics of pre-surge features."""
    feature_cols = [c for c in result_df.columns
                    if c not in ("code", "name", "surge_start", "surge_end",
                                 "gain_pct", "surge_start_price", "surge_end_price")]

    stats = {}
    for col in feature_cols:
        if col in result_df.columns:
            series = result_df[col].dropna()
            stats[col] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "min": float(series.min()),
                "max": float(series.max()),
            }

    # Save features
    features_df = pd.DataFrame(stats).T
    features_df.to_csv(FEATURES_CACHE)
    print(f"[+] Common features saved to {FEATURES_CACHE}")

    return stats


if __name__ == "__main__":
    from data_collector import load_cached_data, get_stock_list

    stock_info = get_stock_list()
    stock_data = load_cached_data()
    result = analyze_all_stocks(stock_data, stock_info)
    if not result.empty:
        stats = compute_common_features(result)
        print(f"\nFound {len(result)} surge events")
        print(f"Average gain: {result['gain_pct'].mean():.1f}%")
        print(f"\nTop 10 surge stocks:")
        top = result.nlargest(10, "gain_pct")
        for _, row in top.iterrows():
            print(f"  {row['code']} {row['name']}: {row['gain_pct']:.0f}% "
                  f"({row['surge_start'].strftime('%Y-%m-%d')} -> "
                  f"{row['surge_end'].strftime('%Y-%m-%d')})")
