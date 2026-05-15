"""
Predict stocks that may surge, based on similarity to historical pre-surge features.
Uses a weighted scoring system comparing current technical indicators to the
average pre-surge profile from historical surge stocks.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from config import FEATURES_CACHE, PREDICTION_RESULT, PRE_SURGE_WINDOW
from data_collector import load_cached_data, get_stock_list
from analyzer import extract_features_for_window, compute_common_features


FEATURE_COLS = [
    "price_change_60d", "recent_5d_return", "recent_10d_return",
    "recent_20d_return", "pct_from_ma5", "pct_from_ma10",
    "pct_from_ma20", "pct_from_ma60", "ma_bullish",
    "macd_dif", "macd_hist", "macd_golden_cross",
    "rsi_6", "rsi_14", "rsi_24",
    "bb_position", "bb_width",
    "vol_ratio_5_20", "vol_trend",
    "volatility_5d", "volatility_20d",
    "avg_amplitude_5d", "avg_amplitude_20d",
    "atr_pct", "kdj_k", "kdj_d", "kdj_j",
    "avg_turnover_5d", "avg_turnover_20d",
    "price_position", "max_drawdown_60d",
    "consecutive_positive", "consecutive_negative",
]


def compute_similarity_score(
    current_features: dict,
    surge_stats: dict,
    feature_weights: dict[str, float] | None = None,
) -> float:
    """
    Compute a similarity score (0-100) between current features and
    the average pre-surge profile.
    """
    if surge_stats is None or not surge_stats:
        return 0.0

    if feature_weights is None:
        feature_weights = {k: 1.0 for k in FEATURE_COLS}

    scores = []
    total_weight = 0.0

    for col in FEATURE_COLS:
        if col not in current_features or col not in surge_stats:
            continue
        val = current_features[col]
        stats = surge_stats[col]
        mean_val = stats["mean"]
        std_val = max(stats["std"], 1e-8)

        if val is None or not np.isfinite(val):
            continue

        # Z-score based similarity (Gaussian)
        z = abs(val - mean_val) / std_val
        similarity = np.exp(-0.5 * z ** 2)  # 1.0 when perfect match, →0 when far
        weight = feature_weights.get(col, 1.0)
        scores.append(similarity * weight)
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return (sum(scores) / total_weight) * 100.0


def build_feature_profile(surge_df: pd.DataFrame) -> dict:
    """Build the average pre-surge feature profile from historical data."""
    stats = {}
    for col in FEATURE_COLS:
        if col in surge_df.columns:
            series = surge_df[col].dropna()
            stats[col] = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
            }
    return stats


def compute_feature_importance(surge_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute feature importance based on stability (inverse of std/mean ratio).
    Features with low variance across surge stocks are more predictive.
    """
    importance = {}
    for col in FEATURE_COLS:
        if col in surge_df.columns:
            series = surge_df[col].dropna()
            mean_val = abs(series.mean())
            std_val = series.std()
            if mean_val > 0:
                cv = std_val / mean_val  # coefficient of variation
                importance[col] = 1.0 / (1.0 + cv)
            else:
                importance[col] = 0.1
    return importance


def screen_current_market(
    stock_data: dict[str, pd.DataFrame],
    stock_info: pd.DataFrame,
    surge_stats: dict,
    feature_weights: dict[str, float],
    top_n: int = 50,
) -> pd.DataFrame:
    """Screen current market for stocks matching pre-surge profile."""
    code_to_name = dict(zip(stock_info["code"], stock_info["name"]))
    results = []

    for code, df in stock_data.items():
        if df is None or len(df) < PRE_SURGE_WINDOW:
            continue

        # Extract features from the most recent window (last 60 days)
        features = extract_features_for_window(df, len(df))
        if features is None:
            continue

        score = compute_similarity_score(features, surge_stats, feature_weights)

        name = code_to_name.get(code, "")
        last_close = df["close"].iloc[-1] if len(df) > 0 else 0

        results.append({
            "code": code,
            "name": name,
            "score": score,
            "last_close": last_close,
            "last_date": df.index[-1] if len(df) > 0 else None,
            **features,
        })

    result_df = pd.DataFrame(results).sort_values("score", ascending=False)
    result_df = result_df.head(top_n).reset_index(drop=True)
    result_df.to_csv(PREDICTION_RESULT, index=False)
    return result_df


def run_prediction() -> pd.DataFrame:
    """Full prediction pipeline."""
    print("[*] Starting prediction pipeline...")

    # Load analysis results
    from config import ANALYSIS_RESULT
    if not os.path.exists(ANALYSIS_RESULT):
        print("[!] No analysis results found. Run analyzer first.")
        return pd.DataFrame()

    surge_df = pd.read_csv(ANALYSIS_RESULT)
    print(f"[*] Loaded {len(surge_df)} historical surge events")

    # Build feature profile
    surge_stats = build_feature_profile(surge_df)
    feature_weights = compute_feature_importance(surge_df)

    # Print top predictive features
    sorted_features = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)
    print("\n[*] Most predictive features (low variance across surge stocks):")
    for name, imp in sorted_features[:10]:
        print(f"    {name}: {imp:.3f}")

    # Load current market data
    stock_info = get_stock_list()
    stock_data = load_cached_data()
    print(f"[*] Loaded {len(stock_data)} stocks for screening")

    # Screen
    predictions = screen_current_market(
        stock_data, stock_info, surge_stats, feature_weights
    )

    print(f"\n[+] Top {len(predictions)} predictions:")
    for _, row in predictions.head(20).iterrows():
        print(f"  {row['code']} {row['name']}: score={row['score']:.1f}")

    return predictions


if __name__ == "__main__":
    run_prediction()
