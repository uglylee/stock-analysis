"""
Flask web dashboard for A-share surge stock analysis.
Displays historical surge stocks, common technical features, and predictions.
"""

import os
import json
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, request

from config import (
    ANALYSIS_RESULT, FEATURES_CACHE, PREDICTION_RESULT,
    DATA_DIR, DAILY_DATA_DIR, PRE_SURGE_WINDOW, SURGE_WINDOW_DAYS,
    STOCK_LIST_CACHE
)
from data_collector import load_cached_data, get_stock_list, ensure_proxy, download_one_stock

app = Flask(__name__)

# Background prediction state
_prediction_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "result_file": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


# --- Data Loaders ---

def load_surge_data() -> pd.DataFrame:
    if os.path.exists(ANALYSIS_RESULT):
        df = pd.read_csv(ANALYSIS_RESULT)
        if "surge_start" in df.columns:
            df["surge_start"] = pd.to_datetime(df["surge_start"])
            df["surge_end"] = pd.to_datetime(df["surge_end"])
        return df
    return pd.DataFrame()


def load_predictions() -> pd.DataFrame:
    if os.path.exists(PREDICTION_RESULT):
        return pd.read_csv(PREDICTION_RESULT)
    return pd.DataFrame()


def load_features_stats() -> dict:
    if os.path.exists(FEATURES_CACHE):
        df = pd.read_csv(FEATURES_CACHE, index_col=0)
        return df.to_dict(orient="index")
    return {}


# --- Plotly Chart Builders ---

def plot_surge_distribution(surge_df: pd.DataFrame) -> str:
    """Histogram of surge gains and time distribution."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("月度涨幅分布", "暴涨时间分布"),
        specs=[[{"type": "histogram"}, {"type": "histogram"}]]
    )

    fig.add_trace(
        go.Histogram(
            x=surge_df["gain_pct"], nbinsx=50,
            marker_color="#ef4444", name="涨幅分布",
            hovertemplate="涨幅: %{x:.0f}%<br>数量: %{y}"
        ),
        row=1, col=1
    )

    surge_df_clean = surge_df.dropna(subset=["surge_start"])
    fig.add_trace(
        go.Histogram(
            x=surge_df_clean["surge_start"], nbinsx=24,
            marker_color="#3b82f6", name="时间分布",
            hovertemplate="日期: %{x}<br>数量: %{y}"
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="A股暴涨股票分析总览",
        template="plotly_dark",
    )
    return fig.to_html(full_html=False)


def plot_top_gainers(surge_df: pd.DataFrame, top_n: int = 20) -> str:
    """Bar chart of top gainers."""
    import plotly.graph_objects as go

    top = surge_df.nlargest(top_n, "gain_pct")
    labels = [f"{r['code']} {r['name']}" for _, r in top.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels[::-1],
        y=top["gain_pct"].values[::-1],
        orientation="v",
        marker=dict(
            color=top["gain_pct"].values[::-1],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="涨幅(%)"),
        ),
        hovertemplate="%{x}<br>涨幅: %{y:.0f}%<extra></extra>",
    ))

    fig.update_layout(
        title=f"涨幅最大的 Top {top_n} 股票",
        xaxis_title="股票",
        yaxis_title="涨幅 (%)",
        height=500,
        template="plotly_dark",
    )
    return fig.to_html(full_html=False)


def plot_stock_detail(df: pd.DataFrame, code: str, name: str,
                      surge_start, surge_end) -> str:
    """Candlestick chart with surge period highlighted."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df is None or df.empty:
        return "<p>无数据</p>"

    # Filter to relevant date range
    surge_start_ts = pd.Timestamp(surge_start)
    surge_end_ts = pd.Timestamp(surge_end)

    start_date = surge_start_ts - pd.DateOffset(days=90)
    end_date = surge_end_ts + pd.DateOffset(days=30)

    mask = (df.index >= start_date) & (df.index <= end_date)
    segment = df[mask].copy()

    if segment.empty:
        segment = df.tail(200)
        surge_start_ts = segment.index[len(segment)//2]
        surge_end_ts = segment.index[min(len(segment)//2 + 20, len(segment)-1)]

    # Pre-surge zone
    pre_start = surge_start_ts - pd.DateOffset(days=60)

    # Compute MAs
    segment["ma5"] = segment["close"].rolling(5).mean()
    segment["ma10"] = segment["close"].rolling(10).mean()
    segment["ma20"] = segment["close"].rolling(20).mean()
    segment["ma60"] = segment["close"].rolling(60).mean()

    # MACD
    ema12 = segment["close"].ewm(span=12).mean()
    ema26 = segment["close"].ewm(span=26).mean()
    segment["macd_dif"] = ema12 - ema26
    segment["macd_dea"] = segment["macd_dif"].ewm(span=9).mean()
    segment["macd_hist"] = 2 * (segment["macd_dif"] - segment["macd_dea"])

    # RSI 14
    delta = segment["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    segment["rsi14"] = 100 - (100 / (1 + (
        gain.ewm(alpha=1/14).mean() / loss.ewm(alpha=1/14).mean()
    )))

    # Volume MA
    segment["vol_ma5"] = segment["volume"].rolling(5).mean()
    segment["vol_ma20"] = segment["volume"].rolling(20).mean()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.2, 0.2, 0.15],
        subplot_titles=(
            f"{code} {name} 价格走势",
            "成交量",
            "MACD",
            "RSI(14)"
        ),
    )

    # --- K-line ---
    colors = ["#ef4444" if segment["close"].iloc[i] >= segment["open"].iloc[i]
              else "#22c55e" for i in range(len(segment))]

    fig.add_trace(go.Candlestick(
        x=segment.index,
        open=segment["open"], high=segment["high"],
        low=segment["low"], close=segment["close"],
        name="K线",
        increasing_line_color="#ef4444",
        decreasing_line_color="#22c55e",
    ), row=1, col=1)

    # MAs
    for ma_col, ma_name, ma_color in [
        ("ma5", "MA5", "#fbbf24"),
        ("ma10", "MA10", "#f97316"),
        ("ma20", "MA20", "#a855f7"),
        ("ma60", "MA60", "#06b6d4"),
    ]:
        fig.add_trace(go.Scatter(
            x=segment.index, y=segment[ma_col],
            mode="lines", name=ma_name,
            line=dict(color=ma_color, width=1),
            visible="legendonly" if ma_col != "ma20" else True,
        ), row=1, col=1)

    # Surge highlight
    fig.add_vrect(
        x0=surge_start_ts, x1=surge_end_ts,
        fillcolor="rgba(239, 68, 68, 0.15)",
        layer="below", line_width=0,
        annotation_text="暴涨区间",
        annotation_position="top left",
        row=1, col=1
    )

    # Pre-surge highlight
    fig.add_vrect(
        x0=pre_start, x1=surge_start_ts,
        fillcolor="rgba(59, 130, 246, 0.1)",
        layer="below", line_width=0,
        annotation_text="暴涨前特征窗口",
        annotation_position="top left",
        row=1, col=1
    )

    # --- Volume ---
    vol_colors = ["#ef4444" if segment["close"].iloc[i] >= segment["open"].iloc[i]
                  else "#22c55e" for i in range(len(segment))]
    fig.add_trace(go.Bar(
        x=segment.index, y=segment["volume"],
        name="成交量", marker_color=vol_colors,
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=segment.index, y=segment["vol_ma5"],
        mode="lines", name="量MA5",
        line=dict(color="#fbbf24", width=1),
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=segment.index, y=segment["vol_ma20"],
        mode="lines", name="量MA20",
        line=dict(color="#a855f7", width=1),
    ), row=2, col=1)

    # --- MACD ---
    fig.add_trace(go.Bar(
        x=segment.index, y=segment["macd_hist"],
        name="MACD柱",
        marker_color=["#ef4444" if v >= 0 else "#22c55e"
                       for v in segment["macd_hist"].fillna(0)],
        showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=segment.index, y=segment["macd_dif"],
        mode="lines", name="DIF",
        line=dict(color="#fbbf24", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=segment.index, y=segment["macd_dea"],
        mode="lines", name="DEA",
        line=dict(color="#a855f7", width=1.5),
    ), row=3, col=1)

    # --- RSI ---
    fig.add_trace(go.Scatter(
        x=segment.index, y=segment["rsi14"],
        mode="lines", name="RSI14",
        line=dict(color="#06b6d4", width=1.5),
        fill="tozeroy", fillcolor="rgba(6, 182, 212, 0.1)",
    ), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                  row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                  row=4, col=1)

    fig.update_layout(
        height=900,
        template="plotly_dark",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_html(full_html=False)


def plot_feature_radar(feature_stats: dict) -> str:
    """Radar/spider chart of common pre-surge technical features."""
    import plotly.graph_objects as go

    # Select key normalized features
    key_features = {
        "price_change_60d": "180日涨跌幅",
        "pct_from_ma20": "距MA20距离",
        "pct_from_ma60": "距MA60距离",
        "volatility_20d": "20日波动率",
        "avg_amplitude_20d": "20日均振幅",
        "rsi_14": "RSI(14)",
        "kdj_k": "KDJ-K",
        "bb_position": "布林带位置",
        "vol_ratio_5_20": "量比(5/20)",
        "max_drawdown_60d": "180日最大回撤",
        "price_position": "价格位置",
        "macd_golden_cross": "MACD金叉",
    }

    # Normalize to 0-100 scale for display
    categories = []
    values = []
    for col, label in key_features.items():
        if col in feature_stats:
            stats = feature_stats[col]
            mean_val = stats["mean"]
            std_val = max(stats["std"], 1e-8)

            # Map to a 0-100 scale centered at mean ± 2std
            norm = min(max((mean_val / (abs(mean_val) + 2 * std_val)), 0), 1) * 100
            categories.append(label)
            values.append(round(norm, 1))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="暴涨前共同特征",
        line_color="#ef4444",
        fillcolor="rgba(239, 68, 68, 0.3)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
            ),
        ),
        title="暴涨前技术特征雷达图 (归一化均值)",
        height=550,
        template="plotly_dark",
        showlegend=True,
    )
    return fig.to_html(full_html=False)


def plot_feature_distributions(feature_stats: dict) -> str:
    """Box plot of key feature distributions."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    key_features = [
        "rsi_14", "rsi_6", "kdj_k", "kdj_j",
        "volatility_20d", "vol_ratio_5_20",
        "pct_from_ma20", "pct_from_ma60",
        "avg_amplitude_20d", "bb_position",
        "price_position", "max_drawdown_60d",
        "macd_golden_cross", "recent_20d_return",
        "consecutive_positive", "price_change_60d",
    ]

    labels = {
        "rsi_14": "RSI(14)",
        "rsi_6": "RSI(6)",
        "kdj_k": "KDJ-K",
        "kdj_j": "KDJ-J",
        "volatility_20d": "20日波动率",
        "vol_ratio_5_20": "量比(5/20)",
        "pct_from_ma20": "距MA20(%)",
        "pct_from_ma60": "距MA60(%)",
        "avg_amplitude_20d": "20日均振幅",
        "bb_position": "布林带位置",
        "price_position": "价格位置",
        "max_drawdown_60d": "180日最大回撤",
        "macd_golden_cross": "MACD金叉",
        "recent_20d_return": "近20日收益",
        "consecutive_positive": "连续阳线",
        "price_change_60d": "180日收益",
    }

    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=[labels.get(f, f) for f in key_features],
    )

    for i, feat in enumerate(key_features):
        if feat not in feature_stats:
            continue
        stats = feature_stats[feat]
        mean_v = stats["mean"]
        q25 = stats["q25"]
        q75 = stats["q75"]
        low = stats["min"]
        high = stats["max"]
        median = stats["median"]

        row = i // 4 + 1
        col = i % 4 + 1

        fig.add_trace(go.Box(
            q1=[q25], median=[median], q3=[q75],
            lowerfence=[low], upperfence=[high],
            mean=[mean_v],
            name=labels.get(feat, feat),
            boxpoints=False,
            marker_color="#ef4444",
            showlegend=False,
        ), row=row, col=col)

    fig.update_layout(
        height=800,
        title_text="暴涨前技术特征分布 (箱线图)",
        template="plotly_dark",
        showlegend=False,
    )
    return fig.to_html(full_html=False)


def plot_prediction_scores(pred_df: pd.DataFrame) -> str:
    """Bar chart of top prediction scores."""
    import plotly.graph_objects as go

    top = pred_df.head(30)
    labels = [f"{r['code']} {r['name']}" for _, r in top.iterrows()]
    scores = top["score"].values

    colors = ["#ef4444" if s > 70 else "#f97316" if s > 50 else "#fbbf24"
              for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scores[::-1],
        y=labels[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{s:.1f}分" for s in scores[::-1]],
        textposition="outside",
        hovertemplate="%{y}<br>匹配度: %{x:.1f}分<extra></extra>",
    ))

    fig.add_vline(x=70, line_dash="dash", line_color="#ef4444",
                  annotation_text="高匹配", annotation_position="top")
    fig.add_vline(x=50, line_dash="dash", line_color="#f97316",
                  annotation_text="中匹配", annotation_position="top")

    fig.update_layout(
        title=f"当前市场暴涨潜力股票预测 (Top {len(top)})",
        xaxis_title="匹配度评分",
        height=700,
        template="plotly_dark",
        xaxis=dict(range=[0, 100]),
    )
    return fig.to_html(full_html=False)


def plot_prediction_compare(pred_df: pd.DataFrame, feature_stats: dict) -> str:
    """Compare top prediction features against the historical profile."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    key_features = [
        "rsi_14", "pct_from_ma20", "pct_from_ma60",
        "vol_ratio_5_20", "volatility_20d", "bb_position",
        "price_position", "max_drawdown_60d",
        "avg_amplitude_20d", "macd_golden_cross",
    ]
    labels = {
        "rsi_14": "RSI(14)",
        "pct_from_ma20": "距MA20",
        "pct_from_ma60": "距MA60",
        "vol_ratio_5_20": "量比",
        "volatility_20d": "波动率",
        "bb_position": "布林位置",
        "price_position": "价格位置",
        "max_drawdown_60d": "最大回撤",
        "avg_amplitude_20d": "振幅",
        "macd_golden_cross": "MACD金叉",
    }

    fig = make_subplots(
        rows=min(5, len(pred_df.head(5))),
        cols=1,
        subplot_titles=[
            f"{r['code']} {r['name']} (评分: {r['score']:.1f})"
            for _, r in pred_df.head(5).iterrows()
        ],
        vertical_spacing=0.08,
    )

    for idx, (_, row) in enumerate(pred_df.head(5).iterrows()):
        current_vals = []
        profile_means = []
        profile_q25 = []
        profile_q75 = []

        for feat in key_features:
            if feat in feature_stats and feat in row.index:
                stats = feature_stats[feat]
                val = row[feat]
                if pd.isna(val) or not np.isfinite(val):
                    current_vals.append(0)
                else:
                    current_vals.append(val)
                profile_means.append(stats["mean"])
                profile_q25.append(stats["q25"])
                profile_q75.append(stats["q75"])

        display_labels = [labels.get(f, f) for f in key_features]

        # Normalize to 0-1 for comparison
        all_vals = current_vals + profile_means + profile_q25 + profile_q75
        vmin = min(all_vals)
        vmax = max(all_vals)
        vrange = max(vmax - vmin, 1e-8)

        current_norm = [(v - vmin) / vrange for v in current_vals]
        means_norm = [(v - vmin) / vrange for v in profile_means]
        q25_norm = [(v - vmin) / vrange for v in profile_q25]
        q75_norm = [(v - vmin) / vrange for v in profile_q75]

        fig.add_trace(go.Scatter(
            x=display_labels, y=current_norm,
            mode="lines+markers",
            name=f"{row['code']} 当前",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=8),
            showlegend=(idx == 0),
        ), row=idx + 1, col=1)

        fig.add_trace(go.Scatter(
            x=display_labels, y=means_norm,
            mode="lines+markers",
            name="历史暴涨均值",
            line=dict(color="#3b82f6", width=2, dash="dash"),
            marker=dict(size=6),
            showlegend=(idx == 0),
        ), row=idx + 1, col=1)

        # Shaded region for interquartile range
        fig.add_trace(go.Scatter(
            x=display_labels + display_labels[::-1],
            y=q75_norm + q25_norm[::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.1)",
            line=dict(color="rgba(59, 130, 246, 0)"),
            name="历史IQR区间",
            showlegend=(idx == 0),
        ), row=idx + 1, col=1)

    fig.update_layout(
        height=250 * min(5, len(pred_df.head(5))),
        title_text="预测股票 vs 历史暴涨特征对比 (归一化)",
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig.to_html(full_html=False)


# --- Routes ---

@app.route("/")
def index():
    """Overview dashboard."""
    surge_df = load_surge_data()
    pred_df = load_predictions()

    summary = {
        "total_surges": len(surge_df),
        "unique_stocks": surge_df["code"].nunique() if not surge_df.empty else 0,
        "avg_gain": round(surge_df["gain_pct"].mean(), 1) if not surge_df.empty else 0,
        "max_gain": round(surge_df["gain_pct"].max(), 1) if not surge_df.empty else 0,
        "predictions_available": len(pred_df),
        "high_score_preds": len(pred_df[pred_df["score"] > 70]) if not pred_df.empty else 0,
    }

    charts = {}
    if not surge_df.empty:
        charts["distribution"] = plot_surge_distribution(surge_df)
        charts["top_gainers"] = plot_top_gainers(surge_df, 20)

    if not pred_df.empty:
        charts["prediction_scores"] = plot_prediction_scores(pred_df)

    # Recent surges table
    recent_surges = pd.DataFrame()
    if not surge_df.empty:
        recent_surges = surge_df.nlargest(20, "gain_pct")[
            ["code", "name", "gain_pct", "surge_start", "surge_end"]
        ].copy()
        recent_surges["surge_start"] = recent_surges["surge_start"].dt.strftime("%Y-%m-%d")
        recent_surges["surge_end"] = recent_surges["surge_end"].dt.strftime("%Y-%m-%d")
        recent_surges["gain_pct"] = recent_surges["gain_pct"].round(1)

    # Prediction table
    pred_table = pd.DataFrame()
    if not pred_df.empty:
        pred_table = pred_df.head(20)[["code", "name", "score", "last_close"]].copy()
        pred_table["score"] = pred_table["score"].round(1)
        pred_table["last_close"] = pred_table["last_close"].round(2)

    return render_template(
        "index.html",
        summary=summary,
        charts=charts,
        recent_surges=recent_surges.to_dict(orient="records"),
        predictions=pred_table.to_dict(orient="records"),
    )


@app.route("/stock/<code>")
def stock_detail(code):
    """Individual stock analysis page."""
    surge_df = load_surge_data()
    stock_surges = surge_df[surge_df["code"] == code] if not surge_df.empty else pd.DataFrame()

    # Load stock data
    stock_data = load_cached_data()
    df = stock_data.get(code)

    name = stock_surges["name"].iloc[0] if not stock_surges.empty else code

    charts = {}
    for i, (_, surge) in enumerate(stock_surges.iterrows()):
        chart_key = f"chart_{i}"
        charts[chart_key] = plot_stock_detail(
            df, code, name,
            surge["surge_start"], surge["surge_end"]
        )

    # Feature summary
    feature_cols = [
        "rsi_14", "rsi_6", "pct_from_ma20", "pct_from_ma60",
        "vol_ratio_5_20", "volatility_20d", "bb_position",
        "price_position", "max_drawdown_60d", "avg_amplitude_20d",
        "macd_golden_cross", "recent_20d_return",
    ]
    feature_labels = {
        "rsi_14": "RSI(14)", "rsi_6": "RSI(6)",
        "pct_from_ma20": "距MA20(%)", "pct_from_ma60": "距MA60(%)",
        "vol_ratio_5_20": "量比(5/20)", "volatility_20d": "20日波动率",
        "bb_position": "布林带位置", "price_position": "价格位置",
        "max_drawdown_60d": "180日最大回撤", "avg_amplitude_20d": "20日均振幅",
        "macd_golden_cross": "MACD金叉", "recent_20d_return": "近20日收益",
    }

    surge_features = []
    for _, surge in stock_surges.iterrows():
        feats = {}
        for col in feature_cols:
            if col in surge.index:
                feats[feature_labels.get(col, col)] = round(surge[col], 4)
        surge_features.append({
            "gain_pct": round(surge["gain_pct"], 1),
            "start": surge["surge_start"].strftime("%Y-%m-%d") if hasattr(surge["surge_start"], "strftime") else str(surge["surge_start"]),
            "features": feats,
        })

    return render_template(
        "stock_detail.html",
        code=code,
        name=name,
        charts=charts,
        surge_count=len(stock_surges),
        surge_features=surge_features,
    )


@app.route("/features")
def common_features():
    """Common technical features analysis page."""
    feature_stats = load_features_stats()
    surge_df = load_surge_data()

    charts = {}
    if feature_stats:
        charts["radar"] = plot_feature_radar(feature_stats)
        charts["distributions"] = plot_feature_distributions(feature_stats)

    # Feature summary table
    feature_table = []
    labels_map = {
        "rsi_14": "RSI(14)", "rsi_6": "RSI(6)", "rsi_24": "RSI(24)",
        "pct_from_ma20": "距MA20距离", "pct_from_ma60": "距MA60距离",
        "volatility_20d": "20日波动率", "avg_amplitude_20d": "20日均振幅",
        "vol_ratio_5_20": "量比(5/20)", "bb_position": "布林带位置",
        "price_position": "价格位置", "max_drawdown_60d": "180日最大回撤",
        "macd_golden_cross": "MACD金叉概率", "kdj_k": "KDJ-K",
        "recent_20d_return": "近20日收益", "price_change_60d": "180日收益",
        "consecutive_positive": "连续阳线天数",
    }

    for col, label in labels_map.items():
        if col in feature_stats:
            stats = feature_stats[col]
            feature_table.append({
                "name": label,
                "key": col,
                "mean": round(stats["mean"], 4),
                "median": round(stats["median"], 4),
                "std": round(stats["std"], 4),
                "q25": round(stats["q25"], 4),
                "q75": round(stats["q75"], 4),
            })

    return render_template(
        "features.html",
        charts=charts,
        feature_table=feature_table,
        surge_count=len(surge_df) if not surge_df.empty else 0,
    )


@app.route("/prediction")
def prediction():
    """Prediction results page."""
    pred_df = load_predictions()
    feature_stats = load_features_stats()

    charts = {}
    summary = {"total": len(pred_df), "high": 0, "medium": 0, "low": 0}

    if not pred_df.empty:
        charts["scores"] = plot_prediction_scores(pred_df)
        if feature_stats:
            charts["compare"] = plot_prediction_compare(pred_df, feature_stats)

        summary["high"] = len(pred_df[pred_df["score"] > 70])
        summary["medium"] = len(pred_df[(pred_df["score"] > 50) & (pred_df["score"] <= 70)])
        summary["low"] = len(pred_df[pred_df["score"] <= 50])

        # Full prediction table
        pred_table = pred_df[[
            "code", "name", "score", "last_close",
            "rsi_14", "pct_from_ma20", "vol_ratio_5_20",
            "volatility_20d", "bb_position", "max_drawdown_60d",
        ]].copy()
        for col in pred_table.columns:
            if col not in ("code", "name"):
                pred_table[col] = pred_table[col].round(2)

        predictions_list = pred_table.to_dict(orient="records")
    else:
        predictions_list = []

    return render_template(
        "prediction.html",
        charts=charts,
        summary=summary,
        predictions=predictions_list,
    )


@app.route("/api/search")
def api_search():
    """Search stocks by code or name."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    surge_df = load_surge_data()
    if surge_df.empty:
        return jsonify([])

    mask = surge_df["code"].str.contains(q) | surge_df["name"].str.contains(q)
    results = surge_df[mask][["code", "name"]].drop_duplicates().head(20)
    return jsonify(results.to_dict(orient="records"))


@app.route("/api/stocks")
def api_stocks():
    """List all surge stocks for autocomplete."""
    surge_df = load_surge_data()
    if surge_df.empty:
        return jsonify([])

    stocks = surge_df[["code", "name"]].drop_duplicates().to_dict(orient="records")
    return jsonify(stocks)


@app.route("/api/stock/<code>/kline")
def api_stock_kline(code):
    """Return last 120 trading days of OHLCV data for a stock."""
    stock_data = load_cached_data()
    df = stock_data.get(code)
    if df is None or df.empty:
        return jsonify({"error": "数据不存在", "data": []})

    df = df.tail(120).copy()
    df.index = df.index.strftime("%Y-%m-%d")
    result = []
    for idx, row in df.iterrows():
        result.append({
            "date": str(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })
    return jsonify({"data": result, "code": code})


@app.route("/api/run-prediction", methods=["POST"])
def api_run_prediction():
    """Trigger a fresh prediction run in the background with current date."""
    global _prediction_state

    if _prediction_state["running"]:
        return jsonify({
            "status": "already_running",
            "message": "预测已在运行中",
            "progress": _prediction_state["progress"],
            "total": _prediction_state["total"],
        })

    _prediction_state = {
        "running": True,
        "progress": 0,
        "total": 0,
        "message": "正在初始化...",
        "result_file": None,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
    }

    thread = threading.Thread(target=_run_prediction_background, daemon=True)
    thread.start()

    return jsonify({"status": "started", "message": "预测已启动"})


@app.route("/api/prediction-status")
def api_prediction_status():
    """Check prediction progress."""
    return jsonify(_prediction_state)


def _run_prediction_background():
    """Run the full prediction pipeline in a background thread."""
    global _prediction_state

    try:
        from predictor import (
            build_feature_profile, compute_feature_importance,
            screen_current_market, FEATURE_COLS
        )
        from analyzer import extract_features_for_window

        _prediction_state["message"] = "正在刷新最新数据..."
        ensure_proxy()

        # Step 1: Refresh stock data for all cached stocks to get latest prices
        stock_info = get_stock_list()
        code_to_name = dict(zip(stock_info["code"], stock_info["name"]))

        # Load existing cached data and refresh each stock's latest data
        cached_data = load_cached_data()
        all_codes = list(cached_data.keys())
        _prediction_state["total"] = len(all_codes)
        _prediction_state["message"] = f"正在更新 {len(all_codes)} 只股票的最新数据..."

        refreshed_data = {}
        for i, code in enumerate(all_codes):
            # Try to download fresh data (will overwrite cache with latest)
            df = download_one_stock(code)
            if df is not None and not df.empty:
                refreshed_data[code] = df
            elif code in cached_data:
                refreshed_data[code] = cached_data[code]

            if (i + 1) % 500 == 0:
                _prediction_state["progress"] = i + 1
                _prediction_state["message"] = (
                    f"数据更新中... ({i + 1}/{len(all_codes)})"
                )

        _prediction_state["progress"] = len(all_codes)
        _prediction_state["message"] = "正在加载分析模型..."

        # Step 2: Load surge profile
        if not os.path.exists(ANALYSIS_RESULT):
            _prediction_state["error"] = "请先运行分析 (python run.py analyze)"
            _prediction_state["running"] = False
            return

        surge_df = pd.read_csv(ANALYSIS_RESULT)
        surge_stats = build_feature_profile(surge_df)
        feature_weights = compute_feature_importance(surge_df)

        # Step 3: Screen current market
        _prediction_state["message"] = f"正在筛选 {len(refreshed_data)} 只股票..."
        _prediction_state["progress"] = 0
        _prediction_state["total"] = len(refreshed_data)

        results = []
        codes_list = list(refreshed_data.keys())
        for i, code in enumerate(codes_list):
            df = refreshed_data[code]
            if df is None or len(df) < PRE_SURGE_WINDOW:
                continue

            features = extract_features_for_window(df, len(df))
            if features is None:
                continue

            from predictor import compute_similarity_score
            score = compute_similarity_score(features, surge_stats, feature_weights)

            name = code_to_name.get(code, "")
            last_close = float(df["close"].iloc[-1]) if len(df) > 0 else 0

            results.append({
                "code": code,
                "name": name,
                "score": score,
                "last_close": last_close,
                "last_date": str(df.index[-1]) if len(df) > 0 else None,
                **features,
            })

            if (i + 1) % 500 == 0:
                _prediction_state["progress"] = i + 1
                _prediction_state["message"] = (
                    f"筛选进度: {i + 1}/{len(codes_list)} | "
                    f"已找到 {len(results)} 只候选"
                )

        _prediction_state["progress"] = len(codes_list)

        # Step 4: Sort and save
        result_df = pd.DataFrame(results).sort_values("score", ascending=False)
        result_df = result_df.head(50).reset_index(drop=True)
        result_df.to_csv(PREDICTION_RESULT, index=False)

        _prediction_state["message"] = (
            f"预测完成! 共 {len(result_df)} 只候选股票, "
            f"高匹配 {len(result_df[result_df['score'] > 70])} 只"
        )
        _prediction_state["finished_at"] = datetime.now().isoformat()
        _prediction_state["running"] = False

    except Exception as e:
        _prediction_state["error"] = str(e)
        _prediction_state["message"] = f"预测失败: {e}"
        _prediction_state["running"] = False


if __name__ == "__main__":
    ensure_proxy()

    # Preload data
    print("[*] Loading data...")
    surge_data = load_surge_data()
    pred_data = load_predictions()
    print(f"    Surge events: {len(surge_data)}")
    print(f"    Predictions: {len(pred_data)}")

    print("\n[*] Starting web server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
