import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

# =========================
# ページ設定
# =========================

st.set_page_config(
    page_title="AI予測精度評価",
    layout="wide"
)

st.title("AI予測精度評価")

# =========================
# 入力
# =========================

tickers_text = st.text_input(
    "銘柄コード",
    "7203.T,6758.T,NVDA"
)

tickers = [
    t.strip()
    for t in tickers_text.split(",")
    if t.strip()
]

selected_ticker = st.selectbox(
    "評価する銘柄",
    tickers
)

backtest_period = st.selectbox(
    "評価期間",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=3
)

# =========================
# データ取得
# =========================

data = yf.download(
    selected_ticker,
    period=backtest_period,
    interval="1d",
    auto_adjust=True,
    progress=False,
    threads=False
)

# =========================
# 精度評価
# =========================

if data.empty or len(data) < 30:

    st.warning("評価に必要なデータ不足")

else:

    close_prices = data["Close"].values.flatten()

    predictions = []
    actuals = []
    dates = []

    train_window = 10

    for i in range(train_window, len(close_prices) - 1):

        train_y = close_prices[
            i - train_window:i
        ]

        train_x = np.arange(
            train_window
        ).reshape(-1, 1)

        model = LinearRegression()

        model.fit(
            train_x,
            train_y
        )

        next_x = np.array([
            [train_window]
        ])

        predicted = model.predict(
            next_x
        )[0]

        actual = close_prices[i + 1]

        predictions.append(predicted)
        actuals.append(actual)
        dates.append(data.index[i + 1])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # =====================
    # 評価指標
    # =====================

    errors = actuals - predictions

    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)

    mape = np.mean(
        abs_errors / actuals
    ) * 100

    actual_direction = np.sign(
        actuals[1:] - actuals[:-1]
    )

    predicted_direction = np.sign(
        predictions[1:] - actuals[:-1]
    )

    direction_accuracy = (
        np.mean(
            actual_direction
            == predicted_direction
        ) * 100
    )

    # =====================
    # KPI表示
    # =====================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "平均誤差 MAE",
        f"{mae:.2f}"
    )

    col2.metric(
        "平均誤差率 MAPE",
        f"{mape:.2f}%"
    )

    col3.metric(
        "方向一致率",
        f"{direction_accuracy:.2f}%"
    )

    # =====================
    # グラフ
    # =====================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actuals,
            mode="lines",
            name="実績"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predictions,
            mode="lines",
            name="予測"
        )
    )

    fig.update_layout(
        title="予測 vs 実績",
        xaxis_title="日付",
        yaxis_title="株価",
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================
    # 日別結果
    # =====================

    st.subheader("📋 日別予測結果")

    result_data = []

    for i in range(len(dates)):

        diff = (
            actuals[i]
            - predictions[i]
        )

        diff_percent = (
            diff / actuals[i]
        ) * 100

        result_data.append({
            "日付": dates[i].strftime(
                "%Y-%m-%d"
            ),
            "予測値": round(
                predictions[i],
                2
            ),
            "実績値": round(
                actuals[i],
                2
            ),
            "差額": round(
                diff,
                2
            ),
            "誤差率%": round(
                diff_percent,
                2
            )
        })

    st.dataframe(
        result_data,
        use_container_width=True
    )