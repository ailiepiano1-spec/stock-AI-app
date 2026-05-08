import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
import os

st.set_page_config(
    page_title="AI株価分析アプリ",
    layout="wide"
)
def get_display_name(ticker):

    try:
        stock = yf.Ticker(ticker)

        return stock.info.get(
            "shortName",
            ticker
        )

    except:

        return ticker


def load_favorites():

    if os.path.exists("favorites.txt"):

        with open(
            "favorites.txt",
            "r",
            encoding="utf-8"
        ) as f:

            return f.read().strip()

    return "7203.T,6758.T,NVDA"


def save_favorites(text):

    with open(
        "favorites.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)

st.title("AI株価分析")
backtest_url = "https://stock-ai-app-4app97coiy8bdxmc9poybk6.streamlit.app/"

st.link_button(
    "AI予測精度評価",
    backtest_url
)
st.write("yfinance version:", yf.__version__)

# =========================
# サイドバー
# =========================
with st.sidebar:
    st.header("設定")
    if st.button("更新"):
         st.rerun()

    st.subheader("お気に入り銘柄")

    favorites_text = st.text_area(
        "お気に入り銘柄コード",
        value=load_favorites(),
        height=100
    )

    if st.button("お気に入りを保存"):
        save_favorites(favorites_text)
        st.success("お気に入りを保存しました")

    tickers = [
        t.strip()
        for t in favorites_text.split(",")
        if t.strip()
    ]

    period = st.selectbox(
        "表示期間",
        ["1mo", "3mo", "6mo", "1y"],
        index=1
    )

def download_daily(ticker, period):
    return yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

def download_daily(ticker, period):

    return yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )
tab1, tab2, tab3, tab4, tab5, tab6= st.tabs([
    "分析結果",
    "ローソク足",
    "ニュース",
    "急騰ランキング",
    "疑似AI分析",
    "今日見るべき銘柄"
])

# =========================
# 分析結果タブ
# =========================
with tab1:
    line_fig = go.Figure()

    for ticker in tickers:
        display_name = get_display_name(ticker)

        data = download_daily(ticker, period)

        if data.empty:
            st.warning(f"{display_name} のデータが取れませんでした")
            continue

        close_prices = data[["Close"]]
        ma5 = close_prices.rolling(5).mean()

        latest_close = close_prices.iloc[-1].values[0]

        try:
            stock = yf.Ticker(ticker)
            fast_price = stock.fast_info.get("last_price")

            if fast_price:
                latest_close = fast_price

        except:
            pass

        latest_ma5 = ma5.iloc[-1].values[0]

        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices.values

        model = LinearRegression()
        model.fit(X, y)

        predicted_price = model.predict(np.array([[len(close_prices)]]))[0][0]
        change_rate = ((predicted_price - latest_close) / latest_close) * 100

        if latest_close > latest_ma5:
            signal = "買い傾向"
            icon = "🟢"
            trend = "上昇傾向"
        else:
            signal = "売り傾向"
            icon = "🔴"
            trend = "下落傾向"

        with st.container(border=True):
            st.subheader(f"{icon} {display_name}")
            st.caption(ticker)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("直近終値", f"{latest_close:.2f}")
            col2.metric("5日平均", f"{latest_ma5:.2f}")
            col3.metric("AI予測", f"{predicted_price:.2f}", f"{change_rate:.2f}%")
            col4.metric("判定", signal)

            st.info(
                f"{display_name} は5日平均線との比較では **{trend}** です。"
                f"AI予測では次回価格を **{predicted_price:.2f}** と見ています。"
            )

        line_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=close_prices["Close"].values.flatten(),
                mode="lines",
                name=display_name
            )
        )

    line_fig.update_layout(
        title="複数銘柄の株価比較",
        xaxis_title="日付",
        yaxis_title="株価",
        hovermode="x unified",
        height=550
    )

    st.plotly_chart(line_fig, use_container_width=True)

# =========================
# ローソク足タブ
# =========================
with tab2:
    selected_ticker = st.selectbox(
        "ローソク足で見る銘柄",
        tickers,
        key="candle_ticker_select"
    )

    selected_name = get_display_name(selected_ticker)

    candle_data = download_daily(
    selected_ticker,
    period
    )

    if candle_data.empty:
        st.warning(f"{selected_name} のデータが取れませんでした")

    else:
        candle_data["MA5"] = candle_data["Close"].rolling(5).mean()
        candle_data["MA25"] = candle_data["Close"].rolling(25).mean()

        latest_ma5 = candle_data["MA5"].iloc[-1]
        latest_ma25 = candle_data["MA25"].iloc[-1]
        prev_ma5 = candle_data["MA5"].iloc[-2]
        prev_ma25 = candle_data["MA25"].iloc[-2]

        if prev_ma5 <= prev_ma25 and latest_ma5 > latest_ma25:
            cross_signal = "🟢 ゴールデンクロス発生：上昇開始サインかも"
        elif prev_ma5 >= prev_ma25 and latest_ma5 < latest_ma25:
            cross_signal = "🔴 デッドクロス発生：下落開始サインかも"
        elif latest_ma5 > latest_ma25:
            cross_signal = "🟢 MA5がMA25より上：上昇トレンド継続っぽい"
        else:
            cross_signal = "🔴 MA5がMA25より下：下落トレンド継続っぽい"

        # RSI
        delta = candle_data["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        candle_data["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = candle_data["Close"].ewm(span=12).mean()
        ema26 = candle_data["Close"].ewm(span=26).mean()
        candle_data["MACD"] = ema12 - ema26
        candle_data["Signal"] = candle_data["MACD"].ewm(span=9).mean()

        candle_fig = go.Figure()

        candle_fig.add_trace(
            go.Candlestick(
                x=candle_data.index,
                open=candle_data["Open"].values.flatten(),
                high=candle_data["High"].values.flatten(),
                low=candle_data["Low"].values.flatten(),
                close=candle_data["Close"].values.flatten(),
                name="ローソク足"
            )
        )

        candle_fig.add_trace(
            go.Scatter(
                x=candle_data.index,
                y=candle_data["MA5"].values.flatten(),
                mode="lines",
                name="MA5"
            )
        )

        candle_fig.add_trace(
            go.Scatter(
                x=candle_data.index,
                y=candle_data["MA25"].values.flatten(),
                mode="lines",
                name="MA25"
            )
        )

        candle_fig.update_layout(
            title=f"{selected_name} ローソク足チャート",
            xaxis_title="日付",
            yaxis_title="株価",
            height=650,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(candle_fig, use_container_width=True)

        st.subheader("クロス判定")
        st.info(cross_signal)

        volume_fig = go.Figure()
        volume_fig.add_trace(
            go.Bar(
                x=candle_data.index,
                y=candle_data["Volume"].values.flatten(),
                name="出来高"
            )
        )

        volume_fig.update_layout(
            title=f"{selected_name} 出来高",
            xaxis_title="日付",
            yaxis_title="出来高",
            height=300
        )

        st.plotly_chart(volume_fig, use_container_width=True)

        st.subheader("RSI")
        latest_rsi = candle_data["RSI"].iloc[-1]

        st.metric("現在のRSI", f"{latest_rsi:.2f}")

        if latest_rsi >= 70:
            st.error("買われすぎ気味かも")
        elif latest_rsi <= 30:
            st.success("売られすぎ気味かも")
        else:
            st.info("通常レンジ")

        macd_fig = go.Figure()

        macd_fig.add_trace(
            go.Scatter(
                x=candle_data.index,
                y=candle_data["MACD"].values.flatten(),
                mode="lines",
                name="MACD"
            )
        )

        macd_fig.add_trace(
            go.Scatter(
                x=candle_data.index,
                y=candle_data["Signal"].values.flatten(),
                mode="lines",
                name="Signal"
            )
        )

        macd_fig.update_layout(
            title=f"{selected_name} MACD",
            xaxis_title="日付",
            yaxis_title="MACD",
            height=300
        )

        st.plotly_chart(macd_fig, use_container_width=True)

        latest_macd = candle_data["MACD"].iloc[-1]
        latest_signal = candle_data["Signal"].iloc[-1]

        if latest_macd > latest_signal:
            st.success("MACD的には上昇サインっぽい")
        else:
            st.error("MACD的には下落サインっぽい")

# =========================
# ニュースタブ
# =========================
with tab3:
    for ticker in tickers:
        display_name = get_display_name(ticker)

        st.subheader(f" {display_name} 関連ニュース")
        st.caption(ticker)

        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            if news:
                for item in news[:5]:
                    title = (
                        item.get("title")
                        or item.get("content", {}).get("title")
                        or "タイトルなし"
                    )

                    link = (
                        item.get("link")
                        or item.get("content", {})
                            .get("canonicalUrl", {})
                            .get("url")
                        or ""
                    )

                    publisher = (
                        item.get("publisher")
                        or item.get("content", {})
                            .get("provider", {})
                            .get("displayName")
                        or "配信元不明"
                    )

                    st.markdown(f"- [{title}]({link})")
                    st.caption(f"配信元: {publisher}")

            else:
                st.write("ニュースは取得できませんでした。")

        except Exception as e:
            st.write("ニュース取得でエラーが出ました。")
            st.code(str(e))

# =========================
# 急騰ランキングタブ
# =========================
with tab4:
    st.subheader("急騰ランキング")

    ranking_data = []

    for ticker in tickers:
        display_name = get_display_name(ticker)
        data = download_daily(ticker,"5d")

        if data.empty or len(data) < 2:
            continue

        close_list = data["Close"].values.flatten()

        latest_close = close_list[-1]
        previous_close = close_list[-2]

        change_rate = ((latest_close - previous_close) / previous_close) * 100

        ranking_data.append({
            "銘柄": display_name,
            "コード": ticker,
            "前日終値": previous_close,
            "直近終値": latest_close,
            "上昇率": change_rate
        })

    ranking_data = sorted(
        ranking_data,
        key=lambda x: x["上昇率"],
        reverse=True
    )

    if ranking_data:
        for i, item in enumerate(ranking_data, start=1):
            if i == 1:
                medal = "🥇"
            elif i == 2:
                medal = "🥈"
            elif i == 3:
                medal = "🥉"
            else:
                medal = f"{i}位"

            with st.container(border=True):
                st.subheader(f"{medal} {item['銘柄']}")
                st.caption(item["コード"])

                col1, col2, col3 = st.columns(3)

                col1.metric("前日終値", f"{item['前日終値']:.2f}")
                col2.metric("直近終値", f"{item['直近終値']:.2f}")
                col3.metric("上昇率", f"{item['上昇率']:.2f}%")

    else:
        st.write("ランキングを作成できませんでした。")

# =========================
# 疑似AI分析タブ
# =========================
with tab5:
    st.subheader("疑似AIニュース分析")

    positive_words = [
        "上方修正", "増益", "最高益", "増配", "好調",
        "成長", "黒字", "受注", "提携", "買収",
        "新製品", "拡大", "過去最高", "record", "growth",
        "profit", "beats", "upgrade"
    ]

    negative_words = [
        "下方修正", "減益", "赤字", "減配", "不祥事",
        "訴訟", "不正", "リコール", "撤退", "損失",
        "悪化", "低迷", "急落", "loss", "misses",
        "downgrade", "lawsuit", "recall"
    ]

    for ticker in tickers:
        display_name = get_display_name(ticker)

        st.subheader(f"{display_name} 分析")
        st.caption(ticker)

        score = 0
        found_positive = []
        found_negative = []
        news_titles = []

        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            for item in news[:10]:
                title = (
                    item.get("title")
                    or item.get("content", {}).get("title")
                    or ""
                )

                news_titles.append(title)

                for word in positive_words:
                    if word.lower() in title.lower():
                        score += 1
                        found_positive.append(word)

                for word in negative_words:
                    if word.lower() in title.lower():
                        score -= 1
                        found_negative.append(word)

        except:
            st.write("ニュース取得でエラーが出ました。")
            continue

        if score >= 2:
            st.success("ニュース面：強気材料が多め")
        elif score <= -2:
            st.error("ニュース面：弱気材料が多め")
        else:
            st.info("ニュース面：中立")

        col1, col2, col3 = st.columns(3)
        col1.metric("ニューススコア", score)
        col2.metric("強気ワード数", len(found_positive))
        col3.metric("弱気ワード数", len(found_negative))

        st.write("検出した強気ワード")
        st.write(found_positive if found_positive else "なし")

        st.write("検出した弱気ワード")
        st.write(found_negative if found_negative else "なし")

        # 総合AI判定
        total_score = score
        reasons = []

        if score >= 2:
            reasons.append("ニュースは強気材料多め")
        elif score <= -2:
            reasons.append("ニュースは弱気材料多め")

        try:
            data = download_daily(ticker,period)

            data["MA5"] = data["Close"].rolling(5).mean()
            data["MA25"] = data["Close"].rolling(25).mean()

            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()

            rs = avg_gain / avg_loss
            data["RSI"] = 100 - (100 / (1 + rs))

            ema12 = data["Close"].ewm(span=12).mean()
            ema26 = data["Close"].ewm(span=26).mean()

            data["MACD"] = ema12 - ema26
            data["Signal"] = data["MACD"].ewm(span=9).mean()

            latest_rsi = data["RSI"].iloc[-1]
            latest_macd = data["MACD"].iloc[-1]
            latest_signal = data["Signal"].iloc[-1]
            latest_ma5 = data["MA5"].iloc[-1]
            latest_ma25 = data["MA25"].iloc[-1]

            if latest_rsi < 30:
                total_score += 2
                reasons.append("RSIが低く反発期待")
            elif latest_rsi > 70:
                total_score -= 2
                reasons.append("RSIが高く過熱気味")

            if latest_macd > latest_signal:
                total_score += 2
                reasons.append("MACDが上昇サイン")
            else:
                total_score -= 2
                reasons.append("MACDが下落サイン")

            if latest_ma5 > latest_ma25:
                total_score += 2
                reasons.append("短期線が長期線を上回る")
            else:
                total_score -= 2
                reasons.append("短期線が長期線を下回る")

        except:
            reasons.append("テクニカル指標の一部取得に失敗")

        st.subheader("総合AI判定")

        if total_score >= 5:
            st.success("かなり強気")
        elif total_score >= 2:
            st.info("やや強気")
        elif total_score <= -5:
            st.error("かなり弱気")
        elif total_score <= -2:
            st.warning("やや弱気")
        else:
            st.write("中立")

        st.metric("AI総合スコア", total_score)

        st.write("判定理由")
        if reasons:
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("明確な判定理由なし")

        st.write("取得ニュース")
        for title in news_titles:
            st.write(f"- {title}")

with tab6:

    st.subheader("👀 今日見るべき銘柄ランキング")

    watch_data = []

    for ticker in tickers:

        display_name = get_display_name(ticker)

        try:
            data = download_daily(ticker, period)

            if data.empty or len(data) < 30:
                continue

            close = data["Close"].values.flatten()
            volume = data["Volume"].values.flatten()

            latest_close = close[-1]
            prev_close = close[-2]

            change_rate = ((latest_close - prev_close) / prev_close) * 100

            ma5 = data["Close"].rolling(5).mean().iloc[-1].item()
            ma25 = data["Close"].rolling(25).mean().iloc[-1].item()

            delta = data["Close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = rsi.iloc[-1].item()

            ema12 = data["Close"].ewm(span=12).mean()
            ema26 = data["Close"].ewm(span=26).mean()

            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()

            latest_macd = macd.iloc[-1].item()
            latest_signal = signal.iloc[-1].item()

            score = 0
            reasons = []

            if change_rate > 0:
                score += 1
                reasons.append("前日比プラス")

            if change_rate >= 3:
                score += 2
                reasons.append("大きく上昇")

            if ma5 > ma25:
                score += 2
                reasons.append("短期線が長期線を上回る")

            if latest_macd > latest_signal:
                score += 2
                reasons.append("MACDが上昇サイン")

            if latest_rsi <= 30:
                score += 2
                reasons.append("RSIが低く反発期待")

            elif latest_rsi >= 70:
                score -= 2
                reasons.append("RSIが高く過熱気味")

            if volume[-1] > np.mean(volume[-10:]):
                score += 1
                reasons.append("出来高が増加")

            watch_data.append({
                "銘柄": display_name,
                "コード": ticker,
                "スコア": score,
                "前日比": change_rate,
                "RSI": latest_rsi,
                "理由": reasons
            })

        except Exception as e:
            st.write(f"{ticker} でエラー")
            st.code(str(e))
            continue

    watch_data = sorted(
        watch_data,
        key=lambda x: x["スコア"],
        reverse=True
    )

    if watch_data:

        for i, item in enumerate(watch_data, start=1):

            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}位"

            with st.container(border=True):

                st.subheader(f"{medal} {item['銘柄']}")
                st.caption(item["コード"])

                col1, col2, col3 = st.columns(3)

                col1.metric("AI注目スコア", item["スコア"])
                col2.metric("前日比", f"{item['前日比']:.2f}%")
                col3.metric("RSI", f"{item['RSI']:.2f}")

                st.write("注目理由")

                if item["理由"]:
                    for reason in item["理由"]:
                        st.write(f"- {reason}")
                else:
                    st.write("明確な注目理由なし")

    else:
        st.write("注目銘柄を作成できませんでした。")