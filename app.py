import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Ticker Advisor", page_icon="📈", layout="wide")


# ---------- Indicators ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, 12) - ema(series, 26)
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------- Pattern helpers ----------
def detect_candle(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "Not enough data"
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last["Close"] - last["Open"])
    full = max(last["High"] - last["Low"], 1e-9)
    lower_wick = min(last["Open"], last["Close"]) - last["Low"]
    upper_wick = last["High"] - max(last["Open"], last["Close"])

    if last["Close"] > last["Open"] and prev["Close"] < prev["Open"]:
        if last["Open"] <= prev["Close"] and last["Close"] >= prev["Open"]:
            return "Bullish engulfing"
    if last["Close"] < last["Open"] and prev["Close"] > prev["Open"]:
        if last["Open"] >= prev["Close"] and last["Close"] <= prev["Open"]:
            return "Bearish engulfing"
    if lower_wick > body * 2 and upper_wick < body:
        return "Hammer-like"
    if upper_wick > body * 2 and lower_wick < body:
        return "Shooting-star-like"
    if body / full < 0.15:
        return "Doji-like"
    return "No major single-candle signal"


def detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 30:
        return "Not enough data"
    recent = df.tail(20)
    close = recent["Close"]
    high = recent["High"]
    low = recent["Low"]

    if close.iloc[-1] > high.iloc[:-1].max():
        return "Breakout above 20-day range"
    if close.iloc[-1] < low.iloc[:-1].min():
        return "Breakdown below 20-day range"

    range_pct = (high.max() - low.min()) / max(close.mean(), 1e-9)
    slope = np.polyfit(range(len(close)), close.values, 1)[0]
    if range_pct < 0.06 and slope > 0:
        return "Tight bullish consolidation"
    if range_pct < 0.06 and slope < 0:
        return "Tight bearish consolidation"
    return "No clear simple pattern"


# ---------- External data ----------
def get_finnhub_key() -> str:
    try:
        return st.secrets.get("FINNHUB_API_KEY", "").strip()
    except Exception:
        return ""


def get_finnhub_json(endpoint: str, params: dict[str, Any]) -> Any:
    key = get_finnhub_key()
    if not key:
        return None
    base = f"https://finnhub.io/api/v1/{endpoint}"
    payload = dict(params)
    payload["token"] = key
    try:
        resp = requests.get(base, params=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_market_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    hist = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist.dropna(how="all")


def market_trend(symbol: str) -> str:
    data = get_market_history(symbol, period="6mo")
    if data.empty or len(data) < 50:
        return "Unknown"
    close = data["Close"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    last = close.iloc[-1]
    if last > sma20 > sma50:
        return "Up"
    if last < sma20 < sma50:
        return "Down"
    return "Mixed"


@st.cache_data(ttl=900)
def analyze_ticker(ticker: str) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Please enter a ticker.")

    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y", interval="1d", auto_adjust=True)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.dropna(how="all")

    if hist.empty or len(hist) < 60:
        raise ValueError(f"Not enough data found for {ticker}.")

    info = tk.fast_info or {}
    close = hist["Close"]
    volume = hist["Volume"]

    price = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(50).mean().iloc[-1])
    rsi14 = float(rsi(close).iloc[-1])
    macd_line, signal_line, hist_line = macd(close)
    atr14 = float(atr(hist).iloc[-1])
    rel_volume = float(volume.iloc[-1] / max(volume.tail(50).mean(), 1))
    candle = detect_candle(hist)
    pattern = detect_pattern(hist)

    trend = "Uptrend" if price > sma20 > sma50 else "Downtrend" if price < sma20 < sma50 else "Mixed"

    spy = market_trend("SPY")
    qqq = market_trend("QQQ")
    dxy = market_trend("DX-Y.NYB")
    tnx = market_trend("^TNX")

    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    # Macro
    if spy == "Up" and qqq in {"Up", "Mixed"}:
        score += 1.0
        reasons.append("Broad market backdrop is supportive.")
    elif spy == "Down" and qqq == "Down":
        score -= 1.0
        reasons.append("Broad market backdrop is weak.")

    if dxy == "Down":
        score += 0.5
        reasons.append("DXY is easing, which is friendlier for risk assets.")
    elif dxy == "Up":
        score -= 0.5
        warnings.append("DXY is rising, which can pressure risk assets.")

    if tnx == "Down":
        score += 0.5
        reasons.append("10Y yield is falling, which supports growth/risk appetite.")
    elif tnx == "Up":
        score -= 0.5
        warnings.append("10Y yield is rising, which can weigh on equities.")

    # Technical
    if trend == "Uptrend":
        score += 2.0
        reasons.append("Price is above key moving averages in an uptrend.")
    elif trend == "Downtrend":
        score -= 2.0
        reasons.append("Price is below key moving averages in a downtrend.")

    if 45 <= rsi14 <= 68:
        score += 1.5
        reasons.append("RSI is in a constructive momentum zone.")
    elif rsi14 > 75:
        score -= 1.0
        warnings.append("RSI is stretched on the upside.")
    elif rsi14 < 35:
        score -= 0.5
        warnings.append("RSI is weak and may still be in a bearish regime.")

    if macd_line.iloc[-1] > signal_line.iloc[-1] and hist_line.iloc[-1] > 0:
        score += 1.5
        reasons.append("MACD is bullish.")
    elif macd_line.iloc[-1] < signal_line.iloc[-1] and hist_line.iloc[-1] < 0:
        score -= 1.5
        reasons.append("MACD is bearish.")

    if rel_volume >= 1.2:
        score += 1.0
        reasons.append("Relative volume is above average.")
    elif rel_volume < 0.8:
        score -= 0.5
        warnings.append("Relative volume is light.")

    if "Breakout" in pattern or "bullish" in pattern.lower() or "Bullish" in candle:
        score += 1.0
        reasons.append(f"Pattern support: {pattern}; candle: {candle}.")
    if "Breakdown" in pattern or "Bearish" in candle:
        score -= 1.0
        warnings.append(f"Pattern caution: {pattern}; candle: {candle}.")

    # News / analysts / insiders
    analyst_text = "Unavailable"
    analyst_count_summary = None
    finnhub_rec = get_finnhub_json("stock/recommendation", {"symbol": ticker})
    if isinstance(finnhub_rec, list) and finnhub_rec:
        latest = finnhub_rec[0]
        buy = latest.get("buy", 0)
        hold = latest.get("hold", 0)
        sell = latest.get("sell", 0)
        analyst_count_summary = {"buy": buy, "hold": hold, "sell": sell}
        analyst_text = f"Buy {buy} / Hold {hold} / Sell {sell}"
        if buy > sell:
            score += 0.5
            reasons.append("Analyst recommendation trend leans positive.")
        elif sell > buy:
            score -= 0.5
            warnings.append("Analyst recommendation trend leans negative.")

    insider_text = "Unavailable"
    insider = get_finnhub_json("stock/insider-sentiment", {"symbol": ticker, "from": f"{datetime.now().year - 1}-01-01"})
    if insider and insider.get("data"):
        item = insider["data"][-1]
        mspr = item.get("mspr")
        insider_text = f"MSPR {mspr}"
        if isinstance(mspr, (int, float)):
            if mspr > 0:
                score += 0.5
                reasons.append("Insider sentiment is positive.")
            elif mspr < 0:
                score -= 0.5
                warnings.append("Insider sentiment is negative.")

    # News
    news: list[dict[str, str]] = []
    for item in (tk.news or [])[:5]:
        title = item.get("title") or item.get("content", {}).get("title") or "Untitled"
        link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url") or ""
        publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName") or "Source"
        news.append({"title": title, "publisher": publisher, "link": link})

    if not news:
        finnhub_news = get_finnhub_json(
            "company-news",
            {
                "symbol": ticker,
                "from": (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat(),
                "to": datetime.now(timezone.utc).date().isoformat(),
            },
        )
        if isinstance(finnhub_news, list):
            for item in finnhub_news[:5]:
                news.append(
                    {
                        "title": item.get("headline", "Untitled"),
                        "publisher": item.get("source", "Source"),
                        "link": item.get("url", ""),
                    }
                )

    if score >= 4:
        recommendation = "BUY"
        recommendation_note = "Setup looks constructive for a short-term long bias."
    elif score <= -2:
        recommendation = "SELL / AVOID"
        recommendation_note = "Setup is weak or bearish. Better to avoid or stay on the defensive."
    else:
        recommendation = "NO ACTION / WATCH"
        recommendation_note = "Mixed setup. Wait for clearer confirmation."

    entry = price
    stop = price - atr14 * 1.2 if recommendation == "BUY" else price + atr14 * 1.2
    target = price + atr14 * 2.4 if recommendation == "BUY" else price - atr14 * 2.4
    rr = abs((target - entry) / max(abs(entry - stop), 1e-9))

    chart_hint = f"https://www.tradingview.com/search/?query={ticker}"
    quote_type = info.get("quoteType")
    exchange = info.get("exchange")
    if quote_type == "EQUITY" and exchange in {"NMS", "NAS", "NASDAQ"}:
        chart_hint = f"https://www.tradingview.com/symbols/NASDAQ-{ticker}/"
    elif quote_type == "EQUITY" and exchange in {"NYQ", "NYE", "NYSE"}:
        chart_hint = f"https://www.tradingview.com/symbols/NYSE-{ticker}/"

    return {
        "ticker": ticker,
        "long_name": info.get("longName") or ticker,
        "price": price,
        "trend": trend,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi14,
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "atr": atr14,
        "rel_volume": rel_volume,
        "candle": candle,
        "pattern": pattern,
        "spy": spy,
        "qqq": qqq,
        "dxy": dxy,
        "tnx": tnx,
        "analyst": analyst_text,
        "analyst_counts": analyst_count_summary,
        "insider": insider_text,
        "score": score,
        "recommendation": recommendation,
        "recommendation_note": recommendation_note,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": rr,
        "reasons": reasons,
        "warnings": warnings,
        "news": news,
        "sources": {
            "Yahoo Finance": f"https://finance.yahoo.com/quote/{ticker}",
            "TradingView": chart_hint,
            "SEC filings": f"https://www.sec.gov/edgar/search/#/q={ticker}",
        },
    }


# ---------- UI ----------
st.title("📈 Ticker Advisor")
st.caption("Enter a ticker only. The app fetches data and returns Buy, Sell/Avoid, or No Action.")

with st.sidebar:
    st.subheader("Deploy note")
    st.write("Works with Yahoo Finance out of the box. Add `FINNHUB_API_KEY` in Streamlit secrets if you want richer analyst, news, and insider data.")
    st.code('FINNHUB_API_KEY = "your_key_here"', language="toml")
    st.write("For Streamlit Cloud, place that in `.streamlit/secrets.toml` or the app Secrets panel.")

col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input("Ticker", placeholder="AAPL").strip().upper()
with col2:
    analyze = st.button("Analyze", use_container_width=True, type="primary")

if analyze:
    try:
        result = analyze_ticker(ticker)

        if result["recommendation"] == "BUY":
            st.success(f"{result['ticker']}: {result['recommendation']}")
        elif "SELL" in result["recommendation"]:
            st.error(f"{result['ticker']}: {result['recommendation']}")
        else:
            st.warning(f"{result['ticker']}: {result['recommendation']}")
        st.write(result["recommendation_note"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"{result['price']:.2f}")
        m2.metric("Trend", result["trend"])
        m3.metric("Score", f"{result['score']:.1f}")
        m4.metric("Risk/Reward", f"{result['rr']:.2f}")

        st.subheader("Why this call")
        if result["reasons"]:
            for item in result["reasons"]:
                st.write(f"- {item}")
        else:
            st.write("- No strong positive reasons were detected.")

        if result["warnings"]:
            st.subheader("Warnings")
            for item in result["warnings"]:
                st.write(f"- {item}")

        left, right = st.columns(2)
        with left:
            st.subheader("Technical snapshot")
            st.write(f"RSI: {result['rsi']:.1f}")
            st.write(f"MACD / signal: {result['macd']:.3f} / {result['macd_signal']:.3f}")
            st.write(f"Relative volume: {result['rel_volume']:.2f}x")
            st.write(f"Candle: {result['candle']}")
            st.write(f"Pattern: {result['pattern']}")
            st.write(f"SMA20 / SMA50 / SMA200: {result['sma20']:.2f} / {result['sma50']:.2f} / {result['sma200']:.2f}")
        with right:
            st.subheader("Macro + sentiment")
            st.write(f"SPY trend: {result['spy']}")
            st.write(f"QQQ trend: {result['qqq']}")
            st.write(f"DXY trend: {result['dxy']}")
            st.write(f"10Y yield trend: {result['tnx']}")
            st.write(f"Analysts: {result['analyst']}")
            st.write(f"Insiders: {result['insider']}")

        st.subheader("Trade plan")
        t1, t2, t3 = st.columns(3)
        t1.metric("Entry", f"{result['entry']:.2f}")
        t2.metric("Stop", f"{result['stop']:.2f}")
        t3.metric("Target", f"{result['target']:.2f}")

        st.subheader("Price chart")
        chart_data = get_market_history(result["ticker"], period="6mo")
        st.line_chart(chart_data["Close"])

        st.subheader("Recent headlines")
        if result["news"]:
            for item in result["news"]:
                if item.get("link"):
                    st.markdown(f"- [{item['title']}]({item['link']}) — {item['publisher']}")
                else:
                    st.write(f"- {item['title']} — {item['publisher']}")
        else:
            st.write("No recent headlines returned from configured sources.")

        st.subheader("Source links")
        for label, url in result["sources"].items():
            st.markdown(f"- [{label}]({url})")

    except Exception as exc:
        st.error(str(exc))
else:
    st.info("Type a ticker and click Analyze.")
