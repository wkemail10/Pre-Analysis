import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

st.set_page_config(page_title="Ticker Advisor Pro", page_icon="📈", layout="wide")

HEADERS = {"User-Agent": "Mozilla/5.0"}


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


# ---------- Small utilities ----------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_to_rating(score: float) -> tuple[int, str, str, str]:
    rating = int(round(clamp(score, 0, 10)))
    if rating >= 7:
        return rating, "Green", "🟢", "good sign"
    if rating <= 4:
        return rating, "Red", "🔴", "bad sign"
    return rating, "Yellow", "🟡", "neutral"


def render_status_badge(color_name: str, text: str) -> str:
    bg = {
        "Green": "#0f5132",
        "Yellow": "#7a5d00",
        "Red": "#842029",
    }.get(color_name, "#495057")
    return f"<span style='background:{bg}; color:white; padding:6px 10px; border-radius:999px; font-weight:600;'>{text}</span>"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def summarize_lines(lines: list[str], fallback: str) -> str:
    clean = [x.strip() for x in lines if x and x.strip()]
    return " ".join(clean[:2]) if clean else fallback


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
    if lower_wick > body * 2 and upper_wick < max(body, 0.01):
        return "Hammer-like"
    if upper_wick > body * 2 and lower_wick < max(body, 0.01):
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
        resp = requests.get(base, params=payload, headers=HEADERS, timeout=20)
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


@st.cache_data(ttl=3600)
def get_google_news(query: str, limit: int = 6) -> list[dict[str, str]]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    items: list[dict[str, str]] = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")
        for item in soup.find_all("item")[:limit]:
            items.append(
                {
                    "title": item.title.text if item.title else "Untitled",
                    "link": item.link.text if item.link else "",
                    "publisher": item.source.text if item.source else "Google News",
                    "pubDate": item.pubDate.text if item.pubDate else "",
                }
            )
    except Exception:
        return []
    return items


@st.cache_data(ttl=3600)
def get_insider_dashboard_summary(ticker: str) -> dict[str, Any]:
    url = f"https://www.insiderdashboard.com/search?query={ticker}"
    result: dict[str, Any] = {
        "status": "Unavailable",
        "summary": "Could not read Insider Dashboard.",
        "filings": [],
        "url": url,
    }
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        text = resp.text
        soup = BeautifulSoup(text, "html.parser")
        plain = soup.get_text(" ", strip=True)

        if "No insider transaction data found" in plain:
            result["status"] = "Neutral"
            result["summary"] = "No insider transaction data found on Insider Dashboard."
            return result

        if "No insider buys found" in plain:
            result["status"] = "Red"
            result["summary"] = "No insider buys found on Insider Dashboard."

        filings: list[dict[str, str]] = []
        current_form = None
        current_date = None
        current_score = None
        for line in [x.strip() for x in soup.get_text("\n").splitlines() if x.strip()]:
            if line == "Form 4":
                current_form = line
                continue
            if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", line):
                current_date = line
                continue
            if re.fullmatch(r"\d{1,3}", line):
                val = int(line)
                if 0 <= val <= 100:
                    current_score = f"{val}%"
                    continue
            if line.startswith("Summary:"):
                summary = line.replace("Summary:", "").strip()
                filings.append(
                    {
                        "form": current_form or "Form 4",
                        "date": current_date or "",
                        "impact": current_score or "",
                        "summary": summary,
                    }
                )
                current_form = None
                current_date = None
                current_score = None
                if len(filings) >= 3:
                    break

        if filings:
            result["filings"] = filings
            joined = " ".join([f"{f['date']}: {f['summary']}" for f in filings[:2]])
            result["summary"] = joined
            if any("buy" in f["summary"].lower() for f in filings):
                result["status"] = "Green"
            elif any("award" in f["summary"].lower() or "rsu" in f["summary"].lower() for f in filings):
                result["status"] = result["status"] if result["status"] != "Unavailable" else "Yellow"
            else:
                result["status"] = result["status"] if result["status"] != "Unavailable" else "Yellow"
        elif result["status"] == "Unavailable":
            result["summary"] = "Insider Dashboard page loaded, but no easy public filing summary was parsed."

    except Exception as exc:
        result["summary"] = f"Insider Dashboard fetch failed: {exc}"

    return result


# ---------- Scoring / levels ----------
def compute_support_resistance(hist: pd.DataFrame, price: float, atr14: float) -> dict[str, float]:
    recent = hist.tail(30)
    highs = recent["High"]
    lows = recent["Low"]

    recent_highs = sorted(set(round(x, 2) for x in highs.nlargest(5).tolist()))
    recent_lows = sorted(set(round(x, 2) for x in lows.nsmallest(5).tolist()))

    resistance_candidates = [x for x in recent_highs if x > price]
    support_candidates = [x for x in recent_lows if x < price]

    r1 = resistance_candidates[0] if resistance_candidates else round(price + atr14, 2)
    r2 = resistance_candidates[1] if len(resistance_candidates) > 1 else round(r1 + atr14 * 0.8, 2)
    s1 = support_candidates[-1] if support_candidates else round(price - atr14, 2)
    s2 = support_candidates[-2] if len(support_candidates) > 1 else round(s1 - atr14 * 0.8, 2)

    return {"s1": s1, "s2": s2, "r1": r1, "r2": r2}


def analyze_section_scores(
    ticker: str,
    price: float,
    hist: pd.DataFrame,
    rsi14: float,
    macd_value: float,
    macd_signal: float,
    rel_volume: float,
    candle: str,
    pattern: str,
    trend: str,
    spy: str,
    qqq: str,
    dxy: str,
    tnx: str,
    vix: float,
    analyst_text: str,
    analyst_counts: dict[str, int] | None,
    insider_data: dict[str, Any],
    political_news: list[dict[str, str]],
    economic_news: list[dict[str, str]],
    company_news: list[dict[str, str]],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []

    # Macro & Political
    macro_score = 5.0
    macro_notes: list[str] = []
    if vix and vix < 16:
        macro_score += 1.5
        macro_notes.append(f"VIX is calm at about {vix:.1f}.")
    elif vix and vix > 22:
        macro_score -= 1.5
        macro_notes.append(f"VIX is elevated at about {vix:.1f}.")
    if dxy == "Down":
        macro_score += 1.0
        macro_notes.append("DXY is weakening, which is usually a mild tailwind for risk assets.")
    elif dxy == "Up":
        macro_score -= 1.0
        macro_notes.append("DXY is rising, which can pressure risk assets.")
    if tnx == "Down":
        macro_score += 1.0
        macro_notes.append("10Y yield trend is down, which helps growth names.")
    elif tnx == "Up":
        macro_score -= 1.0
        macro_notes.append("10Y yield trend is up, which can be a headwind.")
    if political_news:
        macro_notes.append(f"Political headline: {political_news[0]['title']}")
    if economic_news:
        macro_notes.append(f"Economic headline: {economic_news[0]['title']}")
    rating, color, emoji, meaning = score_to_rating(macro_score)
    sections.append(
        {
            "section": "Macro & Political",
            "score": rating,
            "color": color,
            "emoji": emoji,
            "meaning": meaning,
            "summary": summarize_lines(macro_notes, "Macro backdrop is mixed."),
            "details": macro_notes,
        }
    )

    # Market context
    market_score = 5.0
    market_notes: list[str] = []
    if spy == "Up":
        market_score += 1.5
        market_notes.append("SPY trend is up.")
    elif spy == "Down":
        market_score -= 1.5
        market_notes.append("SPY trend is down.")
    if qqq == "Up":
        market_score += 1.0
        market_notes.append("QQQ trend is up.")
    elif qqq == "Down":
        market_score -= 1.0
        market_notes.append("QQQ trend is down.")
    if company_news:
        market_notes.append(f"Company/news context: {company_news[0]['title']}")
    rating, color, emoji, meaning = score_to_rating(market_score)
    sections.append(
        {
            "section": "Market Context",
            "score": rating,
            "color": color,
            "emoji": emoji,
            "meaning": meaning,
            "summary": summarize_lines(market_notes, "Market context is mixed."),
            "details": market_notes,
        }
    )

    # Stock trend
    trend_score = 5.0
    trend_notes: list[str] = []
    sma20 = safe_float(hist["Close"].rolling(20).mean().iloc[-1])
    sma50 = safe_float(hist["Close"].rolling(50).mean().iloc[-1])
    sma100 = safe_float(hist["Close"].rolling(100).mean().iloc[-1])
    sma200 = safe_float(hist["Close"].rolling(200).mean().iloc[-1]) if len(hist) >= 200 else sma100
    wk52_high = safe_float(hist["High"].tail(252).max())
    wk52_low = safe_float(hist["Low"].tail(252).min())
    if trend == "Uptrend":
        trend_score += 2.0
        trend_notes.append("Price is above short-term moving averages in an uptrend.")
    elif trend == "Downtrend":
        trend_score -= 2.0
        trend_notes.append("Price is below key moving averages in a downtrend.")
    else:
        trend_notes.append("Trend is mixed, not fully aligned.")
    trend_notes.append(f"52-week range is about ${wk52_low:.2f} to ${wk52_high:.2f}.")
    trend_notes.append(f"Price vs SMAs: 20D ${sma20:.2f}, 50D ${sma50:.2f}, 100D ${sma100:.2f}, 200D ${sma200:.2f}.")
    rating, color, emoji, meaning = score_to_rating(trend_score)
    sections.append(
        {
            "section": "Stock Trend",
            "score": rating,
            "color": color,
            "emoji": emoji,
            "meaning": meaning,
            "summary": summarize_lines(trend_notes, "Trend is mixed."),
            "details": trend_notes,
        }
    )

    # Technical analysis
    technical_score = 5.0
    technical_notes: list[str] = []
    if 50 <= rsi14 <= 68:
        technical_score += 2.0
        technical_notes.append(f"RSI is constructive at {rsi14:.2f}.")
    elif rsi14 > 70:
        technical_score -= 1.5
        technical_notes.append(f"RSI is overbought at {rsi14:.2f}.")
    elif rsi14 < 35:
        technical_score -= 1.5
        technical_notes.append(f"RSI is weak at {rsi14:.2f}.")
    else:
        technical_notes.append(f"RSI is neutral at {rsi14:.2f}.")
    if macd_value > macd_signal:
        technical_score += 1.5
        technical_notes.append(f"MACD is bullish ({macd_value:.2f} above {macd_signal:.2f}).")
    else:
        technical_score -= 1.5
        technical_notes.append(f"MACD is bearish ({macd_value:.2f} below {macd_signal:.2f}).")
    technical_notes.append(f"Candle: {candle}. Pattern: {pattern}.")
    rating, color, emoji, meaning = score_to_rating(technical_score)
    sections.append(
        {
            "section": "Technical Analysis",
            "score": rating,
            "color": color,
            "emoji": emoji,
            "meaning": meaning,
            "summary": summarize_lines(technical_notes, "Technicals are mixed."),
            "details": technical_notes,
        }
    )

    # Volume & smart money
    volume_score = 5.0
    volume_notes: list[str] = []
    avg30 = safe_float(hist["Volume"].tail(30).mean())
    if rel_volume >= 1.2:
        volume_score += 1.5
        volume_notes.append(f"Volume is strong at {hist['Volume'].iloc[-1] / 1_000_000:.2f}M vs 30-day average {avg30 / 1_000_000:.2f}M.")
    elif rel_volume < 0.8:
        volume_score -= 1.0
        volume_notes.append("Volume is below normal, so conviction is weaker.")
    else:
        volume_notes.append("Volume is near average.")
    if price > hist["Close"].tail(10).mean():
        volume_notes.append("Recent price action is holding above short-term average, which supports the move.")
    rating, color, emoji, meaning = score_to_rating(volume_score)
    sections.append(
        {
            "section": "Volume & Smart Money",
            "score": rating,
            "color": color,
            "emoji": emoji,
            "meaning": meaning,
            "summary": summarize_lines(volume_notes, "Volume profile is average."),
            "details": volume_notes,
        }
    )

    # Insider / whale
    insider_score = 5.0
    insider_notes: list[str] = []
    status = insider_data.get("status", "Unavailable")
    insider_notes.append(insider_data.get("summary", "No insider summary available."))
    if status == "Green":
        insider_score += 2.0
    elif status == "Red":
        insider_score -= 1.5
    if analyst_counts:
        insider_notes.append(f"Analyst mix: Buy {analyst_counts.get('buy', 0)}, Hold {analyst_counts.get('hold', 0)}, Sell {analyst_counts.get('sell', 0)}.")
    rating, color, emoji, meaning = score_to_rating(insider_score)
    sections.append(
        {
            "section": "Insider / Whale Activity",
            "score": rating,
            "color": color,
            "emoji": emoji,
            "meaning": meaning,
            "summary": summarize_lines(insider_notes, "Insider data is limited."),
            "details": insider_notes,
        }
    )

    # Sentiment / analysts
    sentiment_score = 5.0
    sentiment_notes: list[str] = []
    if analyst_counts:
        buy = analyst_counts.get("buy", 0)
        sell = analyst_counts.get("sell", 0)
        hold = analyst_counts.get("hold", 0)
        if buy > sell:
            sentiment_score += 2.0
            sentiment_notes.append(f"Analysts lean positive: {analyst_text}.")
        elif sell > buy:
            sentiment_score -= 2.0
            sentiment_notes.append(f"Analysts lean negative: {analyst_text}.")
        else:
            sentiment_notes.append(f"Analyst view is balanced: {analyst_text}.")
    else:
        sentiment_notes.append("No live analyst recommendation feed was available.")
    if company_news:
        sentiment_notes.append(f"Latest company headline: {company_news[0]['title']}")
    rating, color, emoji, meaning = score_to_rating(sentiment_score)
    sections.append(
        {
            "section": "Sentiment & Analysts",
            "score": rating,
            "color": color,
            "emoji": emoji,
            "meaning": meaning,
            "summary": summarize_lines(sentiment_notes, "Sentiment is mixed."),
            "details": sentiment_notes,
        }
    )

    return sections


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
    vix_hist = get_market_history("^VIX", period="3mo")
    vix = safe_float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else 0.0

    # News buckets
    company_news: list[dict[str, str]] = []
    for item in (tk.news or [])[:6]:
        title = item.get("title") or item.get("content", {}).get("title") or "Untitled"
        link = item.get("link") or item.get("content", {}).get("canonicalUrl", {}).get("url") or ""
        publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("displayName") or "Source"
        company_news.append({"title": title, "publisher": publisher, "link": link})

    political_news = get_google_news("US politics stock market tariff geopolitical", 4)
    economic_news = get_google_news("Federal Reserve CPI jobs report stock market", 4)
    if not company_news:
        company_news = get_google_news(f"{ticker} stock company news", 5)

    # Analysts
    analyst_text = "Unavailable"
    analyst_count_summary = None
    finnhub_rec = get_finnhub_json("stock/recommendation", {"symbol": ticker})
    if isinstance(finnhub_rec, list) and finnhub_rec:
        latest = finnhub_rec[0]
        buy = int(latest.get("buy", 0) or 0)
        hold = int(latest.get("hold", 0) or 0)
        sell = int(latest.get("sell", 0) or 0)
        analyst_count_summary = {"buy": buy, "hold": hold, "sell": sell}
        analyst_text = f"Buy {buy} / Hold {hold} / Sell {sell}"

    insider_dashboard = get_insider_dashboard_summary(ticker)

    sections = analyze_section_scores(
        ticker=ticker,
        price=price,
        hist=hist,
        rsi14=rsi14,
        macd_value=float(macd_line.iloc[-1]),
        macd_signal=float(signal_line.iloc[-1]),
        rel_volume=rel_volume,
        candle=candle,
        pattern=pattern,
        trend=trend,
        spy=spy,
        qqq=qqq,
        dxy=dxy,
        tnx=tnx,
        vix=vix,
        analyst_text=analyst_text,
        analyst_counts=analyst_count_summary,
        insider_data=insider_dashboard,
        political_news=political_news,
        economic_news=economic_news,
        company_news=company_news,
    )

    total_score = sum(s["score"] for s in sections)
    max_score = len(sections) * 10
    pass_threshold = 52

    levels = compute_support_resistance(hist, price, atr14)

    # Entry / exit logic
    if total_score >= 58:
        recommendation = "BUY"
        entry_zone = f"${levels['s1']:.2f}-${min(price, levels['r1']):.2f} on pullback or reclaim"
        stop = round(levels["s2"] - atr14 * 0.25, 2)
        target1 = round(levels["r1"], 2)
        target2 = round(levels["r2"], 2)
        recommendation_note = "Constructive setup, but still wait for price to hold support or confirm the breakout."
    elif total_score <= 38:
        recommendation = "SELL / AVOID"
        entry_zone = f"${levels['r1']:.2f}-${levels['r2']:.2f} on failed bounce / rejection"
        stop = round(levels["r2"] + atr14 * 0.25, 2)
        target1 = round(levels["s1"], 2)
        target2 = round(levels["s2"], 2)
        recommendation_note = "Setup is weak or extended. Better to avoid longs or only consider a short on a failed bounce."
    else:
        recommendation = "NO ACTION / WATCH"
        entry_zone = f"Wait for move through ${levels['r1']:.2f} or pullback toward ${levels['s1']:.2f}"
        stop = round(levels["s2"] - atr14 * 0.25, 2)
        target1 = round(levels["r1"], 2)
        target2 = round(levels["r2"], 2)
        recommendation_note = "Mixed setup. Let the stock prove direction first."

    entry_ref = (levels["s1"] + min(price, levels["r1"])) / 2 if recommendation == "BUY" else price
    rr = abs((target1 - entry_ref) / max(abs(entry_ref - stop), 1e-9))

    quote_type = info.get("quoteType")
    exchange = info.get("exchange")
    chart_hint = f"https://www.tradingview.com/search/?query={ticker}"
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
        "vix": vix,
        "analyst": analyst_text,
        "analyst_counts": analyst_count_summary,
        "insider_dashboard": insider_dashboard,
        "political_news": political_news,
        "economic_news": economic_news,
        "company_news": company_news,
        "sections": sections,
        "total_score": total_score,
        "max_score": max_score,
        "pass_threshold": pass_threshold,
        "recommendation": recommendation,
        "recommendation_note": recommendation_note,
        "entry_zone": entry_zone,
        "support1": levels["s1"],
        "support2": levels["s2"],
        "resistance1": levels["r1"],
        "resistance2": levels["r2"],
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rr": rr,
        "sources": {
            "Yahoo Finance": f"https://finance.yahoo.com/quote/{ticker}",
            "TradingView": chart_hint,
            "Insider Dashboard": insider_dashboard["url"],
            "SEC filings": f"https://www.sec.gov/edgar/search/#/q={ticker}",
        },
    }


# ---------- UI ----------
st.title("📈 Ticker Advisor Pro")
st.caption("Enter only a ticker. The app fetches the data, rates each section 1-10, color-codes the result, summarizes political/economic/news context, and suggests support/resistance-based entries and exits.")

with st.sidebar:
    st.subheader("Optional upgrade")
    st.write("The app works with Yahoo Finance and public web sources. Add `FINNHUB_API_KEY` in Streamlit secrets if you want a stronger analyst feed.")
    st.code('FINNHUB_API_KEY = "your_key_here"', language="toml")

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
        m1.metric("Current Price", f"${result['price']:.2f}")
        m2.metric("Total Score", f"{result['total_score']:.0f}/{result['max_score']}")
        m3.metric("Threshold", f"{result['pass_threshold']}/{result['max_score']}")
        m4.metric("Risk/Reward", f"{result['rr']:.2f}")

        threshold_text = "PASSES" if result["total_score"] >= result["pass_threshold"] else "DOES NOT PASS"
        st.markdown(
            f"**Minimum threshold to trade:** {result['total_score']:.0f}/{result['max_score']} — {threshold_text}"
        )

        st.subheader("Section Scores")
        rows = []
        for idx, sec in enumerate(result["sections"], start=1):
            rows.append(
                {
                    "#": idx,
                    "Section": sec["section"],
                    "Score": f"{sec['score']}/10",
                    "Signal": sec["color"],
                    "Key Finding": sec["summary"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for sec in result["sections"]:
            badge = render_status_badge(sec["color"], f"{sec['emoji']} {sec['color']} · {sec['score']}/10")
            st.markdown(f"### {sec['section']}  {badge}", unsafe_allow_html=True)
            st.write(sec["summary"])
            with st.expander(f"Show {sec['section']} details"):
                for line in sec["details"]:
                    st.write(f"- {line}")

        st.subheader("Political & Economic News Summary")
        news_col1, news_col2 = st.columns(2)
        with news_col1:
            st.markdown("#### Political / Geopolitical")
            macro_sec = next((s for s in result["sections"] if s["section"] == "Macro & Political"), None)
            if macro_sec:
                st.markdown(render_status_badge(macro_sec["color"], f"{macro_sec['score']}/10"), unsafe_allow_html=True)
            if result["political_news"]:
                for item in result["political_news"][:3]:
                    st.markdown(f"- [{item['title']}]({item['link']})")
            else:
                st.write("No political headlines were fetched.")
        with news_col2:
            st.markdown("#### Economic / Fed / Macro")
            if result["economic_news"]:
                for item in result["economic_news"][:3]:
                    st.markdown(f"- [{item['title']}]({item['link']})")
            else:
                st.write("No economic headlines were fetched.")

        st.subheader("Insider Dashboard Summary")
        insider = result["insider_dashboard"]
        insider_color = insider.get("status", "Yellow")
        st.markdown(render_status_badge(insider_color, insider_color), unsafe_allow_html=True)
        st.write(insider.get("summary", "No insider summary available."))
        if insider.get("filings"):
            for filing in insider["filings"]:
                st.write(f"- {filing.get('date', '')} {filing.get('form', '')} {filing.get('impact', '')}: {filing.get('summary', '')}")
        st.markdown(f"[Open Insider Dashboard page]({insider['url']})")

        st.subheader("Key Levels")
        levels_df = pd.DataFrame(
            [
                ["Current Price", f"${result['price']:.2f}"],
                ["Intraday Support (S1)", f"${result['support1']:.2f}"],
                ["Intraday Support (S2)", f"${result['support2']:.2f}"],
                ["Resistance (R1)", f"${result['resistance1']:.2f}"],
                ["Resistance (R2)", f"${result['resistance2']:.2f}"],
                ["Entry Zone", result['entry_zone']],
                ["Stop Loss", f"${result['stop']:.2f}"],
                ["Target 1", f"${result['target1']:.2f}"],
                ["Target 2", f"${result['target2']:.2f}"],
                ["RSI (14)", f"{result['rsi']:.2f}"],
                ["Risk/Reward", f"{result['rr']:.2f}"],
                ["VIX", f"{result['vix']:.2f}" if result['vix'] else "Unavailable"],
                ["DXY Trend", result['dxy']],
            ],
            columns=["Level", "Price / Value"],
        )
        st.table(levels_df)

        st.subheader("Company Headlines")
        if result["company_news"]:
            for item in result["company_news"][:5]:
                if item.get("link"):
                    st.markdown(f"- [{item['title']}]({item['link']}) — {item['publisher']}")
                else:
                    st.write(f"- {item['title']} — {item['publisher']}")
        else:
            st.write("No recent company headlines were returned.")

        st.subheader("Price chart")
        chart_data = get_market_history(result["ticker"], period="6mo")
        st.line_chart(chart_data["Close"])

        st.subheader("Source links")
        for label, url in result["sources"].items():
            st.markdown(f"- [{label}]({url})")

    except Exception as exc:
        st.error(str(exc))
else:
    st.info("Type a ticker and click Analyze.")
