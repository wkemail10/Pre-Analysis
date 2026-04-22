import math
import re
from datetime import datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

st.set_page_config(page_title='Ticker Advisor Pro+', page_icon='📈', layout='wide')
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .metric-card {border:1px solid #2a2f3a; border-radius:14px; padding:14px; margin-bottom:10px;}
    .small {font-size: 0.92rem; color: #c9d1d9;}
    .tiny {font-size: 0.82rem; color: #9aa4b2;}
    .good {color:#16a34a; font-weight:700;}
    .bad {color:#dc2626; font-weight:700;}
    .neutral {color:#d4a017; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Utilities ----------
def safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def badge(status: str) -> str:
    colors = {
        'Green': ('#052e16', '#22c55e'),
        'Red': ('#450a0a', '#ef4444'),
        'Yellow': ('#422006', '#f59e0b'),
    }
    bg, fg = colors.get(status, ('#1f2937', '#e5e7eb'))
    return f"<span style='background:{bg}; color:{fg}; padding:4px 10px; border-radius:999px; font-weight:700;'>{status}</span>"


def rating_to_status(score: float):
    score = int(round(clamp(score, 1, 10)))
    if score >= 7:
        return score, 'Green'
    if score <= 4:
        return score, 'Red'
    return score, 'Yellow'


def keyword_score(text: str):
    t = (text or '').lower()
    pos = [
        'beat', 'beats', 'upgrade', 'upgraded', 'buy', 'bullish', 'record', 'growth', 'partnership',
        'investment', 'approved', 'surge', 'strong', 'rebound', 'launch', 'expansion', 'outperform'
    ]
    neg = [
        'miss', 'downgrade', 'downgraded', 'sell', 'bearish', 'lawsuit', 'probe', 'cut', 'weak',
        'decline', 'fall', 'drop', 'war', 'tariff', 'inflation', 'delay', 'recall', 'warns'
    ]
    score = 0
    for w in pos:
        score += t.count(w)
    for w in neg:
        score -= t.count(w)
    return score


def short_join(lines, max_items=4):
    clean = [x.strip() for x in lines if x and str(x).strip()]
    return ' '.join(clean[:max_items])


# ---------- Indicators ----------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series):
    line = ema(series, 12) - ema(series, 26)
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def stochastic(df, period=14, smooth=3):
    low_min = df['Low'].rolling(period).min()
    high_max = df['High'].rolling(period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(smooth).mean()
    return k, d


def atr(df, period=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ichimoku(df):
    high9 = df['High'].rolling(9).max()
    low9 = df['Low'].rolling(9).min()
    tenkan = (high9 + low9) / 2
    high26 = df['High'].rolling(26).max()
    low26 = df['Low'].rolling(26).min()
    kijun = (high26 + low26) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    high52 = df['High'].rolling(52).max()
    low52 = df['Low'].rolling(52).min()
    span_b = ((high52 + low52) / 2).shift(26)
    return tenkan, kijun, span_a, span_b


def detect_candle(df):
    if len(df) < 2:
        return 'Not enough data'
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['Close'] - last['Open'])
    full = max(last['High'] - last['Low'], 1e-9)
    lower = min(last['Open'], last['Close']) - last['Low']
    upper = last['High'] - max(last['Open'], last['Close'])
    if last['Close'] > last['Open'] and prev['Close'] < prev['Open'] and last['Close'] >= prev['Open'] and last['Open'] <= prev['Close']:
        return 'Bullish engulfing'
    if last['Close'] < last['Open'] and prev['Close'] > prev['Open'] and last['Close'] <= prev['Open'] and last['Open'] >= prev['Close']:
        return 'Bearish engulfing'
    if lower > body * 2 and upper < max(body, 0.01):
        return 'Hammer-like'
    if upper > body * 2 and lower < max(body, 0.01):
        return 'Shooting-star-like'
    if body / full < 0.15:
        return 'Doji-like'
    return 'No major candle signal'


def pivot_levels(df):
    prev = df.iloc[-2]
    p = (prev['High'] + prev['Low'] + prev['Close']) / 3
    r1 = 2 * p - prev['Low']
    s1 = 2 * p - prev['High']
    r2 = p + (prev['High'] - prev['Low'])
    s2 = p - (prev['High'] - prev['Low'])
    return p, s1, s2, r1, r2


def nearest_liquidity_levels(df, last_close):
    closes = df['Close'].tail(120)
    rounded = closes.round(0)
    counts = rounded.value_counts().sort_index()
    if counts.empty:
        return []
    nearby = counts.loc[(counts.index >= last_close - 20) & (counts.index <= last_close + 20)]
    top = nearby.sort_values(ascending=False).head(4)
    return [float(x) for x in sorted(top.index.tolist())]


def price_volume_orderflow_proxy(df):
    recent = df.tail(10).copy()
    signed = np.where(recent['Close'] >= recent['Open'], recent['Volume'], -recent['Volume'])
    ratio = signed.sum() / max(recent['Volume'].sum(), 1)
    if ratio > 0.15:
        return 'Net buying pressure', 7.5
    if ratio < -0.15:
        return 'Net selling pressure', 3.5
    return 'Balanced / mixed flow', 5.5


def trend_of(df):
    if len(df) < 50:
        return 'Unknown'
    close = df['Close']
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    last = close.iloc[-1]
    if last > sma20 > sma50:
        return 'Up'
    if last < sma20 < sma50:
        return 'Down'
    return 'Mixed'


# ---------- Data ----------
@st.cache_data(ttl=1800)
def get_history(symbol, period='1y'):
    df = yf.download(symbol, period=period, interval='1d', auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how='all')


@st.cache_data(ttl=1800)
def get_ticker_data(symbol):
    t = yf.Ticker(symbol)
    info = t.info or {}
    try:
        news = t.news or []
    except Exception:
        news = []
    try:
        calendar = t.calendar
    except Exception:
        calendar = None
    return info, news, calendar


@st.cache_data(ttl=3600)
def google_news_search(query, limit=6):
    url = f'https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en'
    out = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'xml')
        for item in soup.find_all('item')[:limit]:
            out.append({
                'title': item.title.text if item.title else '',
                'link': item.link.text if item.link else '',
                'source': item.source.text if item.source else '',
                'date': item.pubDate.text if item.pubDate else '',
            })
    except Exception:
        pass
    return out


@st.cache_data(ttl=3600)
def get_macro_news():
    queries = {
        'MarketWatch': 'stock market macro political economy site:marketwatch.com',
        'CNBC': 'stock market macro political economy site:cnbc.com',
        'Yahoo Finance': 'stock market macro political economy site:finance.yahoo.com',
    }
    bundle = {}
    for name, q in queries.items():
        bundle[name] = google_news_search(q, limit=3)
    return bundle


@st.cache_data(ttl=3600)
def get_ticker_context_news(ticker):
    queries = {
        'MarketWatch': f'{ticker} company earnings ceo guidance investment site:marketwatch.com',
        'CNBC': f'{ticker} company earnings ceo guidance investment site:cnbc.com',
        'Yahoo Finance': f'{ticker} company earnings ceo guidance investment site:finance.yahoo.com',
    }
    bundle = {}
    for name, q in queries.items():
        bundle[name] = google_news_search(q, limit=3)
    return bundle


@st.cache_data(ttl=3600)
def get_social_mentions(ticker):
    sites = {
        'X': 'x.com',
        'Reddit': 'reddit.com',
        'Quora': 'quora.com',
        'Facebook': 'facebook.com',
    }
    out = {}
    for label, domain in sites.items():
        out[label] = google_news_search(f'{ticker} stock site:{domain}', limit=3)
    return out


@st.cache_data(ttl=3600)
def get_cnn_fear_greed_text():
    url = 'https://edition-prod-cf.sitemirror.cnn.com/markets/fear-and-greed'
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        text = BeautifulSoup(r.text, 'html.parser').get_text(' ', strip=True)
        return text, url
    except Exception:
        return '', url


@st.cache_data(ttl=3600)
def get_insider_dashboard(ticker):
    url = f'https://www.insiderdashboard.com/search?query={ticker}'
    out = {'url': url, 'rows': [], 'summary': 'No public insider transactions parsed.'}
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        text = BeautifulSoup(r.text, 'html.parser').get_text('\n', strip=True)
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        rows = []
        for i, line in enumerate(lines):
            if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', line):
                row = {'date': line, 'type': '', 'shares': '', 'value': ''}
                window = lines[i:i+10]
                for w in window:
                    if 'buy' in w.lower() or 'sell' in w.lower():
                        row['type'] = w
                        break
                nums = [w for w in window if re.search(r'\$[\d,]+', w) or re.fullmatch(r'[\d,]+', w)]
                if nums:
                    row['shares'] = nums[0]
                if len(nums) > 1:
                    row['value'] = nums[1]
                rows.append(row)
        # dedupe by date+type
        seen = set()
        clean = []
        for row in rows:
            key = (row['date'], row['type'])
            if key in seen:
                continue
            seen.add(key)
            clean.append(row)
        out['rows'] = clean[:8]
        if clean:
            last = clean[0]
            out['summary'] = f"Latest public insider item appears around {last['date']} ({last['type'] or 'filing'}). Use the source link to verify transaction size and direction."
    except Exception:
        pass
    return out


# ---------- Summaries ----------
def summarize_macro(news_bundle, spy_trend, qqq_trend, dxy_trend, teny_trend):
    lines = []
    all_titles = []
    for source, items in news_bundle.items():
        if items:
            titles = [x['title'] for x in items[:2]]
            lines.append(f"{source}: " + ' | '.join(titles))
            all_titles.extend(titles)
    base = 5.5
    if spy_trend == 'Up' and qqq_trend == 'Up':
        base += 1.0
    if dxy_trend == 'Down':
        base += 0.5
    if teny_trend == 'Down':
        base += 0.5
    base += clamp(keyword_score(' '.join(all_titles)) * 0.2, -2, 2)
    rating, status = rating_to_status(base)
    summary = short_join([
        f"SPY {spy_trend}, QQQ {qqq_trend}, DXY {dxy_trend}, 10Y {teny_trend}.",
        short_join(lines, 3),
    ], 2)
    return rating, status, summary, lines


def summarize_market_context(ticker, info, news_bundle, yf_news, calendar):
    lines = []
    title_bank = []
    if info.get('shortName'):
        lines.append(f"{info.get('shortName')} ({ticker}) in {info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}. ")
    if info.get('currentPrice') and info.get('targetMeanPrice'):
        lines.append(f"Current price ~{info.get('currentPrice')}, analyst mean target ~{info.get('targetMeanPrice')}.")
    if safe_float(info.get('revenueGrowth')) == safe_float(info.get('revenueGrowth')):
        rg = safe_float(info.get('revenueGrowth')) * 100
        lines.append(f"Revenue growth approx. {rg:.1f}%.")
    if safe_float(info.get('earningsGrowth')) == safe_float(info.get('earningsGrowth')):
        eg = safe_float(info.get('earningsGrowth')) * 100
        lines.append(f"Earnings growth approx. {eg:.1f}%.")

    for _, items in news_bundle.items():
        for item in items[:2]:
            title_bank.append(item['title'])
    for item in (yf_news or [])[:4]:
        title = item.get('title') or item.get('content', {}).get('title') or ''
        if title:
            title_bank.append(title)
    if title_bank:
        lines.append('Recent catalysts: ' + ' | '.join(title_bank[:4]))

    upcoming = []
    try:
        if calendar is not None and not getattr(calendar, 'empty', True):
            upcoming.append('Upcoming calendar event detected.')
    except Exception:
        pass
    if upcoming:
        lines.extend(upcoming)

    base = 5.5 + clamp(keyword_score(' '.join(title_bank)) * 0.25, -3, 3)
    if safe_float(info.get('revenueGrowth')) > 0:
        base += 0.5
    if safe_float(info.get('earningsGrowth')) > 0:
        base += 0.5
    rating, status = rating_to_status(base)
    summary = short_join(lines, 3)
    return rating, status, summary, title_bank[:6]


def technical_pack(df, spy_df, qqq_df, dia_df):
    close = df['Close']
    vol = df['Volume']
    last = float(close.iloc[-1])
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(df) >= 200 else np.nan
    avg30 = vol.rolling(30).mean().iloc[-1]
    rsi14 = rsi(close).iloc[-1]
    macd_line, macd_signal, _ = macd(close)
    k, d = stochastic(df)
    atr14 = atr(df).iloc[-1]
    _, _, span_a, span_b = ichimoku(df)
    p, s1, s2, r1, r2 = pivot_levels(df)
    liquidity_levels = nearest_liquidity_levels(df, last)
    flow_text, flow_score = price_volume_orderflow_proxy(df)
    candle = detect_candle(df)

    pack = []

    score = 8 if last > sma50 and (np.isnan(sma200) or last > sma200) else 3 if last < sma50 and (np.isnan(sma200) or last < sma200) else 5
    pack.append(('SMA 50 / 200', score, f'Price {last:.2f} vs SMA50 {sma50:.2f}' + (f' and SMA200 {sma200:.2f}' if sma200 == sma200 else '')))

    vscore = 7 if vol.iloc[-1] > avg30 * 1.15 else 4 if vol.iloc[-1] < avg30 * 0.85 else 5
    pack.append(('Volume vs Avg', vscore, f'Latest volume {vol.iloc[-1]:,.0f} vs 30-day avg {avg30:,.0f}.'))

    rscore = 8 if 45 <= rsi14 <= 65 else 3 if rsi14 > 72 or rsi14 < 28 else 5
    pack.append(('RSI', rscore, f'RSI(14) {rsi14:.2f}.'))

    mscore = 7 if macd_line.iloc[-1] > macd_signal.iloc[-1] else 3
    pack.append(('MACD', mscore, f'MACD {macd_line.iloc[-1]:.2f} vs signal {macd_signal.iloc[-1]:.2f}.'))

    stoch_last = k.iloc[-1]
    sscore = 7 if 20 <= stoch_last <= 80 else 4
    pack.append(('Stochastic', sscore, f'Stoch %K {stoch_last:.2f}, %D {d.iloc[-1]:.2f}.'))

    atr_pct = atr14 / last * 100 if last else np.nan
    volscore = 7 if atr_pct < 3 else 5 if atr_pct < 5 else 3
    pack.append(('Volatility (Risk)', volscore, f'ATR(14) ~{atr14:.2f} ({atr_pct:.2f}% of price).'))

    lvscore = 7 if last > p else 4
    pack.append(('Support / Resistance', lvscore, f'S1 {s1:.2f}, S2 {s2:.2f}, R1 {r1:.2f}, R2 {r2:.2f}.'))

    cloud_top = np.nanmax([span_a.iloc[-1], span_b.iloc[-1]]) if not (np.isnan(span_a.iloc[-1]) and np.isnan(span_b.iloc[-1])) else np.nan
    cloud_bottom = np.nanmin([span_a.iloc[-1], span_b.iloc[-1]]) if not (np.isnan(span_a.iloc[-1]) and np.isnan(span_b.iloc[-1])) else np.nan
    if cloud_top == cloud_top:
        iscore = 8 if last > cloud_top else 3 if last < cloud_bottom else 5
        idesc = f'Price {last:.2f} vs cloud {cloud_bottom:.2f}-{cloud_top:.2f}.'
    else:
        iscore = 5
        idesc = 'Not enough data for full cloud.'
    pack.append(('Ichimoku Cloud', iscore, idesc))

    pack.append(('Order Flow (2w proxy)', flow_score, flow_text + ' Proxy derived from signed price/volume, not Level-2 tape.'))

    dol_vol = float((df['Close'].tail(20) * df['Volume'].tail(20)).mean())
    liqscore = 8 if dol_vol > 5e8 else 6 if dol_vol > 1e8 else 3
    liqtxt = f'20-day average dollar volume ~${dol_vol/1e6:,.1f}M. Key liquidity nodes: ' + (', '.join(f'{x:.0f}' for x in liquidity_levels) if liquidity_levels else 'not enough data')
    pack.append(('Liquidity Levels', liqscore, liqtxt))

    cscore = 7 if 'Bullish' in candle or 'Hammer' in candle else 3 if 'Bearish' in candle or 'Shooting' in candle else 5
    pack.append(('Candle', cscore, candle))

    corr_rows = []
    follow_score = 5
    for name, idx_df in [('DJI', dia_df), ('SPY', spy_df), ('QQQ', qqq_df)]:
        joined = pd.concat([df['Close'].pct_change(), idx_df['Close'].pct_change()], axis=1).dropna().tail(60)
        if len(joined) > 20:
            corr = joined.corr().iloc[0, 1]
            corr_rows.append((name, corr))
    corr_rows = sorted(corr_rows, key=lambda x: x[1], reverse=True)
    leader = corr_rows[0][0] if corr_rows else 'N/A'
    if corr_rows and corr_rows[0][1] > 0.8:
        follow_score = 7
    elif corr_rows and corr_rows[0][1] < 0.5:
        follow_score = 4
    pack.append(('Follows Which Index?', follow_score, f'Highest 60-day correlation is with {leader}' + (f' ({corr_rows[0][1]:.2f}).' if corr_rows else '.')))

    overall = sum(float(x[1]) for x in pack) / len(pack)
    summary = f"Technicals are {('constructive' if overall >= 6.5 else 'mixed' if overall >= 4.5 else 'weak')} with strongest signals in {', '.join([x[0] for x in sorted(pack, key=lambda y: y[1], reverse=True)[:2]])}. Main caution comes from {', '.join([x[0] for x in sorted(pack, key=lambda y: y[1])[:2]])}."

    long_entry = min(last, p)
    stop = s1 if s1 < long_entry else long_entry - atr14
    target1 = r1
    target2 = r2

    return pack, overall, summary, {
        'current': last,
        's1': s1,
        's2': s2,
        'r1': r1,
        'r2': r2,
        'entry': long_entry,
        'stop': stop,
        'target1': target1,
        'target2': target2,
        'rsi': rsi14,
    }


def summarize_sentiment_and_analysts(ticker, info):
    text, url = get_cnn_fear_greed_text()
    score = 5.5
    fg_desc = 'Fear & Greed page unavailable.'
    if text:
        m = re.search(r'(Extreme Greed|Greed|Neutral|Fear|Extreme Fear)', text)
        if m:
            label = m.group(1)
            fg_desc = f'CNN Fear & Greed currently reads around {label}.'
            mapping = {'Extreme Greed': 7, 'Greed': 6, 'Neutral': 5, 'Fear': 4, 'Extreme Fear': 3}
            score = mapping.get(label, 5)
    analyst_bits = []
    if info.get('recommendationKey'):
        rk = str(info.get('recommendationKey')).title()
        analyst_bits.append(f'Yahoo analyst key is {rk}.')
        if rk in ['Buy', 'Strong Buy']:
            score += 1
        elif rk in ['Underperform', 'Sell']:
            score -= 1
    if info.get('targetMeanPrice') and info.get('currentPrice'):
        cp = safe_float(info.get('currentPrice'))
        tp = safe_float(info.get('targetMeanPrice'))
        if cp == cp and tp == tp and cp > 0:
            upside = (tp / cp - 1) * 100
            analyst_bits.append(f'Mean target implies about {upside:.1f}% upside/downside.')
            if upside > 10:
                score += 0.5
            elif upside < -5:
                score -= 0.5
    rating, status = rating_to_status(score)
    summary = short_join([fg_desc] + analyst_bits, 4)
    return rating, status, summary, url


def social_summary(ticker):
    bundle = get_social_mentions(ticker)
    lines = []
    for site, items in bundle.items():
        if items:
            titles = '; '.join(x['title'] for x in items[:2])
            lines.append(f'{site}: {titles}')
        else:
            lines.append(f'{site}: no strong indexed discussion found.')
    return lines[:10], bundle


# ---------- UI ----------
st.title('📈 Ticker Advisor Pro+')
st.caption('Ticker in → source-backed summary out. Ratings use Green / Yellow / Red plus 1-10 scores.')

col1, col2 = st.columns([1, 1])
with col1:
    ticker = st.text_input('Ticker', value='AAPL').upper().strip()
with col2:
    run = st.button('Analyze', type='primary')

if run and ticker:
    with st.spinner('Pulling market data, headlines, technicals, and insider summaries...'):
        df = get_history(ticker)
        spy = get_history('SPY', '1y')
        qqq = get_history('QQQ', '1y')
        dia = get_history('DIA', '1y')
        dxy = get_history('DX-Y.NYB', '6mo')
        tnx = get_history('^TNX', '6mo')
        info, yf_news, calendar = get_ticker_data(ticker)

        if df.empty:
            st.error('No price history found for that ticker.')
            st.stop()

        macro_bundle = get_macro_news()
        ticker_news_bundle = get_ticker_context_news(ticker)
        insider = get_insider_dashboard(ticker)
        social_lines, social_bundle = social_summary(ticker)

        spy_tr = trend_of(spy)
        qqq_tr = trend_of(qqq)
        dxy_tr = trend_of(dxy)
        teny_tr = trend_of(tnx)

        macro_rating, macro_status, macro_summary, macro_lines = summarize_macro(macro_bundle, spy_tr, qqq_tr, dxy_tr, teny_tr)
        mkt_rating, mkt_status, mkt_summary, mkt_titles = summarize_market_context(ticker, info, ticker_news_bundle, yf_news, calendar)
        tech_rows, tech_overall, tech_summary, levels = technical_pack(df, spy, qqq, dia)
        tech_rating, tech_status = rating_to_status(tech_overall)
        sent_rating, sent_status, sent_summary, cnn_url = summarize_sentiment_and_analysts(ticker, info)

        insider_score = 5
        if insider['rows']:
            text_blob = ' '.join((x.get('type') or '') for x in insider['rows'])
            if 'buy' in text_blob.lower() and 'sell' not in text_blob.lower():
                insider_score = 7
            elif 'sell' in text_blob.lower() and 'buy' not in text_blob.lower():
                insider_score = 4
        insider_rating, insider_status = rating_to_status(insider_score)

        social_score = 5 + clamp(keyword_score(' '.join(social_lines)) * 0.3, -2, 2)
        social_rating, social_status = rating_to_status(social_score)

        section_df = pd.DataFrame([
            ['Macro & Political', macro_rating, macro_status, macro_summary],
            ['Market Context', mkt_rating, mkt_status, mkt_summary],
            ['Technical Analysis', tech_rating, tech_status, tech_summary],
            ['Insider / Whale Activity', insider_rating, insider_status, insider['summary']],
            ['Sentiment & Analysts', sent_rating, sent_status, sent_summary],
            ['Social Media', social_rating, social_status, short_join(social_lines, 3)],
        ], columns=['Section', 'Score', 'Status', 'Summary'])
        total_score = section_df['Score'].sum()
        threshold = 40
        if total_score >= 48:
            final_call = 'BUY bias'
        elif total_score <= 28:
            final_call = 'SELL / AVOID bias'
        else:
            final_call = 'NO ACTION / WATCH'

    top1, top2, top3 = st.columns(3)
    top1.metric('Ticker', ticker)
    top2.metric('Total Score', f'{int(total_score)}/60')
    top3.metric('Recommendation', final_call)

    st.subheader('Section Scores')
    show_df = section_df.copy()
    show_df['Color'] = show_df['Status'].apply(lambda x: '🟢' if x == 'Green' else '🔴' if x == 'Red' else '🟡')
    st.dataframe(show_df[['Color', 'Section', 'Score', 'Summary']], use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.subheader('Macro & Political')
        st.markdown(f"{badge(macro_status)} **{macro_rating}/10** — {macro_summary}", unsafe_allow_html=True)
        for source, items in macro_bundle.items():
            st.markdown(f'**{source}**')
            if items:
                for it in items[:3]:
                    st.markdown(f"- [{it['title']}]({it['link']})")
            else:
                st.write('- No recent indexed result found.')

        st.subheader('Market Context')
        st.markdown(f"{badge(mkt_status)} **{mkt_rating}/10** — {mkt_summary}", unsafe_allow_html=True)
        if mkt_titles:
            for t in mkt_titles[:6]:
                st.write(f'- {t}')

        st.subheader('Technical Analysis')
        st.markdown(f"{badge(tech_status)} **{tech_rating}/10** — {tech_summary}", unsafe_allow_html=True)
        tech_table = []
        for name, score, desc in tech_rows:
            r, s = rating_to_status(score)
            tech_table.append([('🟢' if s=='Green' else '🔴' if s=='Red' else '🟡'), name, f'{r}/10', desc])
        st.dataframe(pd.DataFrame(tech_table, columns=['Color', 'Indicator', 'Score', 'Reading']), use_container_width=True, hide_index=True)

    with c2:
        st.subheader('Key Levels')
        rr = (levels['target2'] - levels['entry']) / max(levels['entry'] - levels['stop'], 0.01)
        key_levels_df = pd.DataFrame([
            ['Current Price', f"{levels['current']:.2f}"],
            ['Support (S1)', f"{levels['s1']:.2f}"],
            ['Support (S2)', f"{levels['s2']:.2f}"],
            ['Resistance (R1)', f"{levels['r1']:.2f}"],
            ['Resistance (R2)', f"{levels['r2']:.2f}"],
            ['Entry Zone', f"{min(levels['s1'], levels['entry']):.2f} - {levels['entry']:.2f}"],
            ['Stop Loss', f"{levels['stop']:.2f}"],
            ['Target 1', f"{levels['target1']:.2f}"],
            ['Target 2', f"{levels['target2']:.2f}"],
            ['Risk / Reward', f"~{rr:.2f}:1"],
        ], columns=['Level', 'Price'])
        st.dataframe(key_levels_df, use_container_width=True, hide_index=True)

        st.subheader('Insider / Whale Activity')
        st.markdown(f"{badge(insider_status)} **{insider_rating}/10** — {insider['summary']}", unsafe_allow_html=True)
        if insider['rows']:
            rows = []
            for x in insider['rows'][:8]:
                rows.append([x.get('date',''), x.get('type',''), x.get('shares',''), x.get('value','')])
            st.dataframe(pd.DataFrame(rows, columns=['Date', 'Type', 'Quantity', 'Size/Value']), use_container_width=True, hide_index=True)
        st.markdown(f"[Source link]({insider['url']})")
        st.caption('Verify each filing on the source page before trading.')

        st.subheader('Sentiment & Analysts')
        st.markdown(f"{badge(sent_status)} **{sent_rating}/10** — {sent_summary}", unsafe_allow_html=True)
        st.markdown(f"[CNN Fear & Greed]({cnn_url})")

    st.subheader('Social Media')
    st.markdown(f"{badge(social_status)} **{social_rating}/10**", unsafe_allow_html=True)
    for line in social_lines[:10]:
        st.write(f'- {line}')

    with st.expander('Source links used by the app'):
        st.write('- Macro sources are searched from MarketWatch, CNBC, and Yahoo Finance through indexed news results.')
        st.write('- Insider source: Insider Dashboard.')
        st.write('- Sentiment source: CNN Fear & Greed.')
        st.write('- Market/technical data: Yahoo Finance via yfinance.')
