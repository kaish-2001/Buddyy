"""
Buddyy – Your Smart Stock Buddy
====================================================
• Valuation (DCF + Multiples + Monte-Carlo range)
• Technicals (RSI, MACD, Ichimoku, Volume Profile, Support/Resistance...)
• Sector-smart ratios
• 5 Alternative ideas
• Watchlist comparison
• Rule-based + Webhook alerts (Telegram / Email)
• PDF Export
• Live data (NSE .NS / BSE .BO / Global via Yahoo Finance)

Deploy free on Streamlit Cloud → public website link
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from fpdf import FPDF
import base64
import requests
from scipy.stats import norm
import io

st.set_page_config(
    page_title="Buddyy – Your Smart Stock Buddy",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Sector mappings
# -------------------------------------------------
SECTOR_RATIOS = {
    "Technology": ["Trailing P/E", "Forward P/E", "PEG", "Price/Sales", "EV/EBITDA", "ROE", "FCF Yield", "Revenue Growth", "EPS Growth"],
    "Software": ["Trailing P/E", "Forward P/E", "PEG", "Price/Sales", "EV/Sales", "ROE", "FCF Yield", "Revenue Growth"],
    "Financial Services": ["Price/Book", "Trailing P/E", "ROE", "ROA", "Debt/Equity", "Dividend Yield", "Beta"],
    "Banks": ["Price/Book", "Trailing P/E", "ROE", "ROA", "Debt/Equity", "Dividend Yield", "Beta"],
    "Healthcare": ["Trailing P/E", "Forward P/E", "PEG", "EV/EBITDA", "ROE", "Debt/Equity", "Revenue Growth"],
    "Consumer Cyclical": ["Trailing P/E", "Forward P/E", "PEG", "Price/Sales", "EV/EBITDA", "ROE", "Revenue Growth"],
    "Consumer Defensive": ["Trailing P/E", "Forward P/E", "PEG", "EV/EBITDA", "ROE", "Dividend Yield", "Debt/Equity"],
    "Energy": ["Trailing P/E", "EV/EBITDA", "Price/Book", "ROE", "Debt/Equity", "FCF Yield", "Dividend Yield"],
    "Industrials": ["Trailing P/E", "EV/EBITDA", "ROE", "Debt/Equity", "FCF Yield", "Revenue Growth"],
    "Basic Materials": ["Trailing P/E", "EV/EBITDA", "Price/Book", "ROE", "Debt/Equity", "FCF Yield"],
    "Utilities": ["Trailing P/E", "EV/EBITDA", "Dividend Yield", "Debt/Equity", "ROE", "Beta"],
    "Real Estate": ["Price/Book", "Trailing P/E", "Dividend Yield", "Debt/Equity", "ROE", "Beta"],
    "Communication Services": ["Trailing P/E", "Forward P/E", "EV/EBITDA", "ROE", "FCF Yield", "Revenue Growth"],
    "default": ["Trailing P/E", "Forward P/E", "PEG", "Price/Book", "EV/EBITDA", "ROE", "FCF Yield", "Debt/Equity", "Revenue Growth", "EPS Growth", "Dividend Yield", "Beta"]
}

SECTOR_PEERS = {
    "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "AAPL", "MSFT", "GOOGL", "NVDA"],
    "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "JPM"],
    "Banks": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "XOM"],
    "Consumer Cyclical": ["TITAN.NS", "M&M.NS", "MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS"],
    "Consumer Defensive": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Industrials": ["LT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS"],
    "Basic Materials": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS"],
    "Utilities": ["NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS", "TATAPOWER.NS"],
    "Communication Services": ["BHARTIARTL.NS", "IDEA.NS"],
    "default": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "AAPL", "MSFT"]
}

# -------------------------------------------------
# Helpers
# -------------------------------------------------

@st.cache_data(ttl=600)
def fetch_news(ticker):
    """Fetch recent news headlines for the ticker via yfinance"""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        items = []
        for n in news[:8]:
            title = n.get("title") or n.get("content", {}).get("title") or ""
            publisher = n.get("publisher") or n.get("content", {}).get("provider", {}).get("displayName") or ""
            link = n.get("link") or n.get("content", {}).get("clickThroughUrl", {}).get("url") or ""
            if title:
                items.append({"title": title, "publisher": publisher, "link": link})
        return items
    except Exception:
        return []

@st.cache_data(ttl=180, show_spinner=False)
def fetch_data(ticker: str, period: str = "2y", interval: str = "1d"):
    """Fetch with interval support + rate-limit tolerance"""
    import time
    last_err = None
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval, auto_adjust=True)
            if hist.empty or len(hist) < 15:
                return {"ok": False, "error": f"No / insufficient data for {interval} interval. Try a different timeframe."}
            info = t.info
            return {"ok": True, "info": info, "hist": hist}
        except Exception as e:
            last_err = str(e)
            if "Rate" in last_err or "Too Many" in last_err or "429" in last_err:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    return {"ok": False, "error": last_err or "Unknown error"}

def safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None: return default
        cur = cur.get(k) if isinstance(cur, dict) else getattr(cur, k, None)
    return cur if cur is not None else default

def fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "N/A"
    x = float(x)
    ax = abs(x)
    if ax >= 1e12: return f"{x/1e12:.2f}T"
    if ax >= 1e9:  return f"{x/1e9:.2f}B"
    if ax >= 1e6:  return f"{x/1e6:.2f}M"
    return f"{x:,.2f}"

def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "N/A"
    return f"{x*100:.1f}%" if abs(x) < 5 else f"{x:.1f}%"

# -------------------------------------------------
# Technicals + Volume Profile + Ichimoku
# -------------------------------------------------
def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(df)

    # Adaptive windows for short timeframes
    w20 = min(20, max(5, n // 4))
    w50 = min(50, max(8, n // 3))
    w200 = min(200, max(15, n // 2))

    df["SMA_20"] = ta.trend.sma_indicator(c, w20)
    df["SMA_50"] = ta.trend.sma_indicator(c, w50)
    df["SMA_200"] = ta.trend.sma_indicator(c, w200)
    df["EMA_12"] = ta.trend.ema_indicator(c, min(12, max(3, n // 5)))
    df["EMA_26"] = ta.trend.ema_indicator(c, min(26, max(5, n // 4)))

    macd = ta.trend.MACD(c)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    df["RSI"] = ta.momentum.rsi(c, 14)

    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Mid"] = bb.bollinger_mavg()
    df["BB_Low"] = bb.bollinger_lband()

    stoch = ta.momentum.StochasticOscillator(h, l, c)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    df["ATR"] = ta.volatility.average_true_range(h, l, c, 14)
    df["OBV"] = ta.volume.on_balance_volume(c, v)
    df["Volume_SMA"] = v.rolling(20).mean()
    df["ADX"] = ta.trend.adx(h, l, c, 14)

    # Ichimoku
    ichi = ta.trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
    df["Ichimoku_A"] = ichi.ichimoku_a()
    df["Ichimoku_B"] = ichi.ichimoku_b()
    df["Ichimoku_Base"] = ichi.ichimoku_base_line()
    df["Ichimoku_Conv"] = ichi.ichimoku_conversion_line()

    return df

def volume_profile(df, bins=24):
    """Simple volume profile: volume distribution across price levels"""
    if df.empty: return None, None
    recent = df.tail(120)
    price_min, price_max = recent["Low"].min(), recent["High"].max()
    if price_max <= price_min: return None, None
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_at_price = np.zeros(bins)
    for i in range(len(recent)):
        row = recent.iloc[i]
        # distribute volume across the candle range
        low_idx = np.searchsorted(bin_edges, row["Low"], side="right") - 1
        high_idx = np.searchsorted(bin_edges, row["High"], side="right") - 1
        low_idx = max(0, min(low_idx, bins-1))
        high_idx = max(0, min(high_idx, bins-1))
        if high_idx >= low_idx:
            vol_at_price[low_idx:high_idx+1] += row["Volume"] / (high_idx - low_idx + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, vol_at_price



def calc_pivot_points(df):
    """Classic Floor Pivot Points from last completed candle / day"""
    if len(df) < 2:
        return {}
    # Use previous candle as reference (standard)
    prev = df.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    pp = (h + l + c) / 3
    r1 = 2 * pp - l
    s1 = 2 * pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    r3 = h + 2 * (pp - l)
    s3 = l - 2 * (h - pp)
    return {
        "Pivot (PP)": pp,
        "Resistance 1 (R1)": r1,
        "Resistance 2 (R2)": r2,
        "Resistance 3 (R3)": r3,
        "Support 1 (S1)": s1,
        "Support 2 (S2)": s2,
        "Support 3 (S3)": s3,
    }


def calc_fibonacci_levels(df, lookback=60):
    """Fibonacci retracement from recent swing high to swing low"""
    recent = df.tail(lookback)
    if recent.empty:
        return {}
    swing_high = float(recent["High"].max())
    swing_low = float(recent["Low"].min())
    diff = swing_high - swing_low
    if diff <= 0:
        return {}
    levels = {
        "Swing High (0.0%)": swing_high,
        "Fib 23.6%": swing_high - 0.236 * diff,
        "Fib 38.2%": swing_high - 0.382 * diff,
        "Fib 50.0%": swing_high - 0.500 * diff,
        "Fib 61.8%": swing_high - 0.618 * diff,
        "Fib 78.6%": swing_high - 0.786 * diff,
        "Swing Low (100%)": swing_low,
    }
    return levels


def calc_returns(df, price):
    """Calculate returns over multiple periods"""
    if df is None or len(df) < 2 or price is None:
        return {}
    close = df["Close"]
    results = {}
    periods = [
        ("1 Day", 1),
        ("1 Week", 5),
        ("2 Weeks", 10),
        ("1 Month", 21),
        ("3 Months", 63),
        ("6 Months", 126),
        ("1 Year", 252),
    ]
    for label, bars in periods:
        if len(close) > bars:
            past = float(close.iloc[-bars-1])
            if past > 0:
                ret = (price - past) / past * 100
                results[label] = ret
    return results


def detect_candlestick_patterns(df):
    """Detect common candlestick patterns on the last few candles"""
    patterns = []
    if len(df) < 5:
        return patterns

    df = df.copy()
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    
    # Last 3 candles
    for i in [-1, -2, -3]:
        body = abs(c.iloc[i] - o.iloc[i])
        upper_wick = h.iloc[i] - max(c.iloc[i], o.iloc[i])
        lower_wick = min(c.iloc[i], o.iloc[i]) - l.iloc[i]
        full_range = h.iloc[i] - l.iloc[i]
        if full_range == 0:
            continue
        body_ratio = body / full_range
        is_bull = c.iloc[i] > o.iloc[i]
        is_bear = c.iloc[i] < o.iloc[i]

        # Doji
        if body_ratio < 0.1:
            patterns.append(("Doji", "Indecision / possible reversal", i))
        # Hammer (bullish)
        elif lower_wick > 2 * body and upper_wick < body * 0.5 and is_bull:
            patterns.append(("Hammer", "Bullish reversal signal", i))
        # Hanging Man (bearish)
        elif lower_wick > 2 * body and upper_wick < body * 0.5 and is_bear:
            patterns.append(("Hanging Man", "Bearish reversal warning", i))
        # Shooting Star
        elif upper_wick > 2 * body and lower_wick < body * 0.5 and is_bear:
            patterns.append(("Shooting Star", "Bearish reversal signal", i))
        # Inverted Hammer
        elif upper_wick > 2 * body and lower_wick < body * 0.5 and is_bull:
            patterns.append(("Inverted Hammer", "Potential bullish reversal", i))
        # Marubozu
        elif body_ratio > 0.85:
            direction = "Bullish" if is_bull else "Bearish"
            patterns.append((f"{direction} Marubozu", f"Strong {direction.lower()} momentum", i))

    # Engulfing (need 2 candles)
    if len(df) >= 2:
        prev_body = abs(c.iloc[-2] - o.iloc[-2])
        curr_body = abs(c.iloc[-1] - o.iloc[-1])
        # Bullish Engulfing
        if (c.iloc[-2] < o.iloc[-2] and c.iloc[-1] > o.iloc[-1] and
            c.iloc[-1] > o.iloc[-2] and o.iloc[-1] < c.iloc[-2] and curr_body > prev_body):
            patterns.append(("Bullish Engulfing", "Strong bullish reversal", -1))
        # Bearish Engulfing
        if (c.iloc[-2] > o.iloc[-2] and c.iloc[-1] < o.iloc[-1] and
            c.iloc[-1] < o.iloc[-2] and o.iloc[-1] > c.iloc[-2] and curr_body > prev_body):
            patterns.append(("Bearish Engulfing", "Strong bearish reversal", -1))

    return patterns


def detect_chart_patterns(df):
    """Simple chart pattern detection on recent price action"""
    patterns = []
    if len(df) < 30:
        return patterns

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    n = len(close)

    # Higher Highs / Higher Lows (Uptrend)
    recent_highs = high[-20:]
    recent_lows = low[-20:]
    if recent_highs[-1] > recent_highs[-10] and recent_lows[-1] > recent_lows[-10]:
        patterns.append(("Higher Highs + Higher Lows", "Uptrend structure intact", "Bullish"))
    elif recent_highs[-1] < recent_highs[-10] and recent_lows[-1] < recent_lows[-10]:
        patterns.append(("Lower Highs + Lower Lows", "Downtrend structure", "Bearish"))

    # Double Bottom (simplified)
    lows_idx = []
    for i in range(n-25, n-2):
        if low[i] < low[i-1] and low[i] < low[i+1]:
            lows_idx.append(i)
    if len(lows_idx) >= 2:
        l1, l2 = lows_idx[-2], lows_idx[-1]
        if abs(low[l1] - low[l2]) / low[l1] < 0.03 and (l2 - l1) > 5:
            patterns.append(("Possible Double Bottom", "Bullish reversal pattern forming", "Bullish"))

    # Double Top (simplified)
    highs_idx = []
    for i in range(n-25, n-2):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            highs_idx.append(i)
    if len(highs_idx) >= 2:
        h1, h2 = highs_idx[-2], highs_idx[-1]
        if abs(high[h1] - high[h2]) / high[h1] < 0.03 and (h2 - h1) > 5:
            patterns.append(("Possible Double Top", "Bearish reversal pattern forming", "Bearish"))

    # Breakout check
    recent_range_high = high[-20:-1].max()
    recent_range_low = low[-20:-1].min()
    if close[-1] > recent_range_high:
        patterns.append(("Breakout above recent range", "Bullish momentum / continuation", "Bullish"))
    elif close[-1] < recent_range_low:
        patterns.append(("Breakdown below recent range", "Bearish momentum / continuation", "Bearish"))

    return patterns


def timeframe_interpretation(interval, period, trade_horizon):
    """Return human-friendly interpretation"""
    interval_map = {
        "15m": "15-Minute candles (scalping / very short-term)",
        "1h": "1-Hour candles (intraday / short swing)",
        "4h": "4-Hour candles (short to medium swing)",
        "1d": "Daily candles (positional / swing)",
        "1wk": "Weekly candles (medium to long-term trend)",
        "1mo": "Monthly candles (long-term investment view)"
    }
    return f"{interval_map.get(interval, interval)} | History: {period} | Trading focus: {trade_horizon}"


def detect_signals(df):
    signals = []
    if len(df) < 3: return signals
    last, prev = df.iloc[-1], df.iloc[-2]

    if not np.isnan(last["SMA_50"]) and not np.isnan(last["SMA_200"]):
        if prev["SMA_50"] <= prev["SMA_200"] and last["SMA_50"] > last["SMA_200"]:
            signals.append(("🟢 Golden Cross", "Bullish long-term"))
        elif prev["SMA_50"] >= prev["SMA_200"] and last["SMA_50"] < last["SMA_200"]:
            signals.append(("🔴 Death Cross", "Bearish long-term"))

    if not np.isnan(last["MACD"]) and not np.isnan(last["MACD_Signal"]):
        if prev["MACD"] <= prev["MACD_Signal"] and last["MACD"] > last["MACD_Signal"]:
            signals.append(("🟢 MACD Bullish Cross", "Momentum up"))
        elif prev["MACD"] >= prev["MACD_Signal"] and last["MACD"] < last["MACD_Signal"]:
            signals.append(("🔴 MACD Bearish Cross", "Momentum down"))

    rsi = last["RSI"]
    if not np.isnan(rsi):
        if rsi > 70: signals.append(("⚠️ RSI Overbought", f"RSI={rsi:.1f}"))
        elif rsi < 30: signals.append(("🟢 RSI Oversold", f"RSI={rsi:.1f}"))

    if not np.isnan(last["BB_High"]) and last["Close"] > last["BB_High"]:
        signals.append(("⚠️ Above Upper BB", "Extended"))
    elif not np.isnan(last["BB_Low"]) and last["Close"] < last["BB_Low"]:
        signals.append(("🟢 Below Lower BB", "Oversold"))

    if not np.isnan(last["Ichimoku_A"]) and not np.isnan(last["Ichimoku_B"]):
        cloud_top = max(last["Ichimoku_A"], last["Ichimoku_B"])
        cloud_bot = min(last["Ichimoku_A"], last["Ichimoku_B"])
        if last["Close"] > cloud_top: signals.append(("🟢 Above Ichimoku Cloud", "Bullish bias"))
        elif last["Close"] < cloud_bot: signals.append(("🔴 Below Ichimoku Cloud", "Bearish bias"))

    return signals

def support_resistance(df, lookback=90):
    """Return multiple practical technical levels"""
    recent = df.tail(lookback)
    if recent.empty or len(recent) < 20:
        return None, None, None, None, None, None
    # Classic support / resistance
    support1 = recent["Low"].quantile(0.10)
    support2 = recent["Low"].quantile(0.25)
    resistance1 = recent["High"].quantile(0.75)
    resistance2 = recent["High"].quantile(0.90)
    # Recent swing levels
    swing_low = recent["Low"].min()
    swing_high = recent["High"].max()
    return support1, support2, resistance1, resistance2, swing_low, swing_high

# -------------------------------------------------
# DCF + Monte-Carlo
# -------------------------------------------------
def simple_dcf(fcf0, growth_yrs, terminal_g, wacc, shares, net_debt=0.0):
    if fcf0 is None or fcf0 <= 0 or shares is None or shares <= 0:
        return None
    n = len(growth_yrs)
    fcf = fcf0
    fcfs = []
    for g in growth_yrs:
        fcf *= (1 + g)
        fcfs.append(fcf)
    pv = sum(f / ((1 + wacc) ** (i+1)) for i, f in enumerate(fcfs))
    tv = fcfs[-1] * (1 + terminal_g) / (wacc - terminal_g) if wacc > terminal_g else fcfs[-1] * 12
    return (pv + tv / ((1 + wacc)**n) - net_debt) / shares

def monte_carlo_dcf(fcf0, base_growth, years, terminal_g, wacc, shares, net_debt, n_sims=2000):
    """Monte-Carlo around growth, WACC and terminal growth"""
    if fcf0 is None or fcf0 <= 0 or shares is None or shares <= 0:
        return None, None, None, None
    results = []
    for _ in range(n_sims):
        g = np.random.normal(base_growth, base_growth * 0.35)
        g = max(-0.05, min(g, 0.40))
        w = np.random.normal(wacc, 0.012)
        w = max(0.06, min(w, 0.18))
        tg = np.random.normal(terminal_g, 0.006)
        tg = max(0.01, min(tg, 0.045))
        if w <= tg: tg = w - 0.01
        val = simple_dcf(fcf0, [g]*years, tg, w, shares, net_debt)
        if val and val > 0:
            results.append(val)
    if not results:
        return None, None, None, None
    arr = np.array(results)
    return np.percentile(arr, 10), np.percentile(arr, 50), np.percentile(arr, 90), arr


def data_accuracy_meter(info, hist, price):
    """Heuristic 0-100 score of how complete the live data looks"""
    score = 0
    max_score = 0
    checks = []
    def add(cond, points, label):
        nonlocal score, max_score
        max_score += points
        if cond:
            score += points
            checks.append((label, True))
        else:
            checks.append((label, False))
    add(price is not None and price > 0, 15, "Live price available")
    add(safe(info, "marketCap") is not None, 10, "Market Cap")
    add(safe(info, "trailingPE") is not None, 10, "Trailing P/E")
    add(safe(info, "forwardPE") is not None, 8, "Forward P/E")
    add(safe(info, "returnOnEquity") is not None, 8, "ROE")
    add(safe(info, "freeCashflow") is not None and safe(info, "freeCashflow", default=0) > 0, 12, "Free Cash Flow")
    add(safe(info, "revenueGrowth") is not None, 8, "Revenue Growth")
    add(safe(info, "debtToEquity") is not None, 7, "Debt/Equity")
    add(hist is not None and len(hist) > 100, 12, "Sufficient price history")
    add(safe(info, "sector") is not None, 5, "Sector identified")
    add(safe(info, "sharesOutstanding") is not None, 5, "Shares outstanding")
    pct = int(round(score / max_score * 100)) if max_score > 0 else 0
    return pct, checks

def quick_score(info, price):
    score = 50
    pe = safe(info, "trailingPE")
    roe = safe(info, "returnOnEquity")
    de = safe(info, "debtToEquity")
    fcf = safe(info, "freeCashflow")
    mcap = safe(info, "marketCap")
    rev_g = safe(info, "revenueGrowth")
    target = safe(info, "targetMeanPrice")
    if pe and 5 < pe < 28: score += 8
    if roe and roe > 0.15: score += 12
    if de is not None and de < 80: score += 8
    if fcf and mcap and fcf / mcap > 0.04: score += 10
    if rev_g and rev_g > 0.10: score += 8
    if target and price and target > price * 1.08: score += 10
    return min(score, 100)

# -------------------------------------------------
# Webhook helpers
# -------------------------------------------------
def send_telegram(bot_token, chat_id, message):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=8)
        return r.status_code == 200
    except:
        return False

def send_email_webhook(webhook_url, subject, body):
    """Works with any webhook that accepts JSON (e.g. Zapier, Make, Formspree-style, or custom)"""
    try:
        r = requests.post(webhook_url, json={"subject": subject, "body": body, "text": body}, timeout=8)
        return r.status_code < 400
    except:
        return False

# -------------------------------------------------
# PDF
# -------------------------------------------------
def make_pdf(ticker, name, price, fair, p10, p50, p90, verdict, upside, entry, exit_base, horizon, signals, ratios_df, currency, mc_note):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Stock Analysis Report: {ticker}", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"{name} | {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Valuation Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Current Price: {currency} {price:.2f}", ln=True)
    pdf.cell(0, 6, f"Blended Fair Value: {currency} {fair:.2f}", ln=True)
    if p10 and p90:
        pdf.cell(0, 6, f"Monte-Carlo Range (10-90%): {currency} {p10:.2f} – {p90:.2f} (median {p50:.2f})", ln=True)
    pdf.cell(0, 6, f"Verdict: {verdict}", ln=True)
    pdf.cell(0, 6, f"Upside: {upside*100:+.1f}%", ln=True)
    pdf.cell(0, 6, f"Entry Zone: <= {currency} {entry:.2f}", ln=True)
    pdf.cell(0, 6, f"Base Exit: {currency} {exit_base:.2f}", ln=True)
    pdf.cell(0, 6, f"Horizon: {horizon}", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key Signals", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for s, d in signals[:7]:
        pdf.cell(0, 5, f"- {s}: {d}", ln=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sector-Relevant Ratios", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for _, row in ratios_df.iterrows():
        pdf.cell(0, 5, f"{row['Metric']}: {row['Value']}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, "Disclaimer: Educational prototype only. Not investment advice. Data from Yahoo Finance. Always verify with official NSE/BSE filings and company reports. Monte-Carlo is illustrative.")
    return pdf.output()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🤝 Buddyy")
st.caption("Your Smart Stock Buddy — Valuation • Technicals • Fundamentals • Alerts • Insights")
st.info("🌐 **Works anywhere** — Open this link on any phone, tablet or laptop. No install, no Excel needed. Just internet.")

with st.sidebar:
    st.header("⚙️ Main Controls")
    ticker_input = st.text_input("Main Ticker", value="TCS.NS",
                                 help="NSE: RELIANCE.NS | BSE: SBIN.BO | US: AAPL")
    ticker = ticker_input.upper().strip()
    st.markdown("**Chart Timeframe**")
    interval = st.selectbox(
        "Candle Interval",
        ["15m", "1h", "4h", "1d", "1wk", "1mo"],
        index=3,
        help="15m/1h/4h = Intraday (limited history) | 1d = Daily | 1wk/1mo = Higher timeframe"
    )
    
    # Auto-select suitable period based on interval
    if interval in ["15m"]:
        period = st.selectbox("History Length", ["5d", "15d", "30d", "60d"], index=2)
    elif interval in ["1h", "4h"]:
        period = st.selectbox("History Length", ["5d", "15d", "30d", "60d", "3mo"], index=2)
    elif interval == "1d":
        period = st.selectbox("History Length", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    else:  # 1wk, 1mo
        period = st.selectbox("History Length", ["6mo", "1y", "2y", "5y"], index=1)

    st.markdown("**Trading Horizon**")
    trade_horizon = st.selectbox(
        "For trading decisions",
        ["Intraday / Scalping", "7 Days (Short Swing)", "14 Days (Swing)", "30 Days (Positional)", "2 Months", "3-6 Months"],
        index=2
    )

    st.markdown("---")
    st.subheader("DCF / Monte-Carlo")
    wacc = st.slider("WACC %", 6.0, 16.0, 10.0, 0.25) / 100
    terminal_g = st.slider("Terminal Growth %", 1.0, 4.5, 3.0, 0.25) / 100
    high_g_years = st.slider("High-growth Years", 3, 8, 5)
    high_g = st.slider("Base High Growth %", 0.0, 25.0, 12.0, 0.5) / 100
    mos = st.slider("Margin of Safety %", 10, 40, 25) / 100
    horizon = st.selectbox("Horizon", ["Short (0-12m)", "Medium (1-3y)", "Long (3y+)"])
    n_sims = st.slider("Monte-Carlo Simulations", 500, 5000, 2000, 500)

    st.markdown("---")
    st.subheader("Watchlist")
    watchlist_raw = st.text_area("Extra tickers (comma sep)", value="TCS.NS, INFY.NS, HDFCBANK.NS", height=60)
    watchlist = [t.strip().upper() for t in watchlist_raw.split(",") if t.strip()]

    st.markdown("---")
    st.subheader("🔔 Alert Webhooks (optional)")
    tg_token = st.text_input("Telegram Bot Token", type="password", help="From @BotFather")
    tg_chat = st.text_input("Telegram Chat ID", help="Your chat or channel ID")
    email_hook = st.text_input("Email / Generic Webhook URL", help="Zapier / Make / custom webhook")
    send_on_analyze = st.checkbox("Send alerts after analysis", value=False)

    analyze = st.button("🔄 Analyze", type="primary", use_container_width=True)

if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""

if analyze or st.session_state.last_ticker != ticker:
    st.session_state.last_ticker = ticker
    data = fetch_data(ticker, period, interval)

    if not data["ok"]:
        err_msg = data.get("error", "Unknown error")
        st.error(f"Failed to load **{ticker}**: {err_msg}")
        
        if "Rate" in str(err_msg) or "Too Many" in str(err_msg):
            st.warning("⏳ Yahoo Finance is rate-limiting requests right now. Please wait 30–60 seconds and click **Analyze** again.")
        
        st.info("""
**Correct ticker format tips:**
- NSE stocks → add `.NS`  (example: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `SBIN.NS`)
- BSE stocks → add `.BO`  (example: `SBIN.BO`)
- US stocks → no suffix (example: `AAPL`, `MSFT`, `GOOGL`)

**Note:** Old ticker `HDFC.NS` no longer works. Use **`HDFCBANK.NS`** instead.
""")
        st.stop()

    info = data["info"]
    hist = add_technicals(data["hist"])
    last = hist.iloc[-1]

    name = safe(info, "longName") or safe(info, "shortName") or ticker
    price = safe(info, "currentPrice") or safe(info, "regularMarketPrice") or float(last["Close"])
    market_cap = safe(info, "marketCap")
    pe = safe(info, "trailingPE")
    fwd_pe = safe(info, "forwardPE")
    pb = safe(info, "priceToBook")
    ps = safe(info, "priceToSalesTrailing12Months")
    ev_ebitda = safe(info, "enterpriseToEbitda")
    ev_sales = safe(info, "enterpriseToRevenue")
    roe = safe(info, "returnOnEquity")
    roa = safe(info, "returnOnAssets")
    de = safe(info, "debtToEquity")
    fcf = safe(info, "freeCashflow")
    rev_g = safe(info, "revenueGrowth")
    eps_g = safe(info, "earningsGrowth")
    shares = safe(info, "sharesOutstanding")
    total_debt = safe(info, "totalDebt") or 0
    total_cash = safe(info, "totalCash") or 0
    net_debt = total_debt - total_cash
    sector = safe(info, "sector") or "default"
    industry = safe(info, "industry") or ""
    beta = safe(info, "beta")
    div_yield = safe(info, "dividendYield")
    target = safe(info, "targetMeanPrice")
    currency = safe(info, "currency") or ("INR" if ".NS" in ticker or ".BO" in ticker else "USD")

    sector_key = "default"
    for k in SECTOR_RATIOS:
        if k.lower() in sector.lower() or k.lower() in industry.lower():
            sector_key = k
            break

    # Header
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Price", f"{currency} {price:.2f}")
    c2.metric("Market Cap", fmt_money(market_cap))
    c3.metric("P/E", f"{pe:.1f}" if pe else "N/A")
    c4.metric("Fwd P/E", f"{fwd_pe:.1f}" if fwd_pe else "N/A")
    c5.metric("Analyst Target", f"{target:.1f}" if target else "N/A")
    st.markdown(f"**{name}** · **{sector}** · {industry} · Beta {beta or 'N/A'}")

    # ---------- Data Accuracy Meter + Manual Override ----------
    acc_pct, acc_checks = data_accuracy_meter(info, hist, price)
    
    st.markdown("### 📡 Data Accuracy Meter")
    acc_col1, acc_col2 = st.columns([1, 3])
    with acc_col1:
        if acc_pct >= 80:
            st.success(f"**{acc_pct}%** Reliable")
        elif acc_pct >= 55:
            st.warning(f"**{acc_pct}%** Partial")
        else:
            st.error(f"**{acc_pct}%** Low")
    with acc_col2:
        st.progress(acc_pct / 100)
        missing = [lab for lab, ok in acc_checks if not ok]
        if missing:
            st.caption("Missing / weak: " + ", ".join(missing[:5]))
        else:
            st.caption("All key data points present")

    with st.expander("✏️ Manual Data Entry (override if live data looks wrong)"):
        st.caption("Use this when Yahoo data is incomplete or incorrect. Values you enter will be used for valuation & ratios.")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            man_price = st.number_input("Price", value=float(price) if price else 0.0, min_value=0.0, format="%.2f", key="man_price")
            man_pe = st.number_input("Trailing P/E", value=float(pe) if pe else 0.0, min_value=0.0, format="%.2f", key="man_pe")
            man_fwd_pe = st.number_input("Forward P/E", value=float(fwd_pe) if fwd_pe else 0.0, min_value=0.0, format="%.2f", key="man_fwd")
        with mcol2:
            man_roe = st.number_input("ROE (e.g. 0.18 for 18%)", value=float(roe) if roe else 0.0, min_value=-1.0, max_value=5.0, format="%.4f", key="man_roe")
            man_fcf = st.number_input("Free Cash Flow (absolute)", value=float(fcf) if fcf else 0.0, format="%.0f", key="man_fcf")
            man_mcap = st.number_input("Market Cap", value=float(market_cap) if market_cap else 0.0, format="%.0f", key="man_mcap")
        with mcol3:
            man_de = st.number_input("Debt/Equity", value=float(de) if de else 0.0, min_value=0.0, format="%.2f", key="man_de")
            man_revg = st.number_input("Revenue Growth (e.g. 0.12)", value=float(rev_g) if rev_g else 0.0, format="%.4f", key="man_revg")
            man_shares = st.number_input("Shares Outstanding", value=float(shares) if shares else 0.0, format="%.0f", key="man_shares")
        
        apply_manual = st.checkbox("Apply my manual values for this analysis", value=False, key="apply_manual")
        
        if apply_manual:
            price = man_price if man_price > 0 else price
            pe = man_pe if man_pe > 0 else pe
            fwd_pe = man_fwd_pe if man_fwd_pe > 0 else fwd_pe
            roe = man_roe if man_roe != 0 else roe
            fcf = man_fcf if man_fcf > 0 else fcf
            market_cap = man_mcap if man_mcap > 0 else market_cap
            de = man_de if man_de >= 0 else de
            rev_g = man_revg if man_revg != 0 else rev_g
            shares = man_shares if man_shares > 0 else shares
            st.success("✅ Manual values applied for this run.")



    # ---------- Returns Calculator ----------
    st.markdown("### 📈 Returns Calculator")
    returns = calc_returns(hist, price)
    if returns:
        rcols = st.columns(len(returns))
        for i, (label, ret) in enumerate(returns.items()):
            color = "normal" if ret >= 0 else "inverse"
            rcols[i].metric(label, f"{ret:+.2f}%")
    else:
        st.info("Not enough history to calculate returns.")

    # ---------- News ----------
    st.markdown("### 📰 Latest News")
    news_items = fetch_news(ticker)
    if news_items:
        for item in news_items[:6]:
            title = item.get("title", "")
            pub = item.get("publisher", "")
            link = item.get("link", "")
            if link:
                st.markdown(f"- [{title}]({link}) — _{pub}_")
            else:
                st.markdown(f"- **{title}** — _{pub}_")
    else:
        st.info("No recent news available for this ticker right now.")

    st.markdown("---")

    # Valuation core
    if fcf is None or fcf <= 0:
        fcf = (market_cap or 1e11) * 0.025
    dcf_ps = simple_dcf(fcf, [high_g]*high_g_years, terminal_g, wacc,
                        shares or (market_cap/price if market_cap else 1e9), net_debt)
    fair_pe = 25 if any(x in sector for x in ["Tech", "Software"]) else 18
    mult_ps = None
    if pe and fwd_pe:
        mult_ps = price * (fair_pe / ((pe + fwd_pe) / 2))
    elif pe:
        mult_ps = price * (fair_pe / pe)
    fair = 0.6 * dcf_ps + 0.4 * mult_ps if dcf_ps and mult_ps else (dcf_ps or mult_ps or price)

    # Monte-Carlo
    with st.spinner("Running Monte-Carlo simulations..."):
        p10, p50, p90, mc_arr = monte_carlo_dcf(
            fcf, high_g, high_g_years, terminal_g, wacc,
            shares or (market_cap/price if market_cap else 1e9), net_debt, n_sims
        )

    upside = (fair - price) / price
    if upside > mos: verdict = "🟢 UNDERVALUED"
    elif upside < -mos: verdict = "🔴 OVERVALUED"
    else: verdict = "🟡 FAIRLY VALUED"

    # Fundamental targets (value investing)
    entry_fund = fair * (1 - mos)
    exit_base_fund = fair * 1.08
    exit_bull_fund = fair * 1.25

    # Technical levels (for trading around current price)
    support1, support2, resistance1, resistance2, swing_low, swing_high = support_resistance(hist)
    
    # Practical Technical Entry / Exit near current price
    last_close = float(hist["Close"].iloc[-1])
    sma20 = float(hist["SMA_20"].iloc[-1]) if not np.isnan(hist["SMA_20"].iloc[-1]) else last_close
    sma50 = float(hist["SMA_50"].iloc[-1]) if not np.isnan(hist["SMA_50"].iloc[-1]) else last_close
    
    # Technical Entry zones (buy near support)
    tech_entry_aggressive = support1 if support1 else last_close * 0.97
    tech_entry_conservative = support2 if support2 else last_close * 0.98
    
    # Technical Exit / Targets (sell near resistance)
    tech_exit1 = resistance1 if resistance1 else last_close * 1.04
    tech_exit2 = resistance2 if resistance2 else last_close * 1.08
    tech_exit_swing = swing_high if swing_high else last_close * 1.12

    # Keep old variable names for PDF compatibility
    entry = tech_entry_conservative
    exit_base = tech_exit1
    exit_bull = tech_exit2
    support, resistance = support1, resistance1
    signals = detect_signals(hist)

    # Tabs
    tabs = st.tabs([
        "📊 Valuation + Monte-Carlo", "📉 Technicals + Volume Profile", "📋 Sector Ratios",
        "🔔 Alerts", "🔄 5 Alternatives", "👁 Watchlist", "📄 PDF Export"
    ])

    # ---- Tab 1: Valuation + MC ----
    with tabs[0]:
        st.subheader("Intrinsic Valuation + Monte-Carlo Range")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DCF Fair Value", f"{currency} {dcf_ps:.2f}" if dcf_ps else "N/A")
        col2.metric("Blended Fair Value", f"{currency} {fair:.2f}")
        col3.metric("Upside", f"{upside*100:+.1f}%")
        col4.metric("MOS", f"{(fair-price)/fair*100:.1f}%")
        st.markdown(f"### {verdict}")

        if p10 and p90:
            st.markdown("#### Monte-Carlo Fair Value Distribution")
            m1, m2, m3 = st.columns(3)
            m1.metric("Bearish (10th %ile)", f"{currency} {p10:.2f}")
            m2.metric("Median (50th)", f"{currency} {p50:.2f}")
            m3.metric("Bullish (90th %ile)", f"{currency} {p90:.2f}")

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=mc_arr, nbinsx=50, name="Simulated FV", marker_color="#1f77b4"))
            fig_mc.add_vline(x=price, line_dash="dash", line_color="red", annotation_text="Current Price")
            fig_mc.add_vline(x=fair, line_dash="dash", line_color="green", annotation_text="Blended FV")
            fig_mc.update_layout(height=350, title="Monte-Carlo Fair Value Distribution", xaxis_title="Fair Value", yaxis_title="Count")
            st.plotly_chart(fig_mc, use_container_width=True)
            st.caption(f"Based on {n_sims} simulations varying growth, WACC and terminal growth. Range shows uncertainty, not prediction.")

        st.markdown("#### 📍 Technical Entry / Exit Targets (for trading)")
        st.caption("Based on Support, Resistance, Swing High/Low & Moving Averages — useful for short/medium term trades near current price.")
        
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("**Buy / Entry Zones**")
            st.write(f"Aggressive Entry: **{currency} {tech_entry_aggressive:.2f}**")
            st.write(f"Conservative Entry: **{currency} {tech_entry_conservative:.2f}**")
            st.write(f"Near SMA20: {currency} {sma20:.2f}")
        with t2:
            st.markdown("**Sell / Exit Targets**")
            st.write(f"Target 1 (Resistance): **{currency} {tech_exit1:.2f}**")
            st.write(f"Target 2 (Strong Res): **{currency} {tech_exit2:.2f}**")
            st.write(f"Swing High Target: **{currency} {tech_exit_swing:.2f}**")
        with t3:
            st.markdown("**Key Levels**")
            st.write(f"Support 1: {currency} {support1:.2f}" if support1 else "Support 1: N/A")
            st.write(f"Support 2: {currency} {support2:.2f}" if support2 else "Support 2: N/A")
            st.write(f"Resistance 1: {currency} {resistance1:.2f}" if resistance1 else "Res 1: N/A")
            st.write(f"Current Price: **{currency} {price:.2f}**")

        st.markdown("---")
        st.markdown("#### 💰 Fundamental Entry / Exit (Value Investing)")
        st.caption("Based on DCF fair value — use only if you are a long-term value investor.")
        f1, f2, f3 = st.columns(3)
        f1.write(f"**Fund. Entry ≤** {currency} {entry_fund:.2f}")
        f2.write(f"**Fund. Base Exit** {currency} {exit_base_fund:.2f}")
        f3.write(f"**Fund. Bull Exit** {currency} {exit_bull_fund:.2f}")

        # Scenarios
        scen = []
        for lab, mult, wadj, tg in [("Bear", 0.4, 0.02, 0.7), ("Base", 1.0, 0, 1.0), ("Bull", 1.6, -0.01, 1.15)]:
            v = simple_dcf(fcf, [high_g*mult]*high_g_years, terminal_g*tg, wacc+wadj,
                           shares or market_cap/price, net_debt)
            if v:
                scen.append({"Scenario": lab, "Growth": f"{high_g*mult*100:.1f}%",
                             "Fair Value": f"{currency} {v:.2f}", "Upside": f"{(v-price)/price*100:+.1f}%"})
        st.dataframe(pd.DataFrame(scen), use_container_width=True, hide_index=True)

    # ---- Tab 2: Technicals + Volume Profile ----
    with tabs[1]:
        st.subheader("Technical Indicators + Ichimoku + Volume Profile")
        
        # Timeframe interpretation
        st.markdown(f"**Selected Timeframe:** {timeframe_interpretation(interval, period, trade_horizon)}")
        
        if signals:
            for s, d in signals:
                st.markdown(f"**{s}** — {d}")
        else:
            st.info("No major crossover signals.")

        # ---- Candlestick Patterns ----
        st.markdown("#### 🕯️ Live Candlestick Patterns")
        candle_patterns = detect_candlestick_patterns(hist)
        if candle_patterns:
            for name, meaning, idx in candle_patterns:
                candle_label = "Latest candle" if idx == -1 else f"{abs(idx)} candle(s) ago"
                st.success(f"**{name}** ({candle_label}) — {meaning}")
        else:
            st.info("No strong candlestick patterns detected on recent candles.")

        # ---- Chart Patterns ----
        st.markdown("#### 📐 Chart Patterns Forming")
        chart_patterns = detect_chart_patterns(hist)
        if chart_patterns:
            for name, meaning, bias in chart_patterns:
                if bias == "Bullish":
                    st.success(f"**{name}** — {meaning}")
                elif bias == "Bearish":
                    st.warning(f"**{name}** — {meaning}")
                else:
                    st.info(f"**{name}** — {meaning}")

        else:
            st.info("No clear chart pattern currently forming.")

        # ---- Pivot Points ----
        st.markdown("#### 🎯 Pivot Points (Classic)")
        pivots = calc_pivot_points(hist)
        if pivots:
            pcols = st.columns(4)
            keys = list(pivots.keys())
            for i, k in enumerate(keys):
                with pcols[i % 4]:
                    st.metric(k, f"{pivots[k]:.2f}")
        else:
            st.info("Not enough data for pivot points.")

        # ---- Fibonacci Levels ----
        st.markdown("#### 📐 Fibonacci Retracement Levels")
        fibs = calc_fibonacci_levels(hist)
        if fibs:
            fcols = st.columns(4)
            for i, (k, v) in enumerate(fibs.items()):
                with fcols[i % 4]:
                    st.metric(k, f"{v:.2f}")
            st.caption("Levels drawn from recent swing high to swing low. Price near 38.2% / 50% / 61.8% often acts as support/resistance.")
        else:
            st.info("Not enough data for Fibonacci levels.")

        # ---- MA Crossover Status ----
        st.markdown("#### 🔄 Moving Average Crossover Status")
        last = hist.iloc[-1]
        ma_status = []
        if not np.isnan(last.get("SMA_20", np.nan)) and not np.isnan(last.get("SMA_50", np.nan)):
            if last["SMA_20"] > last["SMA_50"]:
                ma_status.append("SMA20 > SMA50 → Short-term bullish")
            else:
                ma_status.append("SMA20 < SMA50 → Short-term bearish")
        if not np.isnan(last.get("SMA_50", np.nan)) and not np.isnan(last.get("SMA_200", np.nan)):
            if last["SMA_50"] > last["SMA_200"]:
                ma_status.append("SMA50 > SMA200 → Golden Cross zone (Long-term bullish)")
            else:
                ma_status.append("SMA50 < SMA200 → Death Cross zone (Long-term bearish)")
        if not np.isnan(last.get("EMA_12", np.nan)) and not np.isnan(last.get("EMA_26", np.nan)):
            if last["EMA_12"] > last["EMA_26"]:
                ma_status.append("EMA12 > EMA26 → MACD line positive bias")
            else:
                ma_status.append("EMA12 < EMA26 → MACD line negative bias")
        if ma_status:
            for s in ma_status:
                st.write(f"• {s}")
        else:
            st.info("MA data not available for crossover status.")


        t1, t2, t3, t4 = st.columns(4)
        t1.metric("RSI (14)", f"{last['RSI']:.1f}" if not np.isnan(last['RSI']) else "N/A")
        t2.metric("MACD Hist", f"{last['MACD_Hist']:.2f}" if not np.isnan(last['MACD_Hist']) else "N/A")
        t3.metric("Support", f"{support:.2f}" if support else "N/A")
        t4.metric("Resistance", f"{resistance:.2f}" if resistance else "N/A")

        # Volume Profile
        centers, vols = volume_profile(hist)
        if centers is not None:
            st.markdown("#### Volume Profile (last ~120 sessions)")
            fig_vp = go.Figure()
            fig_vp.add_trace(go.Bar(y=centers, x=vols, orientation="h", name="Volume at Price",
                                    marker_color="rgba(30,144,255,0.6)"))
            fig_vp.add_hline(y=price, line_dash="dash", line_color="red", annotation_text="Current")
            if support: fig_vp.add_hline(y=support, line_dash="dot", line_color="lime")
            if resistance: fig_vp.add_hline(y=resistance, line_dash="dot", line_color="orange")
            fig_vp.update_layout(height=380, title="Volume Profile – High volume nodes act as support/resistance",
                                 xaxis_title="Volume", yaxis_title="Price", margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig_vp, use_container_width=True)

        # Main chart
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            row_heights=[0.45, 0.2, 0.2, 0.15],
                            subplot_titles=("Price + MAs + Ichimoku Cloud + Fair Value", "MACD", "RSI", "Volume"))
        fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"],
                                     low=hist["Low"], close=hist["Close"], name="OHLC"), row=1, col=1)
        for col, clr in [("SMA_20", "orange"), ("SMA_50", "blue"), ("SMA_200", "purple")]:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[col], name=col, line=dict(width=1.1, color=clr)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Ichimoku_A"], line=dict(width=0.7, color="green"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Ichimoku_B"], line=dict(width=0.7, color="red"),
                                 fill="tonexty", fillcolor="rgba(0,180,0,0.12)", showlegend=False), row=1, col=1)
        fig.add_hline(y=fair, line_dash="dash", line_color="green", row=1, col=1)
        if support: fig.add_hline(y=support, line_dash="dot", line_color="lime", row=1, col=1)
        if resistance: fig.add_hline(y=resistance, line_dash="dot", line_color="red", row=1, col=1)

        fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD"], name="MACD", line=dict(color="blue")), row=2, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD_Signal"], name="Signal", line=dict(color="orange")), row=2, col=1)
        fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_Hist"], marker_color=["green" if v >= 0 else "red" for v in hist["MACD_Hist"]]), row=2, col=1)

        fig.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], name="RSI", line=dict(color="purple")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        vol_c = ["green" if c >= o else "red" for c, o in zip(hist["Close"], hist["Open"])]
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=vol_c), row=4, col=1)

        fig.update_layout(height=920, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02),
                          margin=dict(l=10, r=10, t=40, b=10))
        fig.update_yaxes(range=[0, 100], row=3, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 3: Sector Ratios ----
    with tabs[2]:
        st.subheader(f"Sector-Relevant Ratios — {sector_key}")
        all_ratios = {
            "Trailing P/E": f"{pe:.1f}" if pe else "N/A",
            "Forward P/E": f"{fwd_pe:.1f}" if fwd_pe else "N/A",
            "PEG": f"{pe/(eps_g*100):.2f}" if pe and eps_g and eps_g > 0 else "N/A",
            "Price/Book": f"{pb:.2f}" if pb else "N/A",
            "Price/Sales": f"{ps:.2f}" if ps else "N/A",
            "EV/EBITDA": f"{ev_ebitda:.1f}" if ev_ebitda else "N/A",
            "EV/Sales": f"{ev_sales:.1f}" if ev_sales else "N/A",
            "ROE": fmt_pct(roe) if roe else "N/A",
            "ROA": fmt_pct(roa) if roa else "N/A",
            "FCF Yield": f"{(fcf/market_cap)*100:.1f}%" if fcf and market_cap else "N/A",
            "Debt/Equity": f"{de:.1f}" if de else "N/A",
            "Revenue Growth": fmt_pct(rev_g) if rev_g else "N/A",
            "EPS Growth": fmt_pct(eps_g) if eps_g else "N/A",
            "Dividend Yield": fmt_pct(div_yield) if div_yield else "N/A",
            "Beta": f"{beta:.2f}" if beta else "N/A",
        }
        relevant = SECTOR_RATIOS.get(sector_key, SECTOR_RATIOS["default"])
        ratios_df = pd.DataFrame([{"Metric": m, "Value": all_ratios.get(m, "N/A")} for m in relevant if m in all_ratios])
        st.dataframe(ratios_df, use_container_width=True, hide_index=True)
        st.metric("Quality Score (0-100)", quick_score(info, price))

    # ---- Tab 4: Alerts ----
    with tabs[3]:
        st.subheader("🔔 Alerts & Notifications")
        alerts = []
        if upside > mos:
            alerts.append(("🟢 Valuation", f"Undervalued by {upside*100:.1f}%. Entry zone ≤ {currency} {entry:.2f}"))
        if upside < -mos:
            alerts.append(("🔴 Valuation", f"Overvalued by {abs(upside)*100:.1f}%"))
        for s, d in signals:
            level = "⚠️" if any(x in s for x in ["Overbought", "Bearish", "Death", "Below"]) else "🟢"
            alerts.append((f"{level} Technical", f"{s} — {d}"))
        if not np.isnan(last["RSI"]) and last["RSI"] > 75:
            alerts.append(("⚠️ RSI", "Extremely overbought"))
        if last["Volume"] > 2 * (last["Volume_SMA"] or 1):
            alerts.append(("📢 Volume", "Volume spike > 2× average"))

        if alerts:
            for title, msg in alerts:
                if "⚠️" in title or "🔴" in title:
                    st.warning(f"**{title}**: {msg}")
                else:
                    st.success(f"**{title}**: {msg}")
        else:
            st.info("No active alerts.")

        # Webhook send
        if send_on_analyze and alerts:
            msg = f"<b>{ticker} Alerts</b>\nPrice: {currency} {price:.2f}\nFair: {currency} {fair:.2f}\nVerdict: {verdict}\n\n"
            for t, m in alerts:
                msg += f"• {t}: {m}\n"
            sent = False
            if tg_token and tg_chat:
                sent = send_telegram(tg_token, tg_chat, msg) or sent
            if email_hook:
                sent = send_email_webhook(email_hook, f"{ticker} Analysis Alerts", msg) or sent
            if sent:
                st.success("Alerts sent via webhook(s).")
            elif tg_token or email_hook:
                st.error("Webhook send failed – check token/URL.")

        st.caption("Telegram: create bot via @BotFather, get chat ID via @userinfobot. Email: use Zapier/Make webhook URL.")

    # ---- Tab 5: Alternatives ----
    with tabs[4]:
        st.subheader("🔄 5 Alternative Shares (higher quality/upside signals)")
        peers = [p for p in SECTOR_PEERS.get(sector_key, SECTOR_PEERS["default"]) if p != ticker][:7]
        alt_rows = []
        for p in peers:
            try:
                pd_ = fetch_data(p, "1y")
                if not pd_["ok"]: continue
                i = pd_["info"]
                pr = safe(i, "currentPrice") or safe(i, "regularMarketPrice")
                if not pr: continue
                sc = quick_score(i, pr)
                tg = safe(i, "targetMeanPrice")
                up = ((tg - pr) / pr * 100) if tg and pr else None
                alt_rows.append({
                    "Ticker": p,
                    "Name": (safe(i, "shortName") or p)[:22],
                    "Price": f"{pr:.1f}",
                    "P/E": f"{safe(i,'trailingPE'):.1f}" if safe(i, "trailingPE") else "N/A",
                    "ROE": fmt_pct(safe(i, "returnOnEquity")) if safe(i, "returnOnEquity") else "N/A",
                    "Quality": sc,
                    "Analyst Upside": f"{up:+.1f}%" if up else "N/A"
                })
            except:
                continue
        if alt_rows:
            st.dataframe(pd.DataFrame(alt_rows).sort_values("Quality", ascending=False).head(5),
                         use_container_width=True, hide_index=True)
        else:
            st.warning("Peer data temporarily unavailable (rate limit). Try again in a minute.")

    # ---- Tab 6: Watchlist ----
    with tabs[5]:
        st.subheader("👁 Watchlist Comparison")
        all_tickers = [ticker] + watchlist
        w_rows = []
        for w in all_tickers:
            try:
                wd = fetch_data(w, "6mo")
                if not wd["ok"]: continue
                wi = wd["info"]
                wp = safe(wi, "currentPrice") or safe(wi, "regularMarketPrice")
                w_rows.append({
                    "Ticker": w,
                    "Price": f"{wp:.1f}" if wp else "N/A",
                    "P/E": f"{safe(wi,'trailingPE'):.1f}" if safe(wi, "trailingPE") else "N/A",
                    "Fwd P/E": f"{safe(wi,'forwardPE'):.1f}" if safe(wi, "forwardPE") else "N/A",
                    "ROE": fmt_pct(safe(wi, "returnOnEquity")) if safe(wi, "returnOnEquity") else "N/A",
                    "Mkt Cap": fmt_money(safe(wi, "marketCap")),
                    "Score": quick_score(wi, wp) if wp else 0
                })
            except:
                continue
        if w_rows:
            st.dataframe(pd.DataFrame(w_rows), use_container_width=True, hide_index=True)

    # ---- Tab 7: PDF ----
    with tabs[6]:
        st.subheader("📄 Export Full Report as PDF")
        if st.button("Generate & Download PDF"):
            pdf_bytes = make_pdf(
                ticker, name, price, fair, p10, p50, p90, verdict, upside,
                entry, exit_base, horizon, signals, ratios_df, currency, ""
            )
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{ticker}_full_analysis.pdf">📥 Click to Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("PDF generated.")

    st.success("✅ Full analysis complete.")

# Deployment
with st.expander("🌐 Deploy Buddyy as a public website (free permanent link)"):
    st.markdown("""
**Streamlit Community Cloud (recommended)**

1. Go to https://share.streamlit.io → Login with GitHub  
2. Create a public GitHub repository (you can name it **buddyy**)  
3. Upload:
   - `stock_valuation_prototype.py`
   - `requirements.txt` (content below)
4. New app → select repo → Main file = `stock_valuation_prototype.py` → Deploy  
5. You get a permanent link, ideally set the app name to **buddyy**:  
   `https://buddyy.streamlit.app`  
   Open from any phone or laptop – no install required.

**requirements.txt**
```
streamlit
yfinance
pandas
numpy
plotly
ta
fpdf2
scipy
requests
```

**Data source note**  
Live data comes from Yahoo Finance (covers NSE `.NS`, BSE `.BO` and global markets). It is the most practical free source. Official exchange APIs require keys and are restricted. Always verify important decisions with NSE/BSE official site and company filings.
""")

st.markdown("---")
st.caption("Educational prototype only • Not investment advice • Data: Yahoo Finance • Use Accuracy Meter + Manual Entry when needed • Always cross-check with official sources")
