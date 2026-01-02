import io
import json
import time
import math
import feedparser
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "universe.json"
OUT_DIR = ROOT / "docs" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "stock-analysis-system/0.1 (paper trading)"}

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def stooq_symbol(ticker: str) -> str:
    # Stooq symbols are lower-case. Examples: aapl.us, xro.au
    if ticker.endswith(".AX"):
        return ticker.replace(".AX", "").lower() + ".au"
    return ticker.lower() + ".us"

def fetch_stooq_daily(ticker: str, days: int = 420) -> pd.DataFrame:
    sym = stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").tail(days).reset_index(drop=True)
    df["Ticker"] = ticker
    return df

def gnews_rss_url(ticker: str) -> str:
    q = requests.utils.quote(ticker + " stock")
    return f"https://news.google.com/rss/search?q={q}&hl=en-AU&gl=AU&ceid=AU:en"

def fetch_news_headlines(ticker: str, max_items: int = 8) -> list[dict]:
    url = gnews_rss_url(ticker)
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        items.append({
            "title": getattr(e, "title", "")[:300],
            "link": getattr(e, "link", ""),
            "published": getattr(e, "published", ""),
            "source": getattr(getattr(e, "source", None), "title", "") if hasattr(e, "source") else ""
        })
    return items

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def compute_signals(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float)
    df = df.copy()
    df["ret1"] = close.pct_change()
    df["ma50"] = close.rolling(50).mean()
    df["ma200"] = close.rolling(200).mean()
    df["rsi14"] = rsi(close, 14)
    df["vol20"] = df["ret1"].rolling(20).std() * math.sqrt(252)

    latest = df.iloc[-1]
    trend = 0
    if pd.notna(latest["ma50"]) and pd.notna(latest["ma200"]):
        trend = 1 if latest["ma50"] > latest["ma200"] else (-1 if latest["ma50"] < latest["ma200"] else 0)

    r63 = (close.iloc[-1] / close.iloc[-64] - 1) if len(close) > 64 else float("nan")
    r126 = (close.iloc[-1] / close.iloc[-127] - 1) if len(close) > 127 else float("nan")

    vol = float(latest["vol20"]) if pd.notna(latest["vol20"]) else float("nan")
    rsi_v = float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else float("nan")

    action = "HOLD"
    if trend == 1 and (not math.isnan(r126) and r126 > 0.10) and (math.isnan(rsi_v) or rsi_v < 70):
        action = "BUY"
    if trend == -1 and (not math.isnan(r126) and r126 < -0.05):
        action = "SELL"

    conf = 50
    if action == "BUY":
        conf += 15
    elif action == "SELL":
        conf += 10

    if not math.isnan(r126):
        conf += max(-10, min(10, int(r126 * 50)))

    if not math.isnan(vol):
        conf += int(max(-20, min(0, -vol * 30)))

    conf = max(0, min(100, conf))

    return {
        "last_close": float(close.iloc[-1]),
        "trend": int(trend),
        "mom_3m": None if math.isnan(r63) else float(r63),
        "mom_6m": None if math.isnan(r126) else float(r126),
        "vol_20d_ann": None if math.isnan(vol) else float(vol),
        "rsi14": None if math.isnan(rsi_v) else float(rsi_v),
        "action": action,
        "confidence": int(conf),
    }

def horizon_ranges(sig: dict) -> dict:
    vol = sig.get("vol_20d_ann")
    if vol is None:
        return {"1D": None, "1W": None, "1M": None}
    daily = vol / math.sqrt(252)
    one_week = daily * math.sqrt(5)
    one_month = daily * math.sqrt(21)
    def rng(x): return {"low": -1.0*x, "high": 1.0*x}
    return {"1D": rng(daily), "1W": rng(one_week), "1M": rng(one_month)}

def main():
    universe = json.loads(Path(CONFIG).read_text(encoding="utf-8"))

    all_rows = []
    news_blob = {}
    prices_blob = {}

    for region, cats in universe.items():
        for cat, tickers in cats.items():
            for t in tickers:
                try:
                    df = fetch_stooq_daily(t, days=420)
                except Exception as e:
                    all_rows.append({"region": region, "category": cat, "ticker": t, "status": "NO_DATA", "error": str(e)[:160]})
                    continue

                sig = compute_signals(df)
                ranges = horizon_ranges(sig)

                prices_blob[t] = df.tail(250).to_dict(orient="records")

                try:
                    news_blob[t] = fetch_news_headlines(t)
                except Exception as e:
                    news_blob[t] = [{"title": f"News fetch failed: {str(e)[:160]}", "link": "", "published": "", "source": ""}]

                all_rows.append({
                    "region": region,
                    "category": cat,
                    "ticker": t,
                    **sig,
                    "range_1d": ranges["1D"],
                    "range_1w": ranges["1W"],
                    "range_1m": ranges["1M"],
                    "status": "OK",
                    "asof_utc": now_utc()
                })

                time.sleep(0.2)

    out = {
        "asof_utc": now_utc(),
        "rows": all_rows,
        "news": news_blob,
        "prices": prices_blob,
        "method": {
            "signals": "MA50/MA200 trend + 3M/6M momentum + RSI + 20D volatility; conservative actions; confidence is heuristic (calibrate later).",
            "disclaimer": "Paper-trading research only."
        }
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "snapshot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "snapshot.csv", index=False)

if __name__ == "__main__":
    main()
