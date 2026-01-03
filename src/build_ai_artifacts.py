#!/usr/bin/env python3
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "docs" / "data" / "snapshot.json"
BRIEFING_PATH = ROOT / "docs" / "data" / "ai_briefing.json"
HISTORY_DIR = ROOT / "docs" / "data" / "history"
HISTORY_INDEX_PATH = HISTORY_DIR / "index.json"


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def pct_change(prev_close, last_close):
    p = safe_float(prev_close)
    l = safe_float(last_close)
    if p is None or l is None or p == 0:
        return None
    return (l - p) / p


def load_snapshot():
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Missing snapshot.json at {SNAPSHOT_PATH}")

    with SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
        snap = json.load(f)

    # Dashboard expects asof_utc; some older formats used generated_at_utc etc.
    asof = snap.get("asof_utc") or snap.get("generated_at_utc") or snap.get("generated_at") or utc_now_iso()

    # Rows: prefer snap["rows"] if present; otherwise attempt older layouts
    rows = snap.get("rows")
    if not isinstance(rows, list):
        rows = []

    return asof, snap, rows


def build_winners_losers(rows, top_n=5):
    # Only consider rows with OK status and with both prev_close & last_close
    enriched = []
    for r in rows:
        if str(r.get("status", "")).upper() != "OK":
            continue
        t = r.get("ticker")
        pc = r.get("prev_close")
        lc = r.get("last_close")
        chg = pct_change(pc, lc)
        if t and chg is not None:
            enriched.append((t, chg, pc, lc, r))

    winners = sorted(enriched, key=lambda x: x[1], reverse=True)[:top_n]
    losers = sorted(enriched, key=lambda x: x[1])[:top_n]
    return winners, losers


def build_action_summary(rows):
    counts = {"BUY": 0, "HOLD": 0, "SELL": 0, "OTHER": 0}
    for r in rows:
        if str(r.get("status", "")).upper() != "OK":
            continue
        a = str(r.get("action", "")).upper()
        if a in counts:
            counts[a] += 1
        else:
            counts["OTHER"] += 1
    return counts


def write_briefing(asof_utc, rows):
    winners, losers = build_winners_losers(rows, top_n=5)
    action_counts = build_action_summary(rows)

    def pack(item):
        t, chg, pc, lc, r = item
        return {
            "ticker": t,
            "pct_change": chg,  # decimal (e.g. 0.0123 = +1.23%)
            "prev_close": pc,
            "last_close": lc,
            "action": r.get("action"),
            "confidence": r.get("confidence"),
        }

    briefing = {
        "asof_utc": asof_utc,
        "generated_at_utc": utc_now_iso(),
        "counts": {
            "rows_total": len(rows),
            "rows_ok": sum(1 for r in rows if str(r.get("status", "")).upper() == "OK"),
            "actions": action_counts,
        },
        "top_winners": [pack(x) for x in winners],
        "top_losers": [pack(x) for x in losers],
        # optional lightweight flags you can expand later:
        "flags": [],
    }

    BRIEFING_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIEFING_PATH.write_text(json.dumps(briefing, indent=2), encoding="utf-8")


def history_month_file(asof_utc):
    # asof_utc like "2026-01-03T19:20:00+00:00" or "...Z"
    # We use UTC month partition: YYYY-MM
    ym = asof_utc[:7]
    return HISTORY_DIR / f"{ym}.jsonl"


def append_history(asof_utc, rows):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = history_month_file(asof_utc)

    # Append one line per OK row. This stays small.
    lines = []
    asof_date = asof_utc[:10]

    for r in rows:
        if str(r.get("status", "")).upper() != "OK":
            continue

        rec = {
            "asof_date_utc": asof_date,   # YYYY-MM-DD
            "asof_utc": asof_utc,
            "ticker": r.get("ticker"),
            "region": r.get("region"),
            "category": r.get("category"),
            "action": r.get("action"),
            "confidence": r.get("confidence"),
            "prev_close": r.get("prev_close"),
            "last_close": r.get("last_close"),
            "pred_close_1d": r.get("pred_close_1d"),
            "pred_confidence_1d": r.get("pred_confidence_1d"),
            "range_1d": r.get("range_1d"),
            "vol_20d_ann": r.get("vol_20d_ann"),
            "mom_6m": r.get("mom_6m"),
            "rsi14": r.get("rsi14"),
        }
        lines.append(json.dumps(rec, ensure_ascii=False))

    if not lines:
        return

    with path.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def update_history_index():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in HISTORY_DIR.glob("*.jsonl")])
    index = {
        "generated_at_utc": utc_now_iso(),
        "files": [],
    }

    for p in files:
        # We don’t scan the whole file (keeps it fast). Just store name + size.
        index["files"].append(
            {
                "file": p.name,
                "bytes": p.stat().st_size,
            }
        )

    HISTORY_INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def main():
    asof_utc, snap, rows = load_snapshot()

    # 1) Rolling daily briefing (overwritten)
    write_briefing(asof_utc, rows)

    # 2) Append history (monthly partition)
    append_history(asof_utc, rows)

    # 3) Index for convenience
    update_history_index()

    print("✅ AI artifacts written:")
    print(f" - {BRIEFING_PATH}")
    print(f" - {HISTORY_DIR}/YYYY-MM.jsonl")
    print(f" - {HISTORY_INDEX_PATH}")


if __name__ == "__main__":
    main()
