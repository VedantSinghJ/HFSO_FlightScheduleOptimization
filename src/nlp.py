import re
import pandas as pd
from .metrics import busiest_slots, best_time_to_operate, runway_load_index
from .optimizer import greedy_slot_tuner
from .influence import influence_table

# --------- helpers: intent, kind, airport parsing ----------
_PAT_IATA = re.compile(r"\b([A-Z]{3})\b")
_PAT_CITY = re.compile(r"\b(mumbai|bombay|delhi|new\s*delhi)\b", re.I)
_PAT_CAP  = re.compile(r"(?:cap(?:acity)?\s*=?\s*)(\d+)", re.I)

_IATA_TO_CANON = {"BOM": "BOM", "DEL": "DEL"}
_CITY_TO_IATA  = {"mumbai": "BOM", "bombay": "BOM", "delhi": "DEL", "new delhi": "DEL"}

def _intent(q: str) -> str | None:
    q = q.lower()
    if any(k in q for k in ["busy", "busiest", "most crowded", "peak"]): return "busiest"
    if ("best" in q and any(k in q for k in ["arr","arrival","land"])) or "arrive" in q: return "best_arrival"
    if ("best" in q and any(k in q for k in ["dep","depart","takeoff"])) or "take off" in q: return "best_departure"
    if any(k in q for k in ["rli","load index","congestion","capacity util"]): return "rli"
    if any(k in q for k in ["tune","flatten","shift","reschedul"]): return "tuner"
    if any(k in q for k in ["cascad","propagat","spillover","downstream"]): return "cascade"
    if any(k in q for k in ["weather","visibility","storm"]): return "weather"
    if "capacity" in q or "cap=" in q or _PAT_CAP.search(q): return "capacity"
    if any(k in q for k in ["consolidation","cluster","hourly distribution","concentrated"]): return "consolidation"
    return None

def _kind(q: str) -> str | None:
    q = q.lower()
    if any(k in q for k in ["arriv", "land"]): return "arrival"
    if any(k in q for k in ["depart","takeoff","take off","dep "]): return "departure"
    return None

def _extract_airport(q: str, default: str | None = None) -> str | None:
    # prefer IATA
    for code in _PAT_IATA.findall(q.upper()) or []:
        if code in _IATA_TO_CANON: return _IATA_TO_CANON[code]
    # fallback: city token
    m = _PAT_CITY.search(q)
    if m:
        city = m.group(1).lower().replace("  ", " ").strip().replace("newdelhi","new delhi")
        return _CITY_TO_IATA.get(city, default)
    return default

def _mask_airport_any(df: pd.DataFrame, ap: str | None) -> pd.Series:
    if not ap: return pd.Series(True, index=df.index)
    ap = ap.upper()
    cols = []
    for c in ("origin","destination","airport"):
        if c in df.columns and df[c].notna().any():
            cols.append(df[c].astype(str).str.upper())
    if not cols: return pd.Series(True, index=df.index)
    m = pd.Series(False, index=df.index)
    for s in cols: m = m | s.str.contains(ap, na=False)
    return m

# --------- public entry point ----------
def answer(query: str, df: pd.DataFrame, airport: str | None = None,
           window_min_busiest: int = 15,
           window_min_best: int = 30,
           rli_window_min: int = 5,
           default_capacity: int = 12,
           weather_factor: float = 1.0,
           max_shift_min: int = 15) -> str:
    """
    Text-in → markdown-out using the same intent vocabulary as the UI.
    """
    intent = _intent(query)
    kind   = _kind(query)
    ap     = _extract_airport(query, airport)

    # scope to airport if we can detect one
    d = df.copy()
    if ap:
        m = _mask_airport_any(d, ap)
        d = d.loc[m] if m.any() else d

    # capacity from text (e.g., "capacity 10", "cap=11")
    mcap = _PAT_CAP.findall(query)
    cap = int(mcap[0]) if mcap else int(default_capacity)

    if intent == "busiest":
        out = busiest_slots(d, time_col="sched_dep", window_min=window_min_busiest, top_n=20).head(10)
        return _fmt_table(ap, "Top busiest slots (sched deps)", out)

    if intent == "best_arrival" or (intent == "best_departure" and kind == "arrival"):
        out = best_time_to_operate(d, at_airport=ap, kind="arrival", window_min=window_min_best).head(10)
        return _fmt_table(ap, "Best arrival slots by mean delay", out)

    if intent == "best_departure" or (intent == "best_arrival" and kind == "departure"):
        out = best_time_to_operate(d, at_airport=ap, kind="departure", window_min=window_min_best).head(10)
        return _fmt_table(ap, "Best departure slots by mean delay", out)

    if intent == "rli":
        times = pd.to_datetime(d.get("sched_dep"), errors="coerce")
        out = runway_load_index(times, rli_window_min, float(weather_factor), int(cap)).head(50)
        return _fmt_table(ap, f"Runway Load Index (cap={cap}, weather={weather_factor:.2f})", out)

    if intent == "tuner":
        sugg = greedy_slot_tuner(d, airport=ap or "", window_min=rli_window_min, max_shift_min=int(max_shift_min), per_window_capacity=int(cap))
        return _fmt_table(ap, f"Tuner suggestions (cap={cap}, max_shift={max_shift_min}m)", sugg)

    if intent == "cascade":
        dd = d.copy()
        # if delays sparse, synthesize congestion-based dep delays from slot overload
        dep = pd.to_datetime(dd.get("sched_dep"), errors="coerce")
        slots = dep.dt.floor(f"{rli_window_min}min")
        counts = slots.value_counts().sort_index()
        if "dep_delay_min" not in dd.columns or pd.Series(dd["dep_delay_min"]).notna().sum() < max(5, 0.1 * len(dd)):
            synthetic = ((counts - cap).clip(lower=0) * 3.0)
            dd["dep_delay_min"] = slots.map(synthetic).fillna(0.0).astype(float)
        infl = influence_table(dd, base_delay_col="dep_delay_min", steps=3, airport=ap)
        return _fmt_table(ap, "Cascading impact (higher = more downstream delay)", infl.head(20))

    if intent == "capacity":
        times = pd.to_datetime(d.get("sched_dep"), errors="coerce")
        rli = runway_load_index(times, rli_window_min, float(weather_factor), int(cap))
        peak = 0.0 if rli.empty else float(rli["rli"].max())
        over = 0 if rli.empty else int((rli["rli"] > 1.0).sum())
        return _fmt_msg(ap, f"With capacity={cap}, peak RLI≈{peak:.2f}, overloaded windows={over}.")

    if intent == "consolidation":
        t = pd.to_datetime(d.get("sched_dep"), errors="coerce")
        hourly = t.dt.floor("H").value_counts().sort_index()
        if hourly.empty: return _fmt_msg(ap, "No departures to analyze.")
        total = int(hourly.sum())
        hourly_full = hourly.asfreq("H", fill_value=0)
        roll2 = int(hourly_full.rolling(2).sum().max())
        roll3 = int(hourly_full.rolling(3).sum().max())
        pct2 = (roll2/total*100) if total else 0.0
        pct3 = (roll3/total*100) if total else 0.0
        return _fmt_table(ap, "Peak-hour consolidation", pd.DataFrame({
            "metric": ["total_departures","max_in_any_hour","share_top_2h_block_%","share_top_3h_block_%"],
            "value": [total, int(hourly_full.max()), round(pct2,1), round(pct3,1)]
        }))

    return (
        "Sorry, I didn’t catch that.\n"
        "Try: 'busiest slots in DEL', 'best time to depart Mumbai', 'RLI', 'capacity 10', "
        "'tuner suggestions', 'top cascading flights', 'consolidation'."
    )

# --------- formatting ----------
def _fmt_msg(ap: str | None, msg: str) -> str:
    return (f"[{ap}] " if ap else "") + msg

def _fmt_table(ap: str | None, title: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return _fmt_msg(ap, f"{title}: no data.")
    head = f"[{ap}] " if ap else ""
    return head + title + ":\n" + df.to_markdown(index=False)
