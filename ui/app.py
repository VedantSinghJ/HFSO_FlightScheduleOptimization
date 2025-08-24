import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import duckdb
import altair as alt
import numpy as np
import io
import re
from src import nlp as nlu


from src.metrics import busiest_slots, best_time_to_operate, runway_load_index
from src.optimizer import greedy_slot_tuner
from src.influence import influence_table
from src.processing import read_excel_multiblock, detect_columns, normalize_times, compute_basic_features

st.set_page_config(page_title="Honeywell Flight Scheduling Optimizer (HFSO)", layout="wide")
st.title("Honeywell Flight Scheduling Optimizer (HFSO)")

with st.expander("ℹ️ Quick glossary (tap to learn the terms)"):
    st.markdown("""
- **Slot / Window**: a fixed time bucket (RLI uses **5-min**; busiest/best use **15–30-min**).
- **Movements**: number of departures/arrivals in a slot.
- **RLI (Runway Load Index)** = `movements / (capacity × weather_factor)`. **> 1.0** means overload.
- **Ops perspective**: map historical times onto **today** for live planning. Turn **OFF** to analyze actual past dates.
- **Per-window capacity (5-min)**: max movements a runway can handle each 5-min slot.
- **Weather factor**: capacity multiplier; **0.8** ≈ bad weather (−20%); **1.2** ≈ good (+20%).
- **Max shift (min)**: largest time move the tuner will apply to a flight (e.g., ±15 min).
- **Flights shifted**: unique flights the tuner moved.
- **Est. delay saved (min)**: total departure delay minutes reduced after applying shifts.
- **Efficiency (Saved / Shifted)**: delay saved per minute of total shifting.
- **Cascading flights**: flights whose delays propagate via tight turnarounds / same-slot neighbors.
""")


# ============================== Data loaders ==============================
@st.cache_data(show_spinner=True)
def load_df_cached_from_parquet(parquet_path: str = "data/processed.parquet"):
    try:
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM '{parquet_path}'").fetch_df()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        for c in ["sched_dep", "act_dep", "sched_arr", "act_arr"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=True)
def load_df_from_upload(file_bytes: bytes, filename: str):
    if filename.lower().endswith((".xlsx", ".xls")):
        raw = read_excel_multiblock(io.BytesIO(file_bytes))
    elif filename.lower().endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type. Please upload .xlsx, .xls, or .csv")

    cols = detect_columns(raw)
    raw = normalize_times(raw, cols)
    df = compute_basic_features(raw, cols)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    for c in ["sched_dep", "act_dep", "sched_arr", "act_arr"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a 'date' column exists (derive from any time col if missing)."""
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce").dt.date
        if d.notna().any():
            df["date"] = d
            return df
    for col in ["sched_dep", "sched_arr", "act_dep", "act_arr"]:
        if col in df.columns:
            d = pd.to_datetime(df[col], errors="coerce").dt.date
            if d.notna().any():
                df["date"] = d
                return df
    df["date"] = pd.Timestamp.today().date()
    try:
        st.warning("No 'date' column found and no usable times to derive one. Using today's date as a placeholder.")
    except Exception:
        pass
    return df

# ============== NEW: robust delay fallback so cascade & KPIs never go blank ==============
def _ensure_delay_metric(df_q: pd.DataFrame, cap: int, window_min: int = 5):
    """
    Ensure a usable dep_delay_min exists.
    If actual delays are sparse/missing, synthesize a congestion-based delay:
      delay_per_flight_in_slot = 3 * max(0, count_in_slot - cap) minutes
    Returns (df_with_delay, used_fallback: bool)
    """
    if "dep_delay_min" in df_q.columns and pd.Series(df_q["dep_delay_min"]).notna().sum() >= max(5, 0.1 * len(df_q)):
        return df_q.copy(), False

    d = df_q.copy()
    times = pd.to_datetime(d["sched_dep"], errors="coerce")
    slots = times.dt.floor(f"{window_min}min")
    counts = slots.value_counts().sort_index()
    overload = (counts - cap).clip(lower=0)
    slot_delay = overload * 3.0  # tunable
    d["dep_delay_min"] = slots.map(slot_delay).fillna(0.0).astype(float)
    return d, True

# ============================== Sidebar: source ==============================
st.sidebar.header("Dataset")
uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx/.xls) or CSV", type=["xlsx", "xls", "csv"])
source_choice = st.sidebar.radio(
    "Data source", ["Sample (processed.parquet)", "Uploaded file"],
    index=0 if uploaded is None else 1
)

if source_choice == "Uploaded file":
    if uploaded is None:
        st.warning("Upload a file or switch to Sample.")
        st.stop()
    df = load_df_from_upload(uploaded.getvalue(), uploaded.name)
else:
    df = load_df_cached_from_parquet()

if df is None or df.empty:
    st.warning("No data found. Upload a file or run the pipeline to create `data/processed.parquet`.")
    st.stop()

# ============================== Ensure canonical types ==============================
for c in ["sched_dep", "act_dep", "sched_arr", "act_arr"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")
df = _ensure_date_column(df)
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

# ============================== Date range picker (ROBUST) ==============================
st.sidebar.header("Date range")
valid_dates = pd.Series(df["date"]).dropna()

def _flatten_two_dates(x):
    import itertools, datetime as _dt
    if isinstance(x, (list, tuple)):
        flat = list(itertools.chain.from_iterable(
            (y if isinstance(y, (list, tuple)) else [y]) for y in x
        ))
    else:
        flat = [x]
    if len(flat) == 0:
        a = b = _dt.date.today()
    elif len(flat) == 1:
        a = b = flat[0]
    else:
        a, b = flat[0], flat[1]
    a = pd.to_datetime(a, errors="coerce").date()
    b = pd.to_datetime(b, errors="coerce").date()
    if a > b:
        a, b = b, a
    return a, b

if valid_dates.empty:
    st.sidebar.info("No valid dates found in the dataset.")
    start_date, end_date = _flatten_two_dates(pd.Timestamp.today().date())
else:
    min_d = valid_dates.min()
    max_d = valid_dates.max()
    picked = st.sidebar.date_input(
        "Filter by date (inclusive)",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        format="YYYY/MM/DD",
    )
    start_date, end_date = _flatten_two_dates(picked)
    st.sidebar.caption(
        f"Data covers: {pd.to_datetime(min_d).strftime('%d-%b-%Y')} → "
        f"{pd.to_datetime(max_d).strftime('%d-%b-%Y')} "
        f"({(pd.to_datetime(max_d)-pd.to_datetime(min_d)).days+1} days)"
    )
    st.sidebar.caption(
        f"Selected: {pd.to_datetime(start_date).strftime('%d-%b-%Y')} → "
        f"{pd.to_datetime(end_date).strftime('%d-%b-%Y')} "
        f"({(pd.to_datetime(end_date)-pd.to_datetime(start_date)).days+1} days)"
    )

date_series = pd.to_datetime(pd.Series(df["date"]), errors="coerce").dt.date
mask_date = (date_series >= start_date) & (date_series <= end_date)
df = df.loc[mask_date].copy()

# ============================== Capacity & Weather ==============================
st.sidebar.header("Capacity & Weather")
ops_per_window_capacity = st.sidebar.number_input(
    "Runway capacity per 5-min window", min_value=4, value=12, step=1,
    help="Max movements the runway can safely handle in each 5-min slot under normal conditions."
)
weather_factor = st.sidebar.slider(
    "Weather factor (↓ capacity)", min_value=0.5, max_value=1.5, value=1.0, step=0.05,
    help="Multiplier on effective capacity. 0.8 ≈ bad weather (20% less), 1.2 ≈ good (20% more)."
)

# ============================== Airport + Ops toggle (TOP) ==============================
def _airport_mask_any(df_: pd.DataFrame, airport: str) -> pd.Series:
    ap = (airport or "").upper()
    cols = []
    for col in ("origin", "destination", "airport"):
        if col in df_.columns and df_[col].notna().any():
            cols.append(df_[col].astype(str).str.upper())
    if not cols:
        return pd.Series(True, index=df_.index)
    m = pd.Series(False, index=df_.index)
    for c in cols:
        m = m | c.str.contains(ap, na=False)
    return m

def _select_airport_df(df_all: pd.DataFrame, ap: str) -> pd.DataFrame:
    m = _airport_mask_any(df_all, ap).reindex(df_all.index, fill_value=False)
    out = df_all.loc[m].copy()
    return out if not out.empty else df_all.copy()

def build_airport_options(df_all: pd.DataFrame, top_n: int = 30):
    parts = []
    for col in ("origin", "destination", "airport"):
        if col in df_all.columns:
            parts.append(df_all[col].dropna().astype(str))
    if not parts:
        return ["BOM", "DEL"]
    s = pd.concat(parts)
    try:
        iata = s.str.extractall(r"\b([A-Z]{3})\b")[0].value_counts().index.tolist()
    except Exception:
        iata = []
    try:
        names = s.str.extractall(r"\b([A-Za-z]{4,})\b")[0].str.title().value_counts().index.tolist()
    except Exception:
        names = []
    out, seen = [], set()
    for x in iata + names:
        if x not in seen:
            out.append(x); seen.add(x)
        if len(out) >= top_n:
            break
    return out or ["BOM", "DEL"]

# Top controls
# Top controls
left, right = st.columns([3, 1])
with left:
    airports = build_airport_options(df)
    airport = st.selectbox("Focus Airport / City", options=airports, index=0)

with right:
    # Default ON
    try:
        pattern_view = st.toggle(
            "Ops perspective",
            value=True,
            help="Map historical times onto TODAY for live ops planning."
        )
    except Exception:
        pattern_view = st.checkbox(
            "Ops perspective",
            value=True,
            help="Map historical times onto TODAY for live ops planning."
        )
    # Short, clear guidance line
    st.caption("Ops perspective is **ON** by default (history mapped to today). Turn it **OFF** to analyze actual past dates.")

# scope dataset to airport
mask = _airport_mask_any(df, airport).reindex(df.index, fill_value=False)
df_ap = df.loc[mask].copy()
if df_ap.empty:
    df_ap = df.copy()

# Build view (today-anchored) times
base_times = pd.to_datetime(df_ap["sched_dep"], errors="coerce")
if pattern_view:
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    clock = base_times.dt.strftime("%H:%M:%S")
    view_times = pd.to_datetime(today + " " + clock, errors="coerce")
else:
    view_times = base_times
df_view = df_ap.copy()
df_view["sched_dep"] = view_times



# ============================== Label + table helpers ==============================
def _safe_str(s: pd.Series) -> pd.Series:
    """
    Convert to string, trim, and collapse common null-ish tokens to empty,
    but only when the WHOLE string is a null token (so 'LONDON' is untouched).
    """
    out = s.astype(str).fillna("").str.strip()

    # normalize pandas markers
    out = out.replace({"<NA>": "", "nan": "", "NaN": ""})

    # collapse textual nulls ONLY if the entire value is a null token
    # (case-insensitive, anchors so we don't touch substrings)
    null_full = out.str.match(
        r"^(?i:(?:none|null|nil|n/?a|n\.?a\.?|na|n-a|—|-))$",
        na=False
    )
    out = out.mask(null_full, "")

    return out


def _clean_fno(s: pd.Series) -> pd.Series:
    """
    Normalize flight numbers:
    - remove trailing '.0' from numerics (e.g., '123.0' -> '123')
    - strip spaces; upper-case
    - if there are NO digits at all (e.g., 'NONE', 'NA'), treat as missing
    """
    ss = _safe_str(s)
    ss = ss.str.replace(r"^(\d+)\.0$", r"\1", regex=True) \
           .str.replace(" ", "", regex=False) \
           .str.upper()

    # Require at least one digit in a valid flight number
    ss = ss.where(ss.str.contains(r"\d"), "")

    return ss

def _fmt_when(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce")
    return t.dt.strftime("%d-%b %H:%M").fillna("")

def _compose_label_from_base(df_base: pd.DataFrame) -> pd.Series:
    idx = df_base.index
    carrier = _safe_str(df_base.get("carrier", pd.Series("", index=idx))).str.upper()
    # If carrier equals a null token like 'NONE', blank it
    carrier = carrier.where(~carrier.isin(["NONE", "NULL", "N/A", "NA", "-", "—"]), "")

    fno_raw = _clean_fno(df_base.get("flight_number", pd.Series("", index=idx)))
    fno = fno_raw.replace("", pd.NA).ffill().fillna("")

    origin = _safe_str(df_base.get("origin", pd.Series("", index=idx))).str.upper()
    dest   = _safe_str(df_base.get("destination", pd.Series("", index=idx))).str.upper()
    route = np.where(
        (origin != "") & (dest != ""),
        origin + "→" + dest,
        np.where(origin != "", origin, np.where(dest != "", dest, ""))
    )
    route = pd.Series(route, index=idx, dtype="object")
    fno_pref = fno.str.extract(r"^([A-Z0-9]{1,3})(?=\d)", expand=False).fillna("")
    same_prefix = (carrier != "") & (fno_pref == carrier)
    base = np.where(same_prefix | (carrier == "") | (fno == ""), fno, carrier + fno)
    base = pd.Series(base, index=idx, dtype="object").str.strip()
    when_str = _fmt_when(df_base.get("sched_dep", pd.Series("", index=idx)))
    tail     = _safe_str(df_base.get("tail_number", pd.Series("", index=idx))).str.upper()
    main = np.where(base != "", (base + " " + route).str.strip(), route)
    main = pd.Series(main, index=idx, dtype="object")
    main = (main + (" · " + when_str).where(when_str != "", "")).str.strip()
    main = (main + (" · " + tail).where(tail != "", "")).str.strip()
    return main.fillna("")

def _flight_label_lookup(df_base: pd.DataFrame):
    d = df_base.copy()
    if "flight_id" not in d.columns:
        d = d.reset_index().rename(columns={"index": "flight_id"})
    d["flight_id"] = d["flight_id"].astype(str)
    labels = _compose_label_from_base(d)
    empty = (labels == "") | labels.isna()
    if empty.any():
        labels.loc[empty] = _safe_str(d.loc[empty, "flight_id"])
    mapper = pd.Series(labels.values, index=d["flight_id"])
    mapper = mapper[~mapper.index.duplicated(keep="first")]  # ensure unique keys
    return mapper.to_dict()

def _present_table_with_label_left(table: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    if "flight_id" in t.columns:
        lookup = _flight_label_lookup(base_df)
        shown = t["flight_id"].astype(str).map(lookup).fillna(t["flight_id"].astype(str))
        t = t.drop(columns=["flight_id"])
        t.insert(0, "flight_id", shown)
    else:
        dtemp = t.copy()
        dtemp["__lbl__"] = _compose_label_from_base(dtemp)
        if dtemp["__lbl__"].ne("").any():
            left = dtemp.pop("__lbl__")
            t.insert(0, "flight_id", left)
    return t

def _apply_suggestions(df_: pd.DataFrame, suggestions: pd.DataFrame) -> pd.Series:
    d = df_.copy()
    if "flight_id" not in d.columns:
        d = d.reset_index().rename(columns={"index": "flight_id"})
    d["flight_id"] = d["flight_id"].astype(str)
    d["sched_dep"] = pd.to_datetime(d.get("sched_dep"), errors="coerce")
    if suggestions is None or suggestions.empty or "flight_id" not in suggestions.columns:
        return d["sched_dep"]
    s = suggestions.copy().dropna(subset=["flight_id"])
    s["flight_id"] = s["flight_id"].astype(str)
    s = s.drop_duplicates(subset=["flight_id"], keep="last")
    new_times = pd.to_datetime(s.get("new_time"), errors="coerce")
    map_series = pd.Series(new_times.values, index=s["flight_id"])
    d = d.merge(map_series.rename("new_sched_dep"), left_on="flight_id", right_index=True, how="left")
    d["sched_dep_after"] = pd.to_datetime(d["new_sched_dep"], errors="coerce").fillna(d["sched_dep"])
    return d["sched_dep_after"]

def _rli_line_chart(rli_df: pd.DataFrame, title: str):
    """
    Cleaner RLI line chart:
      - guarantees chronological order
      - downsamples to 15-min for readability
      - shows DATE + time on the axis
    """
    if rli_df is None or rli_df.empty:
        return st.info("No data to plot.")

    t = rli_df.copy()
    t["slot_start"] = pd.to_datetime(t["slot_start"], errors="coerce")
    t = t.dropna(subset=["slot_start"]).sort_values("slot_start")

    # Downsample to reduce overplotting; keep movements sum, rli mean
    t = (
        t.set_index("slot_start")
         .resample("15min")
         .agg({"movements": "sum", "rli": "mean"})
         .reset_index()
    )

    if t.empty:
        return st.info("No data to plot after resampling.")

    chart = (
        alt.Chart(t)
        .mark_line()
        .encode(
            x=alt.X(
                "slot_start:T",
                title="Time",
                axis=alt.Axis(
                    format="%d %b %H:%M",
                    labelAngle=-45,
                    tickCount=8
                )
            ),
            y=alt.Y("rli:Q", title="Runway Load Index"),
            tooltip=[
                alt.Tooltip("slot_start:T", title="Slot"),
                alt.Tooltip("movements:Q", title="Movements"),
                alt.Tooltip("rli:Q", title="RLI", format=".2f"),
            ],
        )
        .properties(title=title, width="container", height=240)
    )

    return st.altair_chart(chart, use_container_width=True)


def build_tuned_schedule(df_: pd.DataFrame, suggestions: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean table showing each flight's scheduled departure
    before/after applying tuner suggestions.

    Supports suggestions with:
      - 'flight_id' + 'new_time' (preferred), or
      - 'flight_id' + 'shift_min' (fallback)
    """
    d = df_.copy()

    # Ensure keys & dtypes
    if "flight_id" not in d.columns:
        d = d.reset_index().rename(columns={"index": "flight_id"})
    d["flight_id"] = d["flight_id"].astype(str)
    for c in ["sched_dep", "act_dep", "sched_arr", "act_arr"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")

    # Always create the column so downstream logic never KeyErrors
    d["new_sched_dep"] = pd.NaT

    # Map suggestions -> new times (if any)
    if suggestions is not None and not suggestions.empty and ("flight_id" in suggestions.columns):
        s = suggestions.copy().dropna(subset=["flight_id"])
        if not s.empty:
            s["flight_id"] = s["flight_id"].astype(str)
            s = s.drop_duplicates(subset=["flight_id"], keep="last")

            # Build a safe mapping: prefer 'new_time', else compute via 'shift_min'
            base_sched = d.set_index("flight_id")["sched_dep"]
            new_map = {}

            for _, row in s.iterrows():
                fid = row["flight_id"]
                newt = pd.to_datetime(row.get("new_time"), errors="coerce")

                if pd.isna(newt) and ("shift_min" in s.columns):
                    shift = row.get("shift_min")
                    base = base_sched.get(fid, pd.NaT)
                    if pd.notna(base) and pd.notna(shift):
                        try:
                            newt = base + pd.to_timedelta(float(shift), unit="m")
                        except Exception:
                            newt = pd.NaT

                new_map[fid] = newt

            if new_map:
                map_series = pd.Series(new_map, name="new_sched_dep")
                d = d.merge(map_series, left_on="flight_id", right_index=True, how="left")

    if "new_sched_dep" not in d.columns:
        d["new_sched_dep"] = pd.NaT

    d["sched_dep_after"] = pd.to_datetime(d["new_sched_dep"], errors="coerce").fillna(d["sched_dep"])

    def _fmt(x):
        x = pd.to_datetime(x, errors="coerce")
        return x.strftime("%Y-%m-%d %H:%M") if pd.notna(x) else ""

    d["scheduled_departure_before"] = d["sched_dep"].map(_fmt)
    d["scheduled_departure_after"]  = d["sched_dep_after"].map(_fmt)

    preferred = [c for c in [
        "flight_id", "carrier", "flight_number", "tail_number",
        "origin", "destination", "airport",
        "scheduled_departure_before", "scheduled_departure_after",
        "dep_delay_min", "arr_delay_min"
    ] if c in d.columns or c in ["scheduled_departure_before","scheduled_departure_after"]]

    return d[preferred].copy()

# ============================== Executive summary ==============================
def _estimate_total_dep_delay(df_subset: pd.DataFrame, times: pd.Series, window_min=5) -> float:
    base = df_subset.copy()
    base["slot"] = pd.to_datetime(base["sched_dep"], errors="coerce").dt.floor(f"{window_min}min")
    slot_mean = base.groupby("slot")["dep_delay_min"].mean().fillna(0.0)
    slots = pd.to_datetime(times, errors="coerce").dt.floor(f"{window_min}min")
    return float(pd.Series(slots).map(slot_mean).fillna(0.0).sum())

def summarize_insights(df_view, df_ap, cap, weather, pattern_view: bool):
    """
    Compute the header KPIs using the SAME lens the user is viewing:
    - if pattern_view=True -> use df_view (today-anchored)
    - else -> use df_ap (historical)
    Also auto-tighten capacity to (peak-1) to preview moves when the plan is within cap.
    """
    base_df = df_view if pattern_view else df_ap
    records = int(pd.to_datetime(base_df["sched_dep"], errors="coerce").notna().sum())

    rli_now = runway_load_index(base_df["sched_dep"], 5, weather, cap)
    peak_rli = 0.0 if rli_now.empty else float(rli_now["rli"].max())
    over_windows = int((rli_now["rli"] > 1.0).sum()) if not rli_now.empty else 0

    best_dep = best_time_to_operate(base_df, kind="departure", window_min=30).head(1)
    best_arr = best_time_to_operate(base_df, kind="arrival",   window_min=30).head(1)

    slot_counts = pd.to_datetime(base_df["sched_dep"], errors="coerce").dt.floor("5min").value_counts()
    observed_peak = int(slot_counts.max()) if not slot_counts.empty else 0
    eff_cap = int(cap)
    demo_mode = False
    if observed_peak and eff_cap >= observed_peak:
        eff_cap = max(1, observed_peak - 1)
        demo_mode = True

    suggestions = greedy_slot_tuner(base_df, "", 5, 15, int(eff_cap))
    moved = 0 if suggestions.empty else int(suggestions["flight_id"].nunique())

    df_for_delay, _ = _ensure_delay_metric(base_df, int(cap), window_min=5)
    dep_before = _estimate_total_dep_delay(df_for_delay, base_df["sched_dep"], window_min=5)
    dep_after  = _estimate_total_dep_delay(df_for_delay, _apply_suggestions(base_df, suggestions), window_min=5)
    saved = dep_before - dep_after

    return {
        "records": records,
        "peak_rli": peak_rli,
        "over_windows": over_windows,
        "best_dep": None if best_dep.empty else best_dep.iloc[0].to_dict(),
        "best_arr": None if best_arr.empty else best_arr.iloc[0].to_dict(),
        "moved": moved,
        "saved": saved,
        "demo_mode": demo_mode,
    }

ins = summarize_insights(df_view, df_ap, int(ops_per_window_capacity), float(weather_factor), pattern_view=pattern_view)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak RLI (current view)", f"{ins['peak_rli']:.2f}")
c2.metric("Windows RLI>1 (view)", f"{ins['over_windows']}")
c3.metric("Sched Tuner: Flights Moved", f"{ins['moved']}")
c4.metric("Est. Delay Saved (min)", f"{ins['saved']:.0f}")
if ins["best_dep"]:
    st.caption(f"🛫 Best depart window: {pd.to_datetime(ins['best_dep'].get('slot_start')).strftime('%H:%M')} | "
               f"avg dep delay ≈ {ins['best_dep'].get('mean_delay_min', ins['best_dep'].get('avg_dep_delay_min', 0)):.1f} min")
if ins["best_arr"]:
    st.caption(f"🛬 Best arrival window: {pd.to_datetime(ins['best_arr'].get('slot_start')).strftime('%H:%M')} | "
               f"avg arr delay ≈ {ins['best_arr'].get('mean_delay_min', ins['best_arr'].get('avg_arr_delay_min', 0)):.1f} min")

def _peak_rli_next2h(df_view_like: pd.DataFrame, weather: float, cap: int):
    """
    Peak RLI in the next 2h. If no flights in [now, now+2h],
    fallback to the next available 2h window starting at the next departure.
    Returns (peak_value, window_label).
    """
    times = pd.to_datetime(df_view_like.get("sched_dep"), errors="coerce").dropna().sort_values()
    if times.empty:
        return 0.0, "no departures in view"

    now = pd.Timestamp.now().floor("min")
    start = now
    end   = now + pd.Timedelta(hours=2)
    window = times[(times >= start) & (times < end)]

    # Fallback: start at next departure if current 2h window is empty
    if window.empty:
        fut = times[times >= now]
        start = fut.iloc[0] if not fut.empty else times.iloc[0]
        end   = start + pd.Timedelta(hours=2)
        window = times[(times >= start) & (times < end)]
        label = f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')} (next window)"
    else:
        label = f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"

    rli = runway_load_index(window, 5, float(weather), int(cap))
    peak = 0.0 if rli.empty else float(rli["rli"].max())
    return peak, label


# ========== Next 2h risk ==========
# ========== Next 2h risk ==========
peak2h, win_label = _peak_rli_next2h(df_view, float(weather_factor), int(ops_per_window_capacity))
st.metric("Peak RLI (next 2h — view)", f"{peak2h:.2f}")
st.caption(f"Window: {win_label}")


# ============================== NLP helpers ==============================
def parse_kind(query: str):
    q = query.lower()
    if any(k in q for k in ["arriv", "land"]): return "arrival"
    if any(k in q for k in ["depart", "takeoff", "take off", "dep "]): return "departure"
    return None

def keyword_route(query: str):
    q = query.lower()
    if "busy" in q or "busiest" in q or "most crowded" in q or "peak" in q: return "busiest"
    if ("best" in q and ("arr" in q or "arrival" in q or "land" in q)) or "arrive" in q: return "best_arrival"
    if ("best" in q and ("dep" in q or "depart" in q or "takeoff" in q)) or "take off" in q: return "best_departure"
    if any(k in q for k in ["rli", "load index", "congestion", "capacity util"]): return "rli"
    if any(k in q for k in ["tune", "flatten", "shift", "reschedul"]): return "tuner"
    if ("cascad" in q) or ("propagat" in q) or ("spillover" in q) or ("downstream" in q): return "cascade"
    if "weather" in q or "visibility" in q or "storm" in q: return "weather"
    if "capacity" in q or "cap=" in q or re.search(r"\bcap\s*\d+\b", q): return "capacity"
    if any(k in q for k in ["consolidation", "cluster", "hourly distribution", "concentrated"]): return "consolidation"
    if any(k in q for k in ["help", "what can i ask", "options"]): return "help"
    return None

# ============================== Tabs ==============================
tab_nlp, tab_analytics, tab_tuner, tab_scen = st.tabs(["🤖 NLP Assistant", "📈 Analytics", "🔧 Tuner", "🧪 Scenarios"])

# ---------------- NLP Assistant ----------------
with tab_nlp:
    def answer_nlp(query: str):
        # Use the same lens the user is viewing:
        base_df = df_view if pattern_view else df_ap

        md = nlu.answer(
            query,
            base_df,
            airport=airport,
            window_min_busiest=15,
            window_min_best=30,
            rli_window_min=5,
            default_capacity=int(ops_per_window_capacity),
            weather_factor=float(weather_factor),
            max_shift_min=15,
        )
        st.markdown(md)

    st.subheader("Ask a question")
    q = st.text_input("Try: 'busiest time slots in DEL', 'best time to depart Mumbai', 'capacity 10', 'top cascading flights'")
    if q:
        try:
            answer_nlp(q)
        except Exception as e:
            st.warning(f"NLP error: {e}")

# ---------------- Analytics tab ----------------
with tab_analytics:
    st.subheader("Busiest Slots (view)")
    st.caption("Top 15-min slots by total movements in the current view (ops perspective if ON).")
    try:
        busiest_tbl = busiest_slots(df_view, time_col="sched_dep", window_min=15, top_n=20)
        st.dataframe(busiest_tbl, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to compute busiest slots: {e}")

    st.subheader("Best Time to Operate (actual schedule)")
    st.caption("30-min windows with the lowest average delays (choose Arrival or Departure).")
    kind_sel = st.selectbox("Arrival or Departure?", ["arrival", "departure"], index=1, key="best_kind")
    try:
        best_tbl = best_time_to_operate(df_ap, at_airport=None, kind=kind_sel, window_min=30)
        st.dataframe(best_tbl.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to compute best-time windows: {e}")

    st.subheader("Runway Load Index (RLI) — view")
    st.caption("RLI = movements / (capacity × weather). RLI > 1.0 indicates overload risk.")
    try:
        rli_tbl = runway_load_index(
            df_view["sched_dep"], window_min=5,
            weather_factor=float(weather_factor),
            ops_per_window_capacity=int(ops_per_window_capacity)
        )
        st.dataframe(rli_tbl.head(60), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to compute RLI: {e}")

    st.subheader("Congestion Heatmap (view)")
    st.caption("RLI by hour (x) and date (y). Darker = more congested.")
    def rli_heatmap(df_ap_like, cap, weather, window_min=5):
        rli = runway_load_index(df_ap_like["sched_dep"], window_min, weather, cap)
        if rli.empty:
            st.info("No data for heatmap.")
            return
        r = rli.copy()
        r["hour"] = pd.to_datetime(r["slot_start"]).dt.hour
        r["day"]  = pd.to_datetime(r["slot_start"]).dt.date
        chart = (
            alt.Chart(r)
            .mark_rect()
            .encode(
                x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("day:O", title="Date"),
                color=alt.Color("rli:Q", title="RLI"),
                tooltip=["slot_start:T","movements:Q","rli:Q"]
            )
            .properties(title="Runway Congestion Heatmap (RLI)", height=220)
        )
        st.altair_chart(chart, use_container_width=True)
    rli_heatmap(df_view, int(ops_per_window_capacity), float(weather_factor))

    if pattern_view:
        st.info("Pattern view anchors historical times to today for congestion preview. Tuner can also run on ops view via the checkbox in the Tuner tab.")

    # NEW: Cascading Impact always visible
    st.subheader("Cascading Impact — Top Flights")
    st.caption("Flights most likely to propagate delays (tight turns / same-slot neighbors).")
    base_df_for_infl, used_fallback_infl = _ensure_delay_metric(df_ap, int(ops_per_window_capacity))
    infl_tbl = influence_table(base_df_for_infl, base_delay_col="dep_delay_min", steps=3)
    st.dataframe(_present_table_with_label_left(infl_tbl, base_df_for_infl), use_container_width=True)
    if used_fallback_infl:
        st.info("Actual delays were sparse — used congestion-based synthetic delays for influence ranking.")

# ---------------- Tuner tab ----------------
with tab_tuner:
    st.subheader("Schedule Tuner (Heuristic)")

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        cap = st.number_input("Per-window capacity (5-min)", min_value=1, value=int(ops_per_window_capacity), step=1, key="cap_tuner",
                              help="Capacity used by the tuner. Set at/under the runway's safe limit per 5-min slot.")
    with c2:
        max_shift = st.number_input("Max shift (min)", min_value=5, value=15, step=5, key="shift_tuner",
                                    help="Largest time move the tuner will apply to a flight (±N minutes).")
    with c3:
        tune_on_view = st.checkbox(
            "Tune using ops perspective (pattern view)",
            value=pattern_view,
            help="When ON, tuner runs on the today-anchored view. When OFF, runs on the actual historical schedule."
        )

    # pick dataset to tune on
    base_df_for_tune = df_view if tune_on_view else df_ap
    base_label = "ops perspective (view: today-anchored)" if tune_on_view else "actual schedule (historical)"
    st.caption(f"Tuning against **{base_label}**.")

    # tighten capacity if not overloaded to surface suggestions (demo mode)
    slot_counts = pd.to_datetime(base_df_for_tune["sched_dep"], errors="coerce").dt.floor("5min").value_counts()
    observed_peak = int(slot_counts.max()) if not slot_counts.empty else 0
    eff_cap = int(cap)
    demo_mode = False
    if observed_peak and eff_cap >= observed_peak:
        eff_cap = max(1, observed_peak - 1)
        demo_mode = True

    # run tuner
    try:
        suggestions = greedy_slot_tuner(
            base_df_for_tune, airport="", window_min=5,
            max_shift_min=int(max_shift), per_window_capacity=int(eff_cap)
        )
        if suggestions.empty:
            st.info("No shifts suggested (already under capacity or constraints too tight). "
                    "Try lowering capacity or increasing max shift.")
        st.dataframe(_present_table_with_label_left(suggestions, base_df_for_tune), use_container_width=True)
        if demo_mode:
            st.caption(f"ℹ️ Observed peak={observed_peak}. Using demo capacity={eff_cap} (peak−1) to surface shifts.")
    except Exception as e:
        suggestions = pd.DataFrame()
        st.error(f"Tuner failed: {e}")

    # RLI before/after computed on the same base (view vs actual)
    window_min = 5
    rli_before = runway_load_index(
        base_df_for_tune["sched_dep"], window_min, float(weather_factor), int(cap)
    )
    sched_dep_after = _apply_suggestions(base_df_for_tune, suggestions if "suggestions" in locals() else pd.DataFrame())
    rli_after = runway_load_index(
        sched_dep_after, window_min, float(weather_factor), int(cap)
    )

    max_before = float(rli_before["rli"].max()) if not rli_before.empty else 0.0
    max_after  = float(rli_after["rli"].max())  if not rli_after.empty  else 0.0
    peak_drop  = max(0.0, max_before - max_after)
    moved_cnt  = 0 if suggestions is None or suggestions.empty else int(suggestions["flight_id"].nunique())

    # delay KPIs — use robust fallback on the SAME base
    df_for_delay_kpi, used_fallback_kpi = _ensure_delay_metric(base_df_for_tune, int(cap), window_min=window_min)
    est_delay_before = _estimate_total_dep_delay(df_for_delay_kpi, base_df_for_tune["sched_dep"], window_min=window_min)
    est_delay_after  = _estimate_total_dep_delay(df_for_delay_kpi, sched_dep_after,    window_min=window_min)
    est_delay_delta  = est_delay_before - est_delay_after

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Peak RLI (Before)", f"{max_before:.2f}")
    k2.metric("Peak RLI (After)",  f"{max_after:.2f}")
    k3.metric("Peak Reduction",    f"{peak_drop:.2f}")
    k4.metric("# Flights Shifted", f"{moved_cnt}")
    k5.metric("Est. Total Dep Delay Δ (min)", f"{est_delay_delta:.0f}")
    st.caption("**Peak Reduction** = Peak RLI Before − After. **Est. Total Dep Delay Δ** = delay minutes saved after shifts.")

    if used_fallback_kpi:
        st.caption("KPIs use synthetic congestion-based delays (actual delays sparse/missing).")

    # total minutes shifted + efficiency
    total_shift_min = 0.0
    if suggestions is not None and not suggestions.empty:
        if "shift_min" in suggestions.columns:
            shifts = suggestions["shift_min"].abs()
        else:
            old = pd.to_datetime(suggestions.get("old_time"), errors="coerce")
            new = pd.to_datetime(suggestions.get("new_time"), errors="coerce")
            shifts = (new - old).dt.total_seconds().abs() / 60.0
        total_shift_min = float(shifts.fillna(0).sum())

    avg_saved_per_flight = (est_delay_delta / moved_cnt) if moved_cnt > 0 else 0.0
    efficiency_ratio = (est_delay_delta / total_shift_min) if total_shift_min > 0 else 0.0

    k6, k7, k8 = st.columns(3)
    k6.metric("Avg Saved per Moved Flight (min)", f"{avg_saved_per_flight:.1f}")
    k7.metric("Total Minutes Shifted", f"{total_shift_min:.0f}")
    k8.metric("Efficiency (Saved / Shifted)", f"{efficiency_ratio:.2f}")
    st.caption("**Efficiency (Saved / Shifted)** = delay saved per minute of total shifting across all moved flights.")


    co = st.columns(2)
    with co[0]:
        _rli_line_chart(rli_before, "RLI Before")
    with co[1]:
        _rli_line_chart(rli_after,  "RLI After")

    # export tuned schedule (on the same base)
    export_df = build_tuned_schedule(base_df_for_tune, suggestions)
    export_df_disp = _present_table_with_label_left(export_df, base_df_for_tune)
    csv_buf = io.StringIO()
    export_df_disp.to_csv(csv_buf, index=False)
    file_tag = "ops_view" if tune_on_view else "actual"
    st.download_button(
        label="📥 Download Tuned Schedule (CSV)",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"tuned_schedule_{airport}_{file_tag}.csv",
        mime="text/csv",
    )

    # recommendations based on the same base
    def recommendation_bullets(df_ap_, cap_, weather_):
        tips = []
        rli_now = runway_load_index(df_ap_["sched_dep"], 5, weather_, cap_)
        if not rli_now.empty and float(rli_now["rli"].max()) > 1.2:
            tips.append("Peak RLI > 1.2 — increase max shift to 20–30 min or add a buffer around peak slots.")
        best_dep_tbl = best_time_to_operate(df_ap_, kind="departure", window_min=30).head(3)
        if not best_dep_tbl.empty:
            s = best_dep_tbl.iloc[0]
            tips.append(f"Schedule more departures near **{pd.to_datetime(s['slot_start']).strftime('%H:%M')}** (lowest avg dep delay).")
        try:
            base_df_infl, _ = _ensure_delay_metric(df_ap_, int(cap_))
            infl_tbl = influence_table(base_df_infl, base_delay_col="dep_delay_min", steps=3).head(5)
            if not infl_tbl.empty:
                lookup = _flight_label_lookup(df_ap_)
                labels = infl_tbl["flight_id"].astype(str).map(lookup).fillna(infl_tbl["flight_id"].astype(str))
                fids = ", ".join(labels.head(3).tolist())
                tips.append(f"Monitor high-impact flights (buffer turnaround): **{fids}**.")
        except Exception:
            pass
        if not tips:
            tips.append("System looks balanced. Stress test with weather factor 0.8 and capacity −2.")
        return tips

    st.subheader("Recommendations")
    for t in recommendation_bullets(base_df_for_tune, int(cap), float(weather_factor)):
        st.markdown(f"- {t}")

    # Ops brief export
    st.download_button(
        "📄 Download Ops Brief (CSV)",
        pd.DataFrame({
            "airport":[airport],
            "capacity":[int(cap)],
            "weather_factor":[float(weather_factor)],
            "peak_rli_before":[max_before],
            "peak_rli_after":[max_after],
            "peak_drop":[peak_drop],
            "flights_moved":[moved_cnt],
            "est_delay_saved_min":[est_delay_delta],
        }).to_csv(index=False).encode("utf-8"),
        file_name=f"ops_brief_{airport}.csv",
        mime="text/csv"
    )

# ---------------- Scenarios tab ----------------
with tab_scen:
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = {}

    st.subheader("Scenario Compare (A/B)")
    with st.expander("Create & compare scenarios"):
        sc_name = st.text_input("Scenario name", value="Case A")
        sc_cap = st.number_input("Capacity (5-min)", 1, 50, int(ops_per_window_capacity), key="sc_cap")
        sc_weather = st.slider("Weather factor", 0.5, 1.5, float(weather_factor), 0.05, key="sc_weather")
        sc_shift = st.number_input("Max shift (min)", 5, 60, 15, 5, key="sc_shift")

        if st.button("➕ Save scenario"):
            rli_b = runway_load_index(df_view["sched_dep"], 5, sc_weather, sc_cap)
            sugg  = greedy_slot_tuner(df_ap, "", 5, int(sc_shift), int(sc_cap))
            after_times = _apply_suggestions(df_ap, sugg)
            rli_a = runway_load_index(after_times, 5, sc_weather, sc_cap)

            df_for_delay_s, _ = _ensure_delay_metric(df_ap, int(sc_cap))
            kpi = {
                "cap": sc_cap, "weather": sc_weather, "shift": sc_shift,
                "peak_before": 0.0 if rli_b.empty else float(rli_b["rli"].max()),
                "peak_after":  0.0 if rli_a.empty else float(rli_a["rli"].max()),
                "moved": 0 if sugg.empty else int(sugg["flight_id"].nunique()),
                "saved_min": _estimate_total_dep_delay(df_for_delay_s, df_ap["sched_dep"]) - _estimate_total_dep_delay(df_for_delay_s, after_times),
            }
            st.session_state.scenarios[sc_name] = kpi

    if st.session_state.scenarios:
        table = pd.DataFrame.from_dict(st.session_state.scenarios, orient="index")
        table["peak_drop"] = (table["peak_before"] - table["peak_after"]).clip(lower=0)
        table = table[["cap","weather","shift","peak_before","peak_after","peak_drop","moved","saved_min"]].sort_values("peak_drop", ascending=False)
        st.dataframe(table, use_container_width=True)

st.caption("© Honeywell Hackathon prototype — HFSO (robust cascade + demo-cap tuner + stable KPIs)")
