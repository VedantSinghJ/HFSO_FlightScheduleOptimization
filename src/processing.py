import re
import pandas as pd
import numpy as np

# =============================== Utilities ===============================

def make_unique(cols):
    """Make a list of column names unique by suffixing .1, .2, ..."""
    seen = {}
    out = []
    for c in cols:
        base = str(c) if c is not None else ""
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}.{seen[base]}")
    return out

HEADER_HINTS = {
    "flight", "flightno", "flightnumber", "flt", "fltno", "fltnr",
    "std", "sta", "atd", "ata",
    "scheduleddepart", "scheduledarrival", "actualdepart", "actualarrival",
    "scheduleddeparture", "scheduledarrival", "actualdeparture", "actualarrival",
    "origin", "from", "source", "destination", "to",
    "airline", "carrier", "operator",
    "tail", "registration", "reg",
    "date", "day", "localdate", "utcdate",
    "runway", "gate", "stand", "bay",
}

def _norm(s):
    return str(s).strip().lower().replace(" ", "").replace("_", "")

def looks_like_header(row_values, min_hits=2):
    vals = [v for v in row_values if v is not None and str(v).strip() != ""]
    if len(vals) < min_hits:
        return False
    hits = 0
    for v in vals:
        if _norm(v) in HEADER_HINTS:
            hits += 1
            if hits >= min_hits:
                return True
    return False

def _clean_token_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype="object")
    out = s.astype(str)
    out = out.replace({"<NA>": "", "NaT": "", "nan": ""})
    return out.str.strip()

# =============================== Excel Reader ===============================

def read_excel_multiblock(path: str) -> pd.DataFrame:
    """
    Read ALL sheets in an Excel workbook.
    For each sheet:
      - try to detect one or more header rows (multi-block tables) and slice blocks,
      - else fall back to header=0 for that sheet.
    Concatenate everything and add a 'sheet' column.
    """
    try:
        # header=None to scan headers ourselves
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str, engine="openpyxl")
    except Exception:
        # Single-sheet fallback
        try:
            solo = pd.read_excel(path, engine="openpyxl")
            solo = solo.dropna(axis=0, how="all").dropna(axis=1, how="all")
            solo["sheet"] = "Sheet1"
            return solo
        except Exception as e:
            raise e

    out_frames = []
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        df = df.where(pd.notnull(df), None)
        header_idxs = [i for i, row in enumerate(df.values.tolist()) if looks_like_header(row)]

        if header_idxs:
            header_idxs.append(len(df))
            for i in range(len(header_idxs) - 1):
                h = header_idxs[i]
                nxt = header_idxs[i + 1]
                header = make_unique(df.iloc[h].tolist())
                block = df.iloc[h + 1 : nxt].copy()
                block.columns = header
                block = block.dropna(axis=1, how="all").dropna(axis=0, how="all")
                if len(block) >= 1 and len(block.columns) >= 3:
                    block["sheet"] = name
                    out_frames.append(block)
        else:
            # Per-sheet simple fallback
            try:
                simple = pd.read_excel(path, sheet_name=name, header=0, engine="openpyxl")
                simple = simple.dropna(axis=0, how="all").dropna(axis=1, how="all")
                if not simple.empty:
                    simple["sheet"] = name
                    out_frames.append(simple)
            except Exception:
                pass

    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True, sort=False)

# =============================== Column Detection ===============================

def detect_columns(df: pd.DataFrame) -> dict:
    """
    Heuristic mapping from raw headers to canonical names.
    Returns dict with keys like: origin, destination, sched_dep, act_dep, sched_arr, act_arr, carrier, flight_number, tail_number, date, airport
    """
    mapping = {}
    candidates = {
        "origin": ["origin","from","source","src","org","departureairport","depairport"],
        "destination": ["destination","to","dest","destinationairport","arrairport"],
        "sched_dep": ["scheduleddeparture","std","scheduleddep","scheddep","scheduleddepart","scheduledtimedeparture","schdep","sch_departure"],
        "act_dep": ["actualdeparture","atd","actualdep","actdep","actualtimedeparture"],
        "sched_arr": ["scheduledarrival","sta","scheduledarr","schedarr","scheduledtimearrival","scharr","sch_arrival"],
        "act_arr": ["actualarrival","ata","actualarr","actarr","actualtimearrival"],
        "carrier": ["airline","carrier","operator"],
        "flight_number": ["flightno","flightnumber","fltno","fltnumber","flight","fl.no","fltnr","flt"],
        "tail_number": ["tailnumber","tail","registration","reg"],
        "runway": ["runway","rwy"],
        "gate": ["gate","stand","bay"],
        "airport": ["airport","apt","ap","station","city"],
        "date": ["date","flightdate","utcdate","localdate","day"],
    }
    cols_norm = {c: _norm(c) for c in df.columns}

    def _match(targets):
        for raw, normed in cols_norm.items():
            if normed in targets:
                return raw
        return None

    for k, targets in candidates.items():
        hit = _match(targets)
        if hit is not None:
            mapping[k] = hit
    return mapping

# =============================== Time Parsing & Combine ===============================

_TIME_ONLY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?\s*([AaPp][Mm])?\s*$")

def parse_time_or_datetime(s: pd.Series) -> pd.Series:
    """
    Parse a column that may be (a) time-only or (b) full datetime.
    IMPORTANT: time-only values must NOT anchor to 'today'. We anchor them to 1900-01-01
    and later combine with a real date column.
    Returns datetime64[ns].
    """
    ss = _clean_token_series(s)
    is_time_only = ss.str.match(_TIME_ONLY_RE, na=False)

    out = pd.Series(pd.NaT, index=ss.index, dtype="datetime64[ns]")

    # Non-time-only → let pandas parse (dayfirst for DD/MM)
    if (~is_time_only).any():
        out.loc[~is_time_only] = pd.to_datetime(ss[~is_time_only], errors="coerce", dayfirst=True)

    # Time-only → prefix a neutral date to avoid "today" anchoring
    if is_time_only.any():
        subset = ss[is_time_only].str.strip()

        # 24h with seconds
        t = pd.to_datetime("1900-01-01 " + subset, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        miss = t.isna()
        # 24h without seconds
        if miss.any():
            t.loc[miss] = pd.to_datetime("1900-01-01 " + subset[miss], format="%Y-%m-%d %H:%M", errors="coerce")
        # 12h with seconds
        miss = t.isna()
        if miss.any():
            t.loc[miss] = pd.to_datetime("1900-01-01 " + subset[miss].str.upper(), format="%Y-%m-%d %I:%M:%S %p", errors="coerce")
        # 12h without seconds
        miss = t.isna()
        if miss.any():
            t.loc[miss] = pd.to_datetime("1900-01-01 " + subset[miss].str.upper(), format="%Y-%m-%d %I:%M %p", errors="coerce")

        out.loc[is_time_only] = t

    return out

def combine_date_and_time(date_col: pd.Series, time_or_dt: pd.Series) -> pd.Series:
    """
    Combine a 'date' column with a 'time or datetime' column:
      - If time_or_dt looks like a full datetime (year > 1971), keep it.
      - Else: merge date + clock from time_or_dt (anchored at 1900-01-01).
    """
    d = pd.to_datetime(date_col, errors="coerce").dt.date
    t = pd.to_datetime(time_or_dt, errors="coerce")

    # If already full datetime (from file), keep it
    full_mask = t.notna() & (t.dt.year > 1971)
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    out.loc[full_mask] = t.loc[full_mask]

    # For 1900-anchored times, combine with date
    m = t.notna() & ~full_mask & pd.Series(d).notna()
    if m.any():
        hh = t.loc[m].dt.strftime("%H:%M:%S")
        dd = pd.Series(d).loc[m].astype(str)
        out.loc[m] = pd.to_datetime(dd + " " + hh, errors="coerce")

    return out

# =============================== Normalization & Features ===============================

def normalize_times(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """
    Returns a copy of df with canonical columns:
      date (date), sched_dep, act_dep, sched_arr, act_arr (datetime64[ns])
      plus origin, destination, carrier, flight_number, tail_number if found.
    """
    d = df.copy()

    # Resolve/clean date column (if exists)
    date_col = cols.get("date")
    if date_col and date_col in d.columns:
        d["date"] = pd.to_datetime(d[date_col], errors="coerce").dt.date
        # Fill small gaps in date (some sheets repeat time blocks w/o date on every row)
        d["date"] = pd.Series(d["date"]).ffill().bfill()
    else:
        d["date"] = pd.NaT

    # Copy over categorical fields if present
    for k in ["origin","destination","carrier","flight_number","tail_number","runway","gate","airport","sheet"]:
        src = cols.get(k) if k in cols else (k if k in d.columns else None)
        if src and src in d.columns:
            d[k] = d[src]

    # Parse schedule/actual columns (robust)
    def _get_series(name, fallbacks):
        src = cols.get(name)
        if src is None:
            for fb in fallbacks:
                if fb in d.columns:
                    src = fb; break
        return parse_time_or_datetime(d[src]) if src in d.columns else pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")

    t_sched_dep = _get_series("sched_dep", ["scheduleddeparture","std","sched_dep","Scheduled Departure","STD"])
    t_act_dep   = _get_series("act_dep",   ["actualdeparture","atd","act_dep","Actual Departure","ATD"])
    t_sched_arr = _get_series("sched_arr", ["scheduledarrival","sta","sched_arr","Scheduled Arrival","STA"])
    t_act_arr   = _get_series("act_arr",   ["actualarrival","ata","act_arr","Actual Arrival","ATA"])

    # Combine with date
    d["sched_dep"] = combine_date_and_time(d["date"], t_sched_dep)
    d["act_dep"]   = combine_date_and_time(d["date"], t_act_dep)
    d["sched_arr"] = combine_date_and_time(d["date"], t_sched_arr)
    d["act_arr"]   = combine_date_and_time(d["date"], t_act_arr)

    return d

def compute_basic_features(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """
    Compute derived fields your app expects: delays, flight_id, etc.
    """
    d = df.copy()

    # Delays
    if "sched_dep" in d.columns and "act_dep" in d.columns:
        d["dep_delay_min"] = (pd.to_datetime(d["act_dep"]) - pd.to_datetime(d["sched_dep"])).dt.total_seconds() / 60.0
    else:
        d["dep_delay_min"] = np.nan

    if "sched_arr" in d.columns and "act_arr" in d.columns:
        d["arr_delay_min"] = (pd.to_datetime(d["act_arr"]) - pd.to_datetime(d["sched_arr"])).dt.total_seconds() / 60.0
    else:
        d["arr_delay_min"] = np.nan

    # Flight id: carrier + flight_number if available, else index
    carrier = (d["carrier"].astype(str).str.upper() if "carrier" in d.columns else pd.Series("", index=d.index))
    fno = (d["flight_number"].astype(str) if "flight_number" in d.columns else pd.Series("", index=d.index))
    # Clean FLT like "123.0"
    fno = fno.str.replace(r"^(\d+)\.0$", r"\1", regex=True).str.replace(" ", "", regex=False).str.upper()
    # If fno already contains prefix equal to carrier, don't duplicate
    prefix = fno.str.extract(r"^([A-Z0-9]{1,3})(?=\d)", expand=False).fillna("")
    same_prefix = (carrier != "") & (prefix == carrier)
    base = np.where(same_prefix | (carrier == "") | (fno == ""), fno, carrier + fno)
    base = pd.Series(base, index=d.index).str.strip()
    # If empty, fall back to index
    fallback = pd.Series(d.index.astype(str), index=d.index)
    d["flight_id"] = base.where(base != "", fallback)

    return d

# =============================== OPTIONAL: Convenience ===============================

def read_and_normalize(input_path: str) -> pd.DataFrame:
    """
    Convenience function (not used by pipeline) to read Excel and normalize in one go.
    """
    raw = read_excel_multiblock(input_path)
    cols = detect_columns(raw)
    norm = normalize_times(raw, cols)
    out = compute_basic_features(norm, cols)
    return out
