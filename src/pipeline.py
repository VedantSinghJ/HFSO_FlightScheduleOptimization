import argparse
import os
import sys
import pandas as pd

# make 'src' imports work when called as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.processing import (
    read_excel_multiblock,
    detect_columns,
    normalize_times,
    compute_basic_features,
)

# -------------------------- helpers --------------------------
def _simple_read_excel(path: str, sheet: str | int | None = None) -> pd.DataFrame:
    """
    Simple pandas read_excel with header inference. If sheet is None, reads all sheets and concatenates.
    """
    try:
        if sheet is None:
            # Read all sheets -> dict[str, DataFrame]
            book = pd.read_excel(path, sheet_name=None)
            frames = [df for df in book.values() if df is not None and not df.empty]
            return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        else:
            return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame()

def _robust_read_csv(path: str) -> pd.DataFrame:
    """
    Try multiple encodings / delimiters for CSV.
    """
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        for sep in (None, ",", ";", "|", "\t"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception:
                continue
    return pd.DataFrame()

def _write_parquet_or_csv(df: pd.DataFrame, out_path: str) -> str:
    """
    Prefer Parquet; if engine missing, fall back to CSV with same basename.
    Returns final path written.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
        print(f"[pipeline] Saved parquet → {out_path}")
        return out_path
    except Exception as e:
        print(f"[pipeline] Parquet write failed ({e}); saving CSV instead.")
        csv_out = os.path.splitext(out_path)[0] + ".csv"
        df.to_csv(csv_out, index=False, encoding="utf-8")
        print(f"[pipeline] Saved CSV → {csv_out}")
        return csv_out

# -------------------------- core --------------------------
def run_pipeline(input_path: str, out_path: str, sheet: str | int | None = None, simple: bool = False) -> str:
    """
    Read raw schedule/ops file(s), normalize times, compute basic features, and persist.
    - input_path: .xlsx/.xls or .csv
    - sheet:      optional sheet name or index to read (Excel only).
                  NOTE: read_excel_multiblock is all-sheets; if you pass --sheet we use simple reader for that sheet.
    - simple:     force simple pandas read (skip multiblock parser).
    """
    inp = input_path.strip()
    low = inp.lower()
    print(f"[pipeline] Reading: {input_path}")

    raw = pd.DataFrame()

    # Excel
    if low.endswith((".xlsx", ".xls")):
        if simple and sheet is not None:
            print("[pipeline] --simple set with --sheet → using pandas.read_excel on that sheet.")
            raw = _simple_read_excel(inp, sheet)
        elif simple and sheet is None:
            print("[pipeline] --simple set → using pandas.read_excel across all sheets.")
            raw = _simple_read_excel(inp, None)
        else:
            # Preferred path: structured multi-block parser (reads ALL sheets)
            try:
                raw = read_excel_multiblock(inp)
                if raw is None or raw.empty:
                    print("[pipeline] Multiblock parser returned 0 rows; falling back to simple all-sheets read.")
                    raw = _simple_read_excel(inp, None if sheet is None else sheet)
            except Exception as e:
                print(f"[pipeline] Multiblock parser errored ({e}); falling back to simple read.")
                raw = _simple_read_excel(inp, None if sheet is None else sheet)

    # CSV
    elif low.endswith(".csv"):
        try:
            raw = pd.read_csv(inp)
        except Exception:
            print("[pipeline] Basic read_csv failed; trying robust CSV fallback …")
            raw = _robust_read_csv(inp)

    else:
        raise ValueError("Unsupported file type. Please provide .xlsx/.xls or .csv")

    if raw is None or raw.empty:
        raise RuntimeError(
            "No rows read from input file (after all fallbacks). "
            "Tip: open the file and copy the clean table to a new sheet with a single header row."
        )

    print(f"[pipeline] Raw shape: {raw.shape}")

    # 2) DETECT & NORMALIZE
    cols = detect_columns(raw)
    if not cols:
        print("[pipeline] Warning: column detection is empty; will still attempt normalization.")
    raw = normalize_times(raw, cols)
    df = compute_basic_features(raw, cols)

    # 3) WRITE
    final_out = _write_parquet_or_csv(df, out_path)

    # 4) SUMMARY
    dmin = pd.to_datetime(df.get("sched_dep"), errors="coerce").min()
    dmax = pd.to_datetime(df.get("sched_dep"), errors="coerce").max()
    print(f"[pipeline] Rows: {len(df)} | Window: {dmin} → {dmax}")
    return final_out

# -------------------------- CLI --------------------------
def main():
    ap = argparse.ArgumentParser("HFSO pipeline")
    ap.add_argument("--input", "-i", required=True, help="Path to Flight_Data.xlsx or CSV")
    ap.add_argument("--out", "-o", default="data/processed.parquet", help="Output parquet/CSV path")
    ap.add_argument("--sheet", help="Excel sheet name or index (optional). If omitted, ALL sheets are read/merged.")
    ap.add_argument("--simple", action="store_true", help="Force simple pandas read (skip multiblock parser)")
    args = ap.parse_args()

    sheet = args.sheet
    if sheet is not None:
        # allow numeric index
        try:
            sheet = int(sheet)
        except ValueError:
            pass

    run_pipeline(args.input, args.out, sheet=sheet, simple=args.simple)

if __name__ == "__main__":
    main()
