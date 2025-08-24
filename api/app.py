
from fastapi import FastAPI
import duckdb, pandas as pd

app = FastAPI(title="HFSO API")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/busiest")
def busiest(limit: int = 10, window_min: int = 15):
    con = duckdb.connect()
    df = con.execute("SELECT * FROM 'data/processed.parquet'").fetch_df()
    df["slot"] = pd.to_datetime(df["sched_dep"]).dt.floor(f"{window_min}min")
    q = df.groupby("slot").size().rename("movements").reset_index().sort_values("movements", ascending=False).head(limit)
    return q.to_dict(orient="records")
