
import pandas as pd
import numpy as np
from typing import Tuple, Dict

def _airport_any_mask(df: pd.DataFrame, airport: str) -> pd.Series:
    ap = (airport or "").upper()
    m_any = pd.Series(True, index=df.index)
    cols = []
    if "origin" in df.columns:
        cols.append(df["origin"].astype(str).str.upper())
    if "destination" in df.columns:
        cols.append(df["destination"].astype(str).str.upper())
    if "airport" in df.columns:
        cols.append(df["airport"].astype(str).str.upper())
    if cols:
        m_any = False
        for c in cols:
            m_any = m_any | c.str.contains(ap, na=False)
    return m_any

def _primary_time(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_datetime(df.get("sched_dep"), errors="coerce")
    arr = pd.to_datetime(df.get("sched_arr"), errors="coerce")
    return dep.fillna(arr)

def _make_uid(df: pd.DataFrame) -> pd.Series:
    base = df["flight_id"].astype(str)
    return base + "_" + base.groupby(base).cumcount().astype(str)

def _enhanced_edge_weights(flight1, flight2, edge_type: str) -> float:
    """
    Calculate more realistic edge weights based on flight characteristics.
    """
    if edge_type == "rotation":
        # Aircraft rotation - very strong dependency
        turn_time = (pd.to_datetime(flight2.get("sched_dep")) - 
                    pd.to_datetime(flight1.get("sched_arr"))).total_seconds() / 60.0
        
        if pd.isna(turn_time):
            return 0.3
        
        # Tighter turnarounds have stronger influence
        if turn_time < 45:
            return 0.9  # Very tight turnaround
        elif turn_time < 90:
            return 0.7  # Normal turnaround
        else:
            return 0.4  # Loose turnaround
            
    elif edge_type == "runway":
        # Runway congestion - moderate dependency
        return 0.25
    
    elif edge_type == "gate":
        # Gate conflicts - moderate dependency  
        return 0.35
    
    elif edge_type == "crew":
        # Crew connections - strong dependency
        return 0.8
        
    return 0.2  # Default

def _build_enhanced_edges(df: pd.DataFrame, slot_min: int = 5):
    """
    Enhanced edge building with multiple dependency types and realistic weights.
    """
    df = df.copy()
    df["uid"] = _make_uid(df)
    uid_to_idx = {u: i for i, u in enumerate(df["uid"].tolist())}
    edges = []

    # 1. Aircraft Rotation edges (strongest dependency)
    if "tail_number" in df.columns and df["tail_number"].notna().any():
        grp = df.dropna(subset=["tail_number"]).sort_values("sched_dep").groupby("tail_number")
        for tail, g in grp:
            flights = list(g.iterrows())
            for i in range(len(flights) - 1):
                _, curr_flight = flights[i]
                _, next_flight = flights[i + 1]
                
                weight = _enhanced_edge_weights(curr_flight, next_flight, "rotation")
                edges.append((uid_to_idx[curr_flight["uid"]], uid_to_idx[next_flight["uid"]], weight))

    # 2. Enhanced runway adjacency with time decay
    df["slot_time"] = _primary_time(df).dt.floor(f"{slot_min}min")
    for slot_time, g in df.dropna(subset=["slot_time"]).sort_values("slot_time").groupby("slot_time"):
        flights = g.sort_values("sched_dep")
        flight_list = list(flights.iterrows())
        
        for i in range(len(flight_list)):
            for j in range(i + 1, min(i + 4, len(flight_list))):  # Look ahead max 3 flights
                curr_idx, curr_flight = flight_list[i]
                next_idx, next_flight = flight_list[j]
                
                # Weight decreases with distance in queue
                base_weight = 0.3
                distance_decay = 0.8 ** (j - i - 1)  # Exponential decay
                weight = base_weight * distance_decay
                
                edges.append((uid_to_idx[curr_flight["uid"]], uid_to_idx[next_flight["uid"]], weight))

    # 3. Gate conflicts (if gate information available)
    if "gate" in df.columns and df["gate"].notna().any():
        gate_groups = df.dropna(subset=["gate"]).groupby("gate")
        for gate, g in gate_groups:
            flights = g.sort_values("sched_dep")
            flight_list = list(flights.iterrows())
            
            for i in range(len(flight_list) - 1):
                _, curr_flight = flight_list[i]
                _, next_flight = flight_list[i + 1]
                
                weight = _enhanced_edge_weights(curr_flight, next_flight, "gate")
                edges.append((uid_to_idx[curr_flight["uid"]], uid_to_idx[next_flight["uid"]], weight))

    # 4. Carrier-based crew connections (heuristic)
    if "carrier" in df.columns and df["carrier"].notna().any():
        carrier_groups = df.dropna(subset=["carrier"]).groupby("carrier")
        for carrier, g in carrier_groups:
            # Only for flights close in time (crew could be same)
            g_sorted = g.sort_values("sched_dep")
            for i, (_, flight1) in enumerate(g_sorted.iterrows()):
                time1 = pd.to_datetime(flight1["sched_dep"])
                
                # Look for flights 2-6 hours later (potential crew connection)
                for j, (_, flight2) in enumerate(g_sorted.iloc[i+1:].iterrows(), i+1):
                    time2 = pd.to_datetime(flight2["sched_dep"])
                    time_diff = (time2 - time1).total_seconds() / 3600.0  # hours
                    
                    if 2 <= time_diff <= 6:  # Reasonable crew connection window
                        weight = 0.4 * max(0, 1 - (time_diff - 2) / 4)  # Decreasing weight
                        edges.append((uid_to_idx[flight1["uid"]], uid_to_idx[flight2["uid"]], weight))
                    elif time_diff > 6:
                        break  # Too far apart

    return edges, uid_to_idx, df[["uid", "flight_id"]]

def influence_table_enhanced(df: pd.DataFrame, base_delay_col: str = "arr_delay_min", steps: int = 4, airport: str | None = None, decay_factor: float = 0.85):
    """
    Enhanced influence calculation with configurable decay and multiple propagation paths.
    """
    d = df.copy()
    if airport:
        d = d[_airport_any_mask(d, airport)]

    if base_delay_col not in d.columns or d[base_delay_col].notna().sum() == 0:
        base_delay_col = "dep_delay_min"

    d["delay_base"] = pd.to_numeric(d[base_delay_col], errors="coerce").fillna(0).clip(lower=0)

    edges, uid_to_idx, map_df = _build_enhanced_edges(d, slot_min=5)
    if not uid_to_idx:
        return pd.DataFrame(columns=["flight_id", "influence_delta_total_delay", "direct_impact", "cascading_impact"])

    N = len(uid_to_idx)

    # Build adjacency matrix with enhanced weights
    W = np.zeros((N, N), dtype=float)
    for u, v, weight in edges:
        W[u, v] = weight

    d["uid"] = _make_uid(d)
    b = np.zeros(N, dtype=float)
    for _, r in d.iterrows():
        i = uid_to_idx.get(r["uid"])
        if i is not None:
            b[i] = float(r["delay_base"])

    # Enhanced propagation with decay
    s = np.ones(N, dtype=float)
    total_influence = s.copy()  # Direct impact
    cascading_influence = np.zeros(N, dtype=float)
    
    current_decay = decay_factor
    for step in range(steps):
        # Matrix-vector multiplication for propagation
        s = W.T @ s  # Transpose because we want column sums
        cascading_influence += s * current_decay
        total_influence += s * current_decay
        current_decay *= decay_factor  # Exponential decay over steps

    # Calculate influence scores
    direct_impact = b
    cascading_impact = b * cascading_influence
    total_delta = b * total_influence

    # Build results
    idx_to_uid = {i: u for u, i in uid_to_idx.items()}
    res = []
    for i in range(N):
        uid = idx_to_uid[i]
        fid = map_df.loc[map_df["uid"] == uid, "flight_id"].iloc[0]
        res.append({
            "flight_id": fid,
            "influence_delta_total_delay": float(total_delta[i]),
            "direct_impact": float(direct_impact[i]),
            "cascading_impact": float(cascading_impact[i]),
            "influence_ratio": float(cascading_impact[i] / max(direct_impact[i], 0.1))  # Avoid div by zero
        })

    out = (pd.DataFrame(res)
           .groupby("flight_id", as_index=False)
           .agg({
               "influence_delta_total_delay": "sum",
               "direct_impact": "sum", 
               "cascading_impact": "sum",
               "influence_ratio": "mean"
           })
           .sort_values("influence_delta_total_delay", ascending=False))
    
    return out

# Backward compatibility
def influence_table(df: pd.DataFrame, base_delay_col: str = "arr_delay_min", steps: int = 3, airport: str | None = None):
    """Enhanced version - drop-in replacement"""
    return influence_table_enhanced(df, base_delay_col, steps, airport)