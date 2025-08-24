# Enhanced metrics.py - More robust statistical analysis

import pandas as pd
import numpy as np
from scipy import stats

def _airport_mask(df, airport, kind):
    airport = (airport or "").upper()
    if kind == "arrival":
        col = "destination"
    else:
        col = "origin"

    if col in df.columns and df[col].notna().any():
        return df[col].astype(str).str.upper().str.contains(airport, na=False)
    if "airport" in df.columns and df["airport"].notna().any():
        return df["airport"].astype(str).str.upper().str.contains(airport, na=False)
    return pd.Series(True, index=df.index)

def busiest_slots_enhanced(df, time_col="sched_dep", window_min=15, top_n=20, include_stats=True):
    """Enhanced busiest slots with statistical confidence and trend analysis."""
    ts = pd.to_datetime(df[time_col], errors="coerce").dropna().sort_values()
    if ts.empty:
        return pd.DataFrame(columns=["slot_start","movements","avg_per_day","confidence_interval","trend"])
    
    slot = f"{window_min}min"
    
    # Group by slot and date to get daily patterns
    df_temp = pd.DataFrame({"timestamp": ts})
    df_temp["slot"] = df_temp["timestamp"].dt.floor(slot)
    df_temp["date"] = df_temp["timestamp"].dt.date
    
    # Daily counts per slot
    daily_counts = df_temp.groupby(["slot", "date"]).size().reset_index(name="daily_count")
    
    # Aggregate statistics
    slot_stats = daily_counts.groupby("slot").agg({
        "daily_count": ["mean", "std", "count", "median"]
    }).round(2)
    
    slot_stats.columns = ["avg_per_day", "std_dev", "days_observed", "median_per_day"]
    slot_stats["total_movements"] = daily_counts.groupby("slot")["daily_count"].sum()
    
    if include_stats:
        # Calculate 95% confidence intervals
        slot_stats["margin_of_error"] = 1.96 * slot_stats["std_dev"] / np.sqrt(slot_stats["days_observed"])
        slot_stats["ci_lower"] = (slot_stats["avg_per_day"] - slot_stats["margin_of_error"]).clip(lower=0)
        slot_stats["ci_upper"] = slot_stats["avg_per_day"] + slot_stats["margin_of_error"]
        slot_stats["confidence_interval"] = slot_stats.apply(
            lambda row: f"{row['ci_lower']:.1f}-{row['ci_upper']:.1f}", axis=1
        )
        
        # Simple trend analysis (increasing/decreasing over time)
        trend_analysis = []
        for slot in slot_stats.index:
            slot_data = daily_counts[daily_counts["slot"] == slot].sort_values("date")
            if len(slot_data) >= 3:
                x = np.arange(len(slot_data))
                y = slot_data["daily_count"].values
                slope, _, r_value, _, _ = stats.linregress(x, y)
                
                if abs(r_value) > 0.3:  # Significant correlation
                    trend = "↗ Increasing" if slope > 0 else "↘ Decreasing"
                else:
                    trend = "→ Stable"
            else:
                trend = "? Insufficient data"
            trend_analysis.append(trend)
        
        slot_stats["trend"] = trend_analysis
    
    result = slot_stats.reset_index()
    result = result.rename(columns={"slot": "slot_start", "total_movements": "movements"})
    result = result.sort_values("movements", ascending=False).head(top_n)
    
    return result

def best_time_to_operate_enhanced(df, at_airport=None, kind="arrival", window_min=30, confidence_level=0.95):
    """Enhanced best time analysis with confidence intervals and reliability scores."""
    if kind == "arrival":
        tcol, dcol = "sched_arr", "arr_delay_min"
    else:
        tcol, dcol = "sched_dep", "dep_delay_min"

    dx = df.copy()
    dx = dx[_airport_mask(dx, at_airport, kind)]

    dx[tcol] = pd.to_datetime(dx[tcol], errors="coerce")
    dx = dx.dropna(subset=[tcol])
    if dx.empty:
        return pd.DataFrame(columns=["slot_start","mean_delay_min","ci_lower","ci_upper","reliability_score","n"])

    delay_used = dx[dcol].copy()
    if delay_used.notna().sum() < max(5, 0.1 * len(dx)):
        delay_used = delay_used.fillna(0.0)
    dx = dx.assign(delay_used=delay_used)

    slot = f"{window_min}min"
    
    # Group by time slots
    grouped = dx.groupby(pd.Grouper(key=tcol, freq=slot))["delay_used"]
    
    # Calculate comprehensive statistics
    stats_df = grouped.agg(["mean", "std", "count", "median"]).reset_index()
    stats_df.columns = [tcol, "mean_delay_min", "std_delay", "n", "median_delay_min"]
    stats_df = stats_df.rename(columns={tcol: "slot_start"})
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    stats_df["margin_of_error"] = stats.t.ppf(1 - alpha/2, stats_df["n"] - 1) * stats_df["std_delay"] / np.sqrt(stats_df["n"])
    stats_df["ci_lower"] = (stats_df["mean_delay_min"] - stats_df["margin_of_error"]).fillna(stats_df["mean_delay_min"])
    stats_df["ci_upper"] = (stats_df["mean_delay_min"] + stats_df["margin_of_error"]).fillna(stats_df["mean_delay_min"])
    
    # Reliability score (higher is more reliable/predictable)
    stats_df["coefficient_of_variation"] = stats_df["std_delay"] / np.maximum(stats_df["mean_delay_min"], 0.1)
    stats_df["sample_reliability"] = np.log(stats_df["n"]) / np.log(stats_df["n"].max())  # Normalized log scale
    stats_df["consistency_score"] = 1 / (1 + stats_df["coefficient_of_variation"])  # Lower CV = higher consistency
    
    # Combined reliability score (0-100)
    stats_df["reliability_score"] = (
        0.6 * stats_df["consistency_score"] + 
        0.4 * stats_df["sample_reliability"]
    ) * 100
    
    # Round for presentation
    stats_df["reliability_score"] = stats_df["reliability_score"].round(1)
    stats_df["mean_delay_min"] = stats_df["mean_delay_min"].round(2)
    stats_df["ci_lower"] = stats_df["ci_lower"].round(2)
    stats_df["ci_upper"] = stats_df["ci_upper"].round(2)
    
    # Select relevant columns
    result_cols = ["slot_start", "mean_delay_min", "ci_lower", "ci_upper", "reliability_score", "n", "median_delay_min"]
    result = stats_df[result_cols].sort_values("mean_delay_min")
    
    return result

def runway_load_index_enhanced(
    times: pd.Series,
    window_min: int = 5,
    weather_factor: float = 1.0,
    ops_per_window_capacity: int = 12,
    occ_per_flight_min: float = 1.5,
    include_predictions: bool = True
):
    """Enhanced RLI with predictive analytics and bottleneck identification."""
    ts = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if ts.empty:
        return pd.DataFrame(columns=["slot_start","movements","rli","bottleneck_risk","predicted_delay"])
    
    slot = f"{window_min}min"
    grp = (
        pd.DataFrame({"t": ts})
        .groupby(pd.Grouper(key="t", freq=slot))
        .size()
        .rename("movements")
        .reset_index()
    )
    
    # Enhanced capacity calculation
    effective_capacity = ops_per_window_capacity * max(weather_factor, 1e-3)
    actual_capacity = effective_capacity * (window_min / 5.0)  # Scale by window size
    
    grp["theoretical_capacity"] = actual_capacity
    grp["rli"] = (grp["movements"] * occ_per_flight_min) / actual_capacity
    grp["capacity_utilization_pct"] = (grp["movements"] / actual_capacity * 100).round(1)
    
    if include_predictions:
        # Bottleneck risk assessment
        grp["bottleneck_risk"] = pd.cut(
            grp["rli"], 
            bins=[0, 0.7, 0.9, 1.1, float('inf')], 
            labels=["Low", "Medium", "High", "Critical"]
        )
        
        # Predicted delay based on congestion
        grp["predicted_delay"] = np.where(
            grp["rli"] <= 1.0,
            0,
            (grp["rli"] - 1.0) * 10 * window_min  # 10 min delay per 0.1 RLI over capacity
        ).round(1)
        
        # Queue length estimation
        grp["estimated_queue_length"] = np.maximum(0, grp["movements"] - actual_capacity).round(0)
    
    grp = grp.rename(columns={"t": "slot_start"})
    return grp

def delay_propagation_analysis(df, time_window_hours=2):
    """
    Analyze how delays propagate through the system over time.
    """
    if "dep_delay_min" not in df.columns:
        return pd.DataFrame(columns=["time_window", "avg_delay_propagation", "affected_flights"])
    
    df_sorted = df.sort_values("sched_dep").copy()
    df_sorted["sched_dep"] = pd.to_datetime(df_sorted["sched_dep"])
    df_sorted["dep_delay_min"] = pd.to_numeric(df_sorted["dep_delay_min"], errors="coerce").fillna(0)
    
    results = []
    window_td = pd.Timedelta(hours=time_window_hours)
    
    for i, row in df_sorted.iterrows():
        current_time = row["sched_dep"]
        current_delay = row["dep_delay_min"]
        
        if current_delay <= 5:  # Skip flights with minimal delay
            continue
            
        # Find flights in the next time window
        future_mask = (
            (df_sorted["sched_dep"] > current_time) & 
            (df_sorted["sched_dep"] <= current_time + window_td)
        )
        future_flights = df_sorted[future_mask]
        
        if len(future_flights) == 0:
            continue
            
        # Calculate correlation/impact
        future_delays = future_flights["dep_delay_min"]
        avg_future_delay = future_delays.mean()
        affected_count = len(future_flights[future_delays > 5])
        
        # Simple propagation score
        propagation_score = min(current_delay * 0.1, avg_future_delay)
        
        results.append({
            "initiating_flight": row.get("flight_id", i),
            "initial_delay": current_delay,
            "time_window_start": current_time,
            "avg_delay_propagation": propagation_score,
            "affected_flights": affected_count,
            "total_future_flights": len(future_flights)
        })
    
    return pd.DataFrame(results).head(20)  # Top 20 propagation events

# Backward compatibility aliases
def busiest_slots(df, time_col="sched_dep", window_min=15, top_n=20):
    """Enhanced version - drop-in replacement"""
    result = busiest_slots_enhanced(df, time_col, window_min, top_n, include_stats=False)
    # Return in original format for compatibility
    return result[["slot_start", "movements"]].copy()

def best_time_to_operate(df, at_airport=None, kind="arrival", window_min=30):
    """Enhanced version - drop-in replacement"""
    result = best_time_to_operate_enhanced(df, at_airport, kind, window_min)
    # Return in original format for compatibility
    return result[["slot_start", "mean_delay_min", "n"]].copy()

def runway_load_index(
    times: pd.Series,
    window_min: int = 5,
    weather_factor: float = 1.0,
    ops_per_window_capacity: int = 12,
    occ_per_flight_min: float = 1.5,
):
    """Enhanced version - drop-in replacement"""
    result = runway_load_index_enhanced(
        times, window_min, weather_factor, ops_per_window_capacity, occ_per_flight_min, include_predictions=False
    )
    # Return in original format for compatibility
    return result[["slot_start", "movements", "rli"]].copy()