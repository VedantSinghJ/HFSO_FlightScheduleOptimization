
import pandas as pd
import numpy as np

def _filter_for_airport(df, airport):
    airport = (airport or "").upper()
    if "origin" in df.columns and df["origin"].notna().any():
        m = df["origin"].astype(str).str.upper().str.contains(airport, na=False)
        if m.any():
            return df[m]
    if "airport" in df.columns and df["airport"].notna().any():
        m = df["airport"].astype(str).str.upper().str.contains(airport, na=False)
        if m.any():
            return df[m]
    return df

def _calculate_priority_score(df_flight_row, base_delay_col="dep_delay_min"):
    """
    Enhanced priority scoring for flight movement decisions.
    Lower score = higher priority to move (easier to reschedule).
    """
    # Base delay impact (lower delay = easier to move)
    delay = pd.to_numeric(df_flight_row.get(base_delay_col, 0), errors="coerce")
    delay = 0 if pd.isna(delay) else abs(float(delay))
    
    # Aircraft type priority (smaller aircraft easier to reschedule)
    # This is a proxy - in real data you'd have aircraft type
    carrier = str(df_flight_row.get("carrier", "")).upper()
    aircraft_penalty = 2.0 if carrier in ["AI", "9W", "6E"] else 1.0  # Major carriers harder to move
    
    # Time-based factors
    sched_time = pd.to_datetime(df_flight_row.get("sched_dep"), errors="coerce")
    if pd.notna(sched_time):
        hour = sched_time.hour
        # Peak hours (6-9, 18-21) harder to reschedule
        peak_penalty = 1.5 if (6 <= hour <= 9) or (18 <= hour <= 21) else 1.0
    else:
        peak_penalty = 1.0
    
    # Turnaround consideration (flights with tight connections harder to move)
    tail = str(df_flight_row.get("tail_number", "")).strip()
    turnaround_penalty = 1.2 if tail != "" else 1.0  # Assume tail number means tight turnaround
    
    return delay * aircraft_penalty * peak_penalty * turnaround_penalty

def greedy_slot_tuner_enhanced(df: pd.DataFrame, airport: str, window_min=5, max_shift_min=15, per_window_capacity=12):
    """
    Enhanced greedy tuner with smarter prioritization and multi-objective optimization.
    """
    d = _filter_for_airport(df.copy(), airport)

    # Time column setup
    d["sched_dep"] = pd.to_datetime(d.get("sched_dep"), errors="coerce")
    if d["sched_dep"].isna().all():
        d["sched_dep"] = pd.to_datetime(d.get("act_dep"), errors="coerce").fillna(
            pd.to_datetime(d.get("sched_arr"), errors="coerce")
        )
    d = d.dropna(subset=["sched_dep"])
    if d.empty:
        return pd.DataFrame(columns=["flight_id","old_time","new_time","shift_min","priority_score"])

    d["slot"] = d["sched_dep"].dt.floor(f"{window_min}min")
    counts = d.groupby("slot").size().rename("count").to_dict()
    slots = sorted(d["slot"].unique())

    # Enhanced: Calculate priority scores for all flights
    d["priority_score"] = d.apply(_calculate_priority_score, axis=1)

    suggestions = []
    
    for s in slots:
        over = counts.get(s, 0) - per_window_capacity
        if over <= 0:
            continue
            
        # Get candidates sorted by priority (lower = easier to move)
        candidates = d[d["slot"] == s].copy().sort_values("priority_score", ascending=True)
        
        moved_this_slot = 0
        for _, r in candidates.iterrows():
            if moved_this_slot >= over:
                break
                
            best_target = None
            best_cost = float('inf')
            
            # Enhanced: Try both directions and multiple shift sizes
            for shift_windows in range(1, max_shift_min // window_min + 1):
                for direction in [-1, 1]:
                    shift_min = direction * shift_windows * window_min
                    target_slot = s + pd.Timedelta(minutes=shift_min)
                    
                    # Check capacity
                    if counts.get(target_slot, 0) >= per_window_capacity:
                        continue
                    
                    # Calculate move cost (prefer smaller shifts, off-peak targets)
                    target_hour = target_slot.hour
                    peak_cost = 1.5 if (6 <= target_hour <= 9) or (18 <= target_hour <= 21) else 1.0
                    shift_cost = abs(shift_min) / max_shift_min  # Normalized shift penalty
                    
                    total_cost = shift_cost + 0.3 * peak_cost  # Weighted combination
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_target = (target_slot, shift_min)
            
            # Make the move if we found a good target
            if best_target:
                target_slot, shift_min = best_target
                counts[target_slot] = counts.get(target_slot, 0) + 1
                counts[s] -= 1
                moved_this_slot += 1
                
                suggestions.append({
                    "flight_id": r.get("flight_id"),
                    "old_time": r["sched_dep"],
                    "new_time": r["sched_dep"] + pd.Timedelta(minutes=shift_min),
                    "shift_min": shift_min,
                    "priority_score": r["priority_score"],
                    "move_cost": best_cost
                })

    return pd.DataFrame(suggestions)

def multi_objective_tuner(df: pd.DataFrame, airport: str, objectives=None, window_min=5, max_shift_min=15, per_window_capacity=12):
    """
    Multi-objective optimization considering delay reduction, passenger impact, and operational constraints.
    """
    if objectives is None:
        objectives = ["minimize_peak_congestion", "minimize_passenger_disruption", "minimize_total_shifts"]
    
    # Get base solution
    base_suggestions = greedy_slot_tuner_enhanced(df, airport, window_min, max_shift_min, per_window_capacity)
    
    if base_suggestions.empty:
        return base_suggestions
    
    # Multi-objective scoring
    base_suggestions = base_suggestions.copy()
    
    # Score 1: Congestion reduction (higher shift from peak slots = better)
    base_suggestions["congestion_score"] = base_suggestions.apply(lambda row: 
        2.0 if pd.to_datetime(row["old_time"]).hour in [7,8,19,20] else 1.0, axis=1)
    
    # Score 2: Passenger disruption (smaller shifts = better)  
    max_shift = base_suggestions["shift_min"].abs().max()
    base_suggestions["disruption_score"] = 1.0 - (base_suggestions["shift_min"].abs() / max_shift)
    
    # Score 3: Operational efficiency (fewer total moves = better)
    base_suggestions["efficiency_score"] = 1.0 / len(base_suggestions)
    
    # Weighted combination
    weights = {"minimize_peak_congestion": 0.4, "minimize_passenger_disruption": 0.4, "minimize_total_shifts": 0.2}
    base_suggestions["composite_score"] = (
        weights.get("minimize_peak_congestion", 0) * base_suggestions["congestion_score"] +
        weights.get("minimize_passenger_disruption", 0) * base_suggestions["disruption_score"] +
        weights.get("minimize_total_shifts", 0) * base_suggestions["efficiency_score"]
    )
    
    # Return top suggestions by composite score
    return base_suggestions.nlargest(min(len(base_suggestions), 50), "composite_score")

# Alias for backward compatibility
def greedy_slot_tuner(df: pd.DataFrame, airport: str, window_min=5, max_shift_min=15, per_window_capacity=12):
    """Enhanced version - drop-in replacement"""
    return greedy_slot_tuner_enhanced(df, airport, window_min, max_shift_min, per_window_capacity)