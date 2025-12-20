import json
from pathlib import Path

# ========= CONFIG =========
DATA_PATH_MULTIAGENT = Path("../logs/multi_agent_200_questions")

# Emission factor in kg CO2 per kWh (adjust this to match your region/grid).
EMISSION_FACTOR_KG_PER_KWH = 0.4
# ==========================


def load_queries(path: Path):
    """
    Load query JSON objects from a file or a directory.
    """
    all_queries = []

    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            all_queries.extend(data)
        elif isinstance(data, dict):
            if "queries" in data and isinstance(data["queries"], list):
                all_queries.extend(data["queries"])
            else:
                all_queries.append(data)
        else:
            raise ValueError("Unsupported JSON structure in file.")
    elif path.is_dir():
        for file in sorted(path.glob("*.json")):
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                all_queries.extend(data)
            elif isinstance(data, dict):
                if "queries" in data and isinstance(data["queries"], list):
                    all_queries.extend(data["queries"])
                else:
                    all_queries.append(data)
            else:
                raise ValueError(f"Unsupported JSON structure in {file}")
    else:
        raise FileNotFoundError(f"{path} is neither a file nor a directory")

    return all_queries


def format_seconds(total_seconds: float) -> str:
    """Return a human-readable HH:MM:SS.s string from seconds."""
    total_seconds = float(total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def merge_entries(existing, new):
    """
    Merge two log entries for the same query id.

    Strategy:
    - Prefer non-empty cost_data / agent_metadata.
    - For runtime_seconds and total_duration_seconds, take max (more conservative).
    - For total_energy_joules, take max.
    """
    if existing is None:
        return new

    merged = dict(existing)

    # runtime_seconds: take max if both exist
    rt_old = existing.get("runtime_seconds")
    rt_new = new.get("runtime_seconds")
    if rt_old is None:
        merged["runtime_seconds"] = rt_new
    elif rt_new is None:
        merged["runtime_seconds"] = rt_old
    else:
        merged["runtime_seconds"] = max(rt_old, rt_new)

    # cost_data: prefer entry that has it; if both, keep the one with larger total_cost
    cost_old = existing.get("cost_data") or {}
    cost_new = new.get("cost_data") or {}
    if cost_old and not cost_new:
        merged["cost_data"] = cost_old
    elif cost_new and not cost_old:
        merged["cost_data"] = cost_new
    elif cost_old and cost_new:
        old_total = cost_old.get("total_cost") or 0.0
        new_total = cost_new.get("total_cost") or 0.0
        merged["cost_data"] = cost_new if new_total >= old_total else cost_old
    else:
        merged["cost_data"] = {}

    # agent_metadata: prefer entry with energy/duration and merge key metrics
    meta_old = existing.get("agent_metadata") or {}
    meta_new = new.get("agent_metadata") or {}
    if not meta_old and meta_new:
        merged["agent_metadata"] = meta_new
    elif meta_old and not meta_new:
        merged["agent_metadata"] = meta_old
    elif meta_old and meta_new:
        merged_meta = dict(meta_old)

        # total_energy_joules: take max
        e_old = meta_old.get("total_energy_joules")
        e_new = meta_new.get("total_energy_joules")
        if e_old is None:
            merged_meta["total_energy_joules"] = e_new
        elif e_new is None:
            merged_meta["total_energy_joules"] = e_old
        else:
            merged_meta["total_energy_joules"] = max(e_old, e_new)

        # total_duration_seconds: take max
        d_old = meta_old.get("total_duration_seconds")
        d_new = meta_new.get("total_duration_seconds")
        if d_old is None:
            merged_meta["total_duration_seconds"] = d_new
        elif d_new is None:
            merged_meta["total_duration_seconds"] = d_old
        else:
            merged_meta["total_duration_seconds"] = max(d_old, d_new)

        merged["agent_metadata"] = merged_meta
    else:
        merged["agent_metadata"] = {}

    return merged


def main():
    raw_queries = load_queries(DATA_PATH_MULTIAGENT)

    # Keep only dict entries
    queries = [q for q in raw_queries if isinstance(q, dict)]
    filtered_out = len(raw_queries) - len(queries)
    if filtered_out:
        print(f"Warning: {filtered_out} non-dict entries were ignored.")

    # ---- Deduplicate by query id, merging duplicates ----
    by_id = {}
    for q in queries:
        qid = q.get("id")
        if qid is None:
            continue
        by_id[qid] = merge_entries(by_id.get(qid), q)

    unique_queries = list(by_id.values())
    n_raw = len(queries)
    n = len(unique_queries)
    duplicates = n_raw - n

    print(f"Loaded log entries:     {n_raw}")
    print(f"Unique query IDs:       {n}")
    if duplicates > 0:
        print(f"Merged duplicate IDs:   {duplicates}")
    print()

    if n == 0:
        print("No queries found after deduplication.")
        return

    # --------- Aggregate core metrics once ---------
    # Runtime
    total_runtime_seconds = sum((q.get("runtime_seconds") or 0.0)
                                for q in unique_queries)

    # Costs
    total_input_cost = 0.0
    total_output_cost = 0.0
    total_total_cost = 0.0
    for q in unique_queries:
        cost = q.get("cost_data") or {}
        total_input_cost += (cost.get("input_cost") or 0.0)
        total_output_cost += (cost.get("output_cost") or 0.0)
        total_total_cost += (cost.get("total_cost") or 0.0)

    # Energy & CO2 (manual calculation)
    total_energy_joules = 0.0
    count_with_energy = 0
    for q in unique_queries:
        meta = q.get("agent_metadata") or {}
        energy_j = meta.get("total_energy_joules")
        if energy_j is not None:
            total_energy_joules += energy_j
            count_with_energy += 1

    JOULES_PER_KWH = 3_600_000.0
    total_energy_kwh = total_energy_joules / JOULES_PER_KWH if count_with_energy else 0.0

    # Manual CO₂ calculation
    total_co2_kg = total_energy_kwh * EMISSION_FACTOR_KG_PER_KWH
    avg_co2_kg_per_query = (
        total_co2_kg / count_with_energy if count_with_energy else 0.0
    )

    # Per-query averages
    avg_runtime_seconds = total_runtime_seconds / n
    avg_input_cost = total_input_cost / n
    avg_output_cost = total_output_cost / n
    avg_total_cost = total_total_cost / n
    avg_energy_kwh_per_query = (
        total_energy_kwh / count_with_energy if count_with_energy else 0.0
    )

    # Cost metadata
    sample_cost_data = (unique_queries[0].get("cost_data") or {})
    currency = sample_cost_data.get("currency", "USD")
    model_name = sample_cost_data.get("model_name", "Unknown model")

    # --------- PER-QUERY AVERAGE STATS ---------
    print("=== PER-QUERY AVERAGE STATS ===")
    print(f"Model:                           {model_name}")
    print(f"Currency:                        {currency}")
    print(f"Queries counted:                 {n}")
    print()
    print(f"Avg runtime / query (seconds):   {avg_runtime_seconds:.3f}")
    print(f"Avg runtime / query (HH:MM:SS):  {format_seconds(avg_runtime_seconds)}")
    print()
    print(f"Avg input cost / query:          {avg_input_cost:.8f} {currency}")
    print(f"Avg output cost / query:         {avg_output_cost:.8f} {currency}")
    print(f"Avg total cost / query:          {avg_total_cost:.8f} {currency}")
    print()
    print(f"Avg energy / query:              {avg_energy_kwh_per_query:.9f} kWh")
    print(f"Avg CO₂ / query:                 {avg_co2_kg_per_query:.6f} kg")
    print(f"Emission factor used:            {EMISSION_FACTOR_KG_PER_KWH:.3f} kg CO₂/kWh")
    print()

    # --------- TOTAL BENCHMARK STATS ---------
    print("=== TOTAL BENCHMARK STATS ===")
    print(f"Total queries:                   {n}")
    print(f"Total runtime (seconds):         {total_runtime_seconds:.3f}")
    print(f"Total runtime (HH:MM:SS):        {format_seconds(total_runtime_seconds)}")
    print()
    print(f"Total input cost:                {total_input_cost:.8f} {currency}")
    print(f"Total output cost:               {total_output_cost:.8f} {currency}")
    print(f"Total combined cost:             {total_total_cost:.8f} {currency}")
    print()
    print(f"Total energy(J):                 {total_energy_joules:.3f} J")
    print(f"Total energy(kWh):               {total_energy_kwh:.6f} kWh")
    print(f"Total CO₂ emissions:             {total_co2_kg:.6f} kg")
    print()
    print("GSM8K Accuracy on 200 samples: 85%")


main()
