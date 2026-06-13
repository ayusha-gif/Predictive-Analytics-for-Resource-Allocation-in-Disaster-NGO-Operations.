from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "odisha_flood_resource_demand.csv"


BLOCKS = [
    {"block": "Ersama", "district": "Jagatsinghpur", "population_exposed": 92000, "road_access_score": 0.52},
    {"block": "Astaranga", "district": "Puri", "population_exposed": 74000, "road_access_score": 0.61},
    {"block": "Rajkanika", "district": "Kendrapara", "population_exposed": 88000, "road_access_score": 0.48},
    {"block": "Chandbali", "district": "Bhadrak", "population_exposed": 96000, "road_access_score": 0.44},
    {"block": "Balikuda", "district": "Jagatsinghpur", "population_exposed": 81000, "road_access_score": 0.58},
]


def build_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    rows = []

    for block in BLOCKS:
        river_base = rng.uniform(2.4, 3.2)
        block_risk = rng.uniform(0.9, 1.25)

        for date in dates:
            month = date.month
            is_monsoon = 1 if month in [6, 7, 8, 9] else 0
            is_peak_monsoon = 1 if month in [7, 8] else 0

            seasonal_rain = rng.gamma(2.0, 14.0) * is_monsoon
            off_season_rain = rng.gamma(1.2, 3.0) * (1 - is_monsoon)
            extreme_event = rng.binomial(1, 0.06 if is_monsoon else 0.01)
            rainfall_mm = seasonal_rain + off_season_rain + extreme_event * rng.uniform(80, 210)

            river_level_m = (
                river_base
                + rainfall_mm * 0.018
                + is_peak_monsoon * rng.uniform(0.2, 0.7)
                + rng.normal(0, 0.18)
            )
            river_level_m = max(river_level_m, 1.2)

            flood_incidents = max(
                0,
                int(
                    rng.poisson(
                        max(0, (rainfall_mm - 65) / 35)
                        + max(0, (river_level_m - 4.2) * 1.8)
                    )
                ),
            )

            access_penalty = 1.0 - block["road_access_score"]
            exposed_factor = block["population_exposed"] / 1000
            base_demand = 8 + 0.02 * exposed_factor

            demand = (
                base_demand
                + rainfall_mm * 0.9
                + max(0, river_level_m - 4.0) * 85
                + flood_incidents * 55
                + exposed_factor * 0.22 * block_risk
                + access_penalty * 35
                + rng.normal(0, 10)
            )

            if rainfall_mm < 35 and river_level_m < 4:
                demand *= rng.uniform(0.65, 0.9)

            rows.append(
                {
                    "date": date,
                    "district": block["district"],
                    "block": block["block"],
                    "rainfall_mm": round(rainfall_mm, 1),
                    "river_level_m": round(river_level_m, 2),
                    "flood_incidents": flood_incidents,
                    "population_exposed": block["population_exposed"],
                    "road_access_score": block["road_access_score"],
                    "dry_ration_kits_needed": max(50, int(round(demand))),
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["block", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    dataset.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(dataset):,} rows to {DATA_PATH}")
