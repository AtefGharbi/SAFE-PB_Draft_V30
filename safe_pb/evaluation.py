"""Bundles metrics.daily_totals() output for several methods into a single
comparison table (pandas DataFrame), mirroring the manuscript's Table 1
structure (Sec. 7.1)."""
from __future__ import annotations
import pandas as pd
from .metrics import daily_totals


def build_comparison_table(results_by_method: dict) -> pd.DataFrame:
    rows = []
    for name, results in results_by_method.items():
        tot = daily_totals(results)
        rows.append({
            "method": name,
            "social_welfare": tot["social_welfare"],
            "utility_profit": tot["total_utility_profit"],
            "total_cost": tot["total_cost"],
            "peak_kw": tot["peak_kw"],
            "PAR": tot["PAR"],
        })
    return pd.DataFrame(rows).set_index("method")
