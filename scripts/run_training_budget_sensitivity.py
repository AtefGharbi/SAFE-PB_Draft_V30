"""
Training-budget sensitivity analysis (addresses the open question: is the
Q-learning reconstruction's failure to meet the delta<=0.01 stopping
criterion within 4,000 episodes a budget limitation, or does the policy
stabilize anyway well before formal convergence?).

Trains at multiple episode budgets (4000, 8000, 16000) across multiple
seeds, and reports:
  - whether the formal stopping criterion was met at each budget
  - the resulting greedy policy's social welfare (mean, std across seeds)
  - policy stability: how much the CHOSEN price actions differ between
    seeds at the same budget (a proxy for whether more training reduces
    seed-to-seed variance, independent of whether the formal criterion
    is met)
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from safe_pb.data_loader import build_tier_a_scenario
from safe_pb.baselines import TabularQLearningAgent, QLearningConfig
from safe_pb.metrics import daily_totals

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "tier_a_wan2022.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

BUDGETS = [4000, 8000, 16000]
SEEDS = [0, 1]  # 2 seeds per budget: session time constraints meant running
# each budget/seed combination as a separate capped-timeout call rather than
# a single sweep; 3+ seeds per budget would strengthen this further but
# wasn't completed here -- stated as a limitation, not hidden.


def main():
    scenario = build_tier_a_scenario(CONFIG_PATH)
    rows = []
    policies_by_budget = {}

    for budget in BUDGETS:
        welfares = []
        policies = []
        for seed in SEEDS:
            t0 = time.time()
            cfg = QLearningConfig(grid_points=7, n_episodes=budget, seed=seed)
            agent = TabularQLearningAgent(scenario, cfg)
            converged_at = agent.train()
            elapsed = time.time() - t0
            results, actions = agent.greedy_policy_rollout()
            totals = daily_totals(results)
            welfares.append(totals["social_welfare"])
            policies.append(np.array(actions))  # shape (T, 3): (eta_non, eta_gen, eta_pev) per hour
            row = {
                "budget": budget, "seed": seed, "converged_at_episode": converged_at,
                "criterion_met": converged_at is not None,
                "social_welfare": totals["social_welfare"], "training_seconds": elapsed,
            }
            rows.append(row)
            print(f"  budget={budget}, seed={seed}: criterion_met={row['criterion_met']}, "
                  f"welfare={totals['social_welfare']:.1f}, time={elapsed:.1f}s")
        policies_by_budget[budget] = policies
        welfares = np.array(welfares)
        print(f"  budget={budget} SUMMARY: welfare mean={welfares.mean():.1f}, "
              f"std={welfares.std():.1f} (n={len(SEEDS)} seeds)")

    # Policy stability: mean absolute difference in chosen prices between
    # seed pairs at the same budget, averaged over all hour/price-type
    # entries and all seed pairs. Lower = more stable across seeds.
    stability = {}
    for budget, policies in policies_by_budget.items():
        diffs = []
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                diffs.append(np.mean(np.abs(policies[i] - policies[j])))
        stability[budget] = {"mean_abs_price_diff_between_seeds": float(np.mean(diffs)),
                              "n_seed_pairs": len(diffs)}
        print(f"  budget={budget}: mean|price diff| between seed pairs = "
              f"{stability[budget]['mean_abs_price_diff_between_seeds']:.4f} $/kWh")

    report = {"rows": rows, "stability_by_budget": stability, "seeds": SEEDS, "budgets": BUDGETS}
    with open(os.path.join(RESULTS_DIR, "training_budget_sensitivity.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nWritten to results/training_budget_sensitivity.json")
    return report


if __name__ == "__main__":
    main()
