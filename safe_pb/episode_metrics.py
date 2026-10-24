"""Episode-level metrics used for manuscript-grade evaluation.

These metrics are computed from actually executed environment trajectories;
no placeholder or estimated values are introduced here.
"""
from __future__ import annotations
import numpy as np
from .metrics import cvar, jain_index, gini_coefficient


def _nanmin_or_nan(values: np.ndarray) -> float:
    """Return nan for an intentionally unmeasured diagnostic without warnings."""
    finite = values[np.isfinite(values)]
    return float(finite.min()) if finite.size else float("nan")


def discomfort_normalizers(scenario) -> np.ndarray:
    """Pre-compute a conservative daily maximum discomfort for each
    controllable household from declared device bounds.  The value is fixed
    before evaluating a policy and is therefore not fitted post hoc.
    """
    vals=[]
    for i, ld in enumerate(scenario.gen_loads):
        e=np.asarray(scenario.gen_demand_kwh[i], float)
        lo=e*(1-ld.dr_max_frac); hi=e*(1-ld.dr_min_frac)
        d0=0.5*ld.alpha*(e-lo)**2 + ld.beta*(e-lo)
        d1=0.5*ld.alpha*(e-hi)**2 + ld.beta*(e-hi)
        vals.append(float(np.maximum(d0,d1).sum()))
    for j, ev in enumerate(scenario.pevs):
        e=np.asarray(scenario.pev_demand_kwh[j], float)
        per=[]
        for t in range(scenario.T):
            # Away window in the training environment is 09:00--17:00.
            if 9 <= t < 17:
                candidates=[0.0]
            else:
                candidates=[-ev.p_rated_kw, ev.p_rated_kw]
            per.append(max(0.5*ev.alpha*(e[t]-p)**2 + ev.beta*(e[t]-p) for p in candidates))
        vals.append(float(np.sum(per)))
    # Batteries have no requested-service discomfort and are therefore not
    # members of the fairness population. Including structural zeros would
    # mechanically depress Jain's index and inflate the Gini coefficient.
    out=np.asarray(vals, float)
    return np.maximum(out, 1e-12)


def summarize_episode(infos: list[dict], scenario) -> dict:
    if not infos:
        raise ValueError('empty episode')
    dt=float(scenario.dt_hours)
    U=np.array([x['U_t'] for x in infos], float)
    C=np.array([x['C_t'] for x in infos], float)
    load=np.array([np.sum(x['p_non'])+np.sum(x['p_gen'])+np.sum(x['p_pev'])+
                   np.sum(x.get('p_bat', [])) for x in infos], float)
    weighted_loss=float(np.sum((1-scenario.rho)*C - scenario.rho*U))
    d=np.sum(np.stack([np.asarray(x.get('fairness_discomfort',
                                        x['household_discomfort_all']), float)
                       for x in infos]), axis=0)
    norm=d/discomfort_normalizers(scenario)
    ac=np.array([x['min_voltage_ac'] for x in infos], float)
    candidate_ac=np.array([x.get('candidate_min_voltage_ac', np.nan) for x in infos], float)
    ac_bad=np.array([bool(x['ac_safety_violated']) for x in infos])
    candidate_ac_bad=np.array([bool(x.get('candidate_ac_safety_violated', False)) for x in infos])
    active=np.array([bool(x.get('projection_active', False)) for x in infos])
    # V23: split violations by whether ANY device-feasible action could have
    # satisfied the band (authority certificate). Only controllable-step
    # violations are attributable to the controller/projection.
    auth=np.array([x.get('authority_feasible') for x in infos], dtype=object)
    auth_known=np.array([a is not None for a in auth])
    auth_ok=np.array([a is True for a in auth])
    bills=np.sum(np.stack([np.asarray(x['household_energy_bill'], float) for x in infos]), axis=0)
    return {
        'weighted_social_loss': weighted_loss,
        'utility_profit': float(U.sum()),
        'customer_cost': float(C.sum()),
        'peak_kw': float(load.max()),
        'mean_kw': float(load.mean()),
        'PAR': float(load.max()/load.mean()) if load.mean()>0 else float('nan'),
        'voltage_violation_minutes': float(ac_bad.sum()*dt*60),
        'min_voltage_ac': _nanmin_or_nan(ac),
        'controllable_violation_minutes': float((ac_bad & auth_ok).sum()*dt*60) if auth_known.all() else float('nan'),
        'exogenous_violation_minutes': float((ac_bad & ~auth_ok).sum()*dt*60) if auth_known.all() else float('nan'),
        'authority_infeasible_steps': int((~auth_ok).sum()) if auth_known.all() else -1,
        'candidate_violation_restored_rate': (float(((candidate_ac_bad & auth_ok) & ~ac_bad).sum() /
                                               max((candidate_ac_bad & auth_ok).sum(), 1))
                                              if auth_known.all() and (candidate_ac_bad & auth_ok).any() else float('nan')),
        'candidate_voltage_violation_minutes': float(candidate_ac_bad.sum()*dt*60),
        'candidate_min_voltage_ac': _nanmin_or_nan(candidate_ac),
        'projection_active_steps': int(active.sum()),
        'projection_active_rate': float(active.mean()),
        'projection_distance_sum': float(sum(x['projection_distance'] for x in infos)),
        'projection_distance_mean': float(np.mean([x['projection_distance'] for x in infos])),
        'projection_delta_l1_sum': float(sum(np.abs(np.asarray(
            x.get('projection_delta_kw', []), float)).sum() for x in infos)),
        'slack_sum': float(sum(x['total_slack'] for x in infos)),
        'normalized_discomfort_variance': float(np.var(norm)),
        'jain_discomfort': jain_index(norm),
        'gini_discomfort': gini_coefficient(norm),
        'normalized_discomfort': norm.tolist(),
        'household_bill': bills.tolist(),
        'terminal_pev_shortfall_kwh': np.asarray(infos[-1].get('terminal_pev_shortfall_kwh', []), float).tolist(),
        'episode_failed_pev': bool(any(x.get('episode_terminated_for_pev_infeasibility', False) for x in infos)),
        # Full audit trail: one vector per time step. CSV serialization keeps
        # these as JSON-like arrays while scalar columns remain analysis-ready.
        'candidate_actions_kw': [np.asarray(x.get('candidate_actions_kw', []), float).tolist()
                                 for x in infos],
        'applied_actions_kw': [np.asarray(x.get('applied_actions_kw', []), float).tolist()
                               for x in infos],
        'projection_delta_kw': [np.asarray(x.get('projection_delta_kw', []), float).tolist()
                                for x in infos],
        'candidate_min_voltage_ac_by_step': candidate_ac.tolist(),
        'post_min_voltage_ac_by_step': ac.tolist(),
    }


def aggregate_episodes(rows: list[dict]) -> dict:
    if not rows:
        return {}
    scalar=['weighted_social_loss','utility_profit','customer_cost','peak_kw','PAR',
            'voltage_violation_minutes','min_voltage_ac',
            'controllable_violation_minutes','exogenous_violation_minutes',
            'authority_infeasible_steps','candidate_violation_restored_rate',
            'candidate_voltage_violation_minutes','candidate_min_voltage_ac',
            'projection_active_steps','projection_active_rate','projection_distance_sum',
            'projection_delta_l1_sum','slack_sum',
            'normalized_discomfort_variance','jain_discomfort','gini_discomfort']
    out={}
    for k in scalar:
        a=np.asarray([r[k] for r in rows], float)
        out[k+'_mean']=float(np.nanmean(a)); out[k+'_std']=float(np.nanstd(a, ddof=1)) if len(a)>1 else 0.0
    losses=np.asarray([r['weighted_social_loss'] for r in rows], float)
    out['CVaR90']=cvar(losses, .90); out['CVaR95']=cvar(losses, .95)
    out['n_episodes']=len(rows)
    out['pev_failure_rate']=float(np.mean([r['episode_failed_pev'] for r in rows]))
    return out
