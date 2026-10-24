"""
Distribution feeder model for the safety projection layer (Command 6).

Network SOURCE = [PUBLIC DATASET]: pandapower's built-in `case33bw()`, a
real published radial test feeder (Baran & Wu, 1989). This is genuine
network data, not synthetic.

Household-to-bus SOURCE = [SAFE-PB EXTENSION / modeling assumption]:
Wan et al. (2022) has no network model at all (their Sec. II-B explicitly
ignores power losses and network topology). Mapping their 15 loads (plus
SAFE-PB's battery/PV households) onto specific feeder buses, and scaling
each household's kW by `homes_per_bus` to reach feeder-representative
loading, is a stated, documented modeling choice for this validation pass
-- not a value taken from any dataset.

Provides two voltage-feasibility checks, matching the manuscript's
three-way distinction (Sec. 5.2 / Command 6):
  1. `linearized_voltages()` -- LinDistFlow-style linear approximation used
     INSIDE the safety QP (fast, differentiable, matches manuscript Eq. 6).
  2. `ac_powerflow_check()` -- full nonlinear AC power flow via pandapower
     (`pp.runpp`), used to INDEPENDENTLY VERIFY schedules after the fact.
     Per the manuscript's rule, a schedule that is linear-feasible is not
     claimed AC-feasible until this check passes.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandapower as pp
import pandapower.networks as pn


@dataclass
class FeederTopology:
    net: object
    root_bus: int
    parent: dict            # bus -> parent bus (tree)
    line_of_bus: dict       # bus -> (from_bus, to_bus, r_ohm, x_ohm) for the line feeding it
    path_to_root: dict      # bus -> ordered list of buses from root to bus (inclusive)
    non_slack_buses: list
    vn_kv: float
    base_p_mw: dict         # native case33bw per-bus load, captured before any mutation
    base_q_mvar: dict
    ac_initialized: bool = False


def build_feeder() -> FeederTopology:
    net = pn.case33bw()
    root = int(net.ext_grid.bus.values[0])
    vn_kv = float(net.bus.vn_kv.iloc[0])

    # Capture the feeder's OWN native loads before we touch anything, so
    # later calls can add household loads ON TOP of realistic pre-existing
    # feeder loading (rather than replacing most buses with zero, which
    # would make the network trivially under-loaded and never stress-test
    # the safety layer).
    base_p_mw = {int(r.bus): float(r.p_mw) for _, r in net.load.iterrows()}
    base_q_mvar = {int(r.bus): float(r.q_mvar) for _, r in net.load.iterrows()}

    # Build adjacency from IN-SERVICE lines only (tie lines are open, matches
    # the standard IEEE 33-bus radial operating configuration -- verified
    # directly from net.line.in_service, not assumed).
    adj = {int(b): [] for b in net.bus.index}
    line_between = {}
    for _, row in net.line[net.line.in_service].iterrows():
        f, t = int(row.from_bus), int(row.to_bus)
        r = row.r_ohm_per_km * row.length_km
        x = row.x_ohm_per_km * row.length_km
        adj[f].append(t)
        adj[t].append(f)
        line_between[(f, t)] = (f, t, r, x)
        line_between[(t, f)] = (f, t, r, x)

    # BFS from root to get a tree (parent pointers + path to root per bus)
    parent = {root: None}
    order = [root]
    frontier = [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    nxt.append(v)
        frontier = nxt

    line_of_bus = {}
    path_to_root = {root: [root]}
    for b in order:
        if b == root:
            continue
        p = parent[b]
        line_of_bus[b] = line_between[(p, b)]
        path_to_root[b] = path_to_root[p] + [b]

    non_slack = [int(b) for b in net.bus.index if int(b) != root]
    return FeederTopology(net=net, root_bus=root, parent=parent, line_of_bus=line_of_bus,
                           path_to_root=path_to_root, non_slack_buses=non_slack, vn_kv=vn_kv,
                           base_p_mw=base_p_mw, base_q_mvar=base_q_mvar)


def assign_households_to_buses(topo: FeederTopology, household_ids: list, homes_per_bus: float,
                                seed: int = 0) -> dict:
    """Round-robin assignment of household ids onto non-slack buses.
    Returns {household_id: bus}. `homes_per_bus` is used by the caller to
    scale kW, not here."""
    buses = topo.non_slack_buses
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(buses))
    return {hid: int(buses[order[i % len(buses)]]) for i, hid in enumerate(household_ids)}


def linearized_voltages(topo: FeederTopology, extra_p_mw: dict, extra_q_mvar: dict,
                         include_base: bool = True, base_scale: float = 1.0) -> dict:
    """LinDistFlow-style linear voltage approximation, manuscript Eq. (6):
        v_j = v_0 - 2 * sum_{l in path(0,j)} (r_l * P_l + x_l * Q_l)
    in per-unit, where P_l/Q_l is the cumulative downstream power carried by
    line l (sum of all bus injections in the subtree beyond that line).

    By default (`include_base=True`) the feeder's own native case33bw loads
    are included, and `extra_p_mw`/`extra_q_mvar` (e.g. household loads) are
    ADDED on top. `base_scale` multiplies the native loads before adding
    (e.g. base_scale=0.5 recalibrates the feeder to a less-saturated
    operating point -- see CALIBRATION note in config/tier_a_wan2022.yaml:
    at base_scale=1.0 the native feeder ALREADY violates 0.95pu with zero
    household load, which makes it impossible to show the safety layer's
    differential effect; base_scale<1 creates the headroom needed for that
    test). Pass `include_base=False` for pure marginal-sensitivity
    calculations (see `safety_projection.voltage_sensitivity`).
    Returns {bus: v_pu}.
    """
    bus_p_mw = {}
    bus_q_mvar = {}
    if include_base:
        bus_p_mw.update({b: v * base_scale for b, v in topo.base_p_mw.items()})
        bus_q_mvar.update({b: v * base_scale for b, v in topo.base_q_mvar.items()})
    for b, v in extra_p_mw.items():
        bus_p_mw[b] = bus_p_mw.get(b, 0.0) + v
    for b, v in extra_q_mvar.items():
        bus_q_mvar[b] = bus_q_mvar.get(b, 0.0) + v

    # cumulative downstream P/Q per bus-line, computed bottom-up
    subtree_p = {b: bus_p_mw.get(b, 0.0) for b in topo.parent}
    subtree_q = {b: bus_q_mvar.get(b, 0.0) for b in topo.parent}
    # process in reverse BFS order (leaves first) via repeated topological pass
    order = list(topo.path_to_root.keys())
    order.sort(key=lambda b: -len(topo.path_to_root[b]))  # deepest first
    for b in order:
        p = topo.parent[b]
        if p is not None:
            subtree_p[p] += subtree_p[b]
            subtree_q[p] += subtree_q[b]

    v_pu = {topo.root_bus: 1.0}
    for b in topo.non_slack_buses:
        drop_kv = 0.0
        for node in topo.path_to_root[b][1:]:
            f, t, r_ohm, x_ohm = topo.line_of_bus[node]
            P_l = subtree_p[node]  # MW carried by the line feeding `node`
            Q_l = subtree_q[node]
            drop_kv += (r_ohm * P_l + x_ohm * Q_l) / topo.vn_kv  # standard ΔV(kV) ≈ (rP+xQ)/V0
        v_pu[b] = 1.0 - drop_kv / topo.vn_kv
    return v_pu


def ac_powerflow_check(topo: FeederTopology, extra_p_mw: dict, extra_q_mvar: dict,
                        include_base: bool = True, base_scale: float = 1.0) -> dict:
    """Independent AC verification via pandapower's Newton-Raphson power flow.
    Sets each bus's load to base_scale*base_p_mw[bus] (+extra_p_mw[bus] if
    given), runs pp.runpp(), and returns per-bus voltage plus line/
    transformer loading."""
    net = topo.net
    # Update the EXISTING load table in place (one row per non-slack bus,
    # already present from case33bw()) rather than reconstructing rows --
    # rebuilding rows from a plain dict lost dtype information (e.g.
    # `controllable` silently became NaN/object instead of bool), which
    # made Newton-Raphson numerically unstable. Keeping the original,
    # correctly-typed table and only overwriting p_mw/q_mvar is both
    # simpler and avoids that failure mode.
    bus_to_loadidx = {int(r.bus): idx for idx, r in net.load.iterrows()}
    for b in topo.non_slack_buses:
        base_p = topo.base_p_mw.get(b, 0.0) * base_scale if include_base else 0.0
        base_q = topo.base_q_mvar.get(b, 0.0) * base_scale if include_base else 0.0
        p = base_p + extra_p_mw.get(b, 0.0)
        q = base_q + extra_q_mvar.get(b, 0.0)
        if b in bus_to_loadidx:
            idx = bus_to_loadidx[b]
            net.load.at[idx, "p_mw"] = p
            net.load.at[idx, "q_mvar"] = q
        else:
            pp.create_load(net, bus=b, p_mw=p, q_mvar=q, name=f"load_{b}")
    try:
        # Only bus injections change across the time series. Reuse the
        # previous voltage solution and invariant network matrices while
        # retaining the same Newton-Raphson AC formulation.
        pp.runpp(
            net,
            numba=False,
            init="results" if topo.ac_initialized else "flat",
            check_connectivity=not topo.ac_initialized,
            recycle={"bus_pq": True, "trafo": False, "gen": False},
        )
        topo.ac_initialized = True
        converged = True
    except Exception as e:
        return {"converged": False, "error": str(e)}
    return {
        "converged": converged,
        "vm_pu": net.res_bus.vm_pu.to_dict(),
        "line_loading_percent": net.res_line.loading_percent.to_dict(),
        "min_vm_pu": float(net.res_bus.vm_pu.min()),
        "max_vm_pu": float(net.res_bus.vm_pu.max()),
        "max_line_loading_percent": float(net.res_line.loading_percent.max()),
    }
