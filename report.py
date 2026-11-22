"""Builds Section 9 (Results) of the proposal as markdown, filled from the simulation output dict R."""
import numpy as np
from sim import METROS, LABEL, M, T, HYD

TARGET = dict(reliability=89.7, losses=8.0, ebitda_worst=12.0)


def tbl(head, rows): return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"


def build_report(R):
    risk, nom, ab, par, ens, sg, fq, tm = R["risk"], R["nominal"], R["ablation"], R["pareto"], R["ens"], R["sage"], R["fq"], R["timing"]
    rf, rr, rb = risk["framework"], risk["routing"], risk["reactive"]; o = []
    o.append("## 9. Results\n")
    o.append("### 9.1 Forecast quality and wait prediction\n")
    o.append(tbl(["Metric"] + METROS, [["Demand MAPE (%)"] + [f1(x) for x in fq["D"]], ["Solar+wind MAPE (%)"] + [f1(x) for x in fq["S"]], ["IEX price MAPE (%)"] + [f1(x) for x in fq["price"]]]))
    o.append(tbl(["Wait predictor (held-out world)", "MAE (min)", "R2"], [[k, f2(v[0]), f2(v[1])] for k, v in sg.items()]))
    o.append("\n### 9.2 Realizations and posterior\n")
    e = ens; o.append(f"{len(e['all_J'])} realizations drawn; {e['nd']} were Pareto non-dominated on the three misfits (demand, renewables+price, battery ageing); {len(e['P'])} kept as representatives.\n")
    o.append(tbl(["Rep", "Members", "kappa", "cloud loss", "wind lull", "price shock", "demand x", "ageing x", "P_omega"],
                 [[i + 1, e["members"][i]] + [f2(e["reps"][i][k]) for k in ("kappa", "cloud", "lull", "pshock", "dmult", "age")] + [f"{e['P'][i]:.3f}"] for i in range(len(e["P"]))]))
    o.append("\n![Posterior](posterior.png)\n\n*" + R["captions"]["posterior.png"] + "*\n")
    o.append("### 9.3 Pareto front\n")
    o.append(f"{len(par)} policies evaluated; {R['n_nd']} non-dominated. The operating point is the default policy (unserved penalty Rs 17/kWh = tariff, wear weight 1.0).\n\n![Pareto](pareto.png)\n\n*"
             + R["captions"]["pareto.png"] + "*\n")
    o.append("### 9.4 Targets under uncertainty (full framework vs baselines)\n")
    o.append("P90 is the conservative case (exceeded with 90% probability), P10 the optimistic one, weighted by P_omega.\n")
    rows = []
    for k, n, tg in (("reliability", "Reliability (%)", ">= 89.7"), ("losses", "Energy losses (%)", "<= 8"), ("ebitda_worst", "EBITDA margin, worst city (%)", ">= 12"),
                     ("lost_mwh", "Lost demand (MWh / 72 h)", "n/a"), ("curt", "Contracted RE unused (%)", "n/a")):
        rows.append([n, f1(rb[k]["P50"]), f1(rf[k]["P90"]), f1(rf[k]["P50"]), f1(rf[k]["P10"]), tg])
    o.append(tbl(["Metric", "Reactive baseline (P50)", "Framework P90", "Framework P50", "Framework P10", "Target"], rows))
    rec = rb["lost_mwh"]["P50"] - rf["lost_mwh"]["P50"]; ok = lambda c: "PASS" if c else "FAIL"
    o.append(f"\nRecovered lost demand (reactive minus framework, P50): **{rec:.1f} MWh per 72 h**.\n")
    o.append("Decision rule check: reliability P50 >= 89.7: **" + ok(rf["reliability"]["P50"] >= 89.7) + "**; losses P50 <= 8: **" + ok(rf["losses"]["P50"] <= 8)
             + "**; worst-city EBITDA P90 >= 12: **" + ok(rf["ebitda_worst"]["P90"] >= 12) + "**. "
             + ("Not every target is met, so the trade-off is shown on the Pareto front: extra storage cycling lifts reliability and margin but adds round-trip losses." if rf["losses"]["P50"] > 8 else "All targets met.") + "\n")
    o.append("![Risk](risk.png)\n\n*" + R["captions"]["risk.png"] + "*\n")
    o.append("### 9.5 Operations (P50 across realizations)\n")
    g = lambda s, k: f1(risk[s][k]["P50"])
    o.append(tbl(["Metric", "Reactive", "Routing only", "Full framework"], [
        ["Average wait per kWh arriving (min)"] + [g(s, "wait") for s in risk], ["Lost demand (MWh / 72 h)"] + [g(s, "lost_mwh") for s in risk],
        ["Sessions redirected (%)"] + [g(s, "redirect") for s in risk], ["Battery wear (Rs lakh / 72 h)"] + [g(s, "wear_lakh") for s in risk],
        ["Renewables in supply (%)"] + [g(s, "re_share") for s in risk], ["Grid in supply (%)"] + [g(s, "grid_share") for s in risk],
        ["Battery FEC per pack-year"] + [g(s, "fec_yr") for s in risk], ["Energy + wear cost (Rs/kWh)"] + [f2(risk[s]["energy_cost"]["P50"]) for s in risk]]))
    o.append("\nTariff tier mix of the framework (0.8 / 0.9 / 1.0 / 1.1 / 1.2): " + " / ".join(f1(x) for x in nom["framework"]["tiers"]) + " %.\n")
    o.append("![Timeline](timeline.png)\n\n*" + R["captions"]["timeline.png"].format(metro=METROS[HYD]) + "*\n")
    o.append("### 9.6 Economics (per 72-hour window)\n")
    o.append(tbl(["EBITDA (Rs lakh / 72 h)", "P90", "P50", "P10"], [[LABEL[s]] + [f1(risk[s]["ebitda_lakh"][p]) for p in ("P90", "P50", "P10")] for s in risk]))
    d = {p: rf["ebitda_lakh"][p] - rb["ebitda_lakh"]["P50"] for p in ("P90", "P50", "P10")}
    o.append(f"\nFramework uplift over the reactive baseline P50: P90 {d['P90']:.1f}, P50 {d['P50']:.1f}, P10 {d['P10']:.1f} Rs lakh.\n")
    o.append("### 9.7 Ablations and scalability (nominal realization)\n")
    o.append(tbl(["Variant", "Reliability (%)", "Losses (%)", "EBITDA (%)", "Wear (Rs lakh)"], [[n, f1(k["reliability"]), f2(k["losses"]), f1(k["ebitda"]), f2(k["wear_lakh"])] for n, k in ab.items()]))
    o.append(f"\nRuntime: one 72 h framework run over {M} cities = {tm['frame_run']:.1f} s ({tm['frame_run']/(T*M)*1e3:.1f} ms per city-hour incl. LP); "
             f"{len(e['P'])} realizations = {tm['risk']:.0f} s. Cost grows linearly with cities (independent LPs) and with N_R.\n")
    return "\n".join(o)
