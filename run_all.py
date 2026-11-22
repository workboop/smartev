"""Runs the whole pipeline and writes everything to ./outputs :  python run_all.py"""
import os, json, time, numpy as np
import sim, viz, report, anim
from sim import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
HIGH = {"reliability", "ebitda", "ebitda_worst", "ebitda_lakh", "re_share"}                  # higher is better -> P90 = 10th percentile
KEYS = ["reliability", "losses", "ebitda", "ebitda_worst", "lost_mwh", "wait", "curt", "wear_lakh", "fec_yr", "redirect", "ebitda_lakh", "energy_cost", "re_share", "grid_share"]


def run_pipeline(n_t=200, n_r=8, seed=7, out=OUT, log=print):
    os.makedirs(out, exist_ok=True); R = {"captions": viz.CAPTIONS}
    log("1/6 training GraphSAGE-lite"); sg, R["sage"] = train_sage()
    w = make_world(seed=seed); log("2/6 nominal runs + ablations")
    res = {s: simulate(w, STRATS[s], sage=sg) for s in STRATS}; R["nominal"] = {s: kpis(r) for s, r in res.items()}
    t0 = time.time(); simulate(w, STRATS["framework"], sage=sg); frame_run = time.time() - t0
    F = STRATS["framework"]; abl = {"Full framework": F, "No routing": dict(F, route=False), "No price tiers": dict(F, tiers=False),
                                    "No degradation cost in scheduler": dict(F, cdeg_lp=False), "No look-ahead (reactive storage)": dict(F, mpc=False),
                                    "Reactive baseline": STRATS["reactive"]}
    R["ablation"] = {n: kpis(simulate(w, c, sage=sg)) for n, c in abl.items()}
    fc = make_forecast(w, 1000); mape = lambda a, b: 100 * np.mean(np.abs(a - b) / np.maximum(np.abs(b), 1e-9))
    R["fq"] = dict(D=[mape(fc["D"][:T, m].sum(1), w["D"][:T, m].sum(1)) for m in range(M)],
                   S=[mape(fc["S"][:T, m][w["S"][:T, m] > 100], w["S"][:T, m][w["S"][:T, m] > 100]) for m in range(M)], price=[mape(fc["price"][:T, m], w["price"][:T, m]) for m in range(M)])
    log("3/6 Pareto sweep"); par = [kpis(simulate(w, dict(F, cu=cu, dm=dm), sage=sg)) for cu in (8, 12, 17, 40) for dm in (.5, 1, 2, 4)]; R["pareto"] = par
    R["n_nd"] = len(viz.fig_pareto(par, R["nominal"], f"{out}/pareto.png"))
    log("4/6 realizations + history matching"); ens = ensemble(n_t, n_r); R["ens"] = ens; viz.fig_post(ens, f"{out}/posterior.png")
    log("5/6 robust evaluation across representatives"); t0 = time.time(); runs = {s: [] for s in STRATS}
    for i, th in enumerate(ens["reps"]):
        wi = make_world(th, 100 + i)
        for s in STRATS: runs[s].append(kpis(simulate(wi, STRATS[s], sage=sg, fseed=i)))
    R["timing"] = dict(frame_run=frame_run, risk=time.time() - t0); wt = np.array(ens["P"])
    risk = {s: {k: {} for k in KEYS} for s in STRATS}
    for s in STRATS:
        for k in KEYS:
            v = np.array([r[k] for r in runs[s]]); qs = (.1, .5, .9) if k in HIGH else (.9, .5, .1)
            for p, q in zip(("P90", "P50", "P10"), qs): risk[s][k][p] = weighted_q(v, wt, q)
    R["risk"] = risk; viz.fig_risk(risk, report.TARGET, f"{out}/risk.png"); viz.fig_timeline(res, w, sim.HYD, f"{out}/timeline.png")
    log("6/6 exporting animation pages"); data = anim.pack(res, w)
    open(f"{out}/network_animation.html", "w").write(anim.network_html(data)); open(f"{out}/procurement_animation.html", "w").write(anim.procure_html(data))
    md = report.build_report(R); open(f"{out}/results_section.md", "w").write(md)
    ser = lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, np.floating) else str(o)
    json.dump({k: R[k] for k in ("nominal", "risk", "ablation", "sage", "timing", "n_nd")}, open(f"{out}/results.json", "w"), default=ser, indent=1)
    return R, md


if __name__ == "__main__":
    t = time.time(); R, md = run_pipeline(); print(md); print(f"\nDone in {time.time()-t:.0f}s -> {OUT}")
