"""Dark-theme static figures. Every figure has a short, plain-language caption (see CAPTIONS)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import sim
from sim import METROS, EVENTS, T, LABEL

BG, FG, GRID = "#0e1117", "#e6e6e6", "#2a2f3a"
COL = {"reactive": "#ff6b6b", "routing": "#ffb84d", "framework": "#4dd0e1"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 9, "legend.facecolor": "#161b22",
                     "legend.edgecolor": GRID})
EVC = ["#ffb84d", "#8e9aaf", "#ff6b9d"]

CAPTIONS = {
    "pareto.png": "Each dot is one operating policy of the full framework. Right means more demand served, up means higher margin, colour shows energy losses. "
                  "Ringed dots are Pareto-optimal; stars mark the three strategies.",
    "timeline.png": "{metro} over 72 hours. Top: worst-station wait. Middle: battery charge (lines) against the IEX price (purple); the framework fills up when power is cheap "
                    "and discharges into the evening peak. Bottom: cumulative demand lost to queue abandonment. Bands mark the stress events.",
    "risk.png": "Bars show the median across history-matched futures, whiskers span the optimistic (P10) to conservative (P90) case, the dashed line is the target.",
    "posterior.png": "200 candidate futures scored against history. Bigger dots carry more posterior weight; ringed dots are the representatives the optimizer plans against.",
}


def shade(ax, lab=False):
    for (n, _, a, b), c in zip(EVENTS, EVC):
        ax.axvspan(a, b, color=c, alpha=.13, lw=0)
        if lab: ax.text((a + b) / 2, ax.get_ylim()[1], n, ha="center", va="top", fontsize=8, color=c, weight="bold")


def save(fig, path): fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_pareto(par, strat, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); P = np.array([[p["reliability"], p["ebitda"], p["losses"]] for p in par])
    nd = [i for i in range(len(P)) if not any((P[j, 0] >= P[i, 0] and P[j, 1] >= P[i, 1] and P[j, 2] <= P[i, 2]) and (P[j] != P[i]).any() for j in range(len(P)))]
    sc = ax.scatter(P[:, 0], P[:, 1], c=P[:, 2], cmap="viridis_r", s=55, zorder=3); ax.scatter(P[nd, 0], P[nd, 1], s=130, fc="none", ec=FG, lw=1.2, zorder=4)
    for s, k in strat.items(): ax.scatter(k["reliability"], k["ebitda"], marker="*", s=260, c=COL[s], ec=BG, zorder=5)
    h = [Line2D([], [], marker="*", ls="", ms=13, c=COL[s], label=sim.SHORT[s]) for s in strat] + [Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=FG, label="Pareto-optimal policy")]
    ax.legend(handles=h, fontsize=8, loc="lower right"); ax.set_xlabel("Reliability  (% of arriving demand served)"); ax.set_ylabel("EBITDA margin (%)"); ax.grid(alpha=.4)
    fig.colorbar(sc, label="Energy losses (%)"); ax.set_title("Reliability vs margin vs losses"); save(fig, path); return nd


def fig_timeline(res, w, m, path):
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True); t = np.arange(T)
    for s, r in res.items():
        ax[0].plot(t, r["wait"][:, m].max(1), c=COL[s], label=sim.SHORT[s], lw=1.6); ax[2].plot(t, np.cumsum(r["lost"][:, m].sum(1)) / 1e3, c=COL[s], lw=1.6)
    for s in ("reactive", "framework"): ax[1].plot(t, res[s]["E"][:, m] / res[s]["Em"][m] * 100, c=COL[s], lw=1.6, label=f"Battery charge, {sim.SHORT[s].lower()}")
    a2 = ax[1].twinx(); a2.fill_between(t, w["price"][:T, m], color="#c792ea", alpha=.18, label="IEX price"); a2.set_ylabel("IEX price (Rs/kWh)", color="#c792ea"); a2.set_ylim(0, 25)
    ax[0].set_ylabel("Worst-station wait (min)"); ax[1].set_ylabel("Battery charge (%)"); ax[2].set_ylabel("Cumulative lost (MWh)"); ax[2].set_xlabel("Hour")
    ax[0].legend(fontsize=8, loc="upper left"); h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    for a in ax: a.grid(alpha=.3); a.set_xlim(0, T - 1)
    for a in ax: shade(a, a is ax[0])
    ax[0].set_title(f"{METROS[m]}: queues, battery and lost demand"); save(fig, path)


def fig_risk(risk, targets, path):
    names = [("reliability", "Reliability (%)"), ("losses", "Energy losses (%)"), ("ebitda_worst", "EBITDA margin, worst city (%)"), ("wait", "Average wait (min)")]
    fig, ax = plt.subplots(1, 4, figsize=(13, 4.1))
    for a, (k, ttl) in zip(ax, names):
        for i, s in enumerate(risk):
            p10, p50, p90 = risk[s][k]["P10"], risk[s][k]["P50"], risk[s][k]["P90"]; lo, hi = min(p10, p90), max(p10, p90)
            a.bar(i, p50, color=COL[s], alpha=.85); a.errorbar(i, p50, yerr=[[p50 - lo], [hi - p50]], c=FG, capsize=4, lw=1.2)
        if k in targets: a.axhline(targets[k], c=FG, ls="--", lw=1)
        a.set_xticks(range(len(risk))); a.set_xticklabels(["Reactive", "Routing", "Framework"], fontsize=8); a.set_title(ttl, fontsize=9); a.grid(axis="y", alpha=.3)
    fig.legend(handles=[Patch(fc="#8e9aaf", label="Median (P50)"), Line2D([], [], c=FG, marker="_", ms=10, label="P10 to P90 range"), Line2D([], [], c=FG, ls="--", label="Target")],
               loc="lower center", ncol=3, fontsize=8, frameon=False); fig.tight_layout(rect=(0, .07, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


def fig_post(ens, path):
    fig, ax = plt.subplots(figsize=(7, 4.5)); TH, post = ens["all_TH"], ens["post"]
    sc = ax.scatter(TH[:, 0], TH[:, 1], s=10 + 2500 * post, c=np.log10(ens["all_J"] + 1), cmap="magma_r", alpha=.8)
    r = ens["reps_idx"]; ax.scatter(TH[r, 0], TH[r, 1], s=170, fc="none", ec=FG, lw=1.4); ax.scatter(*[ens["truth"][k] for k in ("kappa", "cloud")], marker="x", c="#4dd0e1", s=90)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=FG, label="Representative used by the optimizer"), Line2D([], [], marker="x", ls="", ms=9, c="#4dd0e1", label="Hidden truth")], fontsize=8, loc="upper right")
    ax.set_xlabel("Heat sensitivity of demand (per degC)"); ax.set_ylabel("Cloud loss of solar"); fig.colorbar(sc, label="log10(1 + misfit)"); ax.set_title("History matching: 200 realizations"); save(fig, path)
