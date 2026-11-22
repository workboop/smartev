"""Core engine: 3-city EV charging and swapping network (2 + 2 + 3 stations), synthetic data, queues, GraphSAGE-lite,
24 h storage LP, realizations + history matching + Bayesian posterior.  NumPy/SciPy only."""
import numpy as np
from scipy.optimize import linprog

METROS = ["Bengaluru", "Chennai", "Hyderabad"]
HUBS = [2, 2, 3]                                                     # stations per city
M, NH, T, HZ = len(METROS), max(HUBS), 72, 24                        # cities, max stations, sim hours, look-ahead hours
HM = np.arange(NH)[None, :] < np.array(HUBS)[:, None]                # (M, NH) True where a station exists
HS = np.array(HUBS) / 4.                                             # energy-asset scale per city (4 stations = 1.0)
LAYOUT = {2: [(.25, .5), (.75, .5)], 3: [(.2, .3), (.8, .3), (.5, .8)]}
HUB_XY = np.full((M, NH, 2), np.nan)
for _m, _n in enumerate(HUBS): HUB_XY[_m, :_n] = LAYOUT[_n]
DRIVE = np.nan_to_num(np.linalg.norm(HUB_XY[:, :, None] - HUB_XY[:, None], axis=-1)) * 13.0     # drive minutes between stations
ADJ = ((DRIVE > 0) & (DRIVE <= 8.0)).astype(float)                   # redirect only if detour <= 8 min
ADJN = ADJ / np.maximum(ADJ.sum(2, keepdims=True), 1)
BANDOFF = np.array([-1, -.5, 0, .5, 1.])                             # five price tiers: x0.8 ... x1.2
SC = np.array([1.0, .90, .95])                                       # city size
SF = np.array([1.10, 1.0, 1.05])                                     # solar quality
PF = np.array([.95, 1.0, 1.0])                                       # price level
TB = np.array([28., 32, 33])                                         # base temperature
HOT = np.array([0., 1, 1])                                           # cities hit by the heat wave
_HF = {2: [1.5, .65], 3: [1.5, .7, .6]}                              # station 1 is the hot-spot
HF = np.zeros((M, NH))
for _m, _n in enumerate(HUBS): HF[_m, :_n] = _HF[_n]
HYD = METROS.index("Hyderabad")
CAP, BASE, PCONN, PMAX, EMAX, SOLC, WINDC = 1150., 1250., 3000., 1500., 6000., 4200., 700.
OPEX = 30300.                                                        # Rs per city-hour fixed cost at 4 stations
P = dict(eta_conv=.95, eta_c=.95, eta_d=.95, sig=.001 / 24, dist=.015, oa_loss=.03, cdeg=2.25, tariff=17., pi_oa=3.9,
         eps=-.34, comply=.35, fleet=.30, patience=.5, spill=.10)
KAPPA = P["eta_conv"] * (1 - P["dist"])
THETA0 = dict(kappa=.014, cloud=.55, lull=.6, pshock=.25, dmult=1., age=1.)
EVENTS = [("E1", "Heat wave + cloudy day", 31, 47), ("E2", "Price spike + wind lull", 52, 62), ("E3", "Hyderabad station 1 outage", 58, 64)]
CFG = dict(route=False, tiers=False, mpc=False, cdeg_lp=True, cu=17., dm=1., band=.2, comply=.35, bat=1.)
STRATS = {"reactive": dict(CFG), "routing": dict(CFG, route=True), "framework": dict(CFG, route=True, tiers=True, mpc=True)}
LABEL = {"reactive": "Reactive (uninformed drivers, no look-ahead)", "routing": "Informed routing only (no look-ahead)",
         "framework": "Full framework (routing + tiers + 24h MPC)"}
SHORT = {"reactive": "Reactive", "routing": "Routing only", "framework": "Full framework"}
g_ = lambda h, m, s: np.exp(-((h - m) / s) ** 2)


def ar(rng, n, a, s, shape=()):
    e = rng.normal(0, s, (n,) + shape); x = np.zeros_like(e)
    for i in range(1, n): x[i] = a * x[i - 1] + e[i]
    return x


# ------------------------------------------------------------------ synthetic world
def make_world(th=None, seed=0, ev=(1, 1, 1)):
    th = {**THETA0, **(th or {})}; rng = np.random.default_rng(seed); n = T + HZ; t = np.arange(n); hh = t % 24
    prof = .35 + .5 * g_(hh, 9, 2.2) + .9 * g_(hh, 19.5, 2.2) + .35 * g_(hh, 14, 3)
    E1, E2, E3 = [(t >= a) & (t < b) & bool(f) for (_, _, a, b), f in zip(EVENTS, ev)]
    temp = TB + 5 * np.sin(2 * np.pi * (hh - 9) / 24)[:, None] + 7 * E1[:, None] * HOT + ar(rng, n, .9, .5, (M,))
    heat = 1 + th["kappa"] * np.maximum(temp - 32, 0)
    D = (BASE * SC[:, None] * HF)[None] * prof[:, None, None] * heat[:, :, None] * (1 + .04 * ar(rng, n, .6, 1, (M, NH))) * th["dmult"]
    clear = np.where((hh >= 6) & (hh <= 18), np.sin(np.clip(np.pi * (hh - 6) / 12, 0, np.pi)) ** 1.2, 0.)
    sol = SOLC * SC * SF * HS * clear[:, None] * np.clip(.95 + ar(rng, n, .6, .04, (M,)), .7, 1) * (1 - th["cloud"] * E1[:, None])
    wnd = WINDC * SC * HS * np.clip(.35 + .1 * np.cos(2 * np.pi * (hh - 3) / 24)[:, None] + ar(rng, n, .85, .07, (M,)), .03, .95) * (1 - th["lull"] * E2[:, None])
    base = 4 + 5 * g_(hh, 20.5, 2) + 1.5 * g_(hh, 8, 1.8) - 1.8 * g_(hh, 13, 2.5)
    price = np.clip(base[:, None] * PF * (1 + .12 * ar(rng, n, .7, 1, (M,))) * (1 + th["pshock"] * E2[:, None]), 1.5, 10.)
    cap = np.tile((CAP * SC)[None, :, None], (n, 1, NH)); cap[E3, HYD, 0] *= .35; cap[:, ~HM] = 1e-6
    return dict(D=D, S=sol + wnd, sol=sol, wnd=wnd, price=price, cap=cap, temp=temp, th=th)


def make_forecast(w, seed, sd=.08, sp=.10, so=.15):
    rng = np.random.default_rng(seed); n = len(w["price"])
    return dict(D=w["D"] * (1 + sd * ar(rng, n, .8, 1, (M, 1))), S=np.clip(w["S"] * (1 + so * ar(rng, n, .8, 1, (M,))), 0, None),
                price=np.clip(w["price"] * (1 + sp * ar(rng, n, .8, 1, (M,))), 1.5, 10.))


# ------------------------------------------------------------------ GraphSAGE-lite (mean aggregator + ridge readout)
def feats(Q, dem, cap, hh):
    a = 2 * np.pi * hh / 24
    return np.stack([Q / cap, dem / cap, cap / CAP, np.full(Q.shape, np.sin(a)), np.full(Q.shape, np.cos(a))], -1)


class Sage:
    """h_j = ReLU(W [x_j || mean_{k in N(j)} x_k]); readout fit by ridge. Predicts next-hour wait (min) per station."""
    def __init__(s, hid=48, seed=0, nb=True):
        s.nb = nb; s.W = np.random.default_rng(seed).normal(0, 1, (10, hid)) / np.sqrt(10)
    def _z(s, X):
        Z = np.concatenate([X, np.einsum('mij,...mjf->...mif', ADJN, X) if s.nb else 0 * X], -1)
        return np.concatenate([np.maximum(0, Z @ s.W), Z, np.ones(Z.shape[:-1] + (1,))], -1)
    def fit(s, X, y, lam=1e-2):
        mk = np.broadcast_to(HM, X.shape[:-1]).reshape(-1); Z = s._z(X).reshape(-1, s.W.shape[1] + 11)[mk]
        s.beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y.reshape(-1)[mk]); return s
    def predict(s, X): return np.maximum(0, s._z(X) @ s.beta) * HM


def train_sage():
    data = []
    rng = np.random.default_rng(5)
    for sd in range(300, 304):
        th = dict(kappa=rng.uniform(.008, .02), dmult=rng.uniform(.95, 1.1), cloud=rng.uniform(.3, .7))
        simulate(make_world(th, sd), STRATS["reactive"], collect=data)
    X = np.array([d[0] for d in data]); y = np.array([d[1] for d in data]); tr, te = slice(0, 3 * T), slice(3 * T, None)
    sg = Sage().fit(X[tr], y[tr]); own = Sage(nb=False).fit(X[tr], y[tr]); yt = y[te]; mk = np.broadcast_to(HM, yt.shape)
    mae = lambda p: float(np.abs(p - yt)[mk].mean()); r2 = lambda p: float(1 - ((p - yt)[mk] ** 2).sum() / ((yt[mk] - yt[mk].mean()) ** 2).sum())
    pers = X[te][..., 0] * 60
    met = {"GraphSAGE-lite (own + neighbours)": (mae(sg.predict(X[te])), r2(sg.predict(X[te]))),
           "Own features only (no graph)": (mae(own.predict(X[te])), r2(own.predict(X[te]))),
           "Persistence (current queue)": (mae(pers), r2(pers))}
    return sg, met


# ------------------------------------------------------------------ energy: LP planner + dispatch
_A = {}
def _struct(n):
    if n not in _A:
        A = np.zeros((2 * n, 6 * n))                        # x = [g, oa, pc, pd, u, E]
        for h in range(n):
            A[h, 5 * n + h] = 1; A[h, 2 * n + h] = -P["eta_c"]; A[h, 3 * n + h] = 1 / P["eta_d"]
            if h: A[h, 5 * n + h - 1] = -(1 - P["sig"])
            r = n + h; A[r, [h, n + h, 3 * n + h]] = 1; A[r, 2 * n + h] = -1; A[r, 4 * n + h] = 1 / KAPPA
        _A[n] = A
    return _A[n]


def lp_plan(price, Sdel, qhat, E0, emax, pmax, gmax, cu, cd):
    n = len(price); c = np.r_[price, np.zeros(n), np.zeros(n), np.full(n, cd), np.full(n, cu), np.zeros(n)]
    c[5 * n + n - 1] = -price.mean() * P["eta_d"]            # terminal value of stored energy
    b = np.r_[np.zeros(n), qhat / KAPPA]; b[0] = (1 - P["sig"]) * E0
    lo = np.r_[np.zeros(5 * n), np.full(n, .05 * emax)]      # 5% reserve
    hi = np.r_[np.full(n, gmax), Sdel, np.full(n, pmax), np.full(n, pmax), qhat, np.full(n, emax)]
    r = linprog(c, A_eq=_struct(n), b_eq=b, bounds=np.c_[lo, hi], method="highs")
    return (r.x[2 * n], r.x[3 * n]) if r.status == 0 else (0., 0.)


def dispatch(B, oa, pd, pc, E, gmax, emax, pmax):
    pd = min(pd, (1 - P["sig"]) * E * P["eta_d"]); L = B + pc
    oau = min(oa, L); rem = L - oau; pdu = min(pd, rem); rem -= pdu; g = min(rem, gmax); rem -= g
    cut = min(pc, rem); pc -= cut; rem -= cut; f = max(0., B - rem) / B if B > 0 else 1.
    E2 = (1 - P["sig"]) * E + P["eta_c"] * pc - pdu / P["eta_d"]
    return g, oau, pdu, pc, f, min(max(E2, 0), emax)


def react_plan(B, oa, gmax, E, emax, pmax):
    rem = B - min(oa, B); sh = max(0., rem - gmax); pd = min(sh, pmax)
    pc = min(max(0., oa - B), pmax, (emax - E) / P["eta_c"]); return pc, pd


def route(A, Q, cap, Wh, comply):
    A = A.copy(); flow = np.zeros((M, NH, NH)); mv = A * (P["fleet"] + (1 - P["fleet"]) * comply)
    for m in range(M):
        ov = np.maximum(0, Q[m] + A[m] - cap[m] * 1.1)
        for i in np.argsort(-ov):
            if ov[i] <= 1: break
            nb = np.where(ADJ[m, i] > 0)[0]
            for j in nb[np.argsort(Wh[m, nb] + DRIVE[m, i, nb])]:
                if Wh[m, i] <= Wh[m, j] + DRIVE[m, i, j]: continue
                x = min(ov[i], mv[m, i] - flow[m, i].sum(), max(0, cap[m, j] - Q[m, j] - A[m, j]))
                if x > 0: A[m, i] -= x; A[m, j] += x; flow[m, i, j] += x; ov[i] -= x
    return A, flow


# ------------------------------------------------------------------ simulation
def simulate(w, cfg=None, sage=None, fseed=0, collect=None):
    c = {**CFG, **(cfg or {})}; F = make_forecast(w, 1000 + fseed); Q = np.zeros((M, NH))
    Em, Pm, Gm = EMAX * SC * HS * c["bat"], PMAX * SC * HS, PCONN * SC * HS; E = .3 * Em
    keys = "served arr lost Qn wait flow rev enrg wear g oa curt pc pd E sched f tier inp".split()
    R = {k: [] for k in keys}
    for k in range(T):
        hh = k % 24; cap = w["cap"][k]
        pw = F["price"][k:k + HZ].copy(); pw[0] = w["price"][k]
        rk = (pw[None] <= pw[:, None]).mean(1); ti = np.minimum(((rk - 1e-9) * 5).astype(int), 4) if c["tiers"] else np.full((HZ, M), 2)
        mult = 1 + c["band"] * BANDOFF[ti]
        dem = w["D"][k] * (mult[0] ** P["eps"])[:, None]; X = feats(Q, dem, cap, hh)
        Wh = sage.predict(X) if sage else Q / cap * 60
        flow = np.zeros((M, NH, NH))
        if c["route"]: A, flow = route(dem, Q, cap, Wh, c["comply"])
        else: ov = np.maximum(0, Q + dem - cap) * P["spill"]; A = dem - ov + np.einsum('mi,mij->mj', ov, ADJN)
        pot = np.minimum(cap, Q + A); Bm = pot.sum(1) / KAPPA; oad = w["S"][k] * (1 - P["oa_loss"])
        f = np.ones(M); gg = np.zeros(M); oau = np.zeros(M); pcs = np.zeros(M); pds = np.zeros(M); En = E.copy()
        for m in range(M):
            if c["mpc"]:
                idx = slice(k, k + HZ); Dh = F["D"][idx, m] * (mult[:, m] ** P["eps"])[:, None]; Dh[0] = dem[m] + 0 * Dh[0]
                capw = w["cap"][idx, m]; Dq = Dh.copy(); Dq[0] += Q[m]
                q = np.minimum(Dq.sum(1), .97 * capw.sum(1)) if c["route"] else np.minimum(Dq, capw).sum(1)
                Sd = F["S"][idx, m] * (1 - P["oa_loss"]); Sd[0] = oad[m]
                pc, pd = lp_plan(pw[:, m], Sd, q, E[m], Em[m], Pm[m], Gm[m], c["cu"], P["cdeg"] * c["dm"] if c["cdeg_lp"] else 0.)
            else: pc, pd = react_plan(Bm[m], oad[m], Gm[m], E[m], Em[m], Pm[m])
            gg[m], oau[m], pds[m], pcs[m], f[m], En[m] = dispatch(Bm[m], oad[m], pd, pc, E[m], Gm[m], Em[m], Pm[m])
        served = pot * f[:, None]; Qn = Q + A - served; lost = np.maximum(0, Qn - P["patience"] * cap); Qn -= lost
        if collect is not None: collect.append((X, Qn / cap * 60))
        R["served"].append(served); R["arr"].append(dem); R["lost"].append(lost); R["Qn"].append(Qn.copy()); R["wait"].append(Qn / cap * 60)
        R["flow"].append(flow); R["rev"].append(P["tariff"] * mult[0] * served.sum(1)); R["enrg"].append(w["price"][k] * gg + P["pi_oa"] * w["S"][k])
        R["wear"].append(P["cdeg"] * pds); R["g"].append(gg); R["oa"].append(oau); R["curt"].append(oad - oau); R["pc"].append(pcs); R["pd"].append(pds)
        R["E"].append(En.copy()); R["sched"].append(w["S"][k] * (1 - P["oa_loss"])); R["f"].append(f); R["tier"].append(ti[0]); R["inp"].append(gg + oau / (1 - P["oa_loss"]))
        Q, E = Qn, En
    out = {k: np.array(v) for k, v in R.items()}; out["E0"] = .3 * Em; out["Em"] = Em; out["cfg"] = c; return out


def kpis(r):
    sv, ar_ = r["served"].sum(), r["arr"].sum(); ebit = r["rev"] - r["enrg"] - r["wear"] - OPEX * SC * HS
    stored = ((r["E"][-1] - r["E0"]) * P["eta_d"]).sum(); wsum = (r["wait"] * r["arr"]).sum() / ar_
    supply = r["g"].sum() + r["oa"].sum() + r["pd"].sum()
    return dict(reliability=100 * sv / ar_, losses=100 * (1 - (sv + stored) / r["inp"].sum()),
                ebitda=100 * ebit.sum() / r["rev"].sum(), ebitda_worst=100 * (ebit.sum(0) / r["rev"].sum(0)).min(),
                lost_mwh=r["lost"].sum() / 1e3, served_mwh=sv / 1e3, wait=wsum, curt=100 * r["curt"].sum() / r["sched"].sum(),
                wear_lakh=r["wear"].sum() / 1e5, fec_yr=(r["pd"].sum(0) / r["Em"]).mean() * 8760 / T, ebitda_lakh=ebit.sum() / 1e5,
                redirect=100 * r["flow"].sum() / ar_, energy_cost=(r["enrg"].sum() + r["wear"].sum()) / sv,
                re_share=100 * r["oa"].sum() / supply, grid_share=100 * r["g"].sum() / supply,
                tiers=np.bincount(r["tier"].ravel(), minlength=5) / r["tier"].size * 100)


# ------------------------------------------------------------------ realizations, history matching, posterior
def weighted_q(v, wt, q):
    o = np.argsort(v); cw = np.cumsum(wt[o]) / wt.sum(); return float(np.asarray(v)[o][np.searchsorted(cw, q).clip(0, len(v) - 1)])


def ensemble(n_t=200, n_r=8, seed=1):
    rng = np.random.default_rng(seed); truth = dict(kappa=.014, cloud=.55, lull=.6, pshock=.25, dmult=1., age=1.)
    stat = lambda th: (lambda w: (np.log(w["D"][:T].sum()), w["S"][:T].sum(), w["price"][:T].mean()))(make_world(th, 999))
    obs = np.array(stat(truth)) * (1 + .01 * rng.normal(size=3)); age_obs = 1.0 + .03 * rng.normal()
    TH = np.c_[rng.uniform(.005, .025, n_t), rng.uniform(.2, .85, n_t), rng.uniform(.2, .9, n_t), rng.uniform(0, .5, n_t),
               rng.uniform(.9, 1.15, n_t), rng.uniform(.6, 1.5, n_t)]
    keys = list(THETA0); J = np.zeros((n_t, 3))
    for s in range(n_t):
        d, re, pr = stat(dict(zip(keys, TH[s])))
        J[s] = [.5 * ((d - obs[0]) / .02) ** 2, .5 * (((re - obs[1]) / obs[1] / .04) ** 2 + ((pr - obs[2]) / obs[2] / .04) ** 2), .5 * ((TH[s, 5] - age_obs) / .12) ** 2]
    Jt = J.sum(1); post = np.exp(-(Jt - Jt.min())); post /= post.sum()
    nd = [i for i in range(n_t) if not any((J[j] <= J[i]).all() and (J[j] < J[i]).any() for j in range(n_t))]
    order = list(np.argsort(Jt)); reps = [i for i in order if i in nd][:n_r]; reps += [i for i in order if i not in reps][:n_r - len(reps)]
    Z = (TH - TH.mean(0)) / TH.std(0); lab = np.argmin(((Z[:, None] - Z[reps][None]) ** 2).sum(2), 1)
    Pw = np.array([post[lab == o].sum() for o in range(len(reps))]); Pw /= Pw.sum()
    return dict(reps=[dict(zip(keys, TH[i])) for i in reps], P=Pw, members=[int((lab == o).sum()) for o in range(len(reps))], J=J[reps],
                all_J=Jt, all_TH=TH, post=post, reps_idx=reps, nd=len(nd), truth=truth)
