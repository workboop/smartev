"""Streamlit dashboard.  Run:  streamlit run app.py"""
import os, re, tempfile
import numpy as np, streamlit as st, streamlit.components.v1 as components
import sim, viz, anim
from sim import STRATS, LABEL, SHORT, METROS, T, EVENTS, make_world, simulate, kpis, train_sage

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEED = 7
st.set_page_config(page_title="EV network energy allocation", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1250px}
div[data-testid="stMetric"]{background:#161b22;border:1px solid #2a2f3a;border-radius:10px;padding:8px 14px}
section[data-testid="stSidebar"] h3{margin:.2rem 0 .1rem;font-size:.95rem;letter-spacing:.02em;color:#9aa4b2}
.key{display:flex;align-items:center;gap:8px;font-size:.85rem;color:#c9d1d9;margin:2px 0}
.key i{width:10px;height:10px;border-radius:50%;display:inline-block}
</style>""", unsafe_allow_html=True)
st.title("Forecast-driven energy allocation for a 3-city EV charging and swapping network")


@st.cache_resource
def get_sage(): return train_sage()


@st.cache_data(show_spinner="Simulating 3 strategies...")
def live(kappa, dmult, ev, comply, tiers, band, bat):
    sg, _ = get_sage(); w = make_world(dict(kappa=kappa, dmult=dmult), SEED, ev=ev); res = {}
    for s, c in STRATS.items():
        c = dict(c, comply=comply, band=band, bat=bat); c["tiers"] = c["tiers"] and tiers; res[s] = simulate(w, c, sage=sg)
    return w, res, {s: kpis(r) for s, r in res.items()}, anim.pack(res, w, ev)


with st.sidebar:
    st.markdown("### Scenario")
    kappa = st.slider("Heat sensitivity of demand (per degC above 32)", .005, .03, .014, .001)
    dmult = st.slider("Demand level (x)", .8, 1.2, 1.0, .05, format="%.2f")
    st.markdown("### Stress events")
    ev = tuple(int(st.checkbox(f"{i}  {n}", True)) for i, n, _, _ in EVENTS)
    st.markdown("### Operating policy")
    comply = st.slider("Retail drivers who follow a redirect", 0., 1., .35, .05, format="%.2f")
    tiers = st.checkbox("Allow price tiers", True)
    band = st.slider("Price band around the base tariff (+/-)", .10, .30, .20, .05, format="%.2f", disabled=not tiers)
    bat = st.slider("Battery size (x)", .5, 2.0, 1.0, .25, format="%.2f")
    st.markdown("### Strategies")
    for s in STRATS: st.markdown(f'<div class="key"><i style="background:{anim.COL[s]}"></i>{LABEL[s]}</div>', unsafe_allow_html=True)

w, res, K, payload = live(kappa, dmult, ev, comply, tiers, band, bat)
tab = st.tabs(["Overview", "Network", "Procurement", "Results"])

with tab[0]:
    cols = st.columns(3)
    for c, s in zip(cols, STRATS):
        with c:
            st.markdown(f"**{LABEL[s]}**")
            st.metric("Reliability", f"{K[s]['reliability']:.1f}%", f"{K[s]['reliability'] - K['reactive']['reliability']:+.1f} pts vs reactive" if s != "reactive" else None)
            a, b = st.columns(2); a.metric("EBITDA margin", f"{K[s]['ebitda']:.1f}%"); b.metric("Energy losses", f"{K[s]['losses']:.2f}%")
            a, b = st.columns(2); a.metric("Average wait", f"{K[s]['wait']:.1f} min"); b.metric("Renewables in supply", f"{K[s]['re_share']:.0f}%")
    metro = st.radio("City", METROS, index=sim.HYD, horizontal=True); m = METROS.index(metro)
    p = os.path.join(tempfile.gettempdir(), f"tl_{m}.png"); viz.fig_timeline(res, w, m, p)
    st.image(p, caption=viz.CAPTIONS["timeline.png"].format(metro=metro))

with tab[1]:
    components.html(anim.network_html(payload), height=830, scrolling=False)

with tab[2]:
    components.html(anim.procure_html(payload), height=830, scrolling=False)

with tab[3]:
    md = f"{OUT}/results_section.md"
    if not os.path.exists(md): st.info("Run `python run_all.py` to generate the results.")
    else:
        txt = open(md).read(); buf = []
        for line in txt.splitlines():
            mm = re.match(r"!\[.*\]\((.*)\)", line)
            if mm: st.markdown("\n".join(buf)); buf = []; st.image(f"{OUT}/{mm.group(1)}")
            else: buf.append(line)
        st.markdown("\n".join(buf)); st.download_button("Download results_section.md", txt, "results_section.md")
        if st.button("Recompute everything (about 15 s)"):
            import run_all
            with st.status("Running pipeline...") as s: run_all.run_pipeline(log=st.write); s.update(label="Done", state="complete")
            st.rerun()
