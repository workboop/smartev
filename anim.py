"""Self-contained HTML animations (network scene + procurement mix). Fed with simulation arrays, no external libraries."""
import json
import numpy as np
import sim
from sim import METROS, HUBS, EVENTS, T, M, NH, STRATS, SHORT, LABEL

COL = {"reactive": "#ff6b6b", "routing": "#ffb84d", "framework": "#4dd0e1"}
EVC = ["#ffb84d", "#8e9aaf", "#ff6b9d"]


def pack(res, w, ev=(1, 1, 1)):
    r = lambda a, d=0: np.round(np.asarray(a, float), d).tolist()
    D = dict(T=T, metros=METROS, hubs=HUBS, xy=[[list(map(float, p)) for p in sim.HUB_XY[m, :HUBS[m]]] for m in range(M)],
             events=[dict(id=i, name=n, a=a, b=b, color=c) for (i, n, a, b), c, f in zip(EVENTS, EVC, ev) if f],
             band=float(next(iter(res.values()))["cfg"]["band"]), tiers=sim.BANDOFF.tolist(),
             world=dict(price=r(w["price"][:T], 2), sol=r(w["sol"][:T]), wnd=r(w["wnd"][:T]),
                        cap=r(w["cap"][:T] / (sim.CAP * sim.SC)[None, :, None], 2)), strat=[])
    for s, x in res.items():
        flow = [[[[i, j, int(round(x["flow"][k, m, i, j]))] for i in range(NH) for j in range(NH) if x["flow"][k, m, i, j] > 5] for m in range(M)] for k in range(T)]
        D["strat"].append(dict(key=s, label=SHORT[s], long=LABEL[s], color=COL[s], d=dict(
            wait=r(x["wait"], 1), q=r(x["Qn"]), lost=r(x["lost"]), arr=r(x["arr"]), sv=r(x["served"].sum(2)), flow=flow,
            E=r(x["E"] / x["Em"] * 100, 1), g=r(x["g"]), oa=r(x["oa"]), pd=r(x["pd"]), pc=r(x["pc"]), tier=x["tier"].astype(int).tolist(),
            enrg=r(x["enrg"]), f=r(x["f"], 3))))
    return D


COMMON_CSS = """
*{box-sizing:border-box}html,body{margin:0;background:#0e1117;color:#e6e6e6;font:13px/1.35 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:4px 2px 6px}
.seg{display:flex;border:1px solid #2a2f3a;border-radius:8px;overflow:hidden}
.seg button,.btn{background:#161b22;color:#c9d1d9;border:0;padding:6px 12px;cursor:pointer;font:inherit;display:flex;align-items:center;gap:7px}
.seg button+button{border-left:1px solid #2a2f3a}
.seg button.on{background:#202838;color:#fff;box-shadow:inset 0 -2px 0 var(--c,#4dd0e1)}
.btn{border:1px solid #2a2f3a;border-radius:8px;min-width:68px;justify-content:center}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
input[type=range]{flex:1;min-width:150px;accent-color:#4dd0e1}
.clock{font-variant-numeric:tabular-nums;font-weight:600;min-width:120px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.chip{border:1px solid #2a2f3a;border-radius:999px;padding:3px 10px;color:#6b7482;font-size:12px;white-space:nowrap;transition:all .25s}
.chip.on{color:#0e1117;font-weight:600}
.stats{display:flex;gap:10px;flex-wrap:wrap;padding:0 2px 6px}
.stat{background:#161b22;border:1px solid #2a2f3a;border-radius:10px;padding:6px 12px;min-width:150px}
.stat b{display:block;font-size:18px;font-variant-numeric:tabular-nums}.stat span{color:#8b95a5;font-size:11px}
.legend{display:flex;flex-wrap:wrap;gap:4px 18px;padding:6px 4px;color:#9aa4b2;font-size:12px;align-items:center}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-2px}
.ln{display:inline-block;width:20px;height:0;border-top:3px solid;margin-right:6px;vertical-align:3px}
.gr{display:inline-block;width:70px;height:10px;border-radius:5px;margin:0 6px;vertical-align:-1px;background:linear-gradient(90deg,#3ddc84,#ffd166,#ff5d5d)}
svg{width:100%;height:auto;display:block}
text{font-family:inherit}
"""

COMMON_JS = """
const D=__DATA__;const NS="http://www.w3.org/2000/svg";const $=id=>document.getElementById(id);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),lerp=(a,b,u)=>a+(b-a)*u,T=D.T,M=D.metros.length,pad=n=>String(n).padStart(2,'0');
const C={solar:'#fbbf24',wind:'#2dd4bf',grid:'#a78bfa',batt:'#60a5fa'};
function el(tag,at,par,txt){const e=document.createElementNS(NS,tag);for(const k in at)e.setAttribute(k,at[k]);if(txt!=null)e.textContent=txt;if(par)par.appendChild(e);return e}
const lv=(fn,t)=>{const i=Math.floor(t),j=Math.min(i+1,T-1);return lerp(fn(i),fn(j),t-i)};
function mixc(a,b,u){const pa=a.match(/\\w\\w/g).map(h=>parseInt(h,16)),pb=b.match(/\\w\\w/g).map(h=>parseInt(h,16));return 'rgb('+pa.map((v,i)=>Math.round(lerp(v,pb[i],u))).join(',')+')'}
function waitCol(w){const x=clamp(w/30,0,1);return x<.5?mixc('3ddc84','ffd166',x*2):mixc('ffd166','ff5d5d',(x-.5)*2)}
const clockTxt=t=>'Day '+(Math.floor(t/24)+1)+' \\u00b7 '+pad(Math.floor(t%24))+':00';
const S={t:0,play:true,speed:2,strat:D.strat.length-1,phase:0};
function segment(id,items,cur,cb){const box=$(id);box.innerHTML='';items.forEach((it,i)=>{const b=document.createElement('button');b.innerHTML=it.html;if(it.color)b.style.setProperty('--c',it.color);b.onclick=()=>{cur=i;[...box.children].forEach((c,j)=>c.classList.toggle('on',j==i));cb(i)};if(i==cur)b.classList.add('on');box.appendChild(b)})}
function wireClock(update){
  $('play').onclick=()=>{S.play=!S.play;$('play').textContent=S.play?'Pause':'Play'};
  $('scrub').max=T-1;$('scrub').oninput=e=>{S.t=+e.target.value;update()};
  segment('spd',[{html:'0.5x'},{html:'1x'},{html:'2x'},{html:'4x'}],2,i=>{S.speed=[.5,1,2,4][i]});
  D.events.forEach(ev=>{const c=document.createElement('span');c.className='chip';c.id='ev'+ev.id;c.textContent=ev.id+'  '+ev.name;c.dataset.c=ev.color;$('chips').appendChild(c)});
  let last=null;function frame(ts){if(last==null)last=ts;const dt=Math.min((ts-last)/1000,.1);last=ts;
    if(S.play){S.t+=dt*S.speed;S.phase+=dt;if(S.t>=T-1)S.t=0;$('scrub').value=S.t}update(dt);requestAnimationFrame(frame)}
  requestAnimationFrame(frame)}
function evChips(t){D.events.forEach(ev=>{const c=$('ev'+ev.id),on=t>=ev.a&&t<ev.b;c.classList.toggle('on',on);c.style.background=on?ev.color:'transparent';c.style.borderColor=on?ev.color:'#2a2f3a'})}
"""

NET_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>__CSS__</style></head><body>
<div class="bar"><div class="seg" id="strat"></div><button class="btn" id="play">Pause</button><div class="seg" id="spd"></div>
<input type="range" id="scrub" min="0" max="71" step="0.01" value="0"><span class="clock" id="clock"></span><div class="chips" id="chips"></div></div>
<div class="stats"><div class="stat"><b id="k1">-</b><span>Energy served now</span></div><div class="stat"><b id="k2">-</b><span>Average wait now</span></div>
<div class="stat"><b id="k3">-</b><span>Demand lost so far</span></div><div class="stat"><b id="k4">-</b><span>Renewable share of supply now</span></div></div>
<svg id="scene" viewBox="0 0 1000 535"></svg>
<div class="legend">
<span><i class="sw" style="background:#fbbf24"></i>Solar</span><span><i class="sw" style="background:#2dd4bf"></i>Wind</span><span><i class="sw" style="background:#a78bfa"></i>Grid (IEX)</span><span><i class="sw" style="background:#60a5fa"></i>Battery</span>
<span>Line thickness = power drawn from that source</span>
<span>Station wait<i class="gr"></i>0 to 30+ min</span><span><i class="sw" style="background:#e6e6e6;border-radius:50%;width:8px;height:8px"></i>queued energy, 1 dot = 35 kWh</span>
<span><i class="ln" style="border-color:#7dd3fc;border-top-style:dashed"></i>drivers redirected</span><span><i class="sw" style="background:#cbd5e1;border-radius:50%;width:8px;height:8px"></i>inter-city traffic</span>
<span><i class="sw" style="border:1.5px dashed #ff5d5d;background:none"></i>station outage</span></div>
<script>__COMMON__
const POS=[[170,440],[830,440],[500,325]],BUS=165,SX={solar:120,wind:300,grid:880};
const svg=$('scene');const defs=el('defs',{},svg);
const pat=el('pattern',{id:'grid',width:25,height:25,patternUnits:'userSpaceOnUse'},defs);el('path',{d:'M25 0H0V25',fill:'none',stroke:'#1b2231','stroke-width':1},pat);
const mk=el('marker',{id:'arr',viewBox:'0 0 10 10',refX:'8',refY:'5',markerUnits:'userSpaceOnUse',markerWidth:11,markerHeight:11,orient:'auto'},defs);el('path',{d:'M0,0L10,5L0,10z',fill:'#7dd3fc'},mk);
const maxSup=Math.max(1,...D.strat.flatMap(s=>s.d.g.flatMap((r,k)=>r.map((g,m)=>g+s.d.oa[k][m]+s.d.pd[k][m]))));
const wid=v=>1+7*Math.min(1,v/maxSup*1.6);
D.strat.forEach(s=>{s.d.F=s.d.flow.map(hm=>hm.map(list=>{const a=Array.from({length:3},()=>[0,0,0]);list.forEach(([i,j,v])=>a[i][j]=v);return a}))});
const maxArr=Math.max(...D.strat[0].d.arr.flatMap(r=>r.map(a=>a.reduce((x,y)=>x+y,0))));
// ---- backbone and sources
const gBack=el('g',{},svg);el('line',{x1:SX.solar,y1:BUS,x2:SX.grid,y2:BUS,stroke:'#2a2f3a','stroke-width':3,'stroke-linecap':'round'},gBack);
const stub={};['solar','wind','grid'].forEach(k=>{el('line',{x1:SX[k],y1:122,x2:SX[k],y2:BUS,stroke:'#2a2f3a','stroke-width':3},gBack);stub[k]=el('line',{x1:SX[k],y1:122,x2:SX[k],y2:BUS,stroke:C[k],'stroke-dasharray':'5 7'},gBack)});
const sun=el('g',{},svg);el('circle',{cx:SX.solar,cy:58,r:14,fill:C.solar},sun);const rays=el('g',{},sun);
for(let a=0;a<8;a++){const r=a*Math.PI/4;el('line',{x1:SX.solar+Math.cos(r)*20,y1:58+Math.sin(r)*20,x2:SX.solar+Math.cos(r)*29,y2:58+Math.sin(r)*29,stroke:C.solar,'stroke-width':2.5,'stroke-linecap':'round'},rays)}
const glow=el('circle',{cx:SX.solar,cy:58,r:30,fill:C.solar,opacity:0},sun);
const wind=el('g',{},svg);el('line',{x1:SX.wind,y1:60,x2:SX.wind,y2:112,stroke:'#7c8a9a','stroke-width':3},wind);const rot=el('g',{},wind);
for(let a=0;a<3;a++){el('line',{x1:SX.wind,y1:60,x2:SX.wind,y2:32,stroke:C.wind,'stroke-width':4,'stroke-linecap':'round',transform:'rotate('+a*120+' '+SX.wind+' 60)'},rot)}el('circle',{cx:SX.wind,cy:60,r:4,fill:'#e6e6e6'},wind);
const pyl=el('g',{stroke:C.grid,'stroke-width':2.5,fill:'none','stroke-linecap':'round'},svg);
el('path',{d:'M'+SX.grid+' 30L'+(SX.grid-18)+' 112M'+SX.grid+' 30L'+(SX.grid+18)+' 112M'+(SX.grid-12)+' 80H'+(SX.grid+12)+'M'+(SX.grid-8)+' 56H'+(SX.grid+8)+'M'+(SX.grid-22)+' 62H'+(SX.grid+22)+'M'+(SX.grid-22)+' 62v8M'+(SX.grid+22)+' 62v8'},pyl);
const lab={};[['solar','Solar'],['wind','Wind'],['grid','Grid (IEX)']].forEach(([k,n])=>{el('text',{x:SX[k]+14,y:130,'text-anchor':'start','font-size':11,'font-weight':700,fill:'#c9d1d9'},svg,n);lab[k]=el('text',{x:SX[k]+14,y:144,'text-anchor':'start','font-size':11,fill:'#8b95a5'},svg,'')});
// ---- highways
const roads=[[0,2],[2,1],[0,1]].map(([a,b])=>{let d;
  if(a==0&&b==2)d='M295 410Q335 395 375 345';else if(a==2&&b==1)d='M625 345Q665 395 705 410';else d='M295 492H705';
  const g=el('g',{},svg);el('path',{d,stroke:'#202737','stroke-width':9,fill:'none','stroke-linecap':'round'},g);el('path',{d,stroke:'#39414f','stroke-width':1,'stroke-dasharray':'6 6',fill:'none'},g);
  const p=el('path',{d,fill:'none',stroke:'none'},g),L=p.getTotalLength(),cars=[];
  for(let i=0;i<10;i++)cars.push({u:Math.random(),dir:i%2?1:-1,sp:.05+.03*Math.random(),e:el('circle',{r:3,fill:'#cbd5e1'},g)});
  return {a,b,p,L,cars}});
// ---- cities
const city=[];
for(let m=0;m<M;m++){const [cx,cy]=POS[m],G=el('g',{},svg),n=D.hubs[m];
  el('rect',{x:cx-125,y:cy-85,width:250,height:170,rx:14,fill:'#121722',stroke:'#2a2f3a'},G);el('rect',{x:cx-125,y:cy-85,width:250,height:170,rx:14,fill:'url(#grid)'},G);
  el('text',{x:cx-125,y:cy-95,'font-size':13,'font-weight':700,fill:'#e6e6e6'},G,D.metros[m]);
  const price=el('text',{x:cx+125,y:cy-95,'text-anchor':'end','font-size':11.5,fill:'#c9d1d9'},G,'');const lim=el('text',{x:cx+118,y:cy-70,'text-anchor':'end','font-size':9.5,'font-weight':700,fill:'#ffb84d'},G,'');
  const x0=cx-100,sp=[0,1,2].map(()=>0),pts=D.xy[m].map(p=>[cx-115+p[0]*230,cy-60+p[1]*85]);
  // roads between stations (grid style) and redirect overlays
  const rd=(i,j)=>{const [xi,yi]=pts[i],[xj,yj]=pts[j],sx=Math.sign(xj-xi)||1,sy=Math.sign(yj-yi);return yi==yj?['M'+(xi+sx*30)+' '+yi+'H'+(xj-sx*30),'M'+xi+' '+yi+'H'+xj]:['M'+xi+' '+(yi+sy*19)+'V'+yj+'H'+(xj-sx*30),'M'+xi+' '+yi+'V'+yj+'H'+xj]};
  const redirects=[];for(let i=0;i<n;i++)for(let j=0;j<n;j++){if(i==j)continue;const [dd,full]=rd(i,j);if(i<j)el('path',{d:full,stroke:'#252c3b','stroke-width':6,fill:'none','stroke-linecap':'round'},G);redirects.push({i,j,e:el('path',{d:dd,fill:'none',stroke:'#7dd3fc','stroke-dasharray':'6 6','marker-end':'url(#arr)',opacity:0},G)})}
  // stations
  const st=[];for(let j=0;j<n;j++){const [x,y]=pts[j],g=el('g',{},G);
    const dots=[];for(let q=0;q<10;q++)dots.push(el('circle',{cx:x-45+q*10,cy:y-27,r:3,fill:'#e6e6e6',opacity:0},g));
    const r=el('rect',{x:x-30,y:y-17,width:60,height:34,rx:8,fill:'#3ddc84',stroke:'#e6e6e6','stroke-width':1.2},g);
    el('text',{x,y:y-3,'text-anchor':'middle','font-size':10.5,'font-weight':700,fill:'#0e1117'},g,'Station '+(j+1));const wt=el('text',{x,y:y+11,'text-anchor':'middle','font-size':10,fill:'#0e1117'},g,'');
    const lost=el('text',{x,y:y+30,'text-anchor':'middle','font-size':9.5,'font-weight':700,fill:'#ff6b6b'},g,'');const out=el('text',{x:x+28,y:y-21,'text-anchor':'end','font-size':9,'font-weight':700,fill:'#ff5d5d'},g,'');
    st.push({r,wt,lost,out,dots})}
  // battery + mix
  const bx=cx-118,by=cy+50;el('rect',{x:bx,y:by,width:34,height:16,rx:3,fill:'none',stroke:'#c9d1d9'},G);el('rect',{x:bx+34,y:by+4,width:3,height:8,fill:'#c9d1d9'},G);
  const fill=el('rect',{x:bx+2,y:by+2,width:30,height:12,rx:2,fill:C.batt},G);const bt=el('text',{x:bx+44,y:by+11,'font-size':11,'font-weight':700,fill:'#e6e6e6'},G,'');const bs=el('text',{x:bx,y:by+30,'font-size':9.5,fill:'#8b95a5'},G,'');
  const mixX=cx+8,mixW=108;const seg=['solar','wind','grid','batt'].map(k=>el('rect',{x:mixX,y:by+2,width:0,height:12,fill:C[k]},G));el('rect',{x:mixX,y:by+2,width:mixW,height:12,rx:3,fill:'none',stroke:'#39414f'},G);
  const mt=el('text',{x:mixX,y:by+30,'font-size':9.5,fill:'#8b95a5'},G,'');
  // supply taps from the bus
  const tap={};[['solar',-9],['wind',0],['grid',9]].forEach(([k,o])=>{const x=cx+o;el('line',{x1:x,y1:BUS,x2:x,y2:cy-85,stroke:'#202737','stroke-width':3},svg);tap[k]=el('line',{x1:x,y1:BUS,x2:x,y2:cy-85,stroke:C[k],'stroke-dasharray':'5 7'},svg)});
  city.push({st,pts,redirects,fill,bt,bs,seg,mt,mixW,mixX,tap,price,lim,n})}
function update(dt){
  const t=S.t,i=Math.floor(t),s=D.strat[S.strat],d=s.d;$('clock').textContent=clockTxt(t);evChips(t);
  let solT=0,windT=0,pr=0,srv=0,wsum=0,asum=0,re=0,sup=0,lostT=0;
  for(let k=0;k<i;k++)for(let m=0;m<M;m++)lostT+=d.lost[k][m].reduce((a,b)=>a+b,0);
  for(let m=0;m<M;m++)lostT+=(t-i)*d.lost[i][m].reduce((a,b)=>a+b,0);
  for(let m=0;m<M;m++){const c=city[m],sl=lv(k=>D.world.sol[k][m],t),wn=lv(k=>D.world.wnd[k][m],t),g=lv(k=>d.g[k][m],t),oa=lv(k=>d.oa[k][m],t),pd=lv(k=>d.pd[k][m],t),pc=lv(k=>d.pc[k][m],t),E=lv(k=>d.E[k][m],t),p=lv(k=>D.world.price[k][m],t);
    solT+=sl;windT+=wn;pr+=p/M;const tot=sl+wn||1,os=oa*sl/tot,ow=oa*wn/tot;re+=oa;sup+=g+oa+pd;srv+=lv(k=>d.sv[k][m],t);
    const flows={solar:os,wind:ow,grid:g};for(const k in flows){const v=flows[k];c.tap[k].setAttribute('stroke-width',wid(v));c.tap[k].setAttribute('opacity',v>1?.95:.12);c.tap[k].setAttribute('stroke-dashoffset',-S.phase*46)}
    const mult=1+D.band*D.tiers[d.tier[i][m]];
    c.price.textContent='Rs '+p.toFixed(1)+'/kWh'+(s.key=='framework'?'  tariff x'+mult.toFixed(1):'');c.price.setAttribute('fill',p>=9?'#ff6b6b':p<=3.5?'#3ddc84':'#c9d1d9');
    c.lim.textContent=d.f[i][m]<.999?'supply-limited':'';
    c.fill.setAttribute('width',30*clamp(E/100,0,1));c.bt.textContent=Math.round(E)+'%';c.bs.textContent=pc>pd+5?'charging '+Math.round(pc)+' kW':pd>5?'discharging '+Math.round(pd)+' kW':'idle';
    const tt=g+oa+pd||1,sh=[os,ow,g,pd];let x=0;sh.forEach((v,q)=>{const w=c.mixW*v/tt;c.seg[q].setAttribute('x',c.mixX+x);c.seg[q].setAttribute('width',w);x+=w});c.mt.textContent='Renewables '+Math.round(100*oa/tt)+'%  Grid '+Math.round(100*g/tt)+'%';
    for(let j=0;j<c.n;j++){const o=c.st[j],w=lv(k=>d.wait[k][m][j],t),q=lv(k=>d.q[k][m][j],t),lo=d.lost[i][m][j],out=D.world.cap[i][m][j]<.99;
      o.r.setAttribute('fill',waitCol(w));o.r.setAttribute('stroke',out?'#ff5d5d':'#e6e6e6');o.r.setAttribute('stroke-dasharray',out?'4 3':'none');o.r.setAttribute('stroke-width',out?2:1.2);
      o.wt.textContent=Math.round(w)+' min';o.lost.textContent=lo>1?'lost '+Math.round(lo)+' kWh':'';o.out.textContent=out?'OUTAGE':'';
      const nd=Math.min(10,Math.round(q/35));o.dots.forEach((e,z)=>e.setAttribute('opacity',z<nd?1:0));
      const a=lv(k=>d.arr[k][m][j],t);wsum+=w*a;asum+=a}
    c.redirects.forEach(r=>{const v=lv(k=>d.F[k][m][r.i][r.j],t);r.e.setAttribute('opacity',v>5?.95:0);r.e.setAttribute('stroke-width',1.5+v/90);r.e.setAttribute('stroke-dashoffset',-S.phase*40)})}
  const tr=lv(k=>D.strat[0].d.arr[k].reduce((a,b)=>a+b.reduce((x,y)=>x+y,0),0),t)/maxArr/3;
  roads.forEach(r=>{const n=Math.round(2+8*clamp(tr*1.1,0,1));r.cars.forEach((c,q)=>{c.e.setAttribute('opacity',q<n?.9:0);if(q>=n)return;c.u=(c.u+c.dir*c.sp*(dt||0)*(S.play?1:0)+1)%1;
    const p=r.p.getPointAtLength(c.u*r.L),p2=r.p.getPointAtLength(Math.min(r.L,c.u*r.L+1)),nx=-(p2.y-p.y),ny=p2.x-p.x,nl=Math.hypot(nx,ny)||1;c.e.setAttribute('cx',p.x+nx/nl*3.2*c.dir);c.e.setAttribute('cy',p.y+ny/nl*3.2*c.dir)})});
  const so=solT/(D.world.sol.reduce((a,r)=>Math.max(a,r.reduce((x,y)=>x+y,0)),1)),wo=windT/Math.max(1,D.world.wnd.reduce((a,r)=>Math.max(a,r.reduce((x,y)=>x+y,0)),1));
  rays.setAttribute('transform','rotate('+(S.phase*25)+' '+SX.solar+' 58)');glow.setAttribute('opacity',.28*so);stub.solar.setAttribute('opacity',so>.02?.95:.12);stub.wind.setAttribute('opacity',wo>.02?.95:.12);
  stub.solar.setAttribute('stroke-width',wid(solT));stub.wind.setAttribute('stroke-width',wid(windT));stub.grid.setAttribute('stroke-width',wid(city.reduce((a,c,m)=>a+lv(k=>d.g[k][m],t),0)));
  ['solar','wind','grid'].forEach(k=>stub[k].setAttribute('stroke-dashoffset',-S.phase*46));
  rot.setAttribute('transform','rotate('+(S.phase*(40+260*wo))+' '+SX.wind+' 60)');
  lab.solar.textContent=(solT/1000).toFixed(1)+' MW';lab.wind.textContent=(windT/1000).toFixed(1)+' MW';lab.grid.textContent='Rs '+pr.toFixed(1)+'/kWh avg';
  $('k1').textContent=(srv/1000).toFixed(1)+' MW';$('k2').textContent=(wsum/Math.max(asum,1)).toFixed(1)+' min';$('k3').textContent=(lostT/1000).toFixed(1)+' MWh';$('k4').textContent=Math.round(100*re/Math.max(sup,1))+'%'}
segment('strat',D.strat.map(s=>({html:'<i class="dot" style="background:'+s.color+'"></i>'+s.label,color:s.color})),S.strat,i=>{S.strat=i;update(0)});
wireClock(()=>update(0));
</script></body></html>"""

PRO_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>__CSS__</style></head><body>
<div class="bar"><div class="seg" id="strat"></div><div class="seg" id="city"></div><div class="seg" id="mode"></div></div>
<div class="bar"><button class="btn" id="play">Pause</button><div class="seg" id="spd"></div><input type="range" id="scrub" min="0" max="71" step="0.01" value="0"><span class="clock" id="clock"></span><div class="chips" id="chips"></div></div>
<div class="stats"><div class="stat"><b id="k1">-</b><span>IEX price now</span></div><div class="stat"><b id="k2">-</b><span>Blended energy cost now</span></div>
<div class="stat"><b id="k3">-</b><span>Renewables in supply now</span></div><div class="stat"><b id="k4">-</b><span>Grid in supply now</span></div></div>
<div class="legend"><span><i class="sw" style="background:#fbbf24"></i>Solar</span><span><i class="sw" style="background:#2dd4bf"></i>Wind</span><span><i class="sw" style="background:#a78bfa"></i>Grid (IEX)</span><span><i class="sw" style="background:#60a5fa"></i>Battery discharge</span>
<span><i class="ln" style="border-color:#c792ea"></i>IEX price</span><span><i class="ln" style="border-color:#4dd0e1" id="lgc"></i>Blended cost of energy bought</span><span>Shaded bands = stress events</span></div>
<svg id="ch" viewBox="0 0 1000 540"></svg>
<script>__COMMON__
S.city=-1;S.mode='share';
const X0=62,X1=690,PA=[34,214],PB=[262,484],xs=t=>X0+(X1-X0)*t/(T-1);
const svg=$('ch');const defs=el('defs',{},svg);const cp=el('clipPath',{id:'rv'},defs),cr=el('rect',{x:X0,y:0,width:0,height:540},cp);
const NAMES=['Solar','Wind','Grid (IEX)','Battery'],KEYS=['solar','wind','grid','batt'];
function cities(){return S.city<0?[...Array(M).keys()]:[S.city]}
let SER=[],CUM=[],YMAX=1,STEP=1;
function build(){SER=D.strat.map(s=>{const d=s.d,o={sol:[],wnd:[],grid:[],batt:[],price:[],cost:[],en:[]};
  for(let k=0;k<T;k++){let a=0,b=0,g=0,p=0,e=0,n=0,bt=0;cities().forEach(m=>{const oa=d.oa[k][m],sl=D.world.sol[k][m],wn=D.world.wnd[k][m],tt=sl+wn||1;a+=oa*sl/tt;b+=oa*wn/tt;g+=d.g[k][m];bt+=d.pd[k][m];e+=d.enrg[k][m];p+=D.world.price[k][m];n++});
    o.sol.push(a);o.wnd.push(b);o.grid.push(g);o.batt.push(bt);o.price.push(p/n);o.en.push(e);o.cost.push(clamp(e/Math.max(a+b+g,1),0,14))}return o});
  CUM=SER.map(o=>{const c={};['sol','wnd','grid','batt','en'].forEach(k=>{c[k]=[0];o[k].forEach(v=>c[k].push(c[k][c[k].length-1]+v))});return c});
  YMAX=Math.max(1,...SER.flatMap(o=>o.sol.map((v,k)=>(v+o.wnd[k]+o.grid[k]+o.batt[k])/1000)))*1.05;STEP=YMAX<=2.5?.5:YMAX<=6?1:2;YMAX=Math.ceil(YMAX/STEP)*STEP}
const cumAt=(c,t)=>{const i=Math.floor(t);return c[i]+(t-i)*(c[Math.min(i+1,T)]-c[i])};
// ---- static frame
const gS=el('g',{},svg);
D.events.forEach(ev=>{[PA,PB].forEach(P=>el('rect',{x:xs(ev.a),y:P[0],width:xs(ev.b)-xs(ev.a),height:P[1]-P[0],fill:ev.color,opacity:.13},gS));el('text',{x:(xs(ev.a)+xs(ev.b))/2,y:PA[0]+12,'text-anchor':'middle','font-size':10.5,'font-weight':700,fill:ev.color},gS,ev.id)});
for(let h=0;h<72;h+=12){const x=xs(Math.min(h,T-1));[PA,PB].forEach(P=>el('line',{x1:x,y1:P[0],x2:x,y2:P[1],stroke:'#1d2432'},gS));el('text',{x,y:PB[1]+16,'text-anchor':'middle','font-size':10.5,fill:'#8b95a5'},gS,h%24==0?'Day '+(h/24+1):pad(h%24)+':00')}
el('text',{x:(X0+X1)/2,y:PB[1]+34,'text-anchor':'middle','font-size':11,fill:'#8b95a5'},gS,'Simulation time (hours)');
const yA=v=>PA[1]-(PA[1]-PA[0])*v/12;[0,4,8,12].forEach(v=>{el('line',{x1:X0,y1:yA(v),x2:X1,y2:yA(v),stroke:'#1d2432'},gS);el('text',{x:X0-8,y:yA(v)+4,'text-anchor':'end','font-size':10.5,fill:'#8b95a5'},gS,v)});
el('text',{x:16,y:(PA[0]+PA[1])/2,'text-anchor':'middle','font-size':11,fill:'#8b95a5',transform:'rotate(-90 16 '+(PA[0]+PA[1])/2+')'},gS,'Rs per kWh');
const gY=el('g',{},svg),yLab=el('text',{x:16,y:(PB[0]+PB[1])/2,'text-anchor':'middle','font-size':11,fill:'#8b95a5',transform:'rotate(-90 16 '+(PB[0]+PB[1])/2+')'},svg,'');
el('text',{x:X0,y:PA[0]-14,'font-size':12,'font-weight':700,fill:'#c9d1d9'},svg,'What energy costs');el('text',{x:X0,y:PB[0]-14,'font-size':12,'font-weight':700,fill:'#c9d1d9'},svg,'Where the energy comes from');
// ---- dynamic
const gD=el('g',{'clip-path':'url(#rv)'},svg),area=el('path',{fill:'#c792ea','fill-opacity':.2,stroke:'none'},gD),pline=el('path',{fill:'none',stroke:'#c792ea','stroke-width':2},gD),cline=el('path',{fill:'none','stroke-width':2.4},gD);
const layers=KEYS.map(k=>el('path',{fill:C[k],'fill-opacity':.88,stroke:'#0e1117','stroke-width':.6},gD));
const cur1=el('line',{y1:PA[0],y2:PA[1],stroke:'#e6e6e6','stroke-width':1.2},svg),cur2=el('line',{y1:PB[0],y2:PB[1],stroke:'#e6e6e6','stroke-width':1.2},svg);
const d1=el('circle',{r:4,fill:'#c792ea',stroke:'#0e1117'},svg),d2=el('circle',{r:4,stroke:'#0e1117'},svg);
// ---- right panel: cumulative mix per strategy
const RX=730,RW=260;el('text',{x:RX,y:PA[0]-14,'font-size':12,'font-weight':700,fill:'#c9d1d9'},svg,'Procured so far, by strategy');
const bars=D.strat.map((s,i)=>{const y=PA[0]+14+i*82,o={y};
  o.n=el('text',{x:RX,y:y+4,'font-size':12,'font-weight':700,fill:s.color},svg,s.label);o.c=el('text',{x:RX+RW,y:y+4,'text-anchor':'end','font-size':12,fill:'#c9d1d9'},svg,'');
  el('rect',{x:RX,y:y+12,width:RW,height:24,rx:5,fill:'#161b22',stroke:'#2a2f3a'},svg);o.seg=KEYS.map(k=>el('rect',{x:RX,y:y+12,width:0,height:24,fill:C[k]},svg));o.tx=KEYS.map(k=>el('text',{x:RX,y:y+29,'text-anchor':'middle','font-size':10.5,'font-weight':700,fill:'#0e1117'},svg,''));
  o.r=el('text',{x:RX,y:y+54,'font-size':11,fill:'#8b95a5'},svg,'');return o});
const dl=el('text',{x:RX,y:PB[0]+78,'font-size':12,'font-weight':700,fill:'#c9d1d9'},svg,'');const dl2=el('text',{x:RX,y:PB[0]+96,'font-size':11.5,fill:'#9aa4b2'},svg,'');const dl3=el('text',{x:RX,y:PB[0]+114,'font-size':11.5,fill:'#9aa4b2'},svg,'');
function draw(){const o=SER[S.strat],col=D.strat[S.strat].color;$('lgc').style.borderColor=col;cline.setAttribute('stroke',col);d2.setAttribute('fill',col);
  const mw=S.mode=='mw',yB=v=>PB[1]-(PB[1]-PB[0])*v;let tops=Array(T).fill(0),base=Array(T).fill(0);
  const tot=Array.from({length:T},(_,k)=>o.sol[k]+o.wnd[k]+o.grid[k]+o.batt[k]);
  KEYS.forEach((k,q)=>{const arr=[o.sol,o.wnd,o.grid,o.batt][q];const lo=base.slice();const hi=base.map((b,j)=>b+(mw?arr[j]/1000/YMAX:(tot[j]>0?arr[j]/tot[j]:0)));
    layers[q].setAttribute('d','M'+hi.map((v,j)=>xs(j)+' '+yB(v)).join('L')+'L'+lo.map((v,j)=>xs(j)+' '+yB(v)).reverse().join('L')+'Z');base=hi});
  gY.innerHTML='';const tk=mw?Array.from({length:Math.round(YMAX/STEP)+1},(_,q)=>q*STEP/YMAX):[0,.25,.5,.75,1];tk.forEach(v=>{el('line',{x1:X0,y1:yB(v),x2:X1,y2:yB(v),stroke:'#1d2432'},gY);el('text',{x:X0-8,y:yB(v)+4,'text-anchor':'end','font-size':10.5,fill:'#8b95a5'},gY,mw?(v*YMAX).toFixed(STEP<1?1:0):Math.round(v*100)+'%')});
  yLab.textContent=mw?'Supply (MW)':'Share of supply';
  const pp=o.price.map((v,j)=>xs(j)+' '+yA(clamp(v,0,12)));pline.setAttribute('d','M'+pp.join('L'));area.setAttribute('d','M'+pp.join('L')+'L'+xs(T-1)+' '+PA[1]+'L'+xs(0)+' '+PA[1]+'Z');
  cline.setAttribute('d','M'+o.cost.map((v,j)=>xs(j)+' '+yA(clamp(v,0,12))).join('L'))}
function update(){const t=S.t,i=Math.floor(t),o=SER[S.strat],x=xs(t);$('clock').textContent=clockTxt(t);evChips(t);cr.setAttribute('width',x-X0);
  [cur1,cur2].forEach(c=>{c.setAttribute('x1',x);c.setAttribute('x2',x)});const p=lv(k=>o.price[k],t),c=lv(k=>o.cost[k],t);d1.setAttribute('cx',x);d1.setAttribute('cy',yA(clamp(p,0,12)));d2.setAttribute('cx',x);d2.setAttribute('cy',yA(clamp(c,0,12)));
  const a=lv(k=>o.sol[k]+o.wnd[k],t),g=lv(k=>o.grid[k],t),b=lv(k=>o.batt[k],t),tt=a+g+b||1;
  $('k1').textContent='Rs '+p.toFixed(1)+'/kWh';$('k2').textContent='Rs '+c.toFixed(1)+'/kWh';$('k3').textContent=Math.round(100*a/tt)+'%';$('k4').textContent=Math.round(100*g/tt)+'%';
  const res=D.strat.map((s,si)=>{const cu=CUM[si],v=['sol','wnd','grid','batt'].map(k=>cumAt(cu[k],t)),en=cumAt(cu.en,t),su=v[0]+v[1]+v[2]||1,all=v.reduce((p,q)=>p+q,0)||1;return {v,all,re:100*(v[0]+v[1])/all,cost:en/su}});
  bars.forEach((o2,si)=>{const r=res[si];let xx=0;r.v.forEach((val,q)=>{const w=RW*val/r.all;o2.seg[q].setAttribute('x',RX+xx);o2.seg[q].setAttribute('width',w);o2.tx[q].setAttribute('x',RX+xx+w/2);o2.tx[q].textContent=w>34?Math.round(100*val/r.all)+'%':'';xx+=w});
    o2.c.textContent='Rs '+r.cost.toFixed(2)+'/kWh';o2.r.textContent='Renewables '+r.re.toFixed(0)+'%   Grid '+(100*r.v[2]/r.all).toFixed(0)+'%   Battery '+(100*r.v[3]/r.all).toFixed(0)+'%';o2.n.setAttribute('opacity',si==S.strat?1:.6)});
  if(t>4){const f=res[2],rr=res[0];dl.textContent='Full framework vs reactive';dl2.textContent='Energy cost '+((f.cost/rr.cost-1)*100).toFixed(1)+'%  per kWh bought';dl3.textContent='Renewable share '+(f.re-rr.re>=0?'+':'')+(f.re-rr.re).toFixed(1)+' pts'}else{dl.textContent=dl2.textContent=dl3.textContent=''}}
build();draw();
segment('strat',D.strat.map(s=>({html:'<i class="dot" style="background:'+s.color+'"></i>'+s.label,color:s.color})),S.strat,i=>{S.strat=i;draw();update()});
segment('city',[{html:'All cities'},...D.metros.map(n=>({html:n}))],0,i=>{S.city=i-1;build();draw();update()});
segment('mode',[{html:'Share of supply'},{html:'Absolute MW'}],0,i=>{S.mode=i?'mw':'share';draw();update()});
wireClock(update);
</script></body></html>"""


def _fill(tpl, data):
    return tpl.replace("__CSS__", COMMON_CSS).replace("__COMMON__", COMMON_JS).replace("__DATA__", json.dumps(data, separators=(",", ":")))


def network_html(data): return _fill(NET_HTML, data)
def procure_html(data): return _fill(PRO_HTML, data)
