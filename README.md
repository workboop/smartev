# EV charging and swapping network: forecast-driven energy allocation

Three cities (Bengaluru 2 stations, Chennai 2, Hyderabad 3) connected by highways, fed by solar, wind and the IEX grid, with a battery in every city.
Three strategies are compared over 72 hours with three stress events.

## Run
```bash
pip install -r requirements.txt
python run_all.py          # about 15 s: figures, results_section.md, results.json, two standalone animation pages -> ./outputs
streamlit run app.py       # dashboard
```
Deploy: push this folder to GitHub, then on share.streamlit.io choose `app.py` as the main file. Commit `outputs/` so the Results tab is filled immediately.

## Dashboard
| Tab | Content |
|---|---|
| Overview | KPI cards per strategy and the 72 h timeline for a chosen city |
| Network | Animated scene: renewables, grid, batteries, stations, redirects and inter-city traffic |
| Procurement | Animated IEX price, blended cost and supply mix, live comparison of the three strategies |
| Results | Auto-filled Section 9 |

The sidebar changes heat sensitivity, demand level, stress events, redirect acceptance, price tiers and band, and battery size.

## Files
| File | Role |
|---|---|
| `sim.py` | world, queues, GraphSAGE-lite, 24 h storage LP, realizations, history matching, posterior |
| `anim.py` | packs simulation arrays and builds the two self-contained HTML animations |
| `viz.py` | static figures with legends and captions |
| `report.py` | writes Section 9 from the simulation output |
| `run_all.py` | full pipeline |
| `app.py` | Streamlit dashboard |

## Strategies
1. **Reactive**: drivers go to the nearest station, storage only absorbs surplus renewable power visible now.
2. **Routing only**: drivers are told about waits and redirected (up to 8 min detour); no look-ahead.
3. **Full framework**: routing, price tiers and a 24 h rolling look-ahead LP for storage and grid purchases with a wear cost per kWh discharged.
