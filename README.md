# Clean Energy Optimization Microsite

A static, chart-rich site explaining the clean energy optimization puzzle. Designed for GitHub Pages hosting.

## Quick Start

1. **Generate chart data** (requires running from parent directory):
   ```bash
   cd /path/to/Multi-Heatmap\ Test
   python microsite/generate_charts.py
   ```

2. **Serve locally**:
   ```bash
   cd microsite
   python -m http.server 8000
   ```
   Open http://localhost:8000

## Structure

```
microsite/
├── index.html              # Main page (5 parts + takeaways)
├── css/style.css           # Styling (includes assumptions dropdown)
├── js/charts.js            # Chart loading and mapping
├── data/*.json             # Chart data (29 files, generated)
├── generate_charts.py      # Chart generation script (~3500 lines)
├── CHART_DOCUMENTATION.md  # Detailed chart documentation
└── README.md               # This file
```

## Site Narrative Structure

The microsite is organized into 5 parts plus takeaways:

1. **Part 1: The Mystery** - Why does a 95% clean system cost $100/MWh when solar costs $50/MWh?
2. **Part 2: The Energy Problem** - Timing mismatch, solar ceiling, storage limitations
3. **Part 3: The Capacity Problem** - Reliability requirements, capacity credits
4. **Part 4: The Solution** - Clean firm power, optimal resource mix
5. **Part 5: Testing Our Understanding** - Regional variations, sensitivity analysis
6. **Takeaways** - Key implications for clean energy planning

## Key Technologies

- **Frontend**: Vanilla HTML/CSS/JS with Plotly.js for charts
- **Chart Generation**: Python script using the parent project's simulation and optimization modules
- **Simulation**: Numba-JIT compiled 8760-hour chronological simulation
- **Optimization**: Grid search with min-cost objective and hybrid battery dispatch mode

## GitHub Pages Deployment

1. Push the `microsite/` folder to a GitHub repo
2. Go to Settings > Pages
3. Set source to the branch/folder containing the microsite
4. The site will be available at `https://username.github.io/repo-name/`

## Regenerating Charts

Run `generate_charts.py` whenever you want to update the charts with new optimization data or cost assumptions:

```bash
cd /path/to/Multi-Heatmap\ Test
python microsite/generate_charts.py
```

**Note**: Chart generation takes 5-10 minutes due to extensive optimization runs (especially the solar+storage LCOE sweep charts which use grid search with `hybrid_mode=True`).

## Chart Data Files (29 total)

The `data/` directory contains Plotly JSON files for each chart:

| Category | Files |
|----------|-------|
| Core narrative | `mismatch.json`, `lcoe_curve.json`, `resource_mix.json`, `leap.json` |
| Technology comparison | `tech_lcoe.json`, `cost_structure.json` |
| Solar/Wind ceilings | `solar_storage_ceiling.json`, `wind_storage_ceiling.json`, `solar_paradox.json` |
| Solar+Storage economics | `solar_storage_lcoe.json`, `storage_cost_sensitivity.json`, `effective_lcoe.json`, `marginal_energy.json` |
| ELCC/Capacity | `elcc.json`, `elcc_solar_storage.json`, `elcc_wind_storage.json` |
| System economics | `lcoe_breakdown.json`, `gas_lcoe_impact.json`, `gas_capacity.json`, `gas_crossover.json` |
| Regional | `regional.json` |
| Sensitivity | `no_cleanfirm.json`, `no_wind.json`, `cost_sensitivity.json` |
| Financing | `financing.json`, `cleanfirm_tech.json` |
| Optimization | `greedy.json`, `resource_interactions.json` |

See `CHART_DOCUMENTATION.md` for detailed documentation of each chart's generation method.
