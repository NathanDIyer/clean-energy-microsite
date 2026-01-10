# Clean Energy Optimization Microsite

A static, chart-rich site explaining the clean energy optimization puzzle. Designed for GitHub Pages hosting.

## Quick Start

1. **Generate chart data** (requires running from parent directory):
   ```bash
   cd /path/to/Multi-Heatmap\ Test
   python microsite/generate_charts.py
   python microsite/generate_cf_cost_sensitivity.py  # Clean firm cost sensitivity charts
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
├── index.html                       # Main page (5 parts + takeaways)
├── css/style.css                    # Styling (includes assumptions dropdown)
├── js/charts.js                     # Chart loading and mapping
├── data/*.json                      # Chart data (33 files, generated)
├── generate_charts.py               # Main chart generation script (~3500 lines)
├── generate_cf_cost_sensitivity.py  # Clean firm cost sensitivity charts
├── CHART_DOCUMENTATION.md           # Detailed chart documentation
└── README.md                        # This file
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

## Navigation Features

The microsite includes interactive navigation features for improved user experience:

### Visual Navigation
- **Progress Bar**: Fixed bar at the top showing scroll progress (0-100%)
- **Navigation Sidebar**: Fixed sidebar on the right with labeled navigation items
  - Each item shows a dot and section name
  - Active section highlighted with blue color and scaled dot
  - Hover state with light blue background
  - Click any item to jump to that section
- **Section Indicator**: Temporarily displays current section name when jumping between sections
- **Back to Top Button**: Circular button in bottom-right corner (appears after scrolling 500px)

### Keyboard Navigation
The microsite supports keyboard shortcuts for quick navigation:

| Key | Action |
|-----|--------|
| `↓` or `j` | Jump to next section |
| `↑` or `k` | Jump to previous section |
| `Home` | Jump to top (Hero section) |
| `End` | Jump to bottom (Takeaways) |
| `Tab` | Navigate between clickable elements |
| `Enter`/`Space` | Activate focused navigation dot |

### Accessibility Features
- All navigation elements include ARIA labels
- Navigation dots are keyboard accessible (tab + enter/space)
- Smooth scrolling for better UX
- Mobile-responsive (sidebar hidden on mobile, back-to-top button repositioned)

### Implementation
- Progress tracking via scroll event listeners
- Scroll spy automatically detects current section
- CSS transitions for smooth visual feedback
- Mobile breakpoints at 768px and 1024px

## Text Export for AI Analysis

The microsite includes a feature to export all content as structured text, optimized for LLM analysis.

### Features
- **Export Button**: Located at the bottom of the page (footer section)
- **Modal Interface**: Clean, readable text display with actions
- **Copy to Clipboard**: One-click copy for pasting into AI assistants (Claude, ChatGPT, etc.)
- **Download as .txt**: Save the text version locally

### What's Included in the Export
- All section headings and narrative content
- Key insights, callouts, and warnings (marked with >>, ***, **)
- **Actual chart data** extracted from Plotly charts:
  - Chart titles and types
  - Data series with x/y values (up to 50 points per series)
  - Axis labels
  - Heatmap dimensions and value ranges
- Assumption tables and dropdown content
- Structured formatting with clear section separators (using ═══ and ───)
- Timestamp and source information

### Use Cases
- Feed content to LLMs for analysis, questions, or summaries
- Share insights with AI assistants for discussion
- Archive content in plain text format
- Accessibility for text-based readers

### Files
- `js/text-export.js` - Content extraction and modal logic
- Modal styling integrated in `css/style.css`

## GitHub Pages Deployment

1. Push the `microsite/` folder to a GitHub repo
2. Go to Settings > Pages
3. Set source to the branch/folder containing the microsite
4. The site will be available at `https://username.github.io/repo-name/`

## Regenerating Charts

Run the chart generation scripts whenever you want to update the charts with new optimization data or cost assumptions:

```bash
cd /path/to/Multi-Heatmap\ Test
python microsite/generate_charts.py                  # Main charts (~5-10 min)
python microsite/generate_cf_cost_sensitivity.py     # CF cost sensitivity (~2-4 min)
```

**Note**: Chart generation takes time due to extensive optimization runs:
- Main charts: 5-10 minutes (grid search with hybrid battery mode)
- CF cost sensitivity: 2-4 minutes (72 optimizer runs: 3 scenarios × 8 cost points × 3 targets)

## Chart Data Files (33 total)

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
| Clean firm cost | `cf_cost_sensitivity_80pct.json`, `cf_cost_sensitivity_90pct.json`, `cf_cost_sensitivity_100pct.json`, `cf_cost_sensitivity_crossover.json` |
| Financing | `financing.json`, `cleanfirm_tech.json` |
| Optimization | `greedy.json`, `resource_interactions.json` |

See `CHART_DOCUMENTATION.md` for detailed documentation of each chart's generation method.
