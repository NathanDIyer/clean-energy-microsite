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
├── index.html          # Main page
├── css/style.css       # Styling
├── js/charts.js        # Chart loading
├── data/*.json         # Chart data (generated)
└── generate_charts.py  # Data generator
```

## GitHub Pages Deployment

1. Push the `microsite/` folder to a GitHub repo
2. Go to Settings > Pages
3. Set source to the branch/folder containing the microsite
4. The site will be available at `https://username.github.io/repo-name/`

## Regenerating Charts

Run `generate_charts.py` whenever you want to update the charts with new optimization data or cost assumptions. The script uses the V4 optimizer from the main project.
