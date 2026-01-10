import sys
sys.path.insert(0, '/Users/nathaniyer/Desktop/Multi-Heatmap Test')
from microsite.generate_charts import run_optimization, load_zone_data, DEFAULT_COSTS
import numpy as np

zone_data = load_zone_data()
result = run_optimization(zone_data, 'California', 5, DEFAULT_COSTS)
print(f'5% Clean Target Results:')
print(f'  Solar: {result["solar"]:.1f} MW')
print(f'  Wind: {result["wind"]:.1f} MW')
print(f'  Storage: {result["storage"]:.1f} MWh')
print(f'  Clean Firm: {result["clean_firm"]:.1f} MW')
print(f'  LCOE: ${result["lcoe"]:.2f}/MWh')
print(f'  Clean Match: {result["match"]:.2f}%')
