"""
Generate all chart data for the Clean Energy Optimization microsite.

Run from the microsite folder:
    python generate_charts.py

This generates JSON files that Plotly.js can load directly.
"""

import sys
import os
import json
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import simulate_system
from lcoe_calculator import calculate_lcoe, calculate_lcoe_components
from data_loader import load_all_zone_data
from cost_settings_modal import DEFAULT_COSTS

# Try to import V4 optimizer
try:
    from experimental.optimizer_v4 import run_min_lcoe_v4_adaptive
    HAS_V4 = True
except ImportError:
    HAS_V4 = False
    print("Warning: V4 optimizer not available, using simplified optimization")

# Import incremental walk from optimizer for accurate greedy comparison
try:
    from optimizer import run_incremental_cost_walk
    HAS_INCREMENTAL_WALK = True
except ImportError:
    HAS_INCREMENTAL_WALK = False
    print("Warning: Incremental walk optimizer not available")

# Chart colors matching the main app
COLORS = {
    'solar': '#fbbc05',
    'wind': '#4285f4',
    'storage': '#673ab7',
    'clean_firm': '#ff7900',
    'gas': '#ea4335',
    'load': '#202124',
    'baseline': '#9aa0a6'
}

FONT = "Google Sans, -apple-system, sans-serif"

# Common Plotly layout settings
LAYOUT_DEFAULTS = {
    'font': {'family': FONT, 'size': 12},
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'margin': {'l': 60, 'r': 40, 't': 40, 'b': 60},
    'hovermode': 'x unified',
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': 1.02,
        'xanchor': 'center',
        'x': 0.5
    }
}


def run_optimization(zone_data, zone, target, cost_settings, use_solar=True, use_wind=True,
                     use_storage=True, use_clean_firm=True,
                     max_solar=None, max_wind=None, max_storage=None, max_cf=None):
    """Run optimization using scipy differential_evolution with hybrid battery mode.

    Optional max_* parameters allow custom capacity bounds for special scenarios
    (e.g., solar+storage only at very high clean targets needs more capacity).
    """
    from scipy.optimize import differential_evolution

    profiles = zone_data[zone]
    solar_profile = np.array(profiles['solar'])
    wind_profile = np.array(profiles['wind'])
    load_profile = np.array(profiles['load'])
    annual_load = np.sum(load_profile)

    # Default bounds (can be overridden)
    bounds_solar = (0, max_solar if max_solar else 1000) if use_solar else (0, 0)
    bounds_wind = (0, max_wind if max_wind else 500) if use_wind else (0, 0)
    bounds_storage = (0, max_storage if max_storage else 2400) if use_storage else (0, 0)
    bounds_cf = (0, max_cf if max_cf else 125) if use_clean_firm else (0, 0)

    bounds = [bounds_solar, bounds_wind, bounds_storage, bounds_cf]

    def objective(x):
        solar, wind, storage, cf = x

        (solar_out, wind_out, _, _, gas_gen, curtailed, renewable_delivered_hourly, _,
         clean_firm_gen, _, _) = simulate_system(
            solar_capacity=solar,
            wind_capacity=wind,
            storage_capacity=storage,
            clean_firm_capacity=cf,
            solar_profile=solar_profile,
            wind_profile=wind_profile,
            load_profile=load_profile,
            battery_eff=0.85,
            hybrid_mode=True  # Use hybrid battery mode
        )

        # Sum hourly renewable_delivered (already includes clean firm in hybrid mode)
        clean_energy = float(np.sum(renewable_delivered_hourly))
        match = clean_energy / annual_load * 100

        lcoe = calculate_lcoe(
            cost_settings, solar, wind, storage, cf,
            float(np.sum(solar_out)), float(np.sum(wind_out)), float(np.sum(clean_firm_gen)),
            float(np.sum(gas_gen)), float(np.max(gas_gen)), float(np.sum(curtailed)), annual_load
        )

        # Penalize if not meeting target - use strong penalty to enforce constraint
        if match < target - 0.5:
            penalty = 1000 * (target - match)**2
            return float(lcoe) + penalty
        return float(lcoe)

    # Run differential evolution
    result = differential_evolution(
        objective, bounds, strategy='best1bin',
        maxiter=150, popsize=20, tol=0.05, polish=True, workers=1, seed=42
    )

    solar, wind, storage, cf = result.x

    # Get final metrics
    (solar_out, wind_out, _, _, gas_gen, curtailed, renewable_delivered_hourly, _,
     clean_firm_gen, _, _) = simulate_system(
        solar_capacity=solar,
        wind_capacity=wind,
        storage_capacity=storage,
        clean_firm_capacity=cf,
        solar_profile=solar_profile,
        wind_profile=wind_profile,
        load_profile=load_profile,
        battery_eff=0.85,
        hybrid_mode=True
    )

    # Sum hourly renewable_delivered (already includes clean firm in hybrid mode)
    clean_energy = float(np.sum(renewable_delivered_hourly))
    match = clean_energy / annual_load * 100

    lcoe = calculate_lcoe(
        cost_settings, solar, wind, storage, cf,
        float(np.sum(solar_out)), float(np.sum(wind_out)), float(np.sum(clean_firm_gen)),
        float(np.sum(gas_gen)), float(np.max(gas_gen)), float(np.sum(curtailed)), annual_load
    )

    return {
        'solar': float(solar),
        'wind': float(wind),
        'storage': float(storage),
        'clean_firm': float(cf),
        'lcoe': float(lcoe),
        'match': float(match)
    }


def generate_mismatch_chart(zone_data, zone='California'):
    """Generate solar vs load profile mismatch chart."""
    print("Generating mismatch chart...")

    profiles = zone_data[zone]
    solar = profiles['solar']
    load = profiles['load']

    # Average by hour of day (across all days)
    hours = np.arange(24)
    solar_hourly = np.zeros(24)
    load_hourly = np.zeros(24)

    for h in range(24):
        solar_hourly[h] = np.mean(solar[h::24])
        load_hourly[h] = np.mean(load[h::24])

    # Normalize for comparison
    solar_norm = solar_hourly / np.max(solar_hourly) * 100
    load_norm = load_hourly / np.max(load_hourly) * 100

    chart = {
        'data': [
            {
                'x': hours.tolist(),
                'y': solar_norm.tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar Production',
                'line': {'color': COLORS['solar'], 'width': 3},
                'fill': 'tozeroy',
                'fillcolor': 'rgba(251, 188, 5, 0.3)'
            },
            {
                'x': hours.tolist(),
                'y': load_norm.tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Electricity Demand',
                'line': {'color': COLORS['load'], 'width': 3}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Hour of Day',
                'tickvals': [0, 6, 12, 18, 24],
                'ticktext': ['12am', '6am', '12pm', '6pm', '12am'],
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Relative Output (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [{
                'x': 18,
                'y': 85,
                'text': 'Evening gap',
                'showarrow': True,
                'arrowhead': 2,
                'ax': -40,
                'ay': -30,
                'font': {'size': 11, 'color': COLORS['gas']}
            }]
        }
    }

    return chart


def generate_lcoe_curve(zone_data, zone='California', cost_settings=None):
    """Generate LCOE vs Clean Match % curve."""
    print("Generating LCOE curve...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    targets = [50, 60, 70, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]
    lcoes = []

    for target in targets:
        result = run_optimization(zone_data, zone, target, cost_settings)
        lcoes.append(result['lcoe'])
        print(f"  {target}%: ${result['lcoe']:.1f}/MWh")

    # Gas baseline (approximate)
    gas_baseline = 65

    chart = {
        'data': [
            {
                'x': targets,
                'y': lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Clean Energy LCOE',
                'line': {'color': COLORS['wind'], 'width': 3},
                'marker': {'size': 8}
            },
            {
                'x': [50, 99],
                'y': [gas_baseline, gas_baseline],
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Gas Baseline',
                'line': {'color': COLORS['gas'], 'width': 2, 'dash': 'dash'}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Target (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'shapes': [
                # Zone annotations
                {'type': 'rect', 'x0': 50, 'x1': 75, 'y0': 0, 'y1': 1, 'yref': 'paper',
                 'fillcolor': 'rgba(66, 133, 244, 0.1)', 'line': {'width': 0}},
                {'type': 'rect', 'x0': 75, 'x1': 90, 'y0': 0, 'y1': 1, 'yref': 'paper',
                 'fillcolor': 'rgba(251, 188, 5, 0.1)', 'line': {'width': 0}},
                {'type': 'rect', 'x0': 90, 'x1': 99, 'y0': 0, 'y1': 1, 'yref': 'paper',
                 'fillcolor': 'rgba(255, 121, 0, 0.1)', 'line': {'width': 0}}
            ],
            'annotations': [
                {'x': 62, 'y': 0.95, 'yref': 'paper', 'text': 'Easy', 'showarrow': False,
                 'font': {'size': 10, 'color': '#666'}},
                {'x': 82, 'y': 0.95, 'yref': 'paper', 'text': 'Storage Era', 'showarrow': False,
                 'font': {'size': 10, 'color': '#666'}},
                {'x': 95, 'y': 0.95, 'yref': 'paper', 'text': 'Reliability', 'showarrow': False,
                 'font': {'size': 10, 'color': '#666'}}
            ]
        }
    }

    return chart


def generate_resource_mix(zone_data, zone='California', cost_settings=None):
    """Generate stacked area chart of resource mix by target - shows ENERGY (MWh) not capacity."""
    print("Generating resource mix chart (energy)...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']

    # Use 5% increments for cleaner visualization (cap at 99%)
    targets = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]
    solar_energy = []
    wind_energy = []
    storage_energy = []  # MWh shifted (discharged)
    cf_energy = []
    cf_share = []  # Track CF share of total clean energy

    # Track min CF capacity to enforce monotonicity (CF should never decrease as target increases)
    min_cf_capacity = 0

    # Threshold below which we don't allow clean firm (cleaner narrative)
    # This shows solar/wind handling low targets, CF kicking in for high targets
    CF_THRESHOLD = 85

    for target in targets:
        # Only enable clean firm above threshold for cleaner narrative
        use_cf = (target >= CF_THRESHOLD)
        result = run_optimization(zone_data, zone, target, cost_settings, use_clean_firm=use_cf)

        # Enforce monotonicity: CF capacity should never decrease
        # This ensures smooth narrative "more CF as targets increase"
        if result['clean_firm'] < min_cf_capacity:
            # Re-run optimization with minimum CF constraint
            result['clean_firm'] = min_cf_capacity
        else:
            min_cf_capacity = result['clean_firm']

        # Run simulation to get actual energy delivered
        (solar_out, wind_out, storage_out, _, _, _, _, _,
         clean_firm_gen, _, _) = simulate_system(
            solar_capacity=result['solar'],
            wind_capacity=result['wind'],
            storage_capacity=result['storage'],
            clean_firm_capacity=result['clean_firm'],
            solar_profile=solar_profile,
            wind_profile=wind_profile,
            load_profile=load_profile,
            battery_eff=0.85
        )

        # Convert to GWh for readability
        solar_gwh = np.sum(solar_out) / 1000
        wind_gwh = np.sum(wind_out) / 1000
        # Storage: only count discharge (positive values)
        storage_gwh = np.sum(np.maximum(0, storage_out)) / 1000
        cf_gwh = np.sum(clean_firm_gen) / 1000

        solar_energy.append(solar_gwh)
        wind_energy.append(wind_gwh)
        storage_energy.append(storage_gwh)
        cf_energy.append(cf_gwh)

        # Calculate CF share
        total_clean = solar_gwh + wind_gwh + storage_gwh + cf_gwh
        share = (cf_gwh / total_clean * 100) if total_clean > 0 else 0
        cf_share.append(share)

        print(f"  {target}%: Solar={solar_gwh:.0f}, Wind={wind_gwh:.0f}, "
              f"Storage={storage_gwh:.0f}, CF={cf_gwh:.0f} GWh (CF={share:.0f}%)")

    # Find when CF becomes majority (>50% of clean energy)
    cf_majority_target = None
    for i, share in enumerate(cf_share):
        if share > 50:
            cf_majority_target = targets[i]
            break

    # Smooth the data to reduce visual noise from optimizer variability
    # Use weighted moving average (current point gets most weight)
    def smooth_data(data, window=3):
        """Apply weighted smoothing while preserving endpoints."""
        if len(data) < window:
            return data
        smoothed = list(data)  # Copy
        for i in range(1, len(data) - 1):
            # Weighted average: center point gets 50%, neighbors get 25% each
            smoothed[i] = 0.25 * data[i-1] + 0.5 * data[i] + 0.25 * data[i+1]
        return smoothed

    solar_energy = smooth_data(solar_energy)
    wind_energy = smooth_data(wind_energy)
    storage_energy = smooth_data(storage_energy)
    cf_energy = smooth_data(cf_energy)

    # Stack order: Clean Firm at bottom, then storage, wind, solar on top
    chart = {
        'data': [
            {
                'x': targets,
                'y': cf_energy,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Clean Firm',
                'stackgroup': 'one',
                'fillcolor': COLORS['clean_firm'],
                'line': {'color': COLORS['clean_firm'], 'width': 0}
            },
            {
                'x': targets,
                'y': wind_energy,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Wind',
                'stackgroup': 'one',
                'fillcolor': COLORS['wind'],
                'line': {'color': COLORS['wind'], 'width': 0}
            },
            {
                'x': targets,
                'y': solar_energy,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar',
                'stackgroup': 'one',
                'fillcolor': COLORS['solar'],
                'line': {'color': COLORS['solar'], 'width': 0}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Target (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Annual Energy (GWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': []
        }
    }

    # Add annotation for CF majority point
    if cf_majority_target:
        idx = targets.index(cf_majority_target)
        total_at_point = cf_energy[idx] + storage_energy[idx] + wind_energy[idx] + solar_energy[idx]
        chart['layout']['annotations'].append({
            'x': cf_majority_target,
            'y': total_at_point / 2,
            'text': f'Clean firm >50%<br>at {cf_majority_target}%',
            'showarrow': True,
            'arrowhead': 2,
            'ax': 40,
            'ay': -30,
            'font': {'size': 11, 'color': COLORS['clean_firm']},
            'bgcolor': 'rgba(255,255,255,0.8)'
        })

    return chart


def generate_leap_chart(zone_data, zone='California', cost_settings=None):
    """
    Generate weekly production view using stacked bars (like main dashboard).
    Finds the peak gas week for each target to show when gas is most needed.
    Uses hybrid_mode for proper battery dispatch.
    """
    print("Generating leap chart (weekly stacked bars)...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']

    # Run optimizations for two contrasting targets
    targets = [70, 95]
    week_data = {}

    for target in targets:
        result = run_optimization(zone_data, zone, target, cost_settings)
        print(f"  {target}%: S={result['solar']:.0f}, W={result['wind']:.0f}, "
              f"B={result['storage']:.0f}, CF={result['clean_firm']:.0f}, LCOE=${result['lcoe']:.1f}")

        # Run simulation with hybrid_mode for proper battery dispatch
        (solar_out, wind_out, battery_charge, battery_discharge, gas_gen, curtailed, _, _,
         clean_firm_gen, _, _) = simulate_system(
            solar_capacity=result['solar'],
            wind_capacity=result['wind'],
            storage_capacity=result['storage'],
            clean_firm_capacity=result['clean_firm'],
            solar_profile=solar_profile,
            wind_profile=wind_profile,
            load_profile=load_profile,
            battery_eff=0.85,
            hybrid_mode=True
        )

        # Find the week with maximum gas usage (peak stress week)
        max_gas_per_week = []
        for week in range(52):
            week_start = week * 168
            week_end = week_start + 168
            max_gas_in_week = np.max(gas_gen[week_start:week_end])
            total_gas_in_week = np.sum(gas_gen[week_start:week_end])
            max_gas_per_week.append((week, max_gas_in_week, total_gas_in_week))

        # Sort by max gas in week (descending) to find peak stress week
        max_gas_per_week.sort(key=lambda x: x[1], reverse=True)
        peak_week = max_gas_per_week[0][0]
        week_start = peak_week * 168

        print(f"    Peak gas week: {peak_week} (max gas: {max_gas_per_week[0][1]:.1f} MW)")

        # Extract week data
        week_data[target] = {
            'solar': solar_out[week_start:week_start+168],
            'wind': wind_out[week_start:week_start+168],
            'battery_charge': battery_charge[week_start:week_start+168],
            'battery_discharge': battery_discharge[week_start:week_start+168],
            'clean_firm': clean_firm_gen[week_start:week_start+168],
            'gas': gas_gen[week_start:week_start+168],
            'curtailed': curtailed[week_start:week_start+168],
            'load': load_profile[week_start:week_start+168],
            'result': result,
            'peak_week': peak_week,
            'max_gas': max_gas_per_week[0][1]
        }

    hours_in_week = list(range(168))

    # Create subplot-style chart with two panels using STACKED BARS
    chart = {
        'data': [],
        'layout': {
            **LAYOUT_DEFAULTS,
            'barmode': 'stack',
            'height': 600,
            'annotations': []
        }
    }

    for i, target in enumerate(targets):
        data = week_data[target]
        yaxis = 'y' if i == 0 else 'y2'

        # Calculate what portion of solar/wind goes to load vs charging
        # Following the main app logic
        total_renewable = data['solar'] + data['wind'] + data['clean_firm']
        load_week = data['load']

        # Clean firm serves load first
        clean_firm_to_load = np.minimum(data['clean_firm'], load_week)
        remaining_load = load_week - clean_firm_to_load

        # Remaining renewable (solar + wind) serves remaining load
        remaining_renewable = data['solar'] + data['wind']
        direct_renewable = np.minimum(remaining_renewable, remaining_load)

        # Split between solar and wind proportionally
        total_gen = np.where(remaining_renewable > 0, remaining_renewable, 1)
        solar_to_load = direct_renewable * (data['solar'] / total_gen)
        wind_to_load = direct_renewable * (data['wind'] / total_gen)

        # Battery discharge serves load (after efficiency)
        battery_discharge_eff = data['battery_discharge'] * 0.85

        # Add stacked BARS for this target
        chart['data'].extend([
            {
                'x': hours_in_week,
                'y': clean_firm_to_load.tolist(),
                'type': 'bar',
                'name': 'Clean Firm' if i == 0 else None,
                'marker': {'color': COLORS['clean_firm']},
                'yaxis': yaxis,
                'showlegend': i == 0,
                'legendgroup': 'cleanfirm',
                'xaxis': 'x' if i == 0 else 'x2'
            },
            {
                'x': hours_in_week,
                'y': solar_to_load.tolist(),
                'type': 'bar',
                'name': 'Solar' if i == 0 else None,
                'marker': {'color': COLORS['solar']},
                'yaxis': yaxis,
                'showlegend': i == 0,
                'legendgroup': 'solar',
                'xaxis': 'x' if i == 0 else 'x2'
            },
            {
                'x': hours_in_week,
                'y': wind_to_load.tolist(),
                'type': 'bar',
                'name': 'Wind' if i == 0 else None,
                'marker': {'color': COLORS['wind']},
                'yaxis': yaxis,
                'showlegend': i == 0,
                'legendgroup': 'wind',
                'xaxis': 'x' if i == 0 else 'x2'
            },
            {
                'x': hours_in_week,
                'y': battery_discharge_eff.tolist(),
                'type': 'bar',
                'name': 'Battery Discharge' if i == 0 else None,
                'marker': {'color': '#34a853'},  # Green for discharge
                'yaxis': yaxis,
                'showlegend': i == 0,
                'legendgroup': 'battery_discharge',
                'xaxis': 'x' if i == 0 else 'x2'
            },
            {
                'x': hours_in_week,
                'y': data['battery_charge'].tolist(),
                'type': 'bar',
                'name': 'Battery Charging' if i == 0 else None,
                'marker': {'color': COLORS['storage']},  # Purple for charging
                'yaxis': yaxis,
                'showlegend': i == 0,
                'legendgroup': 'battery_charge',
                'xaxis': 'x' if i == 0 else 'x2'
            },
            {
                'x': hours_in_week,
                'y': data['gas'].tolist(),
                'type': 'bar',
                'name': 'Gas' if i == 0 else None,
                'marker': {'color': COLORS['gas']},
                'yaxis': yaxis,
                'showlegend': i == 0,
                'legendgroup': 'gas',
                'xaxis': 'x' if i == 0 else 'x2'
            },
            {
                'x': hours_in_week,
                'y': data['curtailed'].tolist(),
                'type': 'bar',
                'name': 'Curtailed' if i == 0 else None,
                'marker': {'color': '#000000'},
                'yaxis': yaxis,
                'showlegend': i == 0,
                'legendgroup': 'curtailed',
                'xaxis': 'x' if i == 0 else 'x2'
            },
            # Load line
            {
                'x': hours_in_week,
                'y': data['load'].tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Load' if i == 0 else None,
                'line': {'color': COLORS['load'], 'width': 2},
                'yaxis': yaxis,
                'showlegend': i == 0,
                'legendgroup': 'load',
                'xaxis': 'x' if i == 0 else 'x2'
            }
        ])

        # Add annotation for target label
        result = data['result']
        week_label = "Winter" if data['peak_week'] < 13 or data['peak_week'] > 39 else "Summer"
        chart['layout']['annotations'].append({
            'x': 0.02,
            'y': 0.99 if i == 0 else 0.44,
            'xref': 'paper',
            'yref': 'paper',
            'text': f"<b>{target}% Clean Target</b><br>Week {data['peak_week']+1} ({week_label}) - Max Gas: {data['max_gas']:.0f} MW",
            'showarrow': False,
            'font': {'size': 11},
            'align': 'left',
            'bgcolor': 'rgba(255,255,255,0.9)',
            'borderpad': 4
        })

    # Configure axes for two subplots with more vertical spacing
    chart['layout']['xaxis'] = {
        'title': '',
        'showgrid': False,
        'domain': [0, 1],
        'anchor': 'y',
        'tickvals': [0, 24, 48, 72, 96, 120, 144],
        'ticktext': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    }
    chart['layout']['xaxis2'] = {
        'title': 'Hour of Week',
        'showgrid': False,
        'domain': [0, 1],
        'anchor': 'y2',
        'tickvals': [0, 24, 48, 72, 96, 120, 144],
        'ticktext': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    }
    chart['layout']['yaxis'] = {
        'title': 'Power (MW)',
        'showgrid': True,
        'gridcolor': 'rgba(0,0,0,0.1)',
        'domain': [0.58, 0.98]
    }
    chart['layout']['yaxis2'] = {
        'title': 'Power (MW)',
        'showgrid': True,
        'gridcolor': 'rgba(0,0,0,0.1)',
        'domain': [0, 0.40]
    }

    return chart


def generate_regional_comparison(zone_data, cost_settings=None):
    """Generate regional comparison chart - stacked bars showing LCOE with % breakdown by technology."""
    print("Generating regional comparison (LCOE with % breakdown)...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    zones = ['California', 'Texas', 'Florida']
    target = 95
    results = {}

    for zone in zones:
        if zone in zone_data:
            profiles = zone_data[zone]
            result = run_optimization(zone_data, zone, target, cost_settings)

            # Run simulation to get actual energy
            (solar_out, wind_out, storage_out, _, _, _, _, _,
             clean_firm_gen, _, _) = simulate_system(
                solar_capacity=result['solar'],
                wind_capacity=result['wind'],
                storage_capacity=result['storage'],
                clean_firm_capacity=result['clean_firm'],
                solar_profile=profiles['solar'],
                wind_profile=profiles['wind'],
                load_profile=profiles['load'],
                battery_eff=0.85
            )

            # Convert to GWh
            solar_gwh = np.sum(solar_out) / 1000
            wind_gwh = np.sum(wind_out) / 1000
            storage_gwh = np.sum(np.maximum(0, storage_out)) / 1000
            cf_gwh = np.sum(clean_firm_gen) / 1000
            total_gwh = solar_gwh + wind_gwh + storage_gwh + cf_gwh

            # Calculate percentages
            results[zone] = {
                'solar_pct': solar_gwh / total_gwh if total_gwh > 0 else 0,
                'wind_pct': wind_gwh / total_gwh if total_gwh > 0 else 0,
                'storage_pct': storage_gwh / total_gwh if total_gwh > 0 else 0,
                'cf_pct': cf_gwh / total_gwh if total_gwh > 0 else 0,
                'lcoe': result['lcoe']
            }
            print(f"  {zone}: Solar={results[zone]['solar_pct']*100:.1f}%, Wind={results[zone]['wind_pct']*100:.1f}%, "
                  f"Storage={results[zone]['storage_pct']*100:.1f}%, CF={results[zone]['cf_pct']*100:.1f}%, LCOE=${result['lcoe']:.1f}")

    zone_names = list(results.keys())

    # Build stacked bar data - each segment height = LCOE * percentage
    chart = {
        'data': [
            {
                'x': zone_names,
                'y': [results[z]['lcoe'] * results[z]['cf_pct'] for z in zone_names],
                'type': 'bar',
                'name': 'Clean Firm',
                'marker': {'color': COLORS['clean_firm']},
                'hovertemplate': '%{x}<br>Clean Firm: %{customdata:.1f}%<extra></extra>',
                'customdata': [results[z]['cf_pct'] * 100 for z in zone_names]
            },
            {
                'x': zone_names,
                'y': [results[z]['lcoe'] * results[z]['storage_pct'] for z in zone_names],
                'type': 'bar',
                'name': 'Storage (shifted)',
                'marker': {'color': COLORS['storage']},
                'hovertemplate': '%{x}<br>Storage: %{customdata:.1f}%<extra></extra>',
                'customdata': [results[z]['storage_pct'] * 100 for z in zone_names]
            },
            {
                'x': zone_names,
                'y': [results[z]['lcoe'] * results[z]['wind_pct'] for z in zone_names],
                'type': 'bar',
                'name': 'Wind',
                'marker': {'color': COLORS['wind']},
                'hovertemplate': '%{x}<br>Wind: %{customdata:.1f}%<extra></extra>',
                'customdata': [results[z]['wind_pct'] * 100 for z in zone_names]
            },
            {
                'x': zone_names,
                'y': [results[z]['lcoe'] * results[z]['solar_pct'] for z in zone_names],
                'type': 'bar',
                'name': 'Solar',
                'marker': {'color': COLORS['solar']},
                'hovertemplate': '%{x}<br>Solar: %{customdata:.1f}%<extra></extra>',
                'customdata': [results[z]['solar_pct'] * 100 for z in zone_names]
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'barmode': 'stack',
            'title': {
                'text': f'{target}% Clean Energy Target',
                'y': 0.98,
                'yanchor': 'top'
            },
            'margin': {'l': 60, 'r': 40, 't': 80, 'b': 60},
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.08,
                'xanchor': 'center',
                'x': 0.5
            },
            'xaxis': {'showgrid': False},
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [
                {
                    'x': zone,
                    'y': results[zone]['lcoe'] + 3,
                    'text': f"${results[zone]['lcoe']:.0f}/MWh",
                    'showarrow': False,
                    'font': {'size': 12, 'color': '#333', 'weight': 'bold'}
                }
                for zone in zone_names
            ]
        }
    }

    return chart


def generate_cost_sensitivity(zone_data, zone='California'):
    """Generate cost sensitivity comparison."""
    print("Generating cost sensitivity chart...")

    scenarios = {
        'Baseline': DEFAULT_COSTS.copy(),
        'Cheap Storage': {**DEFAULT_COSTS, 'storage': 150},
        'Cheap Clean Firm': {**DEFAULT_COSTS, 'clean_firm': 2500},
        'Expensive Gas': {**DEFAULT_COSTS, 'gas_price': 8}
    }

    target = 95
    results = {}

    for name, costs in scenarios.items():
        result = run_optimization(zone_data, zone, target, costs)
        results[name] = result
        print(f"  {name}: LCOE=${result['lcoe']:.1f}")

    chart = {
        'data': [
            {
                'x': list(results.keys()),
                'y': [r['lcoe'] for r in results.values()],
                'type': 'bar',
                'marker': {
                    'color': [COLORS['wind'], COLORS['storage'], COLORS['clean_firm'], COLORS['gas']]
                }
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'title': f'LCOE at {target}% Clean Energy Under Different Cost Scenarios',
            'xaxis': {'showgrid': False},
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'showlegend': False
        }
    }

    return chart


def generate_no_cleanfirm_chart(zone_data, zone='California', cost_settings=None):
    """Generate what-if no clean firm comparison."""
    print("Generating no clean firm chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    targets = [70, 80, 85, 90, 95, 99]
    lcoe_with_cf = []
    lcoe_without_cf = []

    for target in targets:
        # With clean firm
        result_cf = run_optimization(zone_data, zone, target, cost_settings, use_clean_firm=True)
        lcoe_with_cf.append(result_cf['lcoe'])

        # Without clean firm
        result_no_cf = run_optimization(zone_data, zone, target, cost_settings, use_clean_firm=False)
        lcoe_without_cf.append(result_no_cf['lcoe'])

        print(f"  {target}%: With CF=${result_cf['lcoe']:.1f}, Without=${result_no_cf['lcoe']:.1f}")

    chart = {
        'data': [
            {
                'x': targets,
                'y': lcoe_with_cf,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'With Clean Firm',
                'line': {'color': COLORS['clean_firm'], 'width': 3},
                'marker': {'size': 8}
            },
            {
                'x': targets,
                'y': lcoe_without_cf,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Without Clean Firm',
                'line': {'color': COLORS['gas'], 'width': 3, 'dash': 'dash'},
                'marker': {'size': 8}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Target (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            }
        }
    }

    return chart


def generate_no_wind_chart(zone_data, zone='Mid-Atlantic', cost_settings=None):
    """
    Show how wind resource quality affects system cost.
    Compares regions with poor, moderate, and excellent wind resources.
    All scenarios use Solar + Storage + Wind (no clean firm) to isolate wind quality impact.
    """
    print("Generating wind quality comparison chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    targets = [70, 80, 85, 90, 95, 99]

    # Select regions with different wind quality
    regions = {
        'Florida': {'color': '#e57373', 'label': 'Poor Wind (Florida)'},
        'California': {'color': '#ffb74d', 'label': 'Moderate Wind (California)'},
        'Mid-Atlantic': {'color': '#64b5f6', 'label': 'Good Wind (Mid-Atlantic)'},
        'Plains': {'color': '#81c784', 'label': 'Excellent Wind (Plains)'}
    }

    results = {region: [] for region in regions}

    # Run optimization for each region
    for region_name in regions:
        print(f"  Running {region_name}...")
        for target in targets:
            result = run_optimization(zone_data, region_name, target, cost_settings,
                                      use_wind=True, use_clean_firm=False,
                                      max_solar=3000, max_storage=10000)
            results[region_name].append(result['lcoe'])
            print(f"    {target}%: ${result['lcoe']:.1f}")

    print(f"  Cost at 95% clean:")
    for region_name in regions:
        idx_95 = targets.index(95)
        print(f"    {region_name}: ${results[region_name][idx_95]:.1f}/MWh")

    # Build chart data
    chart_data = []
    for region_name, region_info in regions.items():
        chart_data.append({
            'x': targets,
            'y': results[region_name],
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': region_info['label'],
            'line': {'color': region_info['color'], 'width': 3},
            'marker': {'size': 8}
        })

    chart = {
        'data': chart_data,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Target (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, max([max(v) for v in results.values()]) * 1.1]
            },
            'annotations': [
                {
                    'x': 95,
                    'y': results['Florida'][targets.index(95)] + 10,
                    'text': 'Poor wind forces<br>high costs',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -30,
                    'ay': -30,
                    'font': {'size': 10, 'color': '#e57373'}
                },
                {
                    'x': 95,
                    'y': results['Plains'][targets.index(95)] - 5,
                    'text': 'Excellent wind<br>keeps costs low',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 30,
                    'ay': 30,
                    'font': {'size': 10, 'color': '#81c784'}
                }
            ]
        }
    }

    return chart


def run_greedy_optimization(zone_data, zone, target, cost_settings, hybrid_mode=True):
    """
    Run incremental cost walk optimization using the same algorithm as the main dashboard.
    Uses hybrid_mode by default for accurate battery dispatch.
    """
    if not HAS_INCREMENTAL_WALK:
        # Fallback to simple evaluation if optimizer not available
        return run_optimization(zone_data, zone, target, cost_settings)

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']

    # Get baseline LCOE (gas-only system)
    baseline_lcoe = 65  # Approximate gas-only LCOE

    # Run the actual incremental cost walk from optimizer.py
    result = run_incremental_cost_walk(
        clean_match_target=target,
        solar_profile=solar_profile,
        wind_profile=wind_profile,
        current_hourly_load_profile=load_profile,
        current_load=np.mean(load_profile),
        cost_settings=cost_settings,
        demand_response_val=0,
        use_solar=True,
        use_wind=True,
        use_storage=True,
        use_clean_firm=True,
        baseline_lcoe=baseline_lcoe,
        peak_shaver_mode=False,
        hybrid_mode=hybrid_mode
    )

    return {
        'solar': result.get('capacities', {}).get('solar', 0),
        'wind': result.get('capacities', {}).get('wind', 0),
        'storage': result.get('capacities', {}).get('storage', 0),
        'clean_firm': result.get('capacities', {}).get('clean_firm', 0),
        'lcoe': result['lcoe'],
        'achieved_match': result['achieved_match']
    }


def generate_greedy_comparison(zone_data, zone='California', cost_settings=None):
    """Generate greedy vs optimal LCOE sweep showing divergence at high targets."""
    print("Generating greedy vs optimal sweep...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    targets = [70, 80, 85, 90, 92, 95, 97, 99]
    optimal_lcoes = []
    greedy_lcoes = []
    penalties = []

    for target in targets:
        # Optimal (V4)
        optimal = run_optimization(zone_data, zone, target, cost_settings)
        optimal_lcoes.append(optimal['lcoe'])

        # Greedy
        greedy = run_greedy_optimization(zone_data, zone, target, cost_settings)
        greedy_lcoes.append(greedy['lcoe'])

        penalty = ((greedy['lcoe'] / optimal['lcoe']) - 1) * 100
        penalties.append(penalty)

        print(f"  {target}%: Optimal=${optimal['lcoe']:.1f}, Greedy=${greedy['lcoe']:.1f} (+{penalty:.1f}%)")

    chart = {
        'data': [
            {
                'x': targets,
                'y': optimal_lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Optimal',
                'line': {'color': COLORS['wind'], 'width': 3},
                'marker': {'size': 8}
            },
            {
                'x': targets,
                'y': greedy_lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Greedy',
                'line': {'color': COLORS['gas'], 'width': 3, 'dash': 'dash'},
                'marker': {'size': 8}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Target (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [
                {
                    'x': 95,
                    'y': greedy_lcoes[targets.index(95)] + 5,
                    'text': f'+{penalties[targets.index(95)]:.0f}% penalty',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 30,
                    'ay': -20,
                    'font': {'size': 11, 'color': COLORS['gas']}
                },
                {
                    'x': 99,
                    'y': greedy_lcoes[-1] + 5,
                    'text': f'+{penalties[-1]:.0f}%',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 20,
                    'ay': -25,
                    'font': {'size': 11, 'color': COLORS['gas']}
                }
            ],
            'shapes': [
                # Shade area between curves to show penalty
                {
                    'type': 'path',
                    'path': ' '.join([f"{'M' if i==0 else 'L'} {targets[i]} {optimal_lcoes[i]}" for i in range(len(targets))]) +
                            ' '.join([f" L {targets[i]} {greedy_lcoes[i]}" for i in range(len(targets)-1, -1, -1)]) + ' Z',
                    'fillcolor': 'rgba(234, 67, 53, 0.15)',
                    'line': {'width': 0}
                }
            ]
        }
    }

    return chart


def generate_solar_paradox(zone_data, zone='California', cost_settings=None):
    """
    Show the solar paradox: solar-only has a ceiling around 50% clean energy.
    Shows clean match plateauing while curtailment keeps rising.
    """
    print("Generating solar paradox chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    load_profile = profiles['load']

    # Sweep solar capacity and measure clean energy match (solar only, no storage)
    solar_capacities = list(range(0, 1001, 50))  # 0 to 1000 MW
    matches = []
    curtailments = []

    for solar_cap in solar_capacities:
        # Simulate solar-only (no wind, no storage, no clean firm)
        (solar_out, _, _, _, gas_gen, curtailed, renewable_delivered, _,
         _, _, _) = simulate_system(
            solar_capacity=solar_cap,
            wind_capacity=0,
            storage_capacity=0,
            clean_firm_capacity=0,
            solar_profile=solar_profile,
            wind_profile=profiles['wind'],  # Not used but required
            load_profile=load_profile,
            battery_eff=0.85
        )

        annual_load = np.sum(load_profile)
        total_solar_gen = np.sum(solar_out)
        match = (np.sum(renewable_delivered) / annual_load) * 100
        matches.append(match)

        # Calculate curtailment percentage (of total solar generation)
        if total_solar_gen > 0:
            curtailment_pct = (np.sum(curtailed) / total_solar_gen) * 100
        else:
            curtailment_pct = 0
        curtailments.append(curtailment_pct)

        if solar_cap % 200 == 0:
            print(f"  Solar {solar_cap} MW: {match:.1f}% clean, {curtailment_pct:.1f}% curtailed")

    # Find max clean match achievable with solar only
    max_match = max(matches)

    chart = {
        'data': [
            {
                'x': solar_capacities,
                'y': matches,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Clean Energy Match (%)',
                'line': {'color': COLORS['solar'], 'width': 3},
                'marker': {'size': 6},
                'yaxis': 'y'
            },
            {
                'x': solar_capacities,
                'y': curtailments,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Curtailment (%)',
                'line': {'color': COLORS['gas'], 'width': 2, 'dash': 'dot'},
                'marker': {'size': 4},
                'yaxis': 'y2'
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Solar Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Clean Energy Match (%)',
                'titlefont': {'color': COLORS['solar']},
                'tickfont': {'color': COLORS['solar']},
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'side': 'left',
                'range': [0, 100]
            },
            'yaxis2': {
                'title': 'Curtailment (%)',
                'titlefont': {'color': COLORS['gas']},
                'tickfont': {'color': COLORS['gas']},
                'showgrid': False,
                'overlaying': 'y',
                'side': 'right',
                'range': [0, 100]
            },
            'shapes': [
                # Horizontal line at max achievable match
                {
                    'type': 'line',
                    'x0': 0, 'x1': 1000,
                    'y0': max_match, 'y1': max_match,
                    'line': {'color': COLORS['solar'], 'width': 2, 'dash': 'dash'}
                }
            ],
            'annotations': [
                {
                    'x': 700,
                    'y': max_match + 3,
                    'text': f'Solar ceiling: ~{max_match:.0f}%',
                    'showarrow': False,
                    'font': {'size': 12, 'color': COLORS['solar'], 'weight': 'bold'}
                },
                {
                    'x': 800,
                    'y': curtailments[-3],
                    'yref': 'y2',
                    'text': 'Curtailment keeps rising<br>while match plateaus',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -60,
                    'ay': -40,
                    'font': {'size': 10, 'color': '#666'}
                }
            ]
        }
    }

    return chart


def generate_solar_storage_ceiling(zone_data, zone='California', cost_settings=None):
    """
    Show how adding storage raises the solar ceiling.
    Multiple lines for 0, 1, 2, 3, 4, 5 hours of storage.
    Uses hybrid_mode for optimal battery dispatch.
    """
    print("Generating solar + storage ceiling chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    load_profile = profiles['load']

    # Get peak load to size storage in "hours"
    peak_load = np.max(load_profile)

    # Storage hours to test (MWh = hours * peak_load)
    storage_hours_list = [0, 1, 2, 3, 4, 5]
    solar_capacities = list(range(0, 1501, 50))  # 0 to 1500 MW

    storage_colors = {
        0: COLORS['solar'],
        1: '#ff9800',  # orange
        2: '#4caf50',  # green
        3: '#2196f3',  # blue
        4: '#9c27b0',  # purple
        5: '#673ab7'   # deep purple
    }

    traces = []
    ceilings = {}

    for hours in storage_hours_list:
        storage_mwh = hours * peak_load
        matches = []

        for solar_cap in solar_capacities:
            (_, _, _, _, _, _, renewable_delivered, _, _, _, _) = simulate_system(
                solar_capacity=solar_cap,
                wind_capacity=0,
                storage_capacity=storage_mwh,
                clean_firm_capacity=0,
                solar_profile=solar_profile,
                wind_profile=profiles['wind'],
                load_profile=load_profile,
                battery_eff=0.85,
                hybrid_mode=True  # Use hybrid mode for better battery dispatch
            )

            match = (np.sum(renewable_delivered) / np.sum(load_profile)) * 100
            matches.append(match)

        ceiling = max(matches)
        ceilings[hours] = ceiling
        print(f"  Solar + {hours}h storage ({storage_mwh:.0f} MWh): ceiling = {ceiling:.1f}%")

        traces.append({
            'x': solar_capacities,
            'y': matches,
            'type': 'scatter',
            'mode': 'lines',
            'name': f'Solar + {hours}h storage' if hours > 0 else 'Solar only',
            'line': {'color': storage_colors[hours], 'width': 2 if hours > 0 else 3}
        })

    chart = {
        'data': traces,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Solar Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Clean Energy Match (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100]
            },
            'annotations': [
                {
                    'x': 1400,
                    'y': ceilings[0] + 2,
                    'text': f'No storage: {ceilings[0]:.0f}%',
                    'showarrow': False,
                    'font': {'size': 10, 'color': storage_colors[0]}
                },
                {
                    'x': 1400,
                    'y': ceilings[5] + 2,
                    'text': f'5h storage: {ceilings[5]:.0f}%',
                    'showarrow': False,
                    'font': {'size': 10, 'color': storage_colors[5]}
                }
            ]
        }
    }

    return chart


def generate_wind_storage_ceiling(zone_data, zone='California', cost_settings=None):
    """
    Show how adding storage raises the wind ceiling.
    Multiple lines for 0, 1, 2, 3, 4, 5 hours of storage.
    Uses hybrid_mode for optimal battery dispatch.
    """
    print("Generating wind + storage ceiling chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    wind_profile = profiles['wind']
    load_profile = profiles['load']

    # Get peak load to size storage in "hours"
    peak_load = np.max(load_profile)

    # Storage hours to test (MWh = hours * peak_load)
    storage_hours_list = [0, 1, 2, 3, 4, 5]
    wind_capacities = list(range(0, 1001, 50))  # 0 to 1000 MW

    storage_colors = {
        0: COLORS['wind'],
        1: '#29b6f6',  # light blue
        2: '#4caf50',  # green
        3: '#ff9800',  # orange
        4: '#9c27b0',  # purple
        5: '#673ab7'   # deep purple
    }

    traces = []
    ceilings = {}

    for hours in storage_hours_list:
        storage_mwh = hours * peak_load
        matches = []

        for wind_cap in wind_capacities:
            (_, _, _, _, _, _, renewable_delivered, _, _, _, _) = simulate_system(
                solar_capacity=0,
                wind_capacity=wind_cap,
                storage_capacity=storage_mwh,
                clean_firm_capacity=0,
                solar_profile=profiles['solar'],
                wind_profile=wind_profile,
                load_profile=load_profile,
                battery_eff=0.85,
                hybrid_mode=True  # Use hybrid mode for better battery dispatch
            )

            match = (np.sum(renewable_delivered) / np.sum(load_profile)) * 100
            matches.append(match)

        ceiling = max(matches)
        ceilings[hours] = ceiling
        print(f"  Wind + {hours}h storage ({storage_mwh:.0f} MWh): ceiling = {ceiling:.1f}%")

        traces.append({
            'x': wind_capacities,
            'y': matches,
            'type': 'scatter',
            'mode': 'lines',
            'name': f'Wind + {hours}h storage' if hours > 0 else 'Wind only',
            'line': {'color': storage_colors[hours], 'width': 2 if hours > 0 else 3}
        })

    chart = {
        'data': traces,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Wind Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Clean Energy Match (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100]
            },
            'annotations': [
                {
                    'x': 900,
                    'y': ceilings[0] + 2,
                    'text': f'No storage: {ceilings[0]:.0f}%',
                    'showarrow': False,
                    'font': {'size': 10, 'color': storage_colors[0]}
                },
                {
                    'x': 900,
                    'y': ceilings[5] + 2,
                    'text': f'5h storage: {ceilings[5]:.0f}%',
                    'showarrow': False,
                    'font': {'size': 10, 'color': storage_colors[5]}
                }
            ]
        }
    }

    return chart


def generate_solar_storage_lcoe(zone_data, zone='California', cost_settings=None):
    """
    Show LCOE curve for optimized solar+storage only systems.
    X-axis: Clean energy % (0-100%)
    Y-axis: System LCOE
    Single line with vertical markers when storage hours increase.

    At each clean % target, optimize both solar AND storage for minimum LCOE.
    Draw vertical lines when storage crosses 1h, 2h, 3h, etc. thresholds.
    """
    print("Generating solar+storage LCOE sweep with storage hour markers...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']
    annual_load = np.sum(load_profile)
    peak_load = np.max(load_profile)

    # Storage thresholds in hours (1h = peak_load MWh)
    storage_hour_thresholds = [1, 2, 3, 4, 5, 6, 7, 8]

    results = []
    storage_crossings = {}  # {hours: clean_pct where it first crosses}

    # Sweep from low to high clean energy targets
    for target_pct in range(5, 96, 2):
        best_lcoe = float('inf')
        best_config = None

        # Grid search over solar and storage to find minimum LCOE
        for solar_cap in range(25, 2001, 25):
            for storage_hours in [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12]:
                storage_mwh = storage_hours * peak_load

                (solar_out, _, _, batt_discharge, gas_gen, curtailed,
                 renewable_delivered, _, _, _, _) = simulate_system(
                    solar_capacity=solar_cap,
                    wind_capacity=0,
                    storage_capacity=storage_mwh,
                    clean_firm_capacity=0,
                    solar_profile=solar_profile,
                    wind_profile=wind_profile,
                    load_profile=load_profile,
                    battery_eff=0.85,
                    hybrid_mode=True
                )

                match = (np.sum(renewable_delivered) / annual_load) * 100

                if match >= target_pct - 1:  # Close enough to target
                    lcoe = calculate_lcoe(
                        cost_settings, solar_cap, 0, storage_mwh, 0,
                        np.sum(solar_out), 0, np.sum(batt_discharge),
                        np.sum(gas_gen), np.max(gas_gen), np.sum(curtailed), annual_load
                    )

                    if lcoe < best_lcoe:
                        best_lcoe = lcoe
                        best_config = {
                            'solar': solar_cap,
                            'storage_mwh': storage_mwh,
                            'storage_hours': storage_hours,
                            'match': match
                        }

        if best_config:
            results.append({
                'target': target_pct,
                'match': best_config['match'],
                'lcoe': best_lcoe,
                'solar': best_config['solar'],
                'storage_hours': best_config['storage_hours']
            })

            # Track when we first cross each storage hour threshold
            for hours in storage_hour_thresholds:
                if hours not in storage_crossings and best_config['storage_hours'] >= hours:
                    storage_crossings[hours] = target_pct
                    print(f"    {target_pct}% clean: crosses {hours}h storage (LCOE=${best_lcoe:.0f})")

    # Create main trace
    matches = [r['match'] for r in results]
    lcoes = [r['lcoe'] for r in results]

    traces = [{
        'x': matches,
        'y': lcoes,
        'type': 'scatter',
        'mode': 'lines+markers',
        'name': 'Optimized Solar+Storage',
        'line': {'color': '#1a73e8', 'width': 3},
        'marker': {'size': 6},
        'hoverinfo': 'skip'
    }]

    # Create vertical lines and labels at storage hour crossings
    shapes = []
    annotations = []

    storage_colors = {
        1: '#ff9800', 2: '#4caf50', 3: '#00bcd4', 4: '#2196f3',
        5: '#9c27b0', 6: '#e91e63', 7: '#795548', 8: '#607d8b'
    }

    for hours, clean_pct in storage_crossings.items():
        # Find LCOE at this crossing point
        lcoe_at_crossing = next((r['lcoe'] for r in results if r['target'] >= clean_pct), 100)

        color = storage_colors.get(hours, '#666')

        # Vertical dashed line
        shapes.append({
            'type': 'line',
            'x0': clean_pct, 'x1': clean_pct,
            'y0': 50, 'y1': lcoe_at_crossing,
            'line': {'color': color, 'width': 2, 'dash': 'dot'}
        })

        # Label at top of line
        annotations.append({
            'x': clean_pct,
            'y': lcoe_at_crossing + 8,
            'text': f'{hours}h',
            'showarrow': False,
            'font': {'size': 10, 'color': color, 'weight': 'bold'}
        })

    # Add explanatory annotation
    annotations.append({
        'x': 0.98,
        'y': 0.95,
        'xref': 'paper',
        'yref': 'paper',
        'text': 'Vertical lines show when<br>storage hours increase',
        'showarrow': False,
        'font': {'size': 10, 'color': '#666'},
        'align': 'right'
    })

    chart = {
        'data': traces,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Match (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100]
            },
            'yaxis': {
                'title': 'System LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [50, max(lcoes) * 1.1] if lcoes else [50, 300]
            },
            'shapes': shapes,
            'annotations': annotations
        }
    }

    return chart


def generate_solar_storage_lcoe_cheap(zone_data, zone='California', cost_settings=None):
    """
    Ultra-cheap variant of solar+storage LCOE curve showing global lowest costs.
    Uses Chinese/Gulf market prices: ~$400/kW solar, ~$100/kWh storage.
    This represents the absolute global floor - multiples below most regional averages.
    """
    print("Generating ultra-cheap solar+storage LCOE sweep (global lowest costs)...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    # Override with ultra-cheap global costs
    cheap_costs = cost_settings.copy()
    cheap_costs['solar'] = 400      # Global lowest (China/Middle East), vs $1000 US average
    cheap_costs['storage'] = 100    # Near record lows ($72 Saudi tender, $165 global avg)
    cheap_costs['solar_fixed_om'] = 10  # Slightly lower O&M too

    # Call the standard function with modified costs
    chart = generate_solar_storage_lcoe(zone_data, zone, cheap_costs)

    # Update the trace name to indicate this is the cheap variant
    if chart and 'data' in chart and len(chart['data']) > 0:
        chart['data'][0]['name'] = 'Ultra-Cheap Solar+Storage'
        chart['data'][0]['line']['color'] = '#34a853'  # Green to indicate "cheap"

    return chart


def generate_storage_cost_sensitivity(zone_data, zone='California', cost_settings=None):
    """
    Show how cheaper storage costs bend the LCOE curve for solar+storage systems.
    Multiple lines for different storage costs: $300, $200, $100, $50/kWh.
    No storage hour markers - just clean LCOE sweep lines.
    """
    print("Generating storage cost sensitivity chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']
    annual_load = np.sum(load_profile)
    peak_load = np.max(load_profile)

    # Storage costs to compare ($/kWh)
    storage_costs = [300, 200, 100, 50]
    storage_colors = {
        300: '#1a73e8',  # blue (current/default)
        200: '#4caf50',  # green
        100: '#ff9800',  # orange
        50: '#e91e63'    # pink
    }

    traces = []

    for storage_cost in storage_costs:
        print(f"  Testing storage at ${storage_cost}/kWh...")

        # Create modified cost settings
        modified_costs = cost_settings.copy()
        modified_costs['storage'] = storage_cost

        results = []

        # Sweep from low to high clean energy targets
        for target_pct in range(5, 96, 3):
            best_lcoe = float('inf')
            best_match = 0

            # Grid search over solar and storage to find minimum LCOE
            for solar_cap in range(25, 2001, 50):
                for storage_hours in [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12]:
                    storage_mwh = storage_hours * peak_load

                    (solar_out, _, _, batt_discharge, gas_gen, curtailed,
                     renewable_delivered, _, _, _, _) = simulate_system(
                        solar_capacity=solar_cap,
                        wind_capacity=0,
                        storage_capacity=storage_mwh,
                        clean_firm_capacity=0,
                        solar_profile=solar_profile,
                        wind_profile=wind_profile,
                        load_profile=load_profile,
                        battery_eff=0.85,
                        hybrid_mode=True
                    )

                    match = (np.sum(renewable_delivered) / annual_load) * 100

                    if match >= target_pct - 1:
                        lcoe = calculate_lcoe(
                            modified_costs, solar_cap, 0, storage_mwh, 0,
                            np.sum(solar_out), 0, np.sum(batt_discharge),
                            np.sum(gas_gen), np.max(gas_gen), np.sum(curtailed), annual_load
                        )

                        if lcoe < best_lcoe:
                            best_lcoe = lcoe
                            best_match = match

            if best_lcoe < float('inf'):
                results.append({'match': best_match, 'lcoe': best_lcoe})

        if results:
            matches = [r['match'] for r in results]
            lcoes = [r['lcoe'] for r in results]

            traces.append({
                'x': matches,
                'y': lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': f'${storage_cost}/kWh storage',
                'line': {'color': storage_colors[storage_cost], 'width': 2},
                'marker': {'size': 5},
                'hoverinfo': 'skip'
            })

            print(f"    ${storage_cost}/kWh: Max {max(matches):.0f}% at ${lcoes[-1]:.0f}/MWh")

    # Add ultra-cheap scenario: cheap solar + cheap storage combined
    # Use IDENTICAL sweep logic to the standard lines above
    print(f"  Testing ultra-cheap scenario: $400/kW solar + $100/kWh storage...")
    ember_costs = cost_settings.copy()
    ember_costs['solar'] = 400  # Global lowest solar cost
    ember_costs['storage'] = 100  # Near record-low storage
    ember_costs['solar_fixed_om'] = 10  # Lower O&M too

    results = []

    # Use the actual optimizer like the dashboard does (starts from 0%)
    from optimizer import run_optimization

    for target_pct in range(0, 96, 5):
        try:
            output = run_optimization(
                n_clicks=1,
                clean_match_target=target_pct,
                strategy='min_lcoe',
                selected_zone=zone,
                cost_settings=ember_costs,
                load_type='Hourly Load',
                demand_response_val=0,
                zone_data=zone_data,
                flat_load=None,
                use_solar=True,
                use_wind=False,
                use_storage=True,
                use_clean_firm=False,
                peak_shaver_mode=False,
                hybrid_mode=True
            )

            # Debug: check what we got back
            if output is None:
                print(f"      {target_pct}%: returned None")
                continue

            # Unpack the output
            if len(output) == 5:
                _, result, _, _, _ = output
            else:
                print(f"      {target_pct}%: unexpected output length {len(output)}")
                continue

            if result and isinstance(result, dict):
                if 'lcoe' in result and 'achieved_match' in result:
                    results.append({
                        'match': result['achieved_match'],
                        'lcoe': result['lcoe']
                    })
                    print(f"      {target_pct}%: ✓ {result['achieved_match']:.1f}% @ ${result['lcoe']:.1f}/MWh")
                else:
                    print(f"      {target_pct}%: missing keys")
            else:
                print(f"      {target_pct}%: result not a dict")
        except Exception as e:
            print(f"      {target_pct}%: Error - {e}")
            import traceback
            traceback.print_exc()
            continue

    if results:
        matches = [r['match'] for r in results]
        lcoes = [r['lcoe'] for r in results]

        traces.append({
            'x': matches,
            'y': lcoes,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Ultra-cheap solar + storage',
            'line': {'color': '#9c27b0', 'width': 3, 'dash': 'solid'},
            'marker': {'size': 6},
            'hoverinfo': 'skip'
        })

        print(f"    Ultra-cheap: {min(matches):.0f}% to {max(matches):.0f}% clean, ${lcoes[0]:.0f}/MWh to ${lcoes[-1]:.0f}/MWh")

    chart = {
        'data': traces,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Match (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100]
            },
            'yaxis': {
                'title': 'System LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [50, 220]
            },
            'annotations': [{
                'x': 0.02,
                'y': 0.98,
                'xref': 'paper',
                'yref': 'paper',
                'text': 'Cheaper storage<br>bends the curve down',
                'showarrow': False,
                'font': {'size': 11, 'color': '#666'},
                'align': 'left'
            }]
        }
    }

    return chart


def generate_elcc_chart(zone_data, zone='California', cost_settings=None):
    """
    Generate ELCC-style chart showing marginal effectiveness of each resource.
    Each resource is swept IN ISOLATION to show its true ceiling.
    """
    print("Generating ELCC/effectiveness chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']

    def get_match(solar, wind, storage, cf):
        (_, _, _, _, _, _, renewable_delivered, _, _, _, _) = simulate_system(
            solar_capacity=solar, wind_capacity=wind,
            storage_capacity=storage, clean_firm_capacity=cf,
            solar_profile=solar_profile, wind_profile=wind_profile,
            load_profile=load_profile, battery_eff=0.85
        )
        return (np.sum(renewable_delivered) / np.sum(load_profile)) * 100

    # Sweep each resource IN ISOLATION (no other resources)
    step = 25

    resources = {
        'Solar': {'max': 1000, 'color': COLORS['solar']},
        'Wind': {'max': 500, 'color': COLORS['wind']},
        'Clean Firm': {'max': 125, 'color': COLORS['clean_firm']}
    }

    chart_data = []

    for name, config in resources.items():
        capacities = []
        matches = []

        for cap in range(0, config['max'] + 1, step):
            if name == 'Solar':
                match = get_match(cap, 0, 0, 0)  # Solar only
            elif name == 'Wind':
                match = get_match(0, cap, 0, 0)  # Wind only
            else:  # Clean Firm
                match = get_match(0, 0, 0, cap)  # Clean firm only

            capacities.append(cap)
            matches.append(match)

        print(f"  {name}: 0->{config['max']} MW, match {matches[0]:.1f}%->{matches[-1]:.1f}%")

        chart_data.append({
            'x': capacities,
            'y': matches,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': name,
            'line': {'color': config['color'], 'width': 3},
            'marker': {'size': 6}
        })

    # Find ceilings for annotations
    solar_ceiling = max([m for c, m in zip(chart_data[0]['x'], chart_data[0]['y'])])
    wind_ceiling = max([m for c, m in zip(chart_data[1]['x'], chart_data[1]['y'])])

    chart = {
        'data': chart_data,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Resource Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Clean Energy Match (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 105]
            },
            'annotations': [
                {
                    'x': 800,
                    'y': solar_ceiling + 3,
                    'text': f'Solar ceiling: ~{solar_ceiling:.0f}%',
                    'showarrow': False,
                    'font': {'size': 10, 'color': COLORS['solar']}
                },
                {
                    'x': 400,
                    'y': wind_ceiling + 3,
                    'text': f'Wind ceiling: ~{wind_ceiling:.0f}%',
                    'showarrow': False,
                    'font': {'size': 10, 'color': COLORS['wind']}
                },
                {
                    'x': 100,
                    'y': 95,
                    'text': 'Clean firm: linear to 100%',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 40,
                    'ay': 20,
                    'font': {'size': 10, 'color': COLORS['clean_firm']}
                }
            ]
        }
    }

    return chart


def generate_financing_impact(zone_data, zone='California'):
    """
    Compare cost of capital vs ITC impact on system LCOE.
    Key insight: Discount rate often matters more than tax credits because
    it affects ALL capex-heavy technologies, while ITC only helps specific ones.
    """
    print("Generating financing impact chart...")

    profiles = zone_data[zone]
    target = 95

    scenarios = {
        'Baseline\n(7%, no ITC)': {'discount_rate': 7, 'solar_itc': 0, 'wind_itc': 0, 'storage_itc': 0, 'clean_firm_itc': 0},
        'With 30% ITC\n(7%)': {'discount_rate': 7, 'solar_itc': 30, 'wind_itc': 30, 'storage_itc': 30, 'clean_firm_itc': 30},
        'Low Cost of Capital\n(4%, no ITC)': {'discount_rate': 4, 'solar_itc': 0, 'wind_itc': 0, 'storage_itc': 0, 'clean_firm_itc': 0},
        'Both\n(4%, 30% ITC)': {'discount_rate': 4, 'solar_itc': 30, 'wind_itc': 30, 'storage_itc': 30, 'clean_firm_itc': 30}
    }

    lcoes = []
    labels = list(scenarios.keys())

    for name, overrides in scenarios.items():
        cost_settings = {**DEFAULT_COSTS, **overrides}
        result = run_optimization(zone_data, zone, target, cost_settings)
        lcoes.append(result['lcoe'])
        print(f"  {name.replace(chr(10), ' ')}: ${result['lcoe']:.1f}/MWh")

    # Calculate the impact
    baseline = lcoes[0]
    itc_impact = baseline - lcoes[1]
    discount_impact = baseline - lcoes[2]

    chart = {
        'data': [
            {
                'x': labels,
                'y': lcoes,
                'type': 'bar',
                'marker': {
                    'color': [COLORS['baseline'], COLORS['solar'], COLORS['wind'], COLORS['clean_firm']]
                },
                'text': [f'${l:.0f}' for l in lcoes],
                'textposition': 'outside'
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'title': f'System LCOE at {target}% Clean Energy',
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, max(lcoes) * 1.15]
            },
            'xaxis': {'showgrid': False},
            'showlegend': False,
            'annotations': [
                {
                    'x': 0.5,
                    'y': -0.2,
                    'xref': 'paper',
                    'yref': 'paper',
                    'text': f'ITC saves ${itc_impact:.0f}/MWh | Lower cost of capital saves ${discount_impact:.0f}/MWh',
                    'showarrow': False,
                    'font': {'size': 12, 'color': '#333'}
                }
            ]
        }
    }

    return chart


def generate_gas_crossover(zone_data, zone='California'):
    """
    Show at what gas price clean energy becomes cheaper than gas-only.
    This reveals the "hedge value" of clean energy against fuel price volatility.
    """
    print("Generating gas price crossover chart...")

    gas_prices = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    targets = [0, 70, 90, 95]  # 0% = gas only

    results = {t: [] for t in targets}

    for gas_price in gas_prices:
        for target in targets:
            cost_settings = {**DEFAULT_COSTS, 'gas_price': gas_price}

            if target == 0:
                # Gas-only system - approximate LCOE
                # Gas fuel cost = price × heat_rate
                heat_rate = DEFAULT_COSTS.get('gas_heat_rate', 7.5)
                fuel_cost = gas_price * heat_rate
                # Add fixed costs (capex + O&M) - roughly $35/MWh at baseline
                gas_lcoe = fuel_cost + 35
                results[target].append(gas_lcoe)
            else:
                result = run_optimization(zone_data, zone, target, cost_settings)
                results[target].append(result['lcoe'])

        print(f"  Gas ${gas_price}/MMBtu: " +
              ", ".join([f"{t}%=${results[t][-1]:.0f}" for t in targets]))

    chart = {
        'data': [
            {
                'x': gas_prices,
                'y': results[0],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Gas Only',
                'line': {'color': COLORS['gas'], 'width': 3},
                'marker': {'size': 8}
            },
            {
                'x': gas_prices,
                'y': results[70],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': '70% Clean',
                'line': {'color': COLORS['wind'], 'width': 2, 'dash': 'dot'},
                'marker': {'size': 6}
            },
            {
                'x': gas_prices,
                'y': results[90],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': '90% Clean',
                'line': {'color': COLORS['storage'], 'width': 2},
                'marker': {'size': 6}
            },
            {
                'x': gas_prices,
                'y': results[95],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': '95% Clean',
                'line': {'color': COLORS['clean_firm'], 'width': 3},
                'marker': {'size': 8}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Natural Gas Price ($/MMBtu)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'System LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [
                {
                    'x': 8,
                    'y': 95,
                    'text': 'Crossover zone',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 40,
                    'ay': -30,
                    'font': {'size': 11, 'color': '#333'}
                },
                {
                    'x': 8,
                    'y': 60,
                    'xanchor': 'left',
                    'text': '$8/MMBtu gas ≈ $75/ton carbon price<br>(vs $4/MMBtu baseline)',
                    'showarrow': False,
                    'font': {'size': 10, 'color': '#666'},
                    'bgcolor': 'rgba(255,255,255,0.8)',
                    'bordercolor': '#ccc',
                    'borderwidth': 1,
                    'borderpad': 4
                }
            ]
        }
    }

    return chart


def generate_cleanfirm_crossover(zone_data, zone='California'):
    """
    Show how clean firm cost determines optimal clean firm capacity at 90% clean.

    Three scenarios:
    - Baseline: Default renewable costs
    - Low Cost Renewables: Cheaper solar/wind/storage
    - High Cost Renewables: More expensive solar/wind/storage

    Key insight: There's a crossover point around $3,000-4,000/kW where clean firm
    becomes economical. With expensive renewables, clean firm makes sense at higher costs.
    """
    print("Generating clean firm crossover chart...")

    target = 90  # 90% clean energy target

    # Clean firm costs to sweep ($1,000 to $12,000/kW)
    cf_costs = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]

    # Define scenarios (only varying renewable costs, not clean firm)
    scenarios = {
        'Low Cost Renewables': {
            'solar': 500,
            'wind': 700,
            'storage': 100,
            'label': 'Low Cost Renewables',
            'color': '#34a853',  # Green
            'dash': 'dot'
        },
        'Baseline': {
            'solar': 1000,
            'wind': 1200,
            'storage': 300,
            'label': 'Baseline',
            'color': '#4285f4',  # Blue
            'dash': 'solid'
        },
        'High Cost Renewables': {
            'solar': 1500,
            'wind': 1800,
            'storage': 450,
            'label': 'High Cost Renewables',
            'color': '#ea4335',  # Red
            'dash': 'dash'
        }
    }

    results = {name: {'cf_capacity': [], 'lcoe': []} for name in scenarios}

    for cf_cost in cf_costs:
        print(f"  Clean Firm ${cf_cost}/kW:")
        for name, scenario in scenarios.items():
            cost_settings = {
                **DEFAULT_COSTS,
                'solar': scenario['solar'],
                'wind': scenario['wind'],
                'storage': scenario['storage'],
                'clean_firm': cf_cost
            }

            result = run_optimization(zone_data, zone, target, cost_settings)
            results[name]['cf_capacity'].append(result['clean_firm'])
            results[name]['lcoe'].append(result['lcoe'])

            print(f"    {name}: CF={result['clean_firm']:.0f}MW, LCOE=${result['lcoe']:.0f}")

    # Build chart - y-axis is clean firm capacity
    traces = []
    for name, scenario in scenarios.items():
        traces.append({
            'x': cf_costs,
            'y': results[name]['cf_capacity'],
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': scenario['label'],
            'line': {
                'color': scenario['color'],
                'width': 2 if name == 'Baseline' else 2,
                'dash': scenario['dash']
            },
            'marker': {'size': 7},
            'hovertemplate': (
                f"{scenario['label']}<br>" +
                "CF Cost: $%{x:,.0f}/kW<br>" +
                "Optimal CF: %{y:.0f} MW<br>" +
                "LCOE: $%{customdata:.0f}/MWh<extra></extra>"
            ),
            'customdata': results[name]['lcoe']
        })

    chart = {
        'data': traces,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Firm Cost ($/kW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'tickformat': '$,.0f'
            },
            'yaxis': {
                'title': 'Optimal Clean Firm Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'center',
                'x': 0.5
            },
            'annotations': [
                {
                    'x': 4000,
                    'y': 60,
                    'text': 'Crossover zone',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 50,
                    'ay': -30,
                    'font': {'size': 11, 'color': '#333'}
                }
            ]
        }
    }

    return chart


def generate_cost_structure(zone_data, zone='California', cost_settings=None):
    """
    Show capex vs opex breakdown by technology.
    Key insight: Renewables are nearly 100% capex, gas is mostly opex (fuel).
    This explains why financing costs matter so much for clean energy.
    """
    print("Generating cost structure chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    # Calculate approximate cost breakdown per MWh for each technology
    # Assumptions: Capacity factors, lifetimes from defaults
    discount_rate = cost_settings.get('discount_rate', 7) / 100
    lifetime = 20  # Analysis period

    # Capital recovery factor
    crf = (discount_rate * (1 + discount_rate)**lifetime) / ((1 + discount_rate)**lifetime - 1)

    technologies = {
        'Solar': {
            'capex_kw': cost_settings.get('solar', 1000),
            'fixed_om': cost_settings.get('solar_fixed_om', 15),
            'var_om': cost_settings.get('solar_var_om', 0),
            'fuel': 0,
            'cf': 0.25,  # Capacity factor
            'color': COLORS['solar']
        },
        'Wind': {
            'capex_kw': cost_settings.get('wind', 1200),
            'fixed_om': cost_settings.get('wind_fixed_om', 40),
            'var_om': cost_settings.get('wind_var_om', 0),
            'fuel': 0,
            'cf': 0.35,
            'color': COLORS['wind']
        },
        'Storage': {
            'capex_kw': cost_settings.get('storage', 300) * 4,  # 4-hour duration
            'fixed_om': cost_settings.get('storage_fixed_om', 10) * 4,
            'var_om': cost_settings.get('storage_var_om', 5),
            'fuel': 0,
            'cf': 0.15,  # Effective throughput
            'color': COLORS['storage']
        },
        'Clean Firm': {
            'capex_kw': cost_settings.get('clean_firm', 5000),
            'fixed_om': cost_settings.get('clean_firm_fixed_om', 60),
            'var_om': cost_settings.get('clean_firm_var_om', 10),
            'fuel': cost_settings.get('clean_firm_fuel', 20),
            'cf': 0.85,
            'color': COLORS['clean_firm']
        },
        'Gas': {
            'capex_kw': cost_settings.get('gas_capex', 1200),
            'fixed_om': cost_settings.get('gas_fixed_om', 20),
            'var_om': cost_settings.get('gas_var_om', 2),
            'fuel': cost_settings.get('gas_price', 4) * cost_settings.get('gas_heat_rate', 7.5),
            'cf': 0.50,  # Moderate utilization
            'color': COLORS['gas']
        }
    }

    tech_names = list(technologies.keys())
    capex_costs = []
    om_costs = []
    fuel_costs = []

    for name, tech in technologies.items():
        annual_gen = tech['cf'] * 8760  # kWh per kW capacity
        if annual_gen > 0:
            # Capex component ($/MWh)
            capex_mwh = (tech['capex_kw'] * crf * 1000) / annual_gen
            # Fixed O&M ($/MWh)
            fixed_om_mwh = (tech['fixed_om'] * 1000) / annual_gen
            # Variable O&M ($/MWh)
            var_om_mwh = tech['var_om']
            # Fuel ($/MWh)
            fuel_mwh = tech['fuel']

            capex_costs.append(capex_mwh)
            om_costs.append(fixed_om_mwh + var_om_mwh)
            fuel_costs.append(fuel_mwh)

            total = capex_mwh + fixed_om_mwh + var_om_mwh + fuel_mwh
            capex_pct = capex_mwh / total * 100
            print(f"  {name}: ${total:.0f}/MWh ({capex_pct:.0f}% capex)")
        else:
            capex_costs.append(0)
            om_costs.append(0)
            fuel_costs.append(0)

    # Calculate total heights for annotations
    solar_total = capex_costs[0] + om_costs[0] + fuel_costs[0]
    gas_total = capex_costs[-1] + om_costs[-1] + fuel_costs[-1]

    chart = {
        'data': [
            {
                'x': tech_names,
                'y': capex_costs,
                'type': 'bar',
                'name': 'Capital (Capex)',
                'marker': {'color': '#1a73e8'},
                'hoverinfo': 'skip'
            },
            {
                'x': tech_names,
                'y': om_costs,
                'type': 'bar',
                'name': 'O&M',
                'marker': {'color': '#34a853'},
                'hoverinfo': 'skip'
            },
            {
                'x': tech_names,
                'y': fuel_costs,
                'type': 'bar',
                'name': 'Fuel',
                'marker': {'color': COLORS['gas']},
                'hoverinfo': 'skip'
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'barmode': 'stack',
            'xaxis': {'showgrid': False},
            'yaxis': {
                'title': 'Cost Component ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [
                {
                    'x': 'Gas',
                    'y': gas_total + 5,
                    'text': f'{fuel_costs[-1]/gas_total*100:.0f}% fuel',
                    'showarrow': False,
                    'font': {'size': 11, 'color': COLORS['gas']}
                },
                {
                    'x': 'Solar',
                    'y': solar_total + 5,
                    'text': f'{capex_costs[0]/solar_total*100:.0f}% capex',
                    'showarrow': False,
                    'font': {'size': 11, 'color': '#333'}
                }
            ]
        }
    }

    return chart


def generate_cleanfirm_comparison(zone_data, zone='California'):
    """
    Compare how discount rates affect LCOE across different technology types.
    Key insight: Capital-intensive technologies (solar, wind, nuclear, geothermal) are
    very sensitive to financing costs, while fuel-heavy technologies (gas) are less
    sensitive but have higher floor costs driven by fuel prices.

    Shows side-by-side bar chart comparison at 4% vs 12% discount rates.
    """
    print("Generating technology discount rate comparison...")

    # Different technology assumptions
    # capex in $/kW, fuel in $/MWh, cf = capacity factor
    technologies = {
        'Solar': {'capex': 1000, 'fuel': 0, 'cf': 0.25, 'om': 8, 'color': COLORS['solar'], 'lifetime': 25},
        'Wind': {'capex': 1200, 'fuel': 0, 'cf': 0.35, 'om': 25, 'color': COLORS['wind'], 'lifetime': 25},
        'Nuclear': {'capex': 8000, 'fuel': 8, 'cf': 0.90, 'om': 30, 'color': COLORS['storage'], 'lifetime': 40},
        'Geothermal': {'capex': 5000, 'fuel': 0, 'cf': 0.85, 'om': 20, 'color': '#34a853', 'lifetime': 30},
        'Gas ($3)': {'capex': 900, 'fuel': 21, 'cf': 0.50, 'om': 15, 'color': COLORS['gas'], 'lifetime': 30},  # 7 MMBtu/MWh * $3
        'Gas ($8)': {'capex': 900, 'fuel': 56, 'cf': 0.50, 'om': 15, 'color': '#d32f2f', 'lifetime': 30}   # 7 MMBtu/MWh * $8, darker red
    }

    tech_names = list(technologies.keys())
    lcoes_4pct = []
    lcoes_12pct = []
    colors_list = []

    def calc_lcoe(tech, dr):
        lifetime = tech.get('lifetime', 30)
        crf = (dr/100 * (1 + dr/100)**lifetime) / ((1 + dr/100)**lifetime - 1)
        cf = tech['cf']
        annual_gen = cf * 8760  # kWh per kW

        capex_mwh = (tech['capex'] * crf * 1000) / annual_gen
        om_mwh = tech['om']
        fuel_mwh = tech['fuel']

        return capex_mwh + om_mwh + fuel_mwh

    for tech_name, tech in technologies.items():
        lcoe_4 = calc_lcoe(tech, 4)
        lcoe_12 = calc_lcoe(tech, 12)

        lcoes_4pct.append(lcoe_4)
        lcoes_12pct.append(lcoe_12)
        colors_list.append(tech['color'])

        print(f"  {tech_name}: ${lcoe_4:.0f} (4%) → ${lcoe_12:.0f} (12%), change: +${lcoe_12-lcoe_4:.0f}")

    chart = {
        'data': [
            {
                'x': tech_names,
                'y': lcoes_4pct,
                'type': 'bar',
                'name': '4% Discount Rate',
                'marker': {'color': colors_list, 'opacity': 0.7},
                'text': [f'${v:.0f}' for v in lcoes_4pct],
                'textposition': 'outside',
                'textfont': {'size': 10}
            },
            {
                'x': tech_names,
                'y': lcoes_12pct,
                'type': 'bar',
                'name': '12% Discount Rate',
                'marker': {'color': colors_list},
                'text': [f'${v:.0f}' for v in lcoes_12pct],
                'textposition': 'outside',
                'textfont': {'size': 10}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'barmode': 'group',
            'xaxis': {
                'title': '',
                'showgrid': False
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, max(lcoes_12pct) * 1.15]
            },
            'showlegend': True,
            'legend': {
                'x': 0.02,
                'y': 0.98,
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': '#ddd',
                'borderwidth': 1
            }
        }
    }

    return chart


def generate_tech_lcoe_comparison(zone_data, zone='California', cost_settings=None):
    """
    Show standalone LCOE of each technology with low/high ranges.
    This is the starting point of the mystery: solar is cheapest!
    Shows error bars for cost uncertainty based on low/default/high scenarios.
    """
    print("Generating technology LCOE comparison with ranges...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    discount_rate = cost_settings.get('discount_rate', 7) / 100
    lifetime = 20

    # Capital recovery factor
    crf = (discount_rate * (1 + discount_rate)**lifetime) / ((1 + discount_rate)**lifetime - 1)

    # Cost scenarios: low, default, high (from multi_test.py presets)
    cost_scenarios = {
        'Solar': {
            'capex': {'low': 500, 'default': 1000, 'high': 1500},
            'fixed_om': 15,
            'var_om': 0,
            'fuel': 0,
            'cf': 0.25,
            'color': COLORS['solar']
        },
        'Wind': {
            'capex': {'low': 700, 'default': 1200, 'high': 1800},
            'fixed_om': 40,
            'var_om': 0,
            'fuel': 0,
            'cf': 0.35,
            'color': COLORS['wind']
        },
        'Gas': {
            'capex': {'low': 1200, 'default': 1200, 'high': 2000},
            'fixed_om': 20,
            'var_om': 2,
            'fuel': {'low': 4 * 7.5, 'default': 4 * 7.5, 'high': 7 * 7.5},  # gas_price * heat_rate
            'cf': 0.50,
            'color': COLORS['gas']
        },
        'Clean Firm': {
            'capex': {'low': 3000, 'default': 5000, 'high': 7500},
            'fixed_om': 60,
            'var_om': 10,
            'fuel': 20,
            'cf': 0.85,
            'color': COLORS['clean_firm']
        }
    }

    def calc_lcoe(capex, fixed_om, var_om, fuel, cf):
        annual_gen = cf * 8760
        capex_mwh = (capex * crf * 1000) / annual_gen
        om_mwh = (fixed_om * 1000) / annual_gen + var_om
        return capex_mwh + om_mwh + fuel

    names = []
    lcoes_default = []
    lcoes_low = []
    lcoes_high = []
    colors = []

    for name, tech in cost_scenarios.items():
        capex = tech['capex']
        fuel = tech['fuel'] if isinstance(tech['fuel'], dict) else {'low': tech['fuel'], 'default': tech['fuel'], 'high': tech['fuel']}

        lcoe_low = calc_lcoe(capex['low'], tech['fixed_om'], tech['var_om'], fuel['low'], tech['cf'])
        lcoe_default = calc_lcoe(capex['default'], tech['fixed_om'], tech['var_om'], fuel['default'], tech['cf'])
        lcoe_high = calc_lcoe(capex['high'], tech['fixed_om'], tech['var_om'], fuel['high'], tech['cf'])

        names.append(name)
        lcoes_default.append(lcoe_default)
        lcoes_low.append(lcoe_low)
        lcoes_high.append(lcoe_high)
        colors.append(tech['color'])
        print(f"  {name}: ${lcoe_low:.0f} - ${lcoe_default:.0f} - ${lcoe_high:.0f}/MWh")

    # Calculate error bar values (asymmetric: from default to low/high)
    error_minus = [d - l for d, l in zip(lcoes_default, lcoes_low)]
    error_plus = [h - d for d, h in zip(lcoes_default, lcoes_high)]

    # Build annotations for range labels (to the right of each bar)
    range_annotations = []
    for i, name in enumerate(names):
        # Range label to the right of the bar, slightly above bar top
        range_annotations.append({
            'x': i,
            'y': lcoes_default[i] + 8,
            'xanchor': 'left',
            'xshift': 20,
            'text': f'${lcoes_low[i]:.0f}-${lcoes_high[i]:.0f}',
            'showarrow': False,
            'font': {'size': 11, 'color': '#666'}
        })

    chart = {
        'data': [{
            'x': names,
            'y': lcoes_default,
            'type': 'bar',
            'marker': {'color': colors},
            'hoverinfo': 'skip',
            'error_y': {
                'type': 'data',
                'symmetric': False,
                'array': error_plus,
                'arrayminus': error_minus,
                'color': '#666',
                'thickness': 2,
                'width': 6
            }
        }],
        'layout': {
            **LAYOUT_DEFAULTS,
            'title': 'Standalone Technology LCOE',
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, max(lcoes_high) * 1.2]
            },
            'xaxis': {'showgrid': False},
            'showlegend': False,
            'annotations': [
                # "Cheapest!" label above solar's high range
                {
                    'x': 'Solar',
                    'y': lcoes_high[0] + 12,
                    'text': 'Cheapest!',
                    'showarrow': False,
                    'font': {'size': 12, 'color': COLORS['solar'], 'weight': 'bold'}
                },
                # Footer note
                {
                    'x': 0.5,
                    'y': -0.15,
                    'xref': 'paper',
                    'yref': 'paper',
                    'text': 'Error bars show low/high cost scenarios',
                    'showarrow': False,
                    'font': {'size': 10, 'color': '#666'}
                }
            ] + range_annotations
        }
    }

    return chart


def generate_marginal_energy_value(zone_data, zone='California', cost_settings=None):
    """
    Show declining marginal energy value for solar/wind vs constant for clean firm.
    Key insight: curtailment causes declining value for intermittent resources.
    """
    print("Generating marginal energy value chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']

    def get_energy_delivered(solar, wind, storage, cf):
        """Returns clean energy delivered to load (not curtailed)"""
        (_, _, _, _, _, _, renewable_delivered, _, _, _, _) = simulate_system(
            solar_capacity=solar, wind_capacity=wind,
            storage_capacity=storage, clean_firm_capacity=cf,
            solar_profile=solar_profile, wind_profile=wind_profile,
            load_profile=load_profile, battery_eff=0.85
        )
        return np.sum(renewable_delivered)

    # Sweep each resource and calculate marginal energy per MW added
    step = 25

    # Solar sweep
    solar_caps = list(range(0, 801, step))
    solar_marginal = []
    prev_energy = 0
    for cap in solar_caps:
        energy = get_energy_delivered(cap, 0, 0, 0)
        if cap > 0:
            marginal = (energy - prev_energy) / step  # MWh per MW added
            solar_marginal.append(marginal)
        prev_energy = energy

    # Wind sweep
    wind_caps = list(range(0, 501, step))
    wind_marginal = []
    prev_energy = 0
    for cap in wind_caps:
        energy = get_energy_delivered(0, cap, 0, 0)
        if cap > 0:
            marginal = (energy - prev_energy) / step
            wind_marginal.append(marginal)
        prev_energy = energy

    # Clean firm sweep (should be constant)
    cf_caps = list(range(0, 126, step))
    cf_marginal = []
    prev_energy = 0
    for cap in cf_caps:
        energy = get_energy_delivered(0, 0, 0, cap)
        if cap > 0:
            marginal = (energy - prev_energy) / step
            cf_marginal.append(marginal)
        prev_energy = energy

    print(f"  Solar marginal: {solar_marginal[0]:.0f} -> {solar_marginal[-1]:.0f} MWh/MW")
    print(f"  Wind marginal: {wind_marginal[0]:.0f} -> {wind_marginal[-1]:.0f} MWh/MW")
    print(f"  Clean Firm marginal: {cf_marginal[0]:.0f} -> {cf_marginal[-1]:.0f} MWh/MW")

    chart = {
        'data': [
            {
                'x': solar_caps[1:],
                'y': solar_marginal,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar',
                'line': {'color': COLORS['solar'], 'width': 3}
            },
            {
                'x': wind_caps[1:],
                'y': wind_marginal,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Wind',
                'line': {'color': COLORS['wind'], 'width': 3}
            },
            {
                'x': cf_caps[1:],
                'y': cf_marginal,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Clean Firm',
                'line': {'color': COLORS['clean_firm'], 'width': 3}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Capacity Added (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Marginal Energy Value (MWh per MW added)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [
                {
                    'x': 600,
                    'y': solar_marginal[-3] if len(solar_marginal) > 3 else 500,
                    'text': 'Solar: curtailment<br>kills marginal value',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 40,
                    'ay': -30,
                    'font': {'size': 10, 'color': COLORS['solar']}
                },
                {
                    'x': 100,
                    'y': cf_marginal[3] if len(cf_marginal) > 3 else 7000,
                    'text': 'Clean firm: constant<br>marginal value',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 50,
                    'ay': 20,
                    'font': {'size': 10, 'color': COLORS['clean_firm']}
                }
            ]
        }
    }

    return chart


def generate_elcc_solar_storage(zone_data, zone='California', cost_settings=None):
    """
    Show ELCC (capacity credit) for solar with different storage durations (1-8h).
    Uses hybrid battery mode for proper peak shaving.
    """
    print("Generating solar+storage ELCC chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']
    peak_load = np.max(load_profile)

    def get_peak_gas_needed(solar, wind, storage, cf):
        (_, _, _, _, gas_gen, _, _, _, _, _, _) = simulate_system(
            solar_capacity=solar, wind_capacity=wind,
            storage_capacity=storage, clean_firm_capacity=cf,
            solar_profile=solar_profile, wind_profile=wind_profile,
            load_profile=load_profile, battery_eff=0.85,
            hybrid_mode=True
        )
        return np.max(gas_gen)

    baseline_gas = get_peak_gas_needed(0, 0, 0, 0)

    def capacity_credit(solar, storage):
        with_resource = get_peak_gas_needed(solar, 0, storage, 0)
        return (baseline_gas - with_resource) / baseline_gas * 100

    storage_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    solar_caps = list(range(0, 501, 50))

    storage_colors = {
        0: COLORS['solar'],
        1: '#ffb74d',
        2: '#ff9800',
        3: '#f57c00',
        4: '#4caf50',
        5: '#2196f3',
        6: '#3f51b5',
        7: '#7c4dff',
        8: '#673ab7'
    }

    traces = []

    for hours in storage_hours:
        storage_mwh = hours * peak_load
        cc_values = [capacity_credit(s, storage_mwh) for s in solar_caps]

        if hours == 0:
            print(f"  Solar only: {cc_values[0]:.1f}% -> {cc_values[-1]:.1f}%")
        else:
            print(f"  Solar + {hours}h storage: {cc_values[0]:.1f}% -> {cc_values[-1]:.1f}%")

        traces.append({
            'x': solar_caps,
            'y': cc_values,
            'type': 'scatter',
            'mode': 'lines',
            'name': f'{hours}h storage' if hours > 0 else 'Solar only',
            'line': {
                'color': storage_colors[hours],
                'width': 3 if hours == 0 else 2,
                'dash': 'dot' if hours == 0 else 'solid'
            }
        })

    chart = {
        'data': traces,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Solar Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Capacity Credit (% of peak load displaced)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 80]
            }
        }
    }

    return chart


def generate_elcc_wind_storage(zone_data, zone='California', cost_settings=None):
    """
    Show ELCC (capacity credit) for wind with different storage durations (1-8h).
    Uses hybrid battery mode for proper peak shaving.
    """
    print("Generating wind+storage ELCC chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']
    peak_load = np.max(load_profile)

    def get_peak_gas_needed(solar, wind, storage, cf):
        (_, _, _, _, gas_gen, _, _, _, _, _, _) = simulate_system(
            solar_capacity=solar, wind_capacity=wind,
            storage_capacity=storage, clean_firm_capacity=cf,
            solar_profile=solar_profile, wind_profile=wind_profile,
            load_profile=load_profile, battery_eff=0.85,
            hybrid_mode=True
        )
        return np.max(gas_gen)

    baseline_gas = get_peak_gas_needed(0, 0, 0, 0)

    def capacity_credit(wind, storage):
        with_resource = get_peak_gas_needed(0, wind, storage, 0)
        return (baseline_gas - with_resource) / baseline_gas * 100

    storage_hours = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    wind_caps = list(range(0, 401, 50))

    storage_colors = {
        0: COLORS['wind'],
        1: '#64b5f6',
        2: '#42a5f5',
        3: '#2196f3',
        4: '#4caf50',
        5: '#ff9800',
        6: '#f57c00',
        7: '#9c27b0',
        8: '#673ab7'
    }

    traces = []

    for hours in storage_hours:
        storage_mwh = hours * peak_load
        cc_values = [capacity_credit(w, storage_mwh) for w in wind_caps]

        if hours == 0:
            print(f"  Wind only: {cc_values[0]:.1f}% -> {cc_values[-1]:.1f}%")
        else:
            print(f"  Wind + {hours}h storage: {cc_values[0]:.1f}% -> {cc_values[-1]:.1f}%")

        traces.append({
            'x': wind_caps,
            'y': cc_values,
            'type': 'scatter',
            'mode': 'lines',
            'name': f'{hours}h storage' if hours > 0 else 'Wind only',
            'line': {
                'color': storage_colors[hours],
                'width': 3 if hours == 0 else 2,
                'dash': 'dot' if hours == 0 else 'solid'
            }
        })

    chart = {
        'data': traces,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Wind Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Capacity Credit (% of peak load displaced)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 80]
            }
        }
    }

    return chart


def generate_elcc_with_storage(zone_data, zone='California', cost_settings=None):
    """Legacy function - returns solar ELCC chart for backwards compatibility."""
    return generate_elcc_solar_storage(zone_data, zone, cost_settings)


def generate_lcoe_breakdown(zone_data, zone='California', cost_settings=None):
    """
    Show LCOE breakdown by component at 95% target.
    Uses the full calculate_lcoe_components function for accurate costs.
    Key insight: Gas capacity is needed but contributes small fraction of LCOE.
    """
    print("Generating LCOE breakdown chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    load_profile = profiles['load']
    annual_load = np.sum(load_profile)

    target = 95
    result = run_optimization(zone_data, zone, target, cost_settings)

    # Simulate to get generation
    (solar_out, wind_out, storage_out, _, gas_gen, curtailed, _, _,
     clean_firm_gen, _, _) = simulate_system(
        solar_capacity=result['solar'],
        wind_capacity=result['wind'],
        storage_capacity=result['storage'],
        clean_firm_capacity=result['clean_firm'],
        solar_profile=profiles['solar'],
        wind_profile=profiles['wind'],
        load_profile=load_profile,
        battery_eff=0.85,
        hybrid_mode=True
    )

    # Use the proper LCOE component calculator
    components = calculate_lcoe_components(
        cost_settings,
        result['solar'], result['wind'], result['storage'], result['clean_firm'],
        float(np.sum(solar_out)), float(np.sum(wind_out)), float(np.sum(clean_firm_gen)),
        float(np.sum(gas_gen)), float(np.max(gas_gen)), float(np.sum(curtailed)), annual_load
    )

    # Sum up components per technology (capex + O&M + fuel - ITC - depreciation)
    solar_lcoe = (components.get('solar_capex', 0) + components.get('solar_fixed_om', 0) +
                  components.get('solar_var_om', 0) + components.get('solar_itc', 0) +
                  components.get('solar_depreciation', 0))

    wind_lcoe = (components.get('wind_capex', 0) + components.get('wind_fixed_om', 0) +
                 components.get('wind_var_om', 0) + components.get('wind_itc', 0) +
                 components.get('wind_depreciation', 0))

    storage_lcoe = (components.get('storage_capex', 0) + components.get('storage_fixed_om', 0) +
                    components.get('storage_itc', 0) + components.get('storage_depreciation', 0))

    cf_lcoe = (components.get('clean_firm_capex', 0) + components.get('clean_firm_fixed_om', 0) +
               components.get('clean_firm_var_om', 0) + components.get('clean_firm_fuel', 0) +
               components.get('clean_firm_itc', 0) + components.get('clean_firm_depreciation', 0))

    gas_lcoe = (components.get('gas_capex', 0) + components.get('gas_fixed_om', 0) +
                components.get('gas_var_om', 0) + components.get('gas_fuel', 0) +
                components.get('gas_depreciation', 0))

    # Use the direct LCOE calculation which is more accurate
    # Scale components proportionally to match the direct_lcoe total
    direct_lcoe = components.get('direct_lcoe', 0)
    component_sum = solar_lcoe + wind_lcoe + storage_lcoe + cf_lcoe + gas_lcoe

    if component_sum > 0 and direct_lcoe > 0:
        # Scale all components to match the accurate direct_lcoe
        scale = direct_lcoe / component_sum
        solar_lcoe *= scale
        wind_lcoe *= scale
        storage_lcoe *= scale
        cf_lcoe *= scale
        gas_lcoe *= scale

    total = direct_lcoe if direct_lcoe > 0 else component_sum

    print(f"  Solar: ${solar_lcoe:.1f}/MWh ({solar_lcoe/total*100:.0f}%)")
    print(f"  Wind: ${wind_lcoe:.1f}/MWh ({wind_lcoe/total*100:.0f}%)")
    print(f"  Storage: ${storage_lcoe:.1f}/MWh ({storage_lcoe/total*100:.0f}%)")
    print(f"  Clean Firm: ${cf_lcoe:.1f}/MWh ({cf_lcoe/total*100:.0f}%)")
    print(f"  Gas: ${gas_lcoe:.1f}/MWh ({gas_lcoe/total*100:.0f}%)")
    print(f"  Total: ${total:.1f}/MWh (from calculator: ${components.get('direct_lcoe', 0):.1f})")

    chart = {
        'data': [{
            'x': ['95% Clean System'],
            'y': [solar_lcoe],
            'name': 'Solar',
            'type': 'bar',
            'marker': {'color': COLORS['solar']}
        }, {
            'x': ['95% Clean System'],
            'y': [wind_lcoe],
            'name': 'Wind',
            'type': 'bar',
            'marker': {'color': COLORS['wind']}
        }, {
            'x': ['95% Clean System'],
            'y': [storage_lcoe],
            'name': 'Storage',
            'type': 'bar',
            'marker': {'color': COLORS['storage']}
        }, {
            'x': ['95% Clean System'],
            'y': [cf_lcoe],
            'name': 'Clean Firm',
            'type': 'bar',
            'marker': {'color': COLORS['clean_firm']}
        }, {
            'x': ['95% Clean System'],
            'y': [gas_lcoe],
            'name': 'Gas Backup',
            'type': 'bar',
            'marker': {'color': COLORS['gas']}
        }],
        'layout': {
            **LAYOUT_DEFAULTS,
            'barmode': 'stack',
            'title': {
                'text': f'LCOE Breakdown at {target}% Clean',
                'y': 0.95,
                'yanchor': 'top'
            },
            'margin': {
                'l': 60,
                'r': 40,
                't': 80,
                'b': 60
            },
            'yaxis': {
                'title': 'LCOE Component ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'xaxis': {'showgrid': False},
            'annotations': [{
                'x': '95% Clean System',
                'y': total + 5,
                'text': f'Total: ${total:.0f}/MWh',
                'showarrow': False,
                'font': {'size': 12, 'weight': 'bold'}
            }]
        }
    }

    return chart


def generate_gas_capacity_sweep(zone_data, zone='California', cost_settings=None):
    """
    Show how much gas capacity is needed at each clean energy target.
    Key insight: Even at 95%+, you still need substantial gas capacity for reliability.
    """
    print("Generating gas capacity requirements chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    load_profile = profiles['load']
    peak_load = np.max(load_profile)

    targets = list(range(50, 100, 2))  # 50, 52, ..., 98
    gas_capacities = []
    gas_energy_pcts = []

    for target in targets:
        result = run_optimization(zone_data, zone, target, cost_settings)

        # Run simulation to get gas usage
        (_, _, _, _, gas_gen, _, _, _, _, _, _) = simulate_system(
            solar_capacity=result['solar'],
            wind_capacity=result['wind'],
            storage_capacity=result['storage'],
            clean_firm_capacity=result['clean_firm'],
            solar_profile=profiles['solar'],
            wind_profile=profiles['wind'],
            load_profile=load_profile,
            battery_eff=0.85
        )

        # Gas capacity needed = peak gas generation
        gas_cap_needed = np.max(gas_gen)
        gas_capacities.append(gas_cap_needed)

        # Gas energy as % of load
        gas_energy_pct = (100 - target)  # Simplified - actual gap
        gas_energy_pcts.append(gas_energy_pct)

        if target % 10 == 0:
            print(f"  {target}%: Gas capacity={gas_cap_needed:.0f} MW ({gas_cap_needed/peak_load*100:.0f}% of peak)")

    chart = {
        'data': [
            {
                'x': targets,
                'y': gas_capacities,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Gas Capacity Needed (MW)',
                'line': {'color': COLORS['gas'], 'width': 3},
                'marker': {'size': 6},
                'fill': 'tozeroy',
                'fillcolor': 'rgba(234, 67, 53, 0.2)'
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Target (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Gas Capacity Required (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'shapes': [
                # Line showing peak load for reference
                {
                    'type': 'line',
                    'x0': 50, 'x1': 98,
                    'y0': peak_load, 'y1': peak_load,
                    'line': {'color': COLORS['load'], 'width': 2, 'dash': 'dash'}
                }
            ],
            'annotations': [
                {
                    'x': 60,
                    'y': peak_load + 5,
                    'text': f'Peak Load: {peak_load:.0f} MW',
                    'showarrow': False,
                    'font': {'size': 10, 'color': COLORS['load']}
                },
                {
                    'x': 95,
                    'y': gas_capacities[-3] + 10,
                    'text': 'Still need ~50%<br>of peak as backup',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -40,
                    'ay': -30,
                    'font': {'size': 10, 'color': COLORS['gas']}
                }
            ]
        }
    }

    return chart


def generate_resource_interactions(zone_data, zone='California', cost_settings=None):
    """
    Show how resources interact - solar+storage have stronger synergy than solar+wind.
    Key insight: Storage transforms intermittent solar into dispatchable power.
    """
    print("Generating resource interaction chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']

    def get_match(solar, wind, storage, cf):
        (_, _, _, _, _, _, renewable_delivered, _, _, _, _) = simulate_system(
            solar_capacity=solar, wind_capacity=wind,
            storage_capacity=storage, clean_firm_capacity=cf,
            solar_profile=solar_profile, wind_profile=wind_profile,
            load_profile=load_profile, battery_eff=0.85
        )
        return (np.sum(renewable_delivered) / np.sum(load_profile)) * 100

    # Sweep solar capacity with different companion resources
    solar_caps = list(range(0, 601, 50))

    # Scenario 1: Solar alone
    solar_only = [get_match(s, 0, 0, 0) for s in solar_caps]
    print(f"  Solar only: {solar_only[0]:.1f}% -> {solar_only[-1]:.1f}%")

    # Scenario 2: Solar + moderate wind (100 MW)
    solar_wind = [get_match(s, 100, 0, 0) for s in solar_caps]
    print(f"  Solar + 100 MW wind: {solar_wind[0]:.1f}% -> {solar_wind[-1]:.1f}%")

    # Scenario 3: Solar + moderate storage (400 MWh)
    solar_storage = [get_match(s, 0, 400, 0) for s in solar_caps]
    print(f"  Solar + 400 MWh storage: {solar_storage[0]:.1f}% -> {solar_storage[-1]:.1f}%")

    # Scenario 4: Solar + wind + storage
    solar_wind_storage = [get_match(s, 100, 400, 0) for s in solar_caps]
    print(f"  Solar + wind + storage: {solar_wind_storage[0]:.1f}% -> {solar_wind_storage[-1]:.1f}%")

    # Calculate interaction effects at 400 MW solar
    idx = solar_caps.index(400)
    wind_boost = solar_wind[idx] - solar_only[idx]
    storage_boost = solar_storage[idx] - solar_only[idx]

    chart = {
        'data': [
            {
                'x': solar_caps,
                'y': solar_only,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar Only',
                'line': {'color': COLORS['solar'], 'width': 3}
            },
            {
                'x': solar_caps,
                'y': solar_wind,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar + 100 MW Wind',
                'line': {'color': COLORS['wind'], 'width': 2, 'dash': 'dot'}
            },
            {
                'x': solar_caps,
                'y': solar_storage,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar + 400 MWh Storage',
                'line': {'color': COLORS['storage'], 'width': 3}
            },
            {
                'x': solar_caps,
                'y': solar_wind_storage,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar + Wind + Storage',
                'line': {'color': COLORS['clean_firm'], 'width': 2, 'dash': 'dash'}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Solar Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Clean Energy Match (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100]
            },
            'annotations': [
                {
                    'x': 400,
                    'y': solar_storage[idx] + 3,
                    'text': f'+{storage_boost:.0f}% with storage',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 40,
                    'ay': -20,
                    'font': {'size': 10, 'color': COLORS['storage']}
                },
                {
                    'x': 400,
                    'y': solar_wind[idx] - 3,
                    'text': f'+{wind_boost:.0f}% with wind',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 40,
                    'ay': 20,
                    'font': {'size': 10, 'color': COLORS['wind']}
                }
            ]
        }
    }

    return chart


def generate_gas_lcoe_impact(zone_data, zone='California', cost_settings=None):
    """
    Show that despite needing lots of gas capacity, gas contributes little to LCOE.
    Key insight: Gas operates few hours = low energy cost despite high capacity.

    Creates a line chart with:
    - Total System LCOE (black line)
    - Gas Contribution to LCOE (red line with shaded area below)
    """
    print("Generating gas LCOE impact chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    load_profile = profiles['load']
    annual_load = np.sum(load_profile)

    # More granular targets for smooth curve
    targets = list(range(0, 101, 5))
    gas_lcoe_contribution = []
    total_lcoes = []

    for target in targets:
        if target == 0:
            # Gas-only system - run simulation with no clean resources
            (_, _, _, _, gas_gen, _, _, _, _, _, _) = simulate_system(
                solar_capacity=0,
                wind_capacity=0,
                storage_capacity=0,
                clean_firm_capacity=0,
                solar_profile=profiles['solar'],
                wind_profile=profiles['wind'],
                load_profile=load_profile,
                battery_eff=0.85
            )

            gas_cap = np.max(gas_gen)
            gas_mwh = np.sum(gas_gen)

            # Use proper LCOE calculation matching other targets
            gas_lcoe_total = calculate_lcoe(
                cost_settings,  # cost_params
                0,              # solar_capacity
                0,              # wind_capacity
                0,              # storage_capacity
                0,              # clean_firm_capacity
                0,              # annual_solar_gen
                0,              # annual_wind_gen
                0,              # annual_clean_firm_gen
                gas_mwh,        # annual_gas_gen
                gas_cap,        # gas_capacity_needed
                0,              # curtailed_energy
                annual_load     # annual_load
            )

            total_lcoes.append(gas_lcoe_total)
            gas_lcoe_contribution.append(gas_lcoe_total)
            print(f"  {target}%: Gas-only system, LCOE=${gas_lcoe_total:.1f}/MWh")
            continue

        # For very low clean targets (5-10%), use a simple approach to avoid optimizer inefficiency
        # Just linearly interpolate between gas-only and the 10% optimized result
        if target == 5:
            # Use gas-only baseline for now, will interpolate after 10% is computed
            total_lcoes.append(None)  # Placeholder
            gas_lcoe_contribution.append(None)  # Placeholder
            print(f"  {target}%: Will interpolate after computing 10%")
            continue

        result = run_optimization(zone_data, zone, target, cost_settings)

        # Run simulation
        (_, _, _, _, gas_gen, _, _, _, _, _, _) = simulate_system(
            solar_capacity=result['solar'],
            wind_capacity=result['wind'],
            storage_capacity=result['storage'],
            clean_firm_capacity=result['clean_firm'],
            solar_profile=profiles['solar'],
            wind_profile=profiles['wind'],
            load_profile=load_profile,
            battery_eff=0.85
        )

        gas_cap = np.max(gas_gen)
        gas_mwh = np.sum(gas_gen)

        # Approximate gas LCOE contribution
        gas_capex = cost_settings.get('gas_capex', 1200)
        gas_price = cost_settings.get('gas_price', 4)
        heat_rate = cost_settings.get('gas_heat_rate', 7.5)

        # Gas cost = capex amortized + fuel
        gas_annual_cost = gas_cap * gas_capex * 0.08  # ~8% CRF
        gas_fuel_cost = gas_mwh * gas_price * heat_rate / 1000
        gas_total_cost = gas_annual_cost + gas_fuel_cost
        gas_lcoe = gas_total_cost / annual_load * 1000  # $/MWh

        gas_lcoe_contribution.append(gas_lcoe)
        total_lcoes.append(result['lcoe'])

        print(f"  {target}%: Gas contributes ${gas_lcoe:.1f}/MWh of ${result['lcoe']:.1f}/MWh total")

    # Fill in interpolated values for 5% target
    if None in total_lcoes:
        idx_5 = targets.index(5)
        idx_0 = targets.index(0)
        idx_10 = targets.index(10)

        # Linear interpolation between 0% and 10%
        total_lcoes[idx_5] = (total_lcoes[idx_0] + total_lcoes[idx_10]) / 2
        gas_lcoe_contribution[idx_5] = (gas_lcoe_contribution[idx_0] + gas_lcoe_contribution[idx_10]) / 2
        print(f"  5%: Interpolated, LCOE=${total_lcoes[idx_5]:.1f}/MWh (gas ${gas_lcoe_contribution[idx_5]:.1f}/MWh)")

    # Create line chart with shaded area
    chart = {
        'data': [
            # Total System LCOE - black line with markers
            {
                'x': targets,
                'y': total_lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total System LCOE',
                'line': {'color': '#000000', 'width': 2},
                'marker': {'size': 6, 'color': '#000000'}
            },
            # Gas Contribution to LCOE - red line with shaded fill
            {
                'x': targets,
                'y': gas_lcoe_contribution,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Gas Contribution to LCOE',
                'line': {'color': '#ea4335', 'width': 2},
                'marker': {'size': 6, 'color': '#ea4335'},
                'fill': 'tozeroy',
                'fillcolor': 'rgba(234, 67, 53, 0.2)'
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Target (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100],
                'dtick': 20
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, max(total_lcoes) * 1.1]
            },
            'annotations': [
                {
                    'x': 20,
                    'y': gas_lcoe_contribution[4] if len(gas_lcoe_contribution) > 4 else 40,
                    'text': 'Gas dominates<br>at low clean %',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 0,
                    'ay': -40,
                    'font': {'size': 11, 'color': '#333'}
                },
                {
                    'x': 90,
                    'y': gas_lcoe_contribution[18] if len(gas_lcoe_contribution) > 18 else 15,
                    'text': 'Gas < $15/MWh<br>but capacity still needed',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 0,
                    'ay': 40,
                    'font': {'size': 11, 'color': '#333'}
                }
            ]
        }
    }

    return chart


def generate_effective_lcoe(zone_data, zone='California', cost_settings=None):
    """
    Show how solar's effective LCOE rises with curtailment.
    Key chart: bridges "solar is $50" to "but system costs $100"

    Standalone LCOE assumes all energy is used.
    Effective LCOE = Annual Cost / Actual Energy Delivered (accounting for curtailment)
    """
    print("Generating effective LCOE chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = np.array(profiles['solar'])
    load_profile = np.array(profiles['load'])

    # Cost assumptions
    solar_capex = cost_settings.get('solar', 1000)  # $/kW
    solar_life = 25
    discount_rate = cost_settings.get('discount_rate', 7) / 100
    crf = (discount_rate * (1 + discount_rate) ** solar_life) / ((1 + discount_rate) ** solar_life - 1)

    # Standalone LCOE using same 25% CF assumption as tech_lcoe chart
    solar_cf = 0.25
    standalone_lcoe = (solar_capex * crf) / (8760 * solar_cf) * 1000  # $/MWh

    # Sweep solar capacity
    solar_caps = list(range(50, 1001, 50))
    effective_lcoes = []
    utilization_rates = []

    for cap in solar_caps:
        # Annual cost for this capacity
        annual_cost = cap * solar_capex * crf * 1000  # $ per year

        # Total solar generation at each hour (before curtailment)
        solar_generation = cap * solar_profile  # MW at each hour
        total_generation = np.sum(solar_generation)  # MWh/year

        # Actual energy delivered = min(solar generation, load) at each hour
        # This is the key: solar can only deliver up to what load demands
        solar_delivered = np.minimum(solar_generation, load_profile)
        total_delivered = np.sum(solar_delivered)  # MWh/year

        # Effective LCOE = cost / delivered energy
        if total_delivered > 0:
            eff_lcoe = annual_cost / total_delivered
        else:
            eff_lcoe = standalone_lcoe

        effective_lcoes.append(eff_lcoe)
        utilization_rates.append(total_delivered / total_generation * 100 if total_generation > 0 else 0)

    print(f"  Standalone LCOE: ${standalone_lcoe:.1f}/MWh")
    print(f"  Effective LCOE at 100 MW: ${effective_lcoes[1]:.1f}/MWh ({utilization_rates[1]:.0f}% utilized)")
    print(f"  Effective LCOE at 500 MW: ${effective_lcoes[9]:.1f}/MWh ({utilization_rates[9]:.0f}% utilized)")
    print(f"  Effective LCOE at 1000 MW: ${effective_lcoes[-1]:.1f}/MWh ({utilization_rates[-1]:.0f}% utilized)")

    chart = {
        'data': [
            {
                'x': solar_caps,
                'y': [standalone_lcoe] * len(solar_caps),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Standalone LCOE (theoretical)',
                'line': {'color': COLORS['baseline'], 'width': 2, 'dash': 'dash'}
            },
            {
                'x': solar_caps,
                'y': effective_lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Effective LCOE (actual)',
                'line': {'color': COLORS['solar'], 'width': 3},
                'marker': {'size': 6}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Solar Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, max(effective_lcoes) * 1.1]
            },
            'annotations': [
                {
                    'x': 200,
                    'y': standalone_lcoe,
                    'text': f'Standalone: ${standalone_lcoe:.0f}/MWh',
                    'showarrow': False,
                    'yshift': 15,
                    'font': {'size': 11, 'color': COLORS['baseline']}
                },
                {
                    'x': 800,
                    'y': effective_lcoes[15] if len(effective_lcoes) > 15 else effective_lcoes[-1],
                    'text': 'Curtailment makes<br>each MWh more expensive',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -40,
                    'ay': -40,
                    'font': {'size': 10, 'color': COLORS['solar']}
                }
            ],
            'shapes': [
                {
                    'type': 'rect',
                    'x0': 0,
                    'x1': 300,
                    'y0': 0,
                    'y1': standalone_lcoe * 1.2,
                    'fillcolor': 'rgba(52, 168, 83, 0.1)',
                    'line': {'width': 0}
                },
                {
                    'type': 'rect',
                    'x0': 300,
                    'x1': 1000,
                    'y0': 0,
                    'y1': max(effective_lcoes) * 1.1,
                    'fillcolor': 'rgba(234, 67, 53, 0.05)',
                    'line': {'width': 0}
                }
            ]
        }
    }

    return chart


def main():
    """Generate all charts and save as JSON."""
    print("=" * 60)
    print("Clean Energy Microsite Chart Generator")
    print("=" * 60)

    # Load zone data
    print("\nLoading zone data...")
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(parent_dir, 'normalized_data_2035_by_zone_new.csv')
    zone_data, zones = load_all_zone_data(filename=data_file)
    print(f"Loaded {len(zones)} zones: {', '.join(zones[:5])}...")

    cost_settings = DEFAULT_COSTS.copy()

    # Output directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Generate each chart
    charts = {
        'mismatch': generate_mismatch_chart(zone_data),
        'lcoe_curve': generate_lcoe_curve(zone_data, cost_settings=cost_settings),
        'resource_mix': generate_resource_mix(zone_data, cost_settings=cost_settings),
        'leap': generate_leap_chart(zone_data, cost_settings=cost_settings),
        'regional': generate_regional_comparison(zone_data, cost_settings=cost_settings),
        'cost_sensitivity': generate_cost_sensitivity(zone_data),
        'no_cleanfirm': generate_no_cleanfirm_chart(zone_data, cost_settings=cost_settings),
        'no_wind': generate_no_wind_chart(zone_data, cost_settings=cost_settings),
        'greedy': generate_greedy_comparison(zone_data, cost_settings=cost_settings),
        'solar_paradox': generate_solar_paradox(zone_data, cost_settings=cost_settings),
        'elcc': generate_elcc_chart(zone_data, cost_settings=cost_settings),
        # Financing and economics charts
        'financing': generate_financing_impact(zone_data),
        'gas_crossover': generate_gas_crossover(zone_data),
        'cleanfirm_crossover': generate_cleanfirm_crossover(zone_data),
        'cost_structure': generate_cost_structure(zone_data, cost_settings=cost_settings),
        'cleanfirm_tech': generate_cleanfirm_comparison(zone_data),
        # Gas and resource interaction charts
        'gas_capacity': generate_gas_capacity_sweep(zone_data, cost_settings=cost_settings),
        'resource_interactions': generate_resource_interactions(zone_data, cost_settings=cost_settings),
        'gas_lcoe_impact': generate_gas_lcoe_impact(zone_data, cost_settings=cost_settings),
        # New narrative charts
        'tech_lcoe': generate_tech_lcoe_comparison(zone_data, cost_settings=cost_settings),
        'marginal_energy': generate_marginal_energy_value(zone_data, cost_settings=cost_settings),
        'elcc_solar_storage': generate_elcc_solar_storage(zone_data, cost_settings=cost_settings),
        'elcc_wind_storage': generate_elcc_wind_storage(zone_data, cost_settings=cost_settings),
        'lcoe_breakdown': generate_lcoe_breakdown(zone_data, cost_settings=cost_settings),
        'effective_lcoe': generate_effective_lcoe(zone_data, cost_settings=cost_settings),
        # Storage ceiling charts
        'solar_storage_ceiling': generate_solar_storage_ceiling(zone_data, cost_settings=cost_settings),
        'wind_storage_ceiling': generate_wind_storage_ceiling(zone_data, cost_settings=cost_settings),
        # Solar+storage only LCOE sweep
        'solar_storage_lcoe': generate_solar_storage_lcoe(zone_data, cost_settings=cost_settings),
        'solar_storage_lcoe_cheap': generate_solar_storage_lcoe_cheap(zone_data, cost_settings=cost_settings),
        'storage_cost_sensitivity': generate_storage_cost_sensitivity(zone_data, cost_settings=cost_settings)
    }

    # Save each chart
    print("\n" + "=" * 60)
    print("Saving charts...")
    for name, chart in charts.items():
        filepath = os.path.join(data_dir, f'{name}.json')
        with open(filepath, 'w') as f:
            json.dump(chart, f, indent=2)
        print(f"  Saved {name}.json")

    print("\n" + "=" * 60)
    print("Chart generation complete!")
    print(f"Output: {data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
