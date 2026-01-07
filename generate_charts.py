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
from lcoe_calculator import calculate_lcoe
from data_loader import load_all_zone_data
from cost_settings_modal import DEFAULT_COSTS

# Try to import V4 optimizer
try:
    from experimental.optimizer_v4 import run_min_lcoe_v4_adaptive
    HAS_V4 = True
except ImportError:
    HAS_V4 = False
    print("Warning: V4 optimizer not available, using simplified optimization")

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
                     use_storage=True, use_clean_firm=True):
    """Run optimization for given parameters, returns result dict."""
    if HAS_V4:
        result = run_min_lcoe_v4_adaptive(
            clean_match_target=target,
            zone_data={zone: zone_data[zone]},
            selected_zone=zone,
            cost_settings=cost_settings,
            use_solar=use_solar,
            use_wind=use_wind,
            use_storage=use_storage,
            use_clean_firm=use_clean_firm,
            hybrid_mode=True  # Use hybrid battery for smoother results
        )
        return result
    else:
        # Simplified fallback - just return approximate values
        return {
            'solar': 400 if use_solar else 0,
            'wind': 200 if use_wind else 0,
            'storage': 800 if use_storage else 0,
            'clean_firm': 50 if use_clean_firm and target > 85 else 0,
            'lcoe': 80 + (target - 70) * 0.5,
            'achieved_match': target
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

    targets = [50, 60, 70, 80, 85, 90, 92, 95, 97, 99]
    solar_energy = []
    wind_energy = []
    storage_energy = []  # MWh shifted (discharged)
    cf_energy = []
    cf_share = []  # Track CF share of total clean energy

    for target in targets:
        result = run_optimization(zone_data, zone, target, cost_settings)

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
                'y': storage_energy,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Storage (shifted)',
                'stackgroup': 'one',
                'fillcolor': COLORS['storage'],
                'line': {'color': COLORS['storage'], 'width': 0}
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
            'y': 0.95 if i == 0 else 0.45,
            'xref': 'paper',
            'yref': 'paper',
            'text': f"<b>{target}% Clean Target</b><br>Week {data['peak_week']+1} ({week_label}) - Max Gas: {data['max_gas']:.0f} MW",
            'showarrow': False,
            'font': {'size': 11},
            'align': 'left',
            'bgcolor': 'rgba(255,255,255,0.9)',
            'borderpad': 4
        })

    # Configure axes for two subplots
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
        'domain': [0.55, 1]
    }
    chart['layout']['yaxis2'] = {
        'title': 'Power (MW)',
        'showgrid': True,
        'gridcolor': 'rgba(0,0,0,0.1)',
        'domain': [0, 0.45]
    }

    return chart


def generate_regional_comparison(zone_data, cost_settings=None):
    """Generate regional comparison chart - stacked bars showing energy (GWh)."""
    print("Generating regional comparison (stacked energy)...")

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
            results[zone] = {
                'solar_gwh': np.sum(solar_out) / 1000,
                'wind_gwh': np.sum(wind_out) / 1000,
                'storage_gwh': np.sum(np.maximum(0, storage_out)) / 1000,
                'cf_gwh': np.sum(clean_firm_gen) / 1000,
                'lcoe': result['lcoe']
            }
            print(f"  {zone}: Solar={results[zone]['solar_gwh']:.0f}, Wind={results[zone]['wind_gwh']:.0f}, "
                  f"Storage={results[zone]['storage_gwh']:.0f}, CF={results[zone]['cf_gwh']:.0f} GWh, LCOE=${result['lcoe']:.1f}")

    zone_names = list(results.keys())

    # Calculate totals for annotation positioning
    totals = {z: results[z]['cf_gwh'] + results[z]['storage_gwh'] + results[z]['wind_gwh'] + results[z]['solar_gwh']
              for z in zone_names}

    chart = {
        'data': [
            {
                'x': zone_names,
                'y': [results[z]['cf_gwh'] for z in zone_names],
                'type': 'bar',
                'name': 'Clean Firm',
                'marker': {'color': COLORS['clean_firm']}
            },
            {
                'x': zone_names,
                'y': [results[z]['storage_gwh'] for z in zone_names],
                'type': 'bar',
                'name': 'Storage (shifted)',
                'marker': {'color': COLORS['storage']}
            },
            {
                'x': zone_names,
                'y': [results[z]['wind_gwh'] for z in zone_names],
                'type': 'bar',
                'name': 'Wind',
                'marker': {'color': COLORS['wind']}
            },
            {
                'x': zone_names,
                'y': [results[z]['solar_gwh'] for z in zone_names],
                'type': 'bar',
                'name': 'Solar',
                'marker': {'color': COLORS['solar']}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'barmode': 'stack',
            'title': f'{target}% Clean Energy Target',
            'xaxis': {'showgrid': False},
            'yaxis': {
                'title': 'Annual Energy (GWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [
                {
                    'x': zone,
                    'y': totals[zone] + 10,
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


def generate_no_wind_chart(zone_data, zone='California', cost_settings=None):
    """Generate resource availability comparison - shows 3 scenarios."""
    print("Generating no wind chart (3 scenarios)...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    targets = [70, 80, 85, 90, 95, 99]
    lcoe_solar_storage = []      # Solar + Storage only
    lcoe_solar_storage_cf = []   # Solar + Storage + Clean Firm
    lcoe_full_mix = []           # Solar + Storage + Clean Firm + Wind

    for target in targets:
        # Scenario 1: Solar + Storage only (no wind, no clean firm)
        result_ss = run_optimization(zone_data, zone, target, cost_settings,
                                     use_wind=False, use_clean_firm=False)
        lcoe_solar_storage.append(result_ss['lcoe'])

        # Scenario 2: Solar + Storage + Clean Firm (no wind)
        result_sscf = run_optimization(zone_data, zone, target, cost_settings,
                                       use_wind=False, use_clean_firm=True)
        lcoe_solar_storage_cf.append(result_sscf['lcoe'])

        # Scenario 3: Full mix (all resources)
        result_full = run_optimization(zone_data, zone, target, cost_settings,
                                       use_wind=True, use_clean_firm=True)
        lcoe_full_mix.append(result_full['lcoe'])

        print(f"  {target}%: S+S=${result_ss['lcoe']:.1f}, S+S+CF=${result_sscf['lcoe']:.1f}, "
              f"Full=${result_full['lcoe']:.1f}")

    chart = {
        'data': [
            {
                'x': targets,
                'y': lcoe_solar_storage,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Solar + Storage',
                'line': {'color': COLORS['solar'], 'width': 3},
                'marker': {'size': 8}
            },
            {
                'x': targets,
                'y': lcoe_solar_storage_cf,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Solar + Storage + Clean Firm',
                'line': {'color': COLORS['clean_firm'], 'width': 3},
                'marker': {'size': 8}
            },
            {
                'x': targets,
                'y': lcoe_full_mix,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Solar + Storage + Clean Firm + Wind',
                'line': {'color': COLORS['wind'], 'width': 3},
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
                    'y': lcoe_solar_storage[targets.index(95)] + 5,
                    'text': 'Solar + Storage<br>alone struggles',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -40,
                    'ay': -20,
                    'font': {'size': 10, 'color': COLORS['solar']}
                },
                {
                    'x': 90,
                    'y': lcoe_solar_storage_cf[targets.index(90)] - 5,
                    'text': 'Clean firm<br>helps a lot',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -40,
                    'ay': 25,
                    'font': {'size': 10, 'color': COLORS['clean_firm']}
                },
                {
                    'x': 85,
                    'y': lcoe_full_mix[targets.index(85)] - 5,
                    'text': 'Wind gives<br>final boost',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 30,
                    'ay': 25,
                    'font': {'size': 10, 'color': COLORS['wind']}
                }
            ]
        }
    }

    return chart


def run_greedy_optimization(zone_data, zone, target, cost_settings):
    """
    Run greedy optimization: iteratively add cheapest resource by LCOE impact.
    This represents naive planning that ignores system-level optimization.
    """
    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']

    # Resource limits and step sizes - use finer steps for smoother curves
    max_solar, max_wind, max_storage, max_cf = 1000, 500, 2400, 125
    step = 10  # Smaller steps for smoother optimization

    current = {'solar': 0, 'wind': 0, 'storage': 0, 'clean_firm': 0}

    def evaluate(config):
        (solar_out, wind_out, _, _, gas_gen, curtailed, renewable_delivered, _,
         clean_firm_gen, _, _) = simulate_system(
            solar_capacity=config['solar'],
            wind_capacity=config['wind'],
            storage_capacity=config['storage'],
            clean_firm_capacity=config['clean_firm'],
            solar_profile=solar_profile,
            wind_profile=wind_profile,
            load_profile=load_profile,
            battery_eff=0.85
        )
        annual_load = np.sum(load_profile)
        match = (np.sum(renewable_delivered) / annual_load) * 100
        lcoe = calculate_lcoe(
            cost_settings, config['solar'], config['wind'],
            config['storage'], config['clean_firm'],
            np.sum(solar_out), np.sum(wind_out), np.sum(clean_firm_gen),
            np.sum(gas_gen), np.max(gas_gen), np.sum(curtailed), annual_load
        )
        return lcoe, match

    current_lcoe, current_match = evaluate(current)

    # Greedy: always add the resource with best $/% match improvement
    # Key difference from optimal: no consideration of diminishing returns or system balance
    max_iterations = 300  # More iterations for finer steps
    stuck_count = 0

    for _ in range(max_iterations):
        if current_match >= target:
            break

        best_move = None
        best_efficiency = float('inf')

        # Try adding each resource
        resources = [
            ('solar', max_solar),
            ('wind', max_wind),
            ('storage', max_storage),
            # Greedy rarely picks clean firm because it's "expensive" per MW
            ('clean_firm', max_cf)
        ]

        for res, max_val in resources:
            if current[res] >= max_val:
                continue

            test = current.copy()
            test[res] = min(max_val, test[res] + step)
            test_lcoe, test_match = evaluate(test)

            if test_match > current_match:
                # Greedy metric: LCOE increase per % match gained
                match_gain = test_match - current_match
                lcoe_increase = test_lcoe - current_lcoe
                efficiency = lcoe_increase / match_gain if match_gain > 0.01 else float('inf')

                if efficiency < best_efficiency:
                    best_efficiency = efficiency
                    best_move = (res, test[res], test_lcoe, test_match)

        if best_move is None:
            # If stuck, force add something (greedy gets desperate)
            stuck_count += 1
            if stuck_count > 10:
                break
            # Try adding clean firm even if "inefficient" - greedy eventually realizes it needs it
            for res, max_val in resources:
                if current[res] < max_val:
                    test = current.copy()
                    test[res] = min(max_val, test[res] + step)
                    test_lcoe, test_match = evaluate(test)
                    if test_match > current_match:
                        current[res] = test[res]
                        current_lcoe = test_lcoe
                        current_match = test_match
                        break
            continue

        stuck_count = 0
        current[best_move[0]] = best_move[1]
        current_lcoe = best_move[2]
        current_match = best_move[3]

    return {
        'solar': current['solar'],
        'wind': current['wind'],
        'storage': current['storage'],
        'clean_firm': current['clean_firm'],
        'lcoe': current_lcoe,
        'achieved_match': current_match
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
    Full 0-100% clean energy sweep showing:
    - Total system LCOE
    - Gas contribution to LCOE (starts high, drops dramatically)
    - Gas capacity required (stays high but utilization drops)

    Key insight: You need gas capacity for reliability, but it contributes
    little to LCOE at high clean percentages because it runs so few hours.
    """
    print("Generating gas contribution sweep (0-100% clean)...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    load_profile = profiles['load']
    annual_load = np.sum(load_profile)
    peak_load = np.max(load_profile)

    # Full sweep from 0 to 99%
    targets = list(range(0, 100, 5)) + [95, 99]
    targets = sorted(set(targets))

    results = []

    for target in targets:
        print(f"  Optimizing for {target}% clean...")

        # Use V4 optimizer if available, otherwise fall back to grid search
        if HAS_V4:
            result = run_min_lcoe_v4_adaptive(
                clean_match_target=target,
                zone_data={zone: profiles},
                selected_zone=zone,
                cost_settings=cost_settings,
                use_solar=True,
                use_wind=True,
                use_storage=True,
                use_clean_firm=True,
                hybrid_mode=True
            )
        else:
            result = run_optimization(zone_data, zone, target, cost_settings)

        # Run simulation to get detailed results
        (solar_out, wind_out, _, batt_discharge, gas_gen, curtailed,
         renewable_delivered, _, _, _, _) = simulate_system(
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

        gas_cap = np.max(gas_gen)
        gas_mwh = np.sum(gas_gen)
        gas_cf = gas_mwh / (gas_cap * 8760) * 100 if gas_cap > 0 else 0

        # Calculate gas LCOE contribution
        gas_capex = cost_settings.get('gas_capex', 1200)
        gas_price = cost_settings.get('gas_price', 4)
        heat_rate = cost_settings.get('gas_heat_rate', 7.5)

        # Gas cost = capex amortized + fuel
        gas_annual_cost = gas_cap * gas_capex * 0.08  # ~8% CRF
        gas_fuel_cost = gas_mwh * gas_price * heat_rate / 1000
        gas_total_cost = gas_annual_cost + gas_fuel_cost
        gas_lcoe_contribution = gas_total_cost / annual_load * 1000  # $/MWh

        actual_match = (np.sum(renewable_delivered) / annual_load) * 100

        results.append({
            'target': target,
            'actual_match': actual_match,
            'total_lcoe': result['lcoe'],
            'gas_lcoe_contribution': gas_lcoe_contribution,
            'gas_capacity': gas_cap,
            'gas_cf': gas_cf,
            'gas_energy_pct': (gas_mwh / annual_load) * 100
        })

        print(f"    {target}%: LCOE=${result['lcoe']:.1f}, Gas contributes ${gas_lcoe_contribution:.1f}/MWh, Gas cap={gas_cap:.0f}MW ({gas_cf:.1f}% CF)")

    # Extract data for plotting
    targets_plot = [r['target'] for r in results]
    total_lcoes = [r['total_lcoe'] for r in results]
    gas_contributions = [r['gas_lcoe_contribution'] for r in results]
    gas_capacities = [r['gas_capacity'] for r in results]
    gas_cfs = [r['gas_cf'] for r in results]

    chart = {
        'data': [
            {
                'x': targets_plot,
                'y': total_lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total System LCOE',
                'line': {'color': '#202124', 'width': 3},
                'marker': {'size': 6}
            },
            {
                'x': targets_plot,
                'y': gas_contributions,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Gas Contribution to LCOE',
                'line': {'color': COLORS['gas'], 'width': 3},
                'marker': {'size': 6},
                'fill': 'tozeroy',
                'fillcolor': 'rgba(234, 67, 53, 0.2)'
            },
            {
                'x': targets_plot,
                'y': gas_capacities,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Gas Capacity (MW)',
                'line': {'color': COLORS['gas'], 'width': 2, 'dash': 'dot'},
                'marker': {'size': 4},
                'yaxis': 'y2'
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Clean Energy Target (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100]
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'side': 'left'
            },
            'yaxis2': {
                'title': 'Gas Capacity (MW)',
                'titlefont': {'color': COLORS['gas']},
                'tickfont': {'color': COLORS['gas']},
                'showgrid': False,
                'overlaying': 'y',
                'side': 'right'
            },
            'annotations': [
                {
                    'x': 20,
                    'y': gas_contributions[targets_plot.index(20)] if 20 in targets_plot else gas_contributions[4],
                    'text': 'Gas dominates<br>at low clean %',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 40,
                    'ay': -30,
                    'font': {'size': 10}
                },
                {
                    'x': 90,
                    'y': gas_contributions[-3] if len(gas_contributions) > 3 else gas_contributions[-1],
                    'text': 'Gas < $10/MWh<br>but capacity still needed',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -50,
                    'ay': 30,
                    'font': {'size': 10}
                }
            ]
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

    chart = {
        'data': [
            {
                'x': tech_names,
                'y': capex_costs,
                'type': 'bar',
                'name': 'Capital (Capex)',
                'marker': {'color': '#1a73e8'}
            },
            {
                'x': tech_names,
                'y': om_costs,
                'type': 'bar',
                'name': 'O&M',
                'marker': {'color': '#34a853'}
            },
            {
                'x': tech_names,
                'y': fuel_costs,
                'type': 'bar',
                'name': 'Fuel',
                'marker': {'color': COLORS['gas']}
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
                    'y': sum([capex_costs[-1], om_costs[-1], fuel_costs[-1]]) + 5,
                    'text': f'{fuel_costs[-1]/(capex_costs[-1]+om_costs[-1]+fuel_costs[-1])*100:.0f}% fuel',
                    'showarrow': False,
                    'font': {'size': 10, 'color': COLORS['gas']}
                },
                {
                    'x': 'Solar',
                    'y': capex_costs[0] + 5,
                    'text': f'{capex_costs[0]/(capex_costs[0]+om_costs[0])*100:.0f}% capex',
                    'showarrow': False,
                    'font': {'size': 10, 'color': COLORS['solar']}
                }
            ]
        }
    }

    return chart


def generate_cleanfirm_comparison(zone_data, zone='California'):
    """
    Compare different clean firm technology paths under varying cost of capital.
    Key insight: Zero-fuel technologies (geothermal) are very sensitive to financing,
    while fuel-based ones (hydrogen) are less sensitive but have higher floor costs.
    """
    print("Generating clean firm technology comparison...")

    discount_rates = [3, 5, 7, 9, 12]

    # Different clean firm technology assumptions
    technologies = {
        'Geothermal': {'capex': 5000, 'fuel': 0, 'color': '#34a853'},  # Zero fuel
        'Nuclear': {'capex': 8000, 'fuel': 10, 'color': COLORS['storage']},  # Low fuel
        'Hydrogen': {'capex': 1500, 'fuel': 60, 'color': COLORS['wind']}  # High fuel, low capex
    }

    chart_data = []

    for tech_name, tech in technologies.items():
        lcoes = []
        for dr in discount_rates:
            # Simple LCOE calculation for clean firm
            crf = (dr/100 * (1 + dr/100)**30) / ((1 + dr/100)**30 - 1)
            cf = 0.85  # Capacity factor
            annual_gen = cf * 8760  # kWh per kW

            capex_mwh = (tech['capex'] * crf * 1000) / annual_gen
            om_mwh = 15  # Approximate O&M
            fuel_mwh = tech['fuel']

            lcoe = capex_mwh + om_mwh + fuel_mwh
            lcoes.append(lcoe)

        print(f"  {tech_name}: ${lcoes[0]:.0f} (3%) to ${lcoes[-1]:.0f} (12%)")

        chart_data.append({
            'x': discount_rates,
            'y': lcoes,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': tech_name,
            'line': {'color': tech['color'], 'width': 3},
            'marker': {'size': 8}
        })

    chart = {
        'data': chart_data,
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Discount Rate (%)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Clean Firm LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [
                {
                    'x': 5,
                    'y': 45,
                    'text': 'Geothermal wins<br>at low rates',
                    'showarrow': False,
                    'font': {'size': 10, 'color': '#34a853'}
                },
                {
                    'x': 10,
                    'y': 85,
                    'text': 'Hydrogen has<br>fuel cost floor',
                    'showarrow': False,
                    'font': {'size': 10, 'color': COLORS['wind']}
                }
            ]
        }
    }

    return chart


def generate_tech_lcoe_comparison(zone_data, zone='California', cost_settings=None):
    """
    Show standalone LCOE of each technology.
    This is the starting point of the mystery: solar is cheapest!
    """
    print("Generating technology LCOE comparison...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    discount_rate = cost_settings.get('discount_rate', 7) / 100
    lifetime = 20

    # Capital recovery factor
    crf = (discount_rate * (1 + discount_rate)**lifetime) / ((1 + discount_rate)**lifetime - 1)

    # Calculate LCOE for each technology
    techs = {
        'Solar': {
            'capex': cost_settings.get('solar', 1000),
            'fixed_om': cost_settings.get('solar_fixed_om', 15),
            'var_om': 0,
            'fuel': 0,
            'cf': 0.25,
            'color': COLORS['solar']
        },
        'Wind': {
            'capex': cost_settings.get('wind', 1200),
            'fixed_om': cost_settings.get('wind_fixed_om', 40),
            'var_om': 0,
            'fuel': 0,
            'cf': 0.35,
            'color': COLORS['wind']
        },
        'Gas': {
            'capex': cost_settings.get('gas_capex', 1200),
            'fixed_om': 20,
            'var_om': 2,
            'fuel': cost_settings.get('gas_price', 4) * cost_settings.get('gas_heat_rate', 7.5),
            'cf': 0.50,
            'color': COLORS['gas']
        },
        'Clean Firm': {
            'capex': cost_settings.get('clean_firm', 5000),
            'fixed_om': 60,
            'var_om': 10,
            'fuel': cost_settings.get('clean_firm_fuel', 20),
            'cf': 0.85,
            'color': COLORS['clean_firm']
        }
    }

    names = []
    lcoes = []
    colors = []

    for name, tech in techs.items():
        annual_gen = tech['cf'] * 8760
        capex_mwh = (tech['capex'] * crf * 1000) / annual_gen
        om_mwh = (tech['fixed_om'] * 1000) / annual_gen + tech['var_om']
        fuel_mwh = tech['fuel']
        lcoe = capex_mwh + om_mwh + fuel_mwh

        names.append(name)
        lcoes.append(lcoe)
        colors.append(tech['color'])
        print(f"  {name}: ${lcoe:.0f}/MWh")

    chart = {
        'data': [{
            'x': names,
            'y': lcoes,
            'type': 'bar',
            'marker': {'color': colors},
            'text': [f'${l:.0f}' for l in lcoes],
            'textposition': 'outside'
        }],
        'layout': {
            **LAYOUT_DEFAULTS,
            'title': 'Standalone Technology LCOE',
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, max(lcoes) * 1.2]
            },
            'xaxis': {'showgrid': False},
            'showlegend': False,
            'annotations': [{
                'x': 'Solar',
                'y': lcoes[0] + 8,
                'text': 'Cheapest!',
                'showarrow': False,
                'font': {'size': 11, 'color': COLORS['solar'], 'weight': 'bold'}
            }]
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


def generate_elcc_with_storage(zone_data, zone='California', cost_settings=None):
    """
    Show ELCC (capacity value) with storage interactions.
    Key insight: Solar+storage has much higher capacity value than solar alone.
    Uses hybrid battery mode for proper peak shaving.
    """
    print("Generating ELCC with storage interactions...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    solar_profile = profiles['solar']
    wind_profile = profiles['wind']
    load_profile = profiles['load']
    peak_load = np.max(load_profile)

    def get_peak_gas_needed(solar, wind, storage, cf):
        """Returns peak gas generation needed (lower = higher capacity value)"""
        (_, _, _, _, gas_gen, _, _, _, _, _, _) = simulate_system(
            solar_capacity=solar, wind_capacity=wind,
            storage_capacity=storage, clean_firm_capacity=cf,
            solar_profile=solar_profile, wind_profile=wind_profile,
            load_profile=load_profile, battery_eff=0.85,
            hybrid_mode=True  # Critical for peak shaving
        )
        return np.max(gas_gen)

    def capacity_credit(solar, wind, storage, cf):
        """Capacity credit = how much gas capacity is displaced"""
        baseline_gas = get_peak_gas_needed(0, 0, 0, 0)  # ~peak load
        with_resource = get_peak_gas_needed(solar, wind, storage, cf)
        return (baseline_gas - with_resource) / baseline_gas * 100

    step = 50
    solar_caps = list(range(0, 601, step))

    # Scenario 1: Solar alone
    solar_only_cc = [capacity_credit(s, 0, 0, 0) for s in solar_caps]
    print(f"  Solar only CC: {solar_only_cc[0]:.1f}% -> {solar_only_cc[-1]:.1f}%")

    # Scenario 2: Solar + 2hr storage (200 MWh for 100MW system)
    solar_2hr_cc = [capacity_credit(s, 0, 200, 0) for s in solar_caps]
    print(f"  Solar + 2hr storage CC: {solar_2hr_cc[0]:.1f}% -> {solar_2hr_cc[-1]:.1f}%")

    # Scenario 3: Solar + 4hr storage (400 MWh)
    solar_4hr_cc = [capacity_credit(s, 0, 400, 0) for s in solar_caps]
    print(f"  Solar + 4hr storage CC: {solar_4hr_cc[0]:.1f}% -> {solar_4hr_cc[-1]:.1f}%")

    # Scenario 4: Wind alone
    wind_caps = list(range(0, 401, step))
    wind_only_cc = [capacity_credit(0, w, 0, 0) for w in wind_caps]
    print(f"  Wind only CC: {wind_only_cc[0]:.1f}% -> {wind_only_cc[-1]:.1f}%")

    # Scenario 5: Wind + 4hr storage
    wind_4hr_cc = [capacity_credit(0, w, 400, 0) for w in wind_caps]
    print(f"  Wind + 4hr storage CC: {wind_4hr_cc[0]:.1f}% -> {wind_4hr_cc[-1]:.1f}%")

    chart = {
        'data': [
            {
                'x': solar_caps,
                'y': solar_only_cc,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar Only',
                'line': {'color': COLORS['solar'], 'width': 2, 'dash': 'dot'}
            },
            {
                'x': solar_caps,
                'y': solar_2hr_cc,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar + 2hr Storage',
                'line': {'color': COLORS['solar'], 'width': 2}
            },
            {
                'x': solar_caps,
                'y': solar_4hr_cc,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Solar + 4hr Storage',
                'line': {'color': COLORS['solar'], 'width': 3}
            },
            {
                'x': wind_caps,
                'y': wind_only_cc,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Wind Only',
                'line': {'color': COLORS['wind'], 'width': 2, 'dash': 'dot'}
            },
            {
                'x': wind_caps,
                'y': wind_4hr_cc,
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Wind + 4hr Storage',
                'line': {'color': COLORS['wind'], 'width': 3}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'xaxis': {
                'title': 'Renewable Capacity (MW)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'yaxis': {
                'title': 'Capacity Credit (% of peak load displaced)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100]
            },
            'annotations': [
                {
                    'x': 400,
                    'y': solar_4hr_cc[8] if len(solar_4hr_cc) > 8 else 50,
                    'text': 'Storage boosts<br>solar capacity value',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 50,
                    'ay': -30,
                    'font': {'size': 10, 'color': COLORS['storage']}
                },
                {
                    'x': 300,
                    'y': wind_4hr_cc[6] if len(wind_4hr_cc) > 6 else 40,
                    'text': 'Wind+storage:<br>less improvement',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -50,
                    'ay': 30,
                    'font': {'size': 10, 'color': COLORS['wind']}
                }
            ]
        }
    }

    return chart


def generate_lcoe_breakdown(zone_data, zone='California', cost_settings=None):
    """
    Show LCOE breakdown by component at 95% target.
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
        battery_eff=0.85
    )

    # Calculate LCOE components
    discount_rate = cost_settings.get('discount_rate', 7) / 100
    crf = (discount_rate * (1 + discount_rate)**20) / ((1 + discount_rate)**20 - 1)

    # Solar contribution
    solar_annual_cost = result['solar'] * cost_settings.get('solar', 1000) * crf
    solar_lcoe = solar_annual_cost / annual_load * 1000

    # Wind contribution
    wind_annual_cost = result['wind'] * cost_settings.get('wind', 1200) * crf
    wind_lcoe = wind_annual_cost / annual_load * 1000

    # Storage contribution
    storage_annual_cost = result['storage'] * cost_settings.get('storage', 300) * crf
    storage_lcoe = storage_annual_cost / annual_load * 1000

    # Clean firm contribution
    cf_capex = result['clean_firm'] * cost_settings.get('clean_firm', 5000) * crf
    cf_fuel = np.sum(clean_firm_gen) * cost_settings.get('clean_firm_fuel', 20) / 1000
    cf_lcoe = (cf_capex + cf_fuel) / annual_load * 1000

    # Gas contribution
    gas_cap = np.max(gas_gen)
    gas_capex = gas_cap * cost_settings.get('gas_capex', 1200) * crf
    gas_fuel = np.sum(gas_gen) * cost_settings.get('gas_price', 4) * cost_settings.get('gas_heat_rate', 7.5) / 1000
    gas_lcoe = (gas_capex + gas_fuel) / annual_load * 1000

    total = solar_lcoe + wind_lcoe + storage_lcoe + cf_lcoe + gas_lcoe

    print(f"  Solar: ${solar_lcoe:.1f}/MWh ({solar_lcoe/total*100:.0f}%)")
    print(f"  Wind: ${wind_lcoe:.1f}/MWh ({wind_lcoe/total*100:.0f}%)")
    print(f"  Storage: ${storage_lcoe:.1f}/MWh ({storage_lcoe/total*100:.0f}%)")
    print(f"  Clean Firm: ${cf_lcoe:.1f}/MWh ({cf_lcoe/total*100:.0f}%)")
    print(f"  Gas: ${gas_lcoe:.1f}/MWh ({gas_lcoe/total*100:.0f}%)")
    print(f"  Total: ${total:.1f}/MWh")

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
            'title': f'LCOE Breakdown at {target}% Clean',
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
            }, {
                'x': '95% Clean System',
                'y': solar_lcoe + wind_lcoe + storage_lcoe + cf_lcoe + gas_lcoe/2,
                'text': f'Gas: only ${gas_lcoe:.0f}<br>({gas_lcoe/total*100:.0f}%)',
                'showarrow': True,
                'arrowhead': 2,
                'ax': 80,
                'ay': 0,
                'font': {'size': 10, 'color': COLORS['gas']}
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
    """
    print("Generating gas LCOE impact chart...")

    if cost_settings is None:
        cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    load_profile = profiles['load']
    annual_load = np.sum(load_profile)

    targets = [70, 80, 90, 95, 99]
    gas_caps = []
    gas_energy_mwh = []
    gas_lcoe_contribution = []
    total_lcoes = []

    for target in targets:
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
        gas_cf = gas_mwh / (gas_cap * 8760) * 100 if gas_cap > 0 else 0

        # Approximate gas LCOE contribution
        gas_capex = cost_settings.get('gas_capex', 1200)
        gas_price = cost_settings.get('gas_price', 4)
        heat_rate = cost_settings.get('gas_heat_rate', 7.5)

        # Gas cost = capex amortized + fuel
        gas_annual_cost = gas_cap * gas_capex * 0.08  # ~8% CRF
        gas_fuel_cost = gas_mwh * gas_price * heat_rate / 1000
        gas_total_cost = gas_annual_cost + gas_fuel_cost
        gas_lcoe = gas_total_cost / annual_load * 1000  # $/MWh

        gas_caps.append(gas_cap)
        gas_energy_mwh.append(gas_mwh)
        gas_lcoe_contribution.append(gas_lcoe)
        total_lcoes.append(result['lcoe'])

        print(f"  {target}%: Gas {gas_cap:.0f} MW, CF={gas_cf:.1f}%, "
              f"contributes ${gas_lcoe:.1f}/MWh of ${result['lcoe']:.1f}/MWh total")

    # Create grouped bar chart
    chart = {
        'data': [
            {
                'x': [f"{t}%" for t in targets],
                'y': total_lcoes,
                'type': 'bar',
                'name': 'Total System LCOE',
                'marker': {'color': COLORS['wind']}
            },
            {
                'x': [f"{t}%" for t in targets],
                'y': gas_lcoe_contribution,
                'type': 'bar',
                'name': 'Gas Contribution to LCOE',
                'marker': {'color': COLORS['gas']}
            }
        ],
        'layout': {
            **LAYOUT_DEFAULTS,
            'barmode': 'group',
            'xaxis': {
                'title': 'Clean Energy Target',
                'showgrid': False
            },
            'yaxis': {
                'title': 'LCOE ($/MWh)',
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.1)'
            },
            'annotations': [
                {
                    'x': 0.5,
                    'y': -0.15,
                    'xref': 'paper',
                    'yref': 'paper',
                    'text': 'Gas provides reliability but contributes only ~$5-15/MWh to total LCOE',
                    'showarrow': False,
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
        'cost_structure': generate_cost_structure(zone_data, cost_settings=cost_settings),
        'cleanfirm_tech': generate_cleanfirm_comparison(zone_data),
        # Gas and resource interaction charts
        'gas_capacity': generate_gas_capacity_sweep(zone_data, cost_settings=cost_settings),
        'resource_interactions': generate_resource_interactions(zone_data, cost_settings=cost_settings),
        'gas_lcoe_impact': generate_gas_lcoe_impact(zone_data, cost_settings=cost_settings),
        # New narrative charts
        'tech_lcoe': generate_tech_lcoe_comparison(zone_data, cost_settings=cost_settings),
        'marginal_energy': generate_marginal_energy_value(zone_data, cost_settings=cost_settings),
        'elcc_storage': generate_elcc_with_storage(zone_data, cost_settings=cost_settings),
        'lcoe_breakdown': generate_lcoe_breakdown(zone_data, cost_settings=cost_settings),
        'effective_lcoe': generate_effective_lcoe(zone_data, cost_settings=cost_settings),
        # Storage ceiling charts
        'solar_storage_ceiling': generate_solar_storage_ceiling(zone_data, cost_settings=cost_settings),
        'wind_storage_ceiling': generate_wind_storage_ceiling(zone_data, cost_settings=cost_settings),
        # Solar+storage only LCOE sweep
        'solar_storage_lcoe': generate_solar_storage_lcoe(zone_data, cost_settings=cost_settings)
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
