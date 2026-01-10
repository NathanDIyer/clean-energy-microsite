"""
Generate Clean Firm Cost Sensitivity chart for the microsite.

Shows how clean firm capacity varies with clean firm cost across different
renewable cost scenarios (low/default/high) and clean match targets.

Run from the microsite folder:
    python generate_cf_cost_sensitivity.py
"""

import sys
import os
import json
import numpy as np
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from scipy.interpolate import interp1d

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_all_zone_data
from cost_settings_modal import DEFAULT_COSTS

# Import V4 optimizer
try:
    from experimental.optimizer_v4 import run_min_lcoe_v4_adaptive
    HAS_V4 = True
except ImportError:
    HAS_V4 = False
    print("ERROR: V4 optimizer not available. Cannot generate cost sensitivity chart.")
    sys.exit(1)

# Chart colors matching the cost sweep
COLORS = {
    'low': '#34a853',      # Green
    'default': '#4285f4',  # Blue
    'high': '#ea4335'      # Red
}

FONT = "Google Sans, -apple-system, sans-serif"


def enforce_monotonic_decrease(y_values):
    """
    Enforce monotonic decrease on a series of values.
    If value increases, replace with previous value.

    Args:
        y_values: List of values

    Returns:
        Monotonically decreasing list
    """
    y_mono = [y_values[0]]
    for i in range(1, len(y_values)):
        # Each value must be <= previous value
        y_mono.append(min(y_values[i], y_mono[-1]))
    return y_mono


def linear_start_to_finish(x_values, y_values):
    """
    Create a simple linear line from start to finish point (where it hits zero).
    No smoothing, just a straight line.

    Args:
        x_values: X coordinates
        y_values: Y coordinates

    Returns:
        Linear interpolation from start to end
    """
    # Convert to numpy arrays
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)

    # First, enforce strict monotonic decrease on raw data
    y = np.array(enforce_monotonic_decrease(y.tolist()))

    # Find last non-zero point
    last_nonzero_idx = -1
    for i in range(len(y) - 1, -1, -1):
        if y[i] > 0.5:
            last_nonzero_idx = i
            break

    # If no non-zero points or only one, return as-is
    if last_nonzero_idx < 1:
        return y.tolist()

    # Create linear interpolation from first point to last non-zero point
    # Then zeros for the rest
    start_idx = 0
    end_idx = last_nonzero_idx

    # Get start and end values
    x_start = x[start_idx]
    y_start = y[start_idx]
    x_end = x[end_idx]
    y_end = y[end_idx]

    # Create linear interpolation
    result = []
    for i in range(len(x)):
        if i <= end_idx:
            # Linear interpolation: y = y_start + (y_end - y_start) * (x - x_start) / (x_end - x_start)
            if x_end != x_start:
                y_interp = y_start + (y_end - y_start) * (x[i] - x_start) / (x_end - x_start)
            else:
                y_interp = y_start
            result.append(max(0, y_interp))
        else:
            result.append(0.0)

    return result


def run_cf_cost_sensitivity_sweep(zone_data, zone, clean_match_target, cost_settings):
    """
    Run clean firm cost sweep for a single clean match target.

    Returns dict with results for each renewable cost scenario (low/default/high).
    """
    # Define the 3 wind/solar/battery cost scenarios
    wsb_scenarios = {
        'low': {
            'name': 'Low Cost',
            'label': 'Low Cost Renewables',
            'solar': 500,
            'wind': 700,
            'storage': 100,
        },
        'default': {
            'name': 'Default',
            'label': 'Default Renewables',
            'solar': 1000,
            'wind': 1200,
            'storage': 300,
        },
        'high': {
            'name': 'High Cost',
            'label': 'High Cost Renewables',
            'solar': 1500,
            'wind': 1800,
            'storage': 450,
        }
    }

    # Clean firm cost sweep points - fewer points for smoother curves
    # Using key price points instead of every $1k increment
    clean_firm_costs = [1000, 2000, 3000, 4000, 5000, 7000, 10000, 12000]

    # Zone data for optimizer
    v4_zone_data = {zone: zone_data[zone]}

    results = {}
    total_runs = len(wsb_scenarios) * len(clean_firm_costs)
    current_run = 0

    for scenario_key, scenario in wsb_scenarios.items():
        print(f"\nRunning {scenario['name']} scenario for {clean_match_target}% target...")
        scenario_results = []

        for cf_cost in clean_firm_costs:
            current_run += 1
            print(f"  [{current_run}/{total_runs}] CF ${cf_cost:,}/kW...", end='', flush=True)

            # Create modified cost settings for this run
            run_cost_settings = cost_settings.copy()
            run_cost_settings['solar'] = scenario['solar']
            run_cost_settings['wind'] = scenario['wind']
            run_cost_settings['storage'] = scenario['storage']
            run_cost_settings['clean_firm'] = cf_cost

            try:
                # Run the optimizer
                result, timing = run_min_lcoe_v4_adaptive(
                    clean_match_target=clean_match_target,
                    zone_data=v4_zone_data,
                    selected_zone=zone,
                    cost_settings=run_cost_settings,
                    demand_response_val=0,
                    use_solar=True,
                    use_wind=True,
                    use_storage=True,
                    use_clean_firm=True,
                    peak_shaver_mode=False,
                    hybrid_mode=True,  # Use hybrid mode (default for sweeps)
                    return_timing=True,
                    verbose=False
                )

                cf_capacity = result.get('clean_firm', 0)
                solar_capacity = result.get('solar', 0)
                wind_capacity = result.get('wind', 0)
                storage_capacity = result.get('storage', 0)

                # Calculate clean firm percentage of clean energy capacity
                clean_capacity = solar_capacity + wind_capacity + cf_capacity
                cf_percentage = (cf_capacity / clean_capacity * 100) if clean_capacity > 0 else 0

                scenario_results.append({
                    'clean_firm_cost': cf_cost,
                    'clean_firm_capacity': float(cf_capacity),
                    'cf_percentage': float(cf_percentage),
                    'solar_capacity': float(solar_capacity),
                    'wind_capacity': float(wind_capacity),
                    'storage_capacity': float(storage_capacity),
                    'lcoe': float(result.get('lcoe', 0)),
                    'achieved_match': float(result.get('achieved_match', 0)),
                })

                print(f" CF={cf_capacity:.1f}MW ({cf_percentage:.1f}% of clean capacity)")

            except Exception as e:
                print(f" ERROR: {str(e)}")
                scenario_results.append({
                    'clean_firm_cost': cf_cost,
                    'clean_firm_capacity': 0,
                    'cf_percentage': 0,
                    'solar_capacity': 0,
                    'wind_capacity': 0,
                    'storage_capacity': 0,
                    'lcoe': 0,
                    'achieved_match': 0,
                })

        results[scenario_key] = {
            'name': scenario['name'],
            'label': scenario['label'],
            'data': scenario_results
        }

    return results


def generate_cf_cost_chart_data(zone_data, zone='California'):
    """Generate clean firm cost sensitivity chart data for multiple clean match targets."""

    print(f"\n{'='*80}")
    print("Generating Clean Firm Cost Sensitivity Chart")
    print(f"Zone: {zone}")
    print(f"{'='*80}")

    # Base cost settings
    cost_settings = DEFAULT_COSTS.copy()

    # Clean match targets to analyze
    targets = [80, 90, 100]

    all_results = {}

    for target in targets:
        print(f"\n{'─'*80}")
        print(f"Running sweeps for {target}% clean match target...")
        print(f"{'─'*80}")

        results = run_cf_cost_sensitivity_sweep(zone_data, zone, target, cost_settings)
        all_results[f"{target}pct"] = results

    # Now create the chart data for each target
    charts = {}

    for target in targets:
        target_key = f"{target}pct"
        target_results = all_results[target_key]

        # Create traces for each renewable cost scenario
        traces = []

        for scenario_key in ['low', 'default', 'high']:
            scenario = target_results[scenario_key]
            data = scenario['data']

            x_values = [d['clean_firm_cost'] for d in data]
            y_values = [d['clean_firm_capacity'] for d in data]

            # Create simple linear line from start to finish (no swoopiness)
            y_values_linear = linear_start_to_finish(x_values, y_values)

            trace = {
                'x': x_values,
                'y': y_values_linear,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': scenario['label'],
                'line': {
                    'color': COLORS[scenario_key],
                    'width': 3,
                    'dash': 'solid' if scenario_key == 'default' else 'dash',
                    'shape': 'linear'  # Straight lines, no curves
                },
                'marker': {
                    'size': 6,
                    'color': COLORS[scenario_key]
                }
            }

            traces.append(trace)

        # Create layout
        layout = {
            'title': {
                'text': f'Clean Firm Capacity vs. Cost @ {target}% Clean Match Target',
                'font': {'family': FONT, 'size': 18, 'color': '#202124'}
            },
            'xaxis': {
                'title': 'Clean Firm Cost ($/kW)',
                'gridcolor': '#e8eaed',
                'tickformat': '$,.0f'
            },
            'yaxis': {
                'title': 'Optimal Clean Firm Capacity (MW)',
                'gridcolor': '#e8eaed',
                'rangemode': 'tozero'
            },
            'font': {'family': FONT, 'size': 12},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'white',
            'margin': {'l': 80, 'r': 40, 't': 80, 'b': 80},
            'hovermode': 'x unified',
            'legend': {
                'orientation': 'v',
                'yanchor': 'top',
                'y': 0.98,
                'xanchor': 'right',
                'x': 0.98,
                'bgcolor': 'rgba(255, 255, 255, 0.9)',
                'bordercolor': '#e8eaed',
                'borderwidth': 1
            }
        }

        charts[target_key] = {
            'data': traces,
            'layout': layout
        }

    # Also create a crossover analysis chart showing at what CF cost does CF become 50%+
    print(f"\n{'─'*80}")
    print("Analyzing clean firm crossover points (CF >= 50% of clean capacity)...")
    print(f"{'─'*80}")

    crossover_data = []

    for target in targets:
        target_key = f"{target}pct"
        target_results = all_results[target_key]

        crossover_row = {'target': target}

        for scenario_key in ['low', 'default', 'high']:
            scenario = target_results[scenario_key]
            data = scenario['data']

            # Find the crossover point where CF becomes 50%+ of clean capacity
            crossover_cost = None
            for d in data:
                if d['cf_percentage'] >= 50:
                    crossover_cost = d['clean_firm_cost']
                    break

            crossover_row[scenario_key] = crossover_cost

            if crossover_cost:
                print(f"  {target}% target, {scenario['name']}: CF >= 50% at ${crossover_cost:,}/kW")
            else:
                print(f"  {target}% target, {scenario['name']}: CF never reaches 50%")

        crossover_data.append(crossover_row)

    # Create crossover chart
    crossover_traces = []

    for scenario_key in ['low', 'default', 'high']:
        scenario_name = all_results['90pct'][scenario_key]['label']

        x_values = [d['target'] for d in crossover_data]
        y_values = [d.get(scenario_key) for d in crossover_data]

        # Replace None with null for JSON serialization
        y_values = [y if y is not None else None for y in y_values]

        trace = {
            'x': x_values,
            'y': y_values,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': scenario_name,
            'line': {
                'color': COLORS[scenario_key],
                'width': 3
            },
            'marker': {
                'size': 10,
                'color': COLORS[scenario_key]
            }
        }

        crossover_traces.append(trace)

    crossover_layout = {
        'title': {
            'text': 'Clean Firm Cost Threshold for 50%+ of Clean Capacity',
            'font': {'family': FONT, 'size': 18, 'color': '#202124'}
        },
        'xaxis': {
            'title': 'Clean Match Target (%)',
            'gridcolor': '#e8eaed',
            'tickvals': [80, 90, 100]
        },
        'yaxis': {
            'title': 'Clean Firm Cost Threshold ($/kW)',
            'gridcolor': '#e8eaed',
            'tickformat': '$,.0f',
            'rangemode': 'tozero'
        },
        'font': {'family': FONT, 'size': 12},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'white',
        'margin': {'l': 80, 'r': 40, 't': 80, 'b': 80},
        'hovermode': 'x unified',
        'legend': {
            'orientation': 'v',
            'yanchor': 'top',
            'y': 0.98,
            'xanchor': 'left',
            'x': 0.02,
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'bordercolor': '#e8eaed',
            'borderwidth': 1
        },
        'annotations': [{
            'text': 'Shows minimum CF cost needed for clean firm to exceed 50% of clean energy capacity',
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': -0.15,
            'showarrow': False,
            'font': {'size': 11, 'color': '#5f6368'},
            'xanchor': 'center'
        }]
    }

    charts['crossover'] = {
        'data': crossover_traces,
        'layout': crossover_layout
    }

    return charts


def main():
    """Main execution function."""
    print("\nLoading zone data...")

    # Change to parent directory to find CSV files
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir)

    # Load zone data
    zone_data_result = load_all_zone_data()

    # Handle tuple return from fallback function
    if isinstance(zone_data_result, tuple):
        zone_data, available_zones = zone_data_result
    else:
        zone_data = zone_data_result
        available_zones = list(zone_data.keys())

    print(f"Available zones: {available_zones}")

    print("Generating clean firm cost sensitivity charts...")
    charts = generate_cf_cost_chart_data(zone_data, zone='California')

    # Save each chart to a separate JSON file
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)

    for chart_name, chart_data in charts.items():
        filename = f'cf_cost_sensitivity_{chart_name}.json'
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(chart_data, f, indent=2)

        print(f"\n✓ Saved {filename}")

    print(f"\n{'='*80}")
    print("✓ Clean firm cost sensitivity charts generated successfully!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
