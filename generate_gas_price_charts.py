"""
Generate gas price sensitivity and crossover charts for the microsite.

Chart 1: Gas Price Sensitivity - Shows how LCOE changes with gas prices
        at different clean match levels (demonstrates reduced exposure as you go cleaner)

Chart 2: Gas Price Crossover Points - Shows at what gas price clean energy
        becomes cheaper than gas-only for different renewable cost scenarios

Run from the microsite folder:
    python generate_gas_price_charts.py
"""

import sys
import os
import json
import numpy as np
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
    print("ERROR: V4 optimizer not available. Cannot generate gas price charts.")
    sys.exit(1)

# Chart colors
COLORS = {
    'gas': '#ea4335',      # Red
    'low': '#34a853',      # Green
    'default': '#4285f4',  # Blue
    'high': '#673ab7',     # Purple
    'clean_firm': '#ff7900'
}

FONT = "Google Sans, -apple-system, sans-serif"


def calculate_gas_only_lcoe(gas_price):
    """
    Calculate gas-only system LCOE.

    Args:
        gas_price: Natural gas price in $/MMBtu

    Returns:
        LCOE in $/MWh
    """
    heat_rate = DEFAULT_COSTS.get('gas_heat_rate', 7.5)
    fuel_cost = gas_price * heat_rate  # $/MWh
    # Add fixed costs (CAPEX + O&M) - approximately $35/MWh baseline
    return fuel_cost + 35


def find_crossover_point(gas_prices, clean_lcoes, gas_only_lcoes):
    """
    Find the gas price where clean system becomes cheaper than gas-only.

    Returns:
        Crossover gas price in $/MMBtu, or None if no crossover
    """
    # Find where clean LCOE crosses below gas-only LCOE
    for i in range(len(gas_prices) - 1):
        clean_1, clean_2 = clean_lcoes[i], clean_lcoes[i + 1]
        gas_1, gas_2 = gas_only_lcoes[i], gas_only_lcoes[i + 1]

        # Check if crossover happens in this interval
        if (clean_1 >= gas_1 and clean_2 < gas_2) or (clean_1 > gas_1 and clean_2 <= gas_2):
            # Linear interpolation to find exact crossover
            # clean_1 + t * (clean_2 - clean_1) = gas_1 + t * (gas_2 - gas_1)
            # Solve for t
            if (clean_2 - clean_1) != (gas_2 - gas_1):
                t = (gas_1 - clean_1) / ((clean_2 - clean_1) - (gas_2 - gas_1))
                crossover_price = gas_prices[i] + t * (gas_prices[i + 1] - gas_prices[i])
                return crossover_price

    # Check if clean is always cheaper or always more expensive
    if clean_lcoes[-1] < gas_only_lcoes[-1]:
        # Clean is cheaper even at lowest gas price
        return gas_prices[0]

    return None  # No crossover in range


def generate_gas_price_sensitivity(zone_data, zone='California'):
    """
    Chart 1: Gas Price Sensitivity

    Shows how system LCOE changes with gas prices at different clean match levels.
    Demonstrates that gas price exposure decreases as you go cleaner.

    Uses medium (default) renewable costs.
    """
    print("\n" + "=" * 80)
    print("Generating Gas Price Sensitivity Chart")
    print("=" * 80)

    # Gas prices to sweep
    gas_prices = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

    # Clean match targets to analyze
    targets = [0, 25, 50, 70, 90]

    # Use default (medium) renewable costs
    cost_settings = DEFAULT_COSTS.copy()

    results = {}

    for target in targets:
        print(f"\nClean Match Target: {target}%")
        results[target] = []

        for gas_price in gas_prices:
            # Update gas price
            run_cost_settings = cost_settings.copy()
            run_cost_settings['gas_price'] = gas_price

            if target == 0:
                # Gas-only system
                lcoe = calculate_gas_only_lcoe(gas_price)
                results[target].append(lcoe)
                print(f"  Gas ${gas_price}/MMBtu: LCOE=${lcoe:.1f}/MWh (gas-only)")
            else:
                # Run optimization
                v4_zone_data = {zone: zone_data[zone]}
                result, timing = run_min_lcoe_v4_adaptive(
                    clean_match_target=target,
                    zone_data=v4_zone_data,
                    selected_zone=zone,
                    cost_settings=run_cost_settings,
                    demand_response_val=0,
                    use_solar=True,
                    use_wind=True,
                    use_storage=True,
                    use_clean_firm=True,
                    peak_shaver_mode=False,
                    hybrid_mode=True,
                    return_timing=True,
                    verbose=False
                )
                lcoe = result['lcoe']
                results[target].append(lcoe)
                print(f"  Gas ${gas_price}/MMBtu: LCOE=${lcoe:.1f}/MWh")

    # Create chart
    traces = []

    # Color map for different clean levels
    target_colors = {
        0: COLORS['gas'],
        25: '#f4b400',  # Yellow
        50: '#4285f4',  # Blue
        70: '#673ab7',  # Purple
        90: '#34a853'   # Green
    }

    target_widths = {
        0: 3,
        25: 2,
        50: 2,
        70: 2,
        90: 3
    }

    for target in targets:
        trace = {
            'x': gas_prices,
            'y': results[target],
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': f'{target}% Clean' if target > 0 else 'Gas Only',
            'line': {
                'color': target_colors[target],
                'width': target_widths[target]
            },
            'marker': {
                'size': 6 if target not in [0, 90] else 8
            }
        }
        traces.append(trace)

    layout = {
        'title': {
            'text': 'Gas Price Sensitivity Across Clean Energy Targets',
            'font': {'family': FONT, 'size': 18, 'color': '#202124'}
        },
        'xaxis': {
            'title': 'Natural Gas Price ($/MMBtu)',
            'gridcolor': '#e8eaed'
        },
        'yaxis': {
            'title': 'System LCOE ($/MWh)',
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

    chart = {
        'data': traces,
        'layout': layout
    }

    return chart


def generate_gas_crossover_points(zone_data, zone='California'):
    """
    Chart 2: Gas Price Crossover Points

    Shows at what gas price clean energy becomes cheaper than gas-only,
    across different clean match targets and renewable cost scenarios.
    """
    print("\n" + "=" * 80)
    print("Generating Gas Price Crossover Points Chart")
    print("=" * 80)

    # Gas prices to sweep
    gas_prices = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

    # Clean match targets
    targets = [25, 50, 70, 80, 90, 95, 99]

    # Renewable cost scenarios
    scenarios = {
        'low': {
            'name': 'Low Cost Renewables',
            'solar': 500,
            'wind': 700,
            'storage': 100,
        },
        'default': {
            'name': 'Medium Cost Renewables',
            'solar': 1000,
            'wind': 1200,
            'storage': 300,
        },
        'high': {
            'name': 'High Cost Renewables',
            'solar': 1500,
            'wind': 1800,
            'storage': 450,
        }
    }

    crossover_points = {scenario_key: [] for scenario_key in scenarios.keys()}

    for scenario_key, scenario in scenarios.items():
        print(f"\n{scenario['name']} Scenario:")

        cost_settings = DEFAULT_COSTS.copy()
        cost_settings['solar'] = scenario['solar']
        cost_settings['wind'] = scenario['wind']
        cost_settings['storage'] = scenario['storage']

        for target in targets:
            print(f"\n  {target}% Clean Match:")

            # Run optimization at different gas prices
            clean_lcoes = []
            gas_only_lcoes = []

            for gas_price in gas_prices:
                run_cost_settings = cost_settings.copy()
                run_cost_settings['gas_price'] = gas_price

                # Gas-only LCOE
                gas_only_lcoe = calculate_gas_only_lcoe(gas_price)
                gas_only_lcoes.append(gas_only_lcoe)

                # Clean system LCOE
                v4_zone_data = {zone: zone_data[zone]}
                result, timing = run_min_lcoe_v4_adaptive(
                    clean_match_target=target,
                    zone_data=v4_zone_data,
                    selected_zone=zone,
                    cost_settings=run_cost_settings,
                    demand_response_val=0,
                    use_solar=True,
                    use_wind=True,
                    use_storage=True,
                    use_clean_firm=True,
                    peak_shaver_mode=False,
                    hybrid_mode=True,
                    return_timing=True,
                    verbose=False
                )
                clean_lcoe = result['lcoe']
                clean_lcoes.append(clean_lcoe)

                print(f"    Gas ${gas_price}/MMBtu: Clean=${clean_lcoe:.1f}, Gas-only=${gas_only_lcoe:.1f}")

            # Find crossover point
            crossover = find_crossover_point(gas_prices, clean_lcoes, gas_only_lcoes)

            if crossover:
                print(f"    ✓ Crossover at ${crossover:.2f}/MMBtu")
                crossover_points[scenario_key].append(crossover)
            else:
                print(f"    ✗ No crossover in range (clean always more expensive)")
                crossover_points[scenario_key].append(None)

    # Create chart
    traces = []

    for scenario_key, scenario in scenarios.items():
        # Filter out None values
        x_vals = []
        y_vals = []
        for i, crossover in enumerate(crossover_points[scenario_key]):
            if crossover is not None:
                x_vals.append(targets[i])
                y_vals.append(crossover)

        trace = {
            'x': x_vals,
            'y': y_vals,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': scenario['name'],
            'line': {
                'color': COLORS[scenario_key],
                'width': 3 if scenario_key == 'default' else 2,
                'dash': 'solid' if scenario_key == 'default' else 'dash'
            },
            'marker': {
                'size': 8,
                'color': COLORS[scenario_key]
            }
        }
        traces.append(trace)

    layout = {
        'title': {
            'text': 'Gas Price Crossover Points by Clean Match Target',
            'font': {'family': FONT, 'size': 18, 'color': '#202124'}
        },
        'xaxis': {
            'title': 'Clean Energy Target (%)',
            'gridcolor': '#e8eaed',
            'dtick': 10
        },
        'yaxis': {
            'title': 'Crossover Gas Price ($/MMBtu)',
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
            'xanchor': 'left',
            'x': 0.02,
            'bgcolor': 'rgba(255, 255, 255, 0.9)',
            'bordercolor': '#e8eaed',
            'borderwidth': 1
        },
        'annotations': [{
            'text': 'Shows minimum gas price for clean energy to be cost-competitive',
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': -0.15,
            'showarrow': False,
            'font': {'size': 11, 'color': '#5f6368'},
            'xanchor': 'center'
        }]
    }

    chart = {
        'data': traces,
        'layout': layout
    }

    return chart


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

    # Generate charts
    print("\n" + "=" * 80)
    print("Generating gas price analysis charts...")
    print("=" * 80)

    charts = {
        'gas_price_sensitivity': generate_gas_price_sensitivity(zone_data, zone='California'),
        'gas_crossover_points': generate_gas_crossover_points(zone_data, zone='California')
    }

    # Save charts
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)

    for chart_name, chart_data in charts.items():
        filename = f'{chart_name}.json'
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(chart_data, f, indent=2)

        print(f"\n✓ Saved {filename}")

    print(f"\n{'='*80}")
    print("✓ Gas price analysis charts generated successfully!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
