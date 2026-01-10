"""
Generate gas capacity requirement chart for VRE-only systems (solar+wind+storage).

Shows how much gas backup capacity is still needed even at high renewable penetration.
Demonstrates that while gas capacity stays high, its LCOE contribution drops as it runs less.
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
from optimizer import run_optimization

# Chart colors
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


def run_vre_optimization(zone_data, zone, target, cost_settings):
    """
    Run optimization with only solar, wind, and storage enabled.
    Uses hybrid battery mode and min_lcoe strategy.
    """
    from scipy.optimize import differential_evolution

    profiles = zone_data[zone]
    solar_profile = np.array(profiles['solar'])
    wind_profile = np.array(profiles['wind'])
    load_profile = np.array(profiles['load'])
    annual_load = np.sum(load_profile)

    # Only solar, wind, and storage - no clean firm
    bounds = [
        (0, 1000),   # Solar capacity (MW)
        (0, 500),    # Wind capacity (MW)
        (0, 2400),   # Storage capacity (MWh)
    ]

    def objective(x):
        solar, wind, storage = x

        (solar_out, wind_out, _, _, gas_gen, curtailed, renewable_delivered_hourly, _,
         clean_firm_gen, _, _) = simulate_system(
            solar_capacity=solar,
            wind_capacity=wind,
            storage_capacity=storage,
            clean_firm_capacity=0,  # No clean firm
            solar_profile=solar_profile,
            wind_profile=wind_profile,
            load_profile=load_profile,
            battery_eff=0.85,
            peak_shaver_mode=True,  # Hybrid mode uses peak shaver
            hybrid_mode=True        # Enable hybrid battery dispatch
        )

        annual_solar = np.sum(solar_out)
        annual_wind = np.sum(wind_out)
        annual_gas = np.sum(gas_gen)
        annual_curtailed = np.sum(curtailed)
        gas_capacity_needed = np.max(gas_gen)

        # Calculate clean match percentage
        clean_energy = np.sum(renewable_delivered_hourly) + np.sum(clean_firm_gen)
        clean_match_pct = (clean_energy / annual_load) * 100

        # Penalty if we don't hit the target
        if clean_match_pct < target:
            penalty = (target - clean_match_pct) * 1000
        else:
            penalty = 0

        # Calculate LCOE
        lcoe = calculate_lcoe(
            cost_settings,
            solar_capacity=solar,
            wind_capacity=wind,
            storage_capacity=storage,
            clean_firm_capacity=0,
            annual_solar_gen=annual_solar,
            annual_wind_gen=annual_wind,
            annual_clean_firm_gen=0,
            annual_gas_gen=annual_gas,
            gas_capacity_needed=gas_capacity_needed,
            curtailed_energy=annual_curtailed,
            annual_load=annual_load
        )

        return lcoe + penalty

    # Run optimization
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=300,
        popsize=15,
        tol=0.01,
        seed=42,
        workers=1,
        updating='deferred',
        polish=True
    )

    solar_opt, wind_opt, storage_opt = result.x

    # Run final simulation to get all results
    (solar_out, wind_out, _, _, gas_gen, curtailed, renewable_delivered_hourly, _,
     clean_firm_gen, _, _) = simulate_system(
        solar_capacity=solar_opt,
        wind_capacity=wind_opt,
        storage_capacity=storage_opt,
        clean_firm_capacity=0,
        solar_profile=solar_profile,
        wind_profile=wind_profile,
        load_profile=load_profile,
        battery_eff=0.85,
        peak_shaver_mode=True,
        hybrid_mode=True
    )

    annual_solar = np.sum(solar_out)
    annual_wind = np.sum(wind_out)
    annual_gas = np.sum(gas_gen)
    annual_curtailed = np.sum(curtailed)
    gas_capacity_needed = np.max(gas_gen)
    clean_energy = np.sum(renewable_delivered_hourly) + np.sum(clean_firm_gen)
    clean_match_pct = (clean_energy / annual_load) * 100

    # Calculate total LCOE
    total_lcoe = calculate_lcoe(
        cost_settings,
        solar_capacity=solar_opt,
        wind_capacity=wind_opt,
        storage_capacity=storage_opt,
        clean_firm_capacity=0,
        annual_solar_gen=annual_solar,
        annual_wind_gen=annual_wind,
        annual_clean_firm_gen=0,
        annual_gas_gen=annual_gas,
        gas_capacity_needed=gas_capacity_needed,
        curtailed_energy=annual_curtailed,
        annual_load=annual_load
    )

    # Calculate gas-only LCOE (just the gas portion)
    gas_only_lcoe = calculate_lcoe(
        cost_settings,
        solar_capacity=0,
        wind_capacity=0,
        storage_capacity=0,
        clean_firm_capacity=0,
        annual_solar_gen=0,
        annual_wind_gen=0,
        annual_clean_firm_gen=0,
        annual_gas_gen=annual_gas,
        gas_capacity_needed=gas_capacity_needed,
        curtailed_energy=0,
        annual_load=annual_load
    )

    return {
        'solar': solar_opt,
        'wind': wind_opt,
        'storage': storage_opt,
        'gas_capacity': gas_capacity_needed,
        'total_lcoe': total_lcoe,
        'gas_lcoe': gas_only_lcoe,
        'achieved_match': clean_match_pct,
        'annual_gas': annual_gas
    }


def generate_gas_capacity_vre_sweep():
    """
    Generate two charts:
    1. Gas capacity needed vs clean energy target (0-100%)
    2. Total LCOE and gas contribution vs clean energy target

    Uses only solar, wind, and storage (no clean firm).
    Uses hybrid battery mode and default costs.
    """
    print("Generating VRE-only gas capacity sweep...")
    print("Configuration: Solar + Wind + Storage only, Hybrid mode, Default costs")

    # Load data
    zone_data, available_zones = load_all_zone_data()
    zone = 'California'
    cost_settings = DEFAULT_COSTS.copy()

    profiles = zone_data[zone]
    load_profile = np.array(profiles['load'])
    peak_load = np.max(load_profile)

    # Sweep from 0% to 100% clean energy in 5% steps, plus 98% and 99%
    targets = list(range(0, 96, 5)) + [98, 99, 100]

    gas_capacities = []
    total_lcoes = []
    gas_lcoes = []
    achieved_matches = []

    for target in targets:
        print(f"\n{'='*60}")
        print(f"Optimizing for {target}% clean energy target...")

        result = run_vre_optimization(zone_data, zone, target, cost_settings)

        gas_capacities.append(result['gas_capacity'])
        total_lcoes.append(result['total_lcoe'])
        gas_lcoes.append(result['gas_lcoe'])
        achieved_matches.append(result['achieved_match'])

        print(f"  Solar: {result['solar']:.1f} MW")
        print(f"  Wind: {result['wind']:.1f} MW")
        print(f"  Storage: {result['storage']:.1f} MWh")
        print(f"  Gas Capacity: {result['gas_capacity']:.1f} MW ({result['gas_capacity']/peak_load*100:.1f}% of peak)")
        print(f"  Total LCOE: ${result['total_lcoe']:.2f}/MWh")
        print(f"  Gas Contribution: ${result['gas_lcoe']:.2f}/MWh")
        print(f"  Achieved: {result['achieved_match']:.1f}%")

    # Chart 1: Gas Capacity Requirements
    capacity_chart = {
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
                'gridcolor': 'rgba(0,0,0,0.1)',
                'range': [0, 100],
                'dtick': 20
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
                    'x0': 0, 'x1': 100,
                    'y0': peak_load, 'y1': peak_load,
                    'line': {'color': COLORS['load'], 'width': 2, 'dash': 'dash'}
                }
            ],
            'annotations': [
                {
                    'x': 50,
                    'y': peak_load + 5,
                    'text': f'Peak Load: {peak_load:.0f} MW',
                    'showarrow': False,
                    'font': {'size': 10, 'color': COLORS['load']}
                },
                {
                    'x': 95,
                    'y': gas_capacities[19] if len(gas_capacities) > 19 else 0,  # 95% target (index 19)
                    'text': f'~{gas_capacities[19]/peak_load*100:.0f}% of peak<br>still needed at 95%',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': -50,
                    'ay': -40,
                    'font': {'size': 10, 'color': COLORS['gas']}
                }
            ]
        }
    }

    # Chart 2: LCOE Impact
    lcoe_chart = {
        'data': [
            {
                'x': targets,
                'y': total_lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total System LCOE',
                'line': {'color': '#000000', 'width': 2},
                'marker': {'size': 6, 'color': '#000000'}
            },
            {
                'x': targets,
                'y': gas_lcoes,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Gas Contribution to LCOE',
                'line': {'color': COLORS['gas'], 'width': 2},
                'marker': {'size': 6, 'color': COLORS['gas']},
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
                    'y': gas_lcoes[4] if len(gas_lcoes) > 4 else 0,  # 20% target
                    'text': 'Gas dominates<br>at low clean %',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 0,
                    'ay': -40,
                    'font': {'size': 11, 'color': '#333'}
                },
                {
                    'x': 90,
                    'y': gas_lcoes[18] if len(gas_lcoes) > 18 else 0,  # 90% target (index 18)
                    'text': f'Gas < ${gas_lcoes[18]:.0f}/MWh<br>but capacity still needed',
                    'showarrow': True,
                    'arrowhead': 2,
                    'ax': 0,
                    'ay': 40,
                    'font': {'size': 11, 'color': '#333'}
                }
            ]
        }
    }

    # Save charts
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)

    capacity_file = os.path.join(output_dir, 'gas_capacity_vre.json')
    lcoe_file = os.path.join(output_dir, 'gas_lcoe_vre.json')

    with open(capacity_file, 'w') as f:
        json.dump(capacity_chart, f, indent=2)

    with open(lcoe_file, 'w') as f:
        json.dump(lcoe_chart, f, indent=2)

    print(f"\n{'='*60}")
    print("✓ Generated gas_capacity_vre.json")
    print("✓ Generated gas_lcoe_vre.json")
    print(f"\nKey findings:")
    print(f"  - At 0% clean: {gas_capacities[0]:.0f} MW gas ({gas_capacities[0]/peak_load*100:.0f}% of peak)")
    print(f"  - At 50% clean: {gas_capacities[10]:.0f} MW gas ({gas_capacities[10]/peak_load*100:.0f}% of peak)")
    print(f"  - At 90% clean: {gas_capacities[18]:.0f} MW gas ({gas_capacities[18]/peak_load*100:.0f}% of peak)")
    print(f"  - At 95% clean: {gas_capacities[19]:.0f} MW gas ({gas_capacities[19]/peak_load*100:.0f}% of peak)")
    print(f"  - At 98% clean: {gas_capacities[20]:.0f} MW gas ({gas_capacities[20]/peak_load*100:.0f}% of peak)")
    print(f"  - At 99% clean: {gas_capacities[21]:.0f} MW gas ({gas_capacities[21]/peak_load*100:.0f}% of peak)")
    print(f"  - At 100% clean: {gas_capacities[22]:.0f} MW gas ({gas_capacities[22]/peak_load*100:.0f}% of peak)")
    print(f"\n  - Gas LCOE at 90% clean: ${gas_lcoes[18]:.2f}/MWh (only {gas_lcoes[18]/total_lcoes[18]*100:.1f}% of total)")
    print(f"  - Gas LCOE at 98% clean: ${gas_lcoes[20]:.2f}/MWh (only {gas_lcoes[20]/total_lcoes[20]*100:.1f}% of total)")


if __name__ == '__main__':
    generate_gas_capacity_vre_sweep()
