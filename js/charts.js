/**
 * Chart loading and rendering for the Clean Energy Optimization microsite.
 */

// Chart ID to filename mapping
const CHARTS = {
    // Part 1: The Mystery
    'chart-tech-lcoe': 'tech_lcoe.json',
    'chart-cost-structure': 'cost_structure.json',
    'chart-lcoe-curve': 'lcoe_curve.json',

    // Part 2: The Energy Problem
    'chart-mismatch': 'mismatch.json',
    'chart-solar-paradox': 'solar_paradox.json',
    'chart-solar-storage-ceiling': 'solar_storage_ceiling.json',
    'chart-wind-storage-ceiling': 'wind_storage_ceiling.json',
    'chart-solar-storage-lcoe': 'solar_storage_lcoe.json',
    'chart-storage-cost-sensitivity': 'storage_cost_sensitivity.json',
    'chart-effective-lcoe': 'effective_lcoe.json',
    'chart-marginal-energy': 'marginal_energy.json',

    // Part 3: The Capacity Problem
    'chart-elcc': 'elcc.json',
    'chart-elcc-solar-storage': 'elcc_solar_storage.json',
    'chart-elcc-wind-storage': 'elcc_wind_storage.json',

    // Part 4: The Solution
    'chart-resource-mix': 'resource_mix.json',
    'chart-lcoe-breakdown': 'lcoe_breakdown.json',
    'chart-gas-lcoe-impact': 'gas_lcoe_impact.json',
    'chart-leap': 'leap.json',

    // Part 5: Testing Our Understanding
    'chart-regional': 'regional.json',
    'chart-gas-crossover': 'gas_crossover.json',
    'chart-no-cleanfirm': 'no_cleanfirm.json',
    'chart-cleanfirm-tech': 'cleanfirm_tech.json',
    'chart-financing': 'financing.json',
    'chart-no-wind': 'no_wind.json',
    'chart-greedy': 'greedy.json'
};

// Plotly config for all charts
const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false,
    staticPlot: true,
    displaylogo: false
};

/**
 * Load a single chart from JSON and render it.
 */
async function loadChart(containerId, filename) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Container ${containerId} not found`);
        return;
    }

    try {
        const response = await fetch(`data/${filename}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const chartData = await response.json();

        // Render the chart
        Plotly.newPlot(
            containerId,
            chartData.data,
            chartData.layout,
            PLOTLY_CONFIG
        );

        // Mark as loaded (removes loading message)
        container.classList.add('loaded');

    } catch (error) {
        console.error(`Failed to load chart ${filename}:`, error);
        container.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center;
                        height: 100%; color: #666; flex-direction: column;">
                <p>Chart data not available</p>
                <p style="font-size: 0.8em; color: #999;">Run generate_charts.py to create chart data</p>
            </div>
        `;
    }
}

/**
 * Load all charts on the page.
 */
async function loadAllCharts() {
    const promises = Object.entries(CHARTS).map(([containerId, filename]) =>
        loadChart(containerId, filename)
    );

    await Promise.all(promises);
    console.log('All charts loaded');
}

/**
 * Handle window resize for responsive charts.
 */
function setupResizeHandler() {
    let resizeTimeout;

    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            Object.keys(CHARTS).forEach(containerId => {
                const container = document.getElementById(containerId);
                if (container && container.classList.contains('loaded')) {
                    Plotly.Plots.resize(containerId);
                }
            });
        }, 250);
    });
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    loadAllCharts();
    setupResizeHandler();
});
