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
    'chart-solar-storage-lcoe-cheap': 'solar_storage_lcoe_cheap.json',
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
    'chart-gas-capacity-vre': 'gas_capacity_vre.json',
    'chart-gas-lcoe-vre': 'gas_lcoe_vre.json',
    'chart-gas-lcoe-impact': 'gas_lcoe_impact.json',
    'chart-leap': 'leap.json',

    // Part 5: Testing Our Understanding
    'chart-regional': 'regional.json',
    'chart-gas-price-sensitivity': 'gas_price_sensitivity.json',
    'chart-gas-crossover-points': 'gas_crossover_points.json',
    'chart-no-cleanfirm': 'no_cleanfirm.json',
    'chart-cf-cost-80pct': 'cf_cost_sensitivity_80pct.json',
    'chart-cf-cost-90pct': 'cf_cost_sensitivity_90pct.json',
    'chart-cf-cost-100pct': 'cf_cost_sensitivity_100pct.json',
    'chart-cf-cost-crossover': 'cf_cost_sensitivity_crossover.json',
    'chart-cleanfirm-tech': 'cleanfirm_tech.json',
    'chart-financing': 'financing.json',
    'chart-no-wind': 'no_wind.json',
    'chart-greedy': 'greedy.json'
};

// Plotly config for all charts
const PLOTLY_CONFIG = {
    responsive: true,
    displayModeBar: false,
    scrollZoom: false,
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

        // Disable zoom/pan by fixing axis ranges
        if (chartData.layout.xaxis) {
            chartData.layout.xaxis.fixedrange = true;
        }
        if (chartData.layout.yaxis) {
            chartData.layout.yaxis.fixedrange = true;
        }
        if (chartData.layout.xaxis2) {
            chartData.layout.xaxis2.fixedrange = true;
        }
        if (chartData.layout.yaxis2) {
            chartData.layout.yaxis2.fixedrange = true;
        }

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

/**
 * Navigation and progress features
 */
function setupNavigation() {
    const progressBar = document.getElementById('progress-bar');
    const backToTopButton = document.getElementById('back-to-top');
    const sectionIndicator = document.getElementById('section-indicator');
    const navItems = document.querySelectorAll('.nav-item');

    // Navigation sections
    const sections = [
        { id: 'hero', element: document.getElementById('hero'), label: 'Intro' },
        { id: 'part-1', element: document.getElementById('part-1'), label: 'Part 1: The Mystery' },
        { id: 'part-2', element: document.getElementById('part-2'), label: 'Part 2: Energy Problem' },
        { id: 'part-3', element: document.getElementById('part-3'), label: 'Part 3: Capacity Problem' },
        { id: 'part-4', element: document.getElementById('part-4'), label: 'Part 4: The Solution' },
        { id: 'part-5', element: document.getElementById('part-5'), label: 'Part 5: Testing' },
        { id: 'takeaways', element: document.getElementById('takeaways'), label: 'Takeaways' }
    ].filter(section => section.element !== null);

    let sectionIndicatorTimeout;

    /**
     * Update progress bar based on scroll position
     */
    function updateProgressBar() {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight - windowHeight;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const progress = (scrollTop / documentHeight) * 100;

        progressBar.style.width = `${Math.min(progress, 100)}%`;
    }

    /**
     * Update active navigation item based on current section
     */
    function updateActiveSection() {
        const scrollPosition = window.pageYOffset + window.innerHeight / 3;

        let activeSection = sections[0];

        for (const section of sections) {
            if (section.element && section.element.offsetTop <= scrollPosition) {
                activeSection = section;
            }
        }

        // Update nav items
        navItems.forEach(item => {
            if (item.dataset.target === activeSection.id) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }

    /**
     * Show section indicator temporarily
     */
    function showSectionIndicator(sectionLabel) {
        sectionIndicator.textContent = sectionLabel;
        sectionIndicator.classList.add('visible');

        clearTimeout(sectionIndicatorTimeout);
        sectionIndicatorTimeout = setTimeout(() => {
            sectionIndicator.classList.remove('visible');
        }, 2000);
    }

    /**
     * Show/hide back to top button
     */
    function updateBackToTop() {
        if (window.pageYOffset > 500) {
            backToTopButton.classList.add('visible');
        } else {
            backToTopButton.classList.remove('visible');
        }
    }

    /**
     * Handle scroll events
     */
    let scrollTimeout;
    function handleScroll() {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
            updateProgressBar();
            updateActiveSection();
            updateBackToTop();
        }, 10);
    }

    /**
     * Smooth scroll to section
     */
    function scrollToSection(targetId, showIndicator = true) {
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            const offsetTop = targetElement.offsetTop;
            window.scrollTo({
                top: offsetTop - 20,
                behavior: 'smooth'
            });

            // Show section indicator
            if (showIndicator) {
                const section = sections.find(s => s.id === targetId);
                if (section) {
                    showSectionIndicator(section.label);
                }
            }
        }
    }

    /**
     * Set up event listeners
     */
    window.addEventListener('scroll', handleScroll, { passive: true });

    // Navigation item clicks and keyboard
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetId = item.dataset.target;
            scrollToSection(targetId);
        });

        // Handle Enter/Space key for accessibility
        item.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                const targetId = item.dataset.target;
                scrollToSection(targetId);
            }
        });
    });

    // Back to top button click
    backToTopButton.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        // Ignore if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        const currentIndex = sections.findIndex(s =>
            document.querySelector(`.nav-item[data-target="${s.id}"]`)?.classList.contains('active')
        );

        // Arrow down or J key - next section
        if ((e.key === 'ArrowDown' || e.key === 'j') && currentIndex < sections.length - 1) {
            e.preventDefault();
            scrollToSection(sections[currentIndex + 1].id);
        }

        // Arrow up or K key - previous section
        if ((e.key === 'ArrowUp' || e.key === 'k') && currentIndex > 0) {
            e.preventDefault();
            scrollToSection(sections[currentIndex - 1].id);
        }

        // Home key - scroll to top
        if (e.key === 'Home') {
            e.preventDefault();
            scrollToSection('hero');
        }

        // End key - scroll to bottom
        if (e.key === 'End') {
            e.preventDefault();
            scrollToSection(sections[sections.length - 1].id);
        }
    });

    // Initial update
    handleScroll();
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    loadAllCharts();
    setupResizeHandler();
    setupNavigation();
});
