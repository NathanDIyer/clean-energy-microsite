/**
 * Text Export Feature for AI Analysis
 * Generates a structured text version of the microsite content
 */

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    const exportBtn = document.getElementById('export-text-btn');
    const modal = document.getElementById('text-export-modal');
    const closeBtn = modal.querySelector('.modal-close');
    const copyBtn = document.getElementById('copy-text-btn');
    const downloadBtn = document.getElementById('download-text-btn');
    const textArea = document.getElementById('text-content');

    /**
     * Generate structured text content from the microsite
     */
    function generateTextContent() {
        let content = '';

        // Header
        content += '═══════════════════════════════════════════════════════════\n';
        content += 'THE CLEAN ENERGY PUZZLE - TEXT VERSION FOR AI ANALYSIS\n';
        content += '═══════════════════════════════════════════════════════════\n\n';
        content += 'Source: Clean Energy Optimization Microsite\n';
        content += 'URL: https://www.niyer-energy-dashboards.com/\n';
        content += 'Generated: ' + new Date().toLocaleString() + '\n\n';

        // Hero Section
        const hero = document.querySelector('.hero');
        if (hero) {
            content += '───────────────────────────────────────────────────────────\n';
            content += 'INTRODUCTION\n';
            content += '───────────────────────────────────────────────────────────\n\n';

            const title = hero.querySelector('h1')?.textContent.trim();
            const subtitle = hero.querySelector('.subtitle')?.textContent.trim();
            const cta = hero.querySelector('.hero-cta')?.textContent.trim();

            if (title) content += title + '\n\n';
            if (subtitle) content += subtitle + '\n\n';
            if (cta) content += cta + '\n\n';
        }

        // Process each main part
        const parts = [
            { selector: '#part-1', title: 'PART 1: THE MYSTERY' },
            { selector: '#part-2', title: 'PART 2: THE ENERGY PROBLEM' },
            { selector: '#part-3', title: 'PART 3: THE CAPACITY PROBLEM' },
            { selector: '#part-4', title: 'PART 4: THE SOLUTION' },
            { selector: '#part-5', title: 'PART 5: TESTING OUR UNDERSTANDING' },
            { selector: '#takeaways', title: 'TAKEAWAYS & IMPLICATIONS' }
        ];

        parts.forEach(part => {
            const partElement = document.querySelector(part.selector);
            if (!partElement) return;

            content += '═══════════════════════════════════════════════════════════\n';
            content += part.title + '\n';
            content += '═══════════════════════════════════════════════════════════\n\n';

            // Get part subtitle if exists
            const subtitle = partElement.querySelector('.part-subtitle')?.textContent.trim();
            if (subtitle) {
                content += 'Theme: ' + subtitle + '\n\n';
            }

            // Get all sections within this part
            let currentElement = partElement.nextElementSibling;
            while (currentElement && !currentElement.classList.contains('part-header')) {
                if (currentElement.classList.contains('section')) {
                    content += extractSectionContent(currentElement);
                }
                currentElement = currentElement.nextElementSibling;
            }
        });

        // Footer
        content += '═══════════════════════════════════════════════════════════\n';
        content += 'END OF DOCUMENT\n';
        content += '═══════════════════════════════════════════════════════════\n\n';
        content += 'For interactive charts and full details, visit:\n';
        content += 'https://www.niyer-energy-dashboards.com/\n';

        return content;
    }

    /**
     * Extract data from a Plotly chart
     */
    function extractChartData(chartElement) {
        let content = '';

        try {
            // Check if Plotly chart exists - Plotly stores data directly on the DOM element
            // Try multiple ways to access the data
            let data = chartElement.data;
            let layout = chartElement.layout;

            // If not available directly, try the Plotly internal structure
            if (!data && chartElement._fullData) {
                data = chartElement._fullData;
            }
            if (!layout && chartElement._fullLayout) {
                layout = chartElement._fullLayout;
            }

            // Debug logging
            console.log('Chart element:', chartElement.id);
            console.log('Has data:', !!data, 'Has layout:', !!layout);
            console.log('Data length:', data?.length);

            // If still no data, chart hasn't loaded
            if (!data || !layout || data.length === 0) {
                console.warn('Chart data not available for:', chartElement.id);
                return content;
            }

            // Chart title
            const titleText = layout.title?.text || layout.title || '';
            if (titleText) {
                content += 'CHART: ' + titleText + '\n';
            }

            // Chart type detection
            const chartType = data[0]?.type || 'scatter';
            content += 'Type: ' + chartType + '\n\n';

            // Extract data based on chart type
            if (chartType === 'bar' || chartType === 'scatter' || chartType === 'line') {
                // For bar, scatter, and line charts
                data.forEach((trace, index) => {
                    if (trace.name) {
                        content += 'Series: ' + trace.name + '\n';
                    } else if (data.length > 1) {
                        content += 'Series ' + (index + 1) + '\n';
                    }

                    if (trace.x && trace.y) {
                        const dataPoints = Math.min(trace.x.length, 100); // Show up to 100 points
                        content += 'Data points (showing first ' + dataPoints + ' of ' + trace.x.length + '):\n';

                        for (let i = 0; i < dataPoints; i++) {
                            const xVal = formatValue(trace.x[i]);
                            const yVal = formatValue(trace.y[i]);
                            content += '  ' + xVal + ': ' + yVal + '\n';
                        }

                        if (trace.x.length > dataPoints) {
                            content += '  ... (' + (trace.x.length - dataPoints) + ' more points)\n';
                        }
                    }
                    content += '\n';
                });
            } else if (chartType === 'heatmap') {
                // For heatmaps
                data.forEach((trace, index) => {
                    if (trace.name) {
                        content += 'Series: ' + trace.name + '\n';
                    }

                    if (trace.z && Array.isArray(trace.z)) {
                        content += 'Heatmap dimensions: ' + trace.z.length + ' rows x ' + (trace.z[0]?.length || 0) + ' columns\n';
                        content += 'Value range: ' + findMinMax(trace.z) + '\n';

                        // Include x and y labels if available
                        if (trace.x && trace.x.length > 0) {
                            content += 'X labels: ' + trace.x.slice(0, 10).join(', ');
                            if (trace.x.length > 10) content += ' ... (' + trace.x.length + ' total)';
                            content += '\n';
                        }
                        if (trace.y && trace.y.length > 0) {
                            content += 'Y labels: ' + trace.y.slice(0, 10).join(', ');
                            if (trace.y.length > 10) content += ' ... (' + trace.y.length + ' total)';
                            content += '\n';
                        }
                    }
                    content += '\n';
                });
            }

            // Axis labels
            if (layout.xaxis && layout.xaxis.title) {
                const xTitle = layout.xaxis.title.text || layout.xaxis.title;
                content += 'X-axis: ' + xTitle + '\n';
            }
            if (layout.yaxis && layout.yaxis.title) {
                const yTitle = layout.yaxis.title.text || layout.yaxis.title;
                content += 'Y-axis: ' + yTitle + '\n';
            }

            content += '\n';

        } catch (error) {
            console.warn('Error extracting chart data:', error);
        }

        return content;
    }

    /**
     * Format a value for display
     */
    function formatValue(val) {
        if (typeof val === 'number') {
            return val.toFixed(2);
        }
        return String(val);
    }

    /**
     * Find min and max in a 2D array (for heatmaps)
     */
    function findMinMax(arr) {
        let min = Infinity;
        let max = -Infinity;

        for (let row of arr) {
            for (let val of row) {
                if (typeof val === 'number') {
                    min = Math.min(min, val);
                    max = Math.max(max, val);
                }
            }
        }

        return min.toFixed(2) + ' to ' + max.toFixed(2);
    }

    /**
     * Extract content from a section element
     */
    function extractSectionContent(section) {
        let content = '';

        // Section heading
        const heading = section.querySelector('h2');
        if (heading) {
            content += '───────────────────────────────────────────────────────────\n';
            content += heading.textContent.trim() + '\n';
            content += '───────────────────────────────────────────────────────────\n\n';
        }

        // Get all direct paragraphs and content
        const children = section.children;
        for (let child of children) {
            // Extract chart data
            if (child.classList.contains('chart-container')) {
                const chartId = child.id;
                if (chartId) {
                    content += '...........................................................\n';
                    content += 'CHART DATA: ' + chartId.replace('chart-', '').toUpperCase() + '\n';
                    content += '...........................................................\n\n';

                    // Try to get actual Plotly data
                    const chartData = extractChartData(child);
                    if (chartData) {
                        content += chartData;
                    } else {
                        content += '[Chart data not available - view interactive chart on site]\n\n';
                    }
                }
                continue;
            }

            // Get paragraph text
            if (child.tagName === 'P') {
                const text = child.textContent.trim();
                if (text && !child.classList.contains('chart-note')) {
                    // Check if it's an insight
                    if (child.classList.contains('insight')) {
                        content += '>> KEY INSIGHT: ' + text + '\n\n';
                    } else if (child.classList.contains('transition-text')) {
                        content += '>> ' + text + '\n\n';
                    } else {
                        content += text + '\n\n';
                    }
                }
            }

            // Get callouts
            if (child.classList.contains('callout')) {
                const calloutText = child.textContent.trim();
                if (child.classList.contains('warning')) {
                    content += '*** IMPORTANT: ' + calloutText + '\n\n';
                } else if (child.classList.contains('danger')) {
                    content += '*** CRITICAL: ' + calloutText + '\n\n';
                } else {
                    content += '** NOTE: ' + calloutText + '\n\n';
                }
            }

            // Get lists
            if (child.tagName === 'UL' && child.classList.contains('region-notes')) {
                const items = child.querySelectorAll('li');
                items.forEach(item => {
                    content += '  - ' + item.textContent.trim() + '\n';
                });
                content += '\n';
            }

            // Get two-column content
            if (child.classList.contains('two-column')) {
                const columns = child.querySelectorAll('.column');
                columns.forEach(col => {
                    const colHeading = col.querySelector('h3')?.textContent.trim();
                    const colText = col.querySelector('p')?.textContent.trim();
                    if (colHeading && colText) {
                        content += colHeading + ': ' + colText + '\n\n';
                    }
                });
            }

            // Get implication cards
            if (child.classList.contains('implications-grid')) {
                const cards = child.querySelectorAll('.implication-card');
                cards.forEach(card => {
                    const cardTitle = card.querySelector('h3')?.textContent.trim();
                    const cardText = card.querySelector('p')?.textContent.trim();
                    if (cardTitle && cardText) {
                        content += '  > ' + cardTitle + '\n    ' + cardText + '\n\n';
                    }
                });
            }

            // Get assumption tables
            if (child.classList.contains('assumptions-dropdown')) {
                const summaryText = child.querySelector('summary')?.textContent.trim();
                if (summaryText) {
                    content += 'ASSUMPTIONS: ' + summaryText + '\n';
                }

                // Get table data if exists
                const table = child.querySelector('table');
                if (table) {
                    const headers = Array.from(table.querySelectorAll('th')).map(th => th.textContent.trim());
                    content += headers.join(' | ') + '\n';
                    content += '-'.repeat(headers.join(' | ').length) + '\n';

                    const rows = table.querySelectorAll('tbody tr');
                    rows.forEach(row => {
                        const cells = Array.from(row.querySelectorAll('td')).map(td => td.textContent.trim());
                        content += cells.join(' | ') + '\n';
                    });
                    content += '\n';
                }

                // Get other assumption content
                const assumptionText = child.querySelector('.assumptions-content p, .assumptions-content ul');
                if (assumptionText) {
                    content += assumptionText.textContent.trim() + '\n\n';
                }
            }
        }

        return content;
    }

    /**
     * Open modal and generate content
     */
    function openModal() {
        // Check if charts are loaded
        const chartContainers = document.querySelectorAll('.chart-container');
        let loadedCharts = 0;
        let totalCharts = 0;

        chartContainers.forEach(container => {
            totalCharts++;
            if (container.classList.contains('loaded') || container.data) {
                loadedCharts++;
            }
        });

        const textContent = generateTextContent();
        textArea.value = textContent;

        // Add note about chart loading status
        if (loadedCharts < totalCharts) {
            textArea.value = 'NOTE: Some charts may not have loaded yet (' + loadedCharts + ' of ' + totalCharts + ' loaded). ' +
                           'Chart data will be limited. For complete data, wait for all charts to load before exporting.\n\n' +
                           '═══════════════════════════════════════════════════════════\n\n' +
                           textContent;
        }

        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    /**
     * Close modal
     */
    function closeModal() {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }

    /**
     * Copy text to clipboard
     */
    async function copyToClipboard() {
        try {
            await navigator.clipboard.writeText(textArea.value);

            // Visual feedback
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            copyBtn.style.background = '#34a853';

            setTimeout(() => {
                copyBtn.textContent = originalText;
                copyBtn.style.background = '';
            }, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
            alert('Failed to copy to clipboard. Please select and copy manually.');
        }
    }

    /**
     * Download text as file
     */
    function downloadText() {
        const blob = new Blob([textArea.value], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'clean-energy-puzzle-text-version.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        // Visual feedback
        const originalText = downloadBtn.textContent;
        downloadBtn.textContent = 'Downloaded!';

        setTimeout(() => {
            downloadBtn.textContent = originalText;
        }, 2000);
    }

    // Event listeners
    exportBtn.addEventListener('click', openModal);
    closeBtn.addEventListener('click', closeModal);
    copyBtn.addEventListener('click', copyToClipboard);
    downloadBtn.addEventListener('click', downloadText);

    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeModal();
        }
    });
});
