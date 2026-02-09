/**
 * CPU Memory Chart controller using Chart.js.
 * Displays 4 lines: RSS Total, RSS Delta, tracemalloc current, tracemalloc peak.
 */
class MemoryChart {
    constructor(canvasId) {
        this.maxPoints = 500;
        this.labels = [];
        this.ctx = document.getElementById(canvasId).getContext('2d');

        this.chart = new Chart(this.ctx, {
            type: 'line',
            data: {
                labels: this.labels,
                datasets: [
                    {
                        label: 'RSS Total (MB)',
                        data: [],
                        borderColor: '#4f6ef7',
                        backgroundColor: 'rgba(79,110,247,0.1)',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        yAxisID: 'y',
                        fill: true,
                    },
                    {
                        label: 'RSS Delta (MB)',
                        data: [],
                        borderColor: '#f0ad4e',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        yAxisID: 'y1',
                    },
                    {
                        label: 'tracemalloc current (MB)',
                        data: [],
                        borderColor: '#2ea043',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        yAxisID: 'y1',
                    },
                    {
                        label: 'tracemalloc peak (MB)',
                        data: [],
                        borderColor: '#d33',
                        borderWidth: 1.5,
                        borderDash: [5, 3],
                        pointRadius: 0,
                        yAxisID: 'y1',
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                scales: {
                    x: {
                        display: true,
                        title: { display: true, text: 'Chunk Index', color: '#888', font: { size: 10 } },
                        ticks: { color: '#666', font: { size: 9 }, maxTicksLimit: 10 },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                    },
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: { display: true, text: 'RSS (MB)', color: '#4f6ef7', font: { size: 10 } },
                        ticks: { color: '#4f6ef7', font: { size: 9 } },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        title: { display: true, text: 'Delta / Alloc (MB)', color: '#f0ad4e', font: { size: 10 } },
                        ticks: { color: '#f0ad4e', font: { size: 9 } },
                        grid: { drawOnChartArea: false },
                    },
                },
                plugins: {
                    legend: {
                        labels: { color: '#aaa', font: { size: 10 }, usePointStyle: true, pointStyle: 'line' },
                    },
                },
            },
        });
    }

    addDataPoint(chunkIndex, memory) {
        this.labels.push(chunkIndex);
        this.chart.data.datasets[0].data.push(memory.rss_mb);
        this.chart.data.datasets[1].data.push(memory.rss_delta_mb);
        this.chart.data.datasets[2].data.push(memory.tracemalloc_current_mb);
        this.chart.data.datasets[3].data.push(memory.tracemalloc_peak_mb);

        // Sliding window
        if (this.labels.length > this.maxPoints) {
            this.labels.shift();
            this.chart.data.datasets.forEach(ds => ds.data.shift());
        }

        this.chart.update();
    }

    clear() {
        this.labels.length = 0;
        this.chart.data.datasets.forEach(ds => { ds.data.length = 0; });
        this.chart.update();
    }
}
