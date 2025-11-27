// Global state
let currentModel = null;
let optimizationInProgress = false;
let modelLibrary = [];
let connectedDevices = [];
let optimizationHistory = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupNavigation();
    loadInitialData();
    updateDeviceConfiguration(); // Initialize device config UI
    updateOptimizeButton();
});

// Set up all event listeners
function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            navigateToPage(page);
        });
    });

    // Model selection events (optimize page)
    const modelSelect = document.getElementById('modelSelect');
    const clearSelectionBtn = document.getElementById('clearSelectionBtn');
    
    if (modelSelect) {
        modelSelect.addEventListener('change', handleModelSelection);
    }
    if (clearSelectionBtn) {
        clearSelectionBtn.addEventListener('click', clearModelSelection);
    }
    
    // Optimization events
    const optimizeBtn = document.getElementById('optimizeBtn');
    if (optimizeBtn) optimizeBtn.addEventListener('click', startOptimization);
    
    // Other buttons
    const addNewModelBtn = document.getElementById('addNewModelBtn');
    if (addNewModelBtn) addNewModelBtn.addEventListener('click', addNewModel);
    
    const refreshModelsBtn = document.getElementById('refreshModelsBtn');
    if (refreshModelsBtn) refreshModelsBtn.addEventListener('click', refreshModelLibrary);
    
    const githubBtn = document.getElementById('githubBtn');
    if (githubBtn) githubBtn.addEventListener('click', openGitHubRepo);
    
    // Menu events
    if (window.electronAPI) {
        window.electronAPI.onMenuImportModel(() => {
            navigateToPage('models'); // Navigate to models page instead
        });
    }
    
    // Configuration change events
    const deviceSelect = document.getElementById('deviceSelect');
    const precisionSelect = document.getElementById('precisionSelect');
    if (deviceSelect) {
        deviceSelect.addEventListener('change', () => {
            updateDeviceConfiguration();
            updateOptimizeButton();
        });
    }
    if (precisionSelect) precisionSelect.addEventListener('change', updateOptimizeButton);
    
    // History page events
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', clearOptimizationHistory);
    
    // Export/Report buttons (optimize page results section)
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) exportBtn.addEventListener('click', saveOptimizedModelToFile);
    
    const reportBtn = document.getElementById('reportBtn');
    if (reportBtn) reportBtn.addEventListener('click', generateOptimizationReport);
    
    // Modal events
    const parameterModal = document.getElementById('parameterModal');
    if (parameterModal) {
        parameterModal.addEventListener('click', (e) => {
            if (e.target === parameterModal) {
                closeParameterModal();
            }
        });
    }
    
    // Setup range sliders
    setupRangeSliders();
}

// Navigation functions
function setupNavigation() {
    // Show home page by default
    navigateToPage('home');
}

function navigateToPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.style.display = 'none';
    });
    
    // Remove active class from all nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(pageId + 'Page');
    if (targetPage) {
        targetPage.style.display = 'block';
    }
    
    // Add active class to selected nav item
    const targetNavItem = document.querySelector(`[data-page="${pageId}"]`);
    if (targetNavItem) {
        targetNavItem.classList.add('active');
    }
    
    // Load page-specific data
    loadPageData(pageId);
}

function loadPageData(pageId) {
    switch(pageId) {
        case 'home':
            updateDashboard();
            break;
        case 'optimize':
            updateModelSelector();
            break;
        case 'batch':
            initializeBatchOptimizePage();
            break;
        case 'models':
            updateModelsGrid();
            break;
        case 'history':
            updateHistoryTable();
            break;
    }
}

// Dashboard functions
async function updateDashboard() {
    try {
        // Get statistics from backend
        const statsResult = await window.electronAPI.getHistoryStatistics();
        
        if (statsResult.success) {
            const stats = statsResult.statistics;
            
            // Update dashboard elements with backend statistics
            const modelCount = document.getElementById('modelCount');
            const optimizationCount = document.getElementById('optimizationCount');
            const avgPerformance = document.getElementById('avgPerformance');
            const successRate = document.getElementById('successRate');
            const avgMemorySaved = document.getElementById('avgMemorySaved');
            const lastOptimization = document.getElementById('lastOptimization');
            
            // Basic counts
            const originalModelsCount = modelLibrary.filter(model => model.is_original !== false).length;
            if (modelCount) modelCount.textContent = `${originalModelsCount} models in your library`;
            if (optimizationCount) optimizationCount.textContent = `${stats.total_optimizations || 0} total optimizations`;
            
            // Success rate
            if (successRate) {
                if (stats.total_optimizations > 0) {
                    successRate.textContent = `${stats.success_rate.toFixed(1)}% success rate (${stats.successful_optimizations}/${stats.total_optimizations})`;
                } else {
                    successRate.textContent = 'No optimizations yet';
                }
            }
            
            // Average performance gain
            if (avgPerformance) {
                if (stats.average_performance_gain > 0) {
                    avgPerformance.textContent = `${stats.average_performance_gain}% average improvement`;
                } else {
                    avgPerformance.textContent = 'No optimizations yet';
                }
            }
            
            // Average memory reduction
            if (avgMemorySaved) {
                if (stats.average_memory_reduction > 0) {
                    avgMemorySaved.textContent = `${stats.average_memory_reduction}% average reduction`;
                } else {
                    avgMemorySaved.textContent = 'No optimizations yet';
                }
            }
            
            // Last optimization
            if (lastOptimization) {
                if (optimizationHistory.length > 0) {
                    const lastOpt = optimizationHistory[0]; // First item is the most recent
                    const timeAgo = getTimeAgo(new Date(lastOpt.timestamp));
                    lastOptimization.textContent = `${lastOpt.model_info?.name || 'Unknown model'} - ${timeAgo}`;
                } else {
                    lastOptimization.textContent = 'No recent optimizations';
                }
            }
            
            // Update dashboard stats with backend data
            updateDashboardStatsFromBackend(stats);
        } else {
            // Fallback to local calculation if backend fails
            updateDashboardLocal();
        }
    } catch (error) {
        console.error('Failed to update dashboard:', error);
        // Fallback to local calculation
        updateDashboardLocal();
    }
    
    updateActivityFeed();
}

function updateDashboardLocal() {
    // Fallback function using local data
    const modelCount = document.getElementById('modelCount');
    const optimizationCount = document.getElementById('optimizationCount');
    const avgPerformance = document.getElementById('avgPerformance');
    const successRate = document.getElementById('successRate');
    const avgMemorySaved = document.getElementById('avgMemorySaved');
    const lastOptimization = document.getElementById('lastOptimization');
    
    // Basic counts
    const originalModelsCount = modelLibrary.filter(model => model.is_original !== false).length;
    if (modelCount) modelCount.textContent = `${originalModelsCount} models in your library`;
    
    const optimizedModels = modelLibrary.filter(model => !model.isOriginal);
    if (optimizationCount) optimizationCount.textContent = `${optimizedModels.length} models optimized`;
    
    // Success rate calculation
    const successfulOptimizations = optimizationHistory.filter(h => h.results?.success === true).length;
    const totalOptimizations = optimizationHistory.length;
    const successPercentage = totalOptimizations > 0 ? (successfulOptimizations / totalOptimizations * 100).toFixed(1) : 0;
    
    if (successRate) {
        successRate.textContent = totalOptimizations > 0 
            ? `${successPercentage}% success rate (${successfulOptimizations}/${totalOptimizations})`
            : 'No optimizations yet';
    }
    
    // Average performance gain
    const successfulModels = optimizedModels.filter(model => model.performanceGain);
    const avgGain = successfulModels.length > 0 
        ? successfulModels.reduce((sum, model) => {
            const gain = parseFloat(model.performanceGain.replace('%', '')) || 0;
            return sum + gain;
        }, 0) / successfulModels.length
        : 0;
    
    if (avgPerformance) {
        avgPerformance.textContent = successfulModels.length > 0 
            ? `${avgGain.toFixed(1)}% average improvement`
            : 'No optimizations yet';
    }
    
    // Average memory reduction
    const modelsWithMemoryData = optimizedModels.filter(model => model.memoryReduction);
    const avgMemory = modelsWithMemoryData.length > 0 
        ? modelsWithMemoryData.reduce((sum, model) => {
            const reduction = parseFloat(model.memoryReduction.replace('%', '')) || 0;
            return sum + reduction;
        }, 0) / modelsWithMemoryData.length
        : 0;
    
    if (avgMemorySaved) {
        avgMemorySaved.textContent = modelsWithMemoryData.length > 0 
            ? `${avgMemory.toFixed(1)}% average reduction`
            : 'No optimizations yet';
    }
    
    // Last optimization
    const lastOpt = optimizationHistory.length > 0 
        ? optimizationHistory[0]
        : null;
    
    if (lastOptimization) {
        if (lastOpt) {
            const timeAgo = getTimeAgo(new Date(lastOpt.timestamp));
            lastOptimization.textContent = `${lastOpt.model_info?.name || lastOpt.modelName || 'Unknown'} - ${timeAgo}`;
        } else {
            lastOptimization.textContent = 'No recent optimizations';
        }
    }
}

function updateDashboardStatsFromBackend(stats) {
    const popularDevice = document.getElementById('popularDevice');
    const popularPrecision = document.getElementById('popularPrecision');
    const totalOptimizations = document.getElementById('totalOptimizations');
    const weeklyOptimizations = document.getElementById('weeklyOptimizations');
    
    // Most used device from backend stats
    if (popularDevice) {
        popularDevice.textContent = stats.most_used_device ? stats.most_used_device.toUpperCase() : '-';
    }
    
    // Most used precision from backend stats
    if (popularPrecision) {
        popularPrecision.textContent = stats.most_used_precision ? stats.most_used_precision.toUpperCase() : '-';
    }
    
    // Total optimizations
    if (totalOptimizations) {
        totalOptimizations.textContent = stats.total_optimizations || 0;
    }
    
    // Weekly optimizations (calculate from recent history)
    if (weeklyOptimizations) {
        // Filter optimizations from the last 7 days
        const weekAgo = new Date();
        weekAgo.setDate(weekAgo.getDate() - 7);
        
        const recentOptimizations = optimizationHistory.filter(opt => {
            try {
                const optDate = new Date(opt.timestamp);
                return optDate >= weekAgo;
            } catch (e) {
                return false;
            }
        });
        
        weeklyOptimizations.textContent = recentOptimizations.length;
    }
}

function updateDashboardStats() {
    const popularDevice = document.getElementById('popularDevice');
    const popularPrecision = document.getElementById('popularPrecision');
    const totalOptimizations = document.getElementById('totalOptimizations');
    const weeklyOptimizations = document.getElementById('weeklyOptimizations');
    
    // Most used device
    const deviceCounts = {};
    optimizationHistory.forEach(opt => {
        const device = opt.targetDevice;
        deviceCounts[device] = (deviceCounts[device] || 0) + 1;
    });
    
    const mostUsedDevice = Object.keys(deviceCounts).reduce((a, b) => 
        deviceCounts[a] > deviceCounts[b] ? a : b, Object.keys(deviceCounts)[0]
    );
    
    if (popularDevice) {
        popularDevice.textContent = mostUsedDevice ? mostUsedDevice.toUpperCase() : '-';
    }
    
    // Most used precision (from parameters)
    const precisionCounts = {};
    optimizationHistory.forEach(opt => {
        if (opt.parameters && opt.parameters.precision) {
            const precision = opt.parameters.precision;
            precisionCounts[precision] = (precisionCounts[precision] || 0) + 1;
        }
    });
    
    const mostUsedPrecision = Object.keys(precisionCounts).reduce((a, b) => 
        precisionCounts[a] > precisionCounts[b] ? a : b, Object.keys(precisionCounts)[0]
    );
    
    if (popularPrecision) {
        popularPrecision.textContent = mostUsedPrecision ? mostUsedPrecision.toUpperCase() : '-';
    }
    
    // Total optimizations
    if (totalOptimizations) {
        totalOptimizations.textContent = optimizationHistory.length.toString();
    }
    
    // Weekly optimizations
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    const weeklyCount = optimizationHistory.filter(opt => 
        new Date(opt.timestamp) > weekAgo
    ).length;
    
    if (weeklyOptimizations) {
        weeklyOptimizations.textContent = weeklyCount.toString();
    }
}

function getTimeAgo(date) {
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);
    
    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d ago`;
    return `${Math.floor(diffInSeconds / 604800)}w ago`;
}

function openGitHubRepo() {
    // Replace with your actual GitHub repository URL
    const repoUrl = 'https://github.com/yourusername/tsurutune-app';
    
    // In Electron, we need to use the shell API to open external URLs
    if (window.electronAPI && window.electronAPI.openExternal) {
        window.electronAPI.openExternal(repoUrl);
    } else {
        // Fallback for development/browser environment
        window.open(repoUrl, '_blank');
    }
}

function updateActivityFeed() {
    const activityList = document.getElementById('activityList');
    if (!activityList) return;
    
    // Show recent optimizations (last 5)
    const recentOptimizations = optimizationHistory.slice(-5).reverse();
    
    if (recentOptimizations.length === 0) {
        activityList.innerHTML = `
            <div class="activity-item">
                <div class="activity-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                    </svg>
                </div>
                <div class="activity-content">
                    <div class="activity-title">Welcome to TsuruTune!</div>
                    <div class="activity-time">Start by adding your first model</div>
                </div>
            </div>
        `;
        return;
    }
    
    const activities = recentOptimizations.map(opt => {
        const statusIcon = opt.status === 'success' ? 
            `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="13,2 3,14 12,14 11,22 21,10 12,10 13,2"/>
            </svg>` :
            `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
            </svg>`;
        
        const statusColor = opt.status === 'success' ? '#388e3c' : '#d32f2f';
        const details = opt.status === 'success' 
            ? `${opt.performanceGain || '+0%'} performance • ${opt.memoryReduction || '0%'} memory`
            : 'Optimization failed';
        
        return `
            <div class="activity-item">
                <div class="activity-icon" style="color: ${statusColor};">
                    ${statusIcon}
                </div>
                <div class="activity-content">
                    <div class="activity-title">${opt.modelName} (${opt.targetDevice.toUpperCase()})</div>
                    <div class="activity-time">${getTimeAgo(new Date(opt.timestamp))} • ${details}</div>
                </div>
            </div>
        `;
    }).join('');
    
    activityList.innerHTML = activities;
}

function showOptimizationHistory() {
    if (optimizationHistory.length === 0) {
        alert('No optimizations yet. Start by adding models and optimizing them!');
        return;
    }
    
    const historyText = optimizationHistory.map(opt => 
        `${opt.modelName}: ${opt.performanceGain} improvement, ${opt.memoryReduction} memory reduction`
    ).join('\n');
    
    alert('Optimization History:\n\n' + historyText);
}

// Models page functions
function updateModelsGrid() {
    const modelsTableBody = document.getElementById('modelsTableBody');
    const emptyModelsState = document.getElementById('emptyModelsState');
    const modelsTable = document.querySelector('.models-table');
    
    if (!modelsTableBody) return;
    
    // Filter to show only original models (imported models)
    const originalModels = modelLibrary.filter(model => model.is_original !== false);
    
    if (originalModels.length === 0) {
        if (modelsTable) modelsTable.style.display = 'none';
        if (emptyModelsState) emptyModelsState.style.display = 'block';
        return;
    }
    
    if (modelsTable) modelsTable.style.display = 'table';
    if (emptyModelsState) emptyModelsState.style.display = 'none';
    
    const modelsHTML = originalModels.map(model => {
        const addedDate = model.imported_at ? new Date(model.imported_at).toLocaleDateString() : 
                         (model.dateAdded ? new Date(model.dateAdded).toLocaleDateString() : 'Unknown');
        const fileSize = model.size_mb ? `${model.size_mb} MB` : (model.size ? `${(model.size / (1024*1024)).toFixed(1)} MB` : 'Unknown');
        const modelType = model.type || (model.extension ? model.extension.toUpperCase() : 'Unknown');
        
        return `
            <tr>
                <td>
                    <strong>${model.name}</strong>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">
                        ${model.description || (model.auto_discovered ? 'Auto-discovered model' : 'Imported model')}
                    </div>
                </td>
                <td>
                    <span class="table-model-type original">
                        ${modelType}
                    </span>
                </td>
                <td>
                    <span class="model-size">
                        ${fileSize}
                    </span>
                </td>
                <td>${addedDate}</td>
                <td>
                    <div class="table-actions">
                        <button class="btn btn-sm btn-primary" onclick="optimizeExistingModel('${model.id}')">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                            </svg>
                            Optimize
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="deleteModel('${model.id}')">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3,6 5,6 21,6"/>
                                <path d="m19,6v14a2,2 0 0,1 -2,2H7a2,2 0 0,1 -2,-2V6m3,0V4a2,2 0 0,1 2,-2h4a2,2 0 0,1 2,2v2"/>
                            </svg>
                            Delete
                        </button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
    
    modelsTableBody.innerHTML = modelsHTML;
}

// Devices page functions
function updateDevicesGrid() {
    const devicesGrid = document.getElementById('devicesGrid');
    if (!devicesGrid) return;
    
    if (connectedDevices.length === 0) {
        devicesGrid.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1; text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; color: var(--text-muted);">
                    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>
                        <rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>
                        <line x1="6" y1="6" x2="6.01" y2="6"/>
                        <line x1="6" y1="18" x2="6.01" y2="18"/>
                    </svg>
                </div>
                <h3>No devices detected</h3>
                <p>Connect your Jetson device or scan for available devices</p>
                <button class="card-action" onclick="scanForDevices()">Scan for Devices</button>
            </div>
        `;
        return;
    }
    
    const devicesHTML = connectedDevices.map(device => `
        <div class="device-card ${device.status}">
            <div class="device-header">
                <h3>${device.name}</h3>
                <span class="device-status ${device.status}">${device.status}</span>
            </div>
            <div class="device-specs">
                <div class="spec">
                    <span class="spec-label">Memory:</span>
                    <span>${device.memory}</span>
                </div>
                <div class="spec">
                    <span class="spec-label">CUDA Cores:</span>
                    <span>${device.cudaCores}</span>
                </div>
                <div class="spec">
                    <span class="spec-label">Tensor Cores:</span>
                    <span>${device.tensorCores}</span>
                </div>
            </div>
            <div class="device-actions">
                <button class="device-btn primary" onclick="connectDevice('${device.id}')">
                    ${device.status === 'connected' ? 'Disconnect' : 'Connect'}
                </button>
                <button class="device-btn secondary" onclick="deviceDetails('${device.id}')">Details</button>
            </div>
        </div>
    `).join('');
    
    devicesGrid.innerHTML = devicesHTML;
}

// Device configuration functions
function updateDeviceConfiguration() {
    const deviceSelect = document.getElementById('deviceSelect');
    const cudaConfig = document.getElementById('cudaConfig');
    const cpuConfig = document.getElementById('cpuConfig');
    
    if (!deviceSelect || !cudaConfig || !cpuConfig) return;
    
    const selectedDevice = deviceSelect.value;
    
    if (selectedDevice === 'cuda') {
        cudaConfig.style.display = 'block';
        cpuConfig.style.display = 'none';
    } else if (selectedDevice === 'cpu') {
        cudaConfig.style.display = 'none';
        cpuConfig.style.display = 'block';
    }
}

// Setup range sliders with value displays
function setupRangeSliders() {
    // CUDA sparsity target
    const cudaSparsityTarget = document.getElementById('cudaSparsityTarget');
    const cudaSparsityValue = document.getElementById('cudaSparsityValue');
    if (cudaSparsityTarget && cudaSparsityValue) {
        cudaSparsityTarget.addEventListener('input', (e) => {
            cudaSparsityValue.textContent = e.target.value + '%';
        });
    }
    
    // CPU channel pruning
    const cpuChannelPruning = document.getElementById('cpuChannelPruning');
    const cpuChannelPruningValue = document.getElementById('cpuChannelPruningValue');
    if (cpuChannelPruning && cpuChannelPruningValue) {
        cpuChannelPruning.addEventListener('input', (e) => {
            cpuChannelPruningValue.textContent = e.target.value + '%';
        });
    }
}

// Benchmark functions
function updateBenchmarkOptions() {
    const benchmarkModels = document.getElementById('benchmarkModels');
    if (!benchmarkModels) return;
    
    if (modelLibrary.length === 0) {
        benchmarkModels.innerHTML = '<p>No models available for benchmarking</p>';
        return;
    }
    
    const modelsHTML = modelLibrary.map(model => `
        <div class="checkbox-item">
            <label class="toggle-switch">
                <input type="checkbox" value="${model.id}">
                <span class="toggle-slider"></span>
            </label>
            <label>${model.name}</label>
        </div>
    `).join('');
    
    benchmarkModels.innerHTML = modelsHTML;
}

async function runBenchmark() {
    const selectedModels = Array.from(document.querySelectorAll('#benchmarkModels input:checked'))
        .map(input => input.value);
    
    if (selectedModels.length === 0) {
        alert('Please select at least one model to benchmark');
        return;
    }
    
    // Simulate benchmark results
    const results = selectedModels.map(modelId => {
        const model = modelLibrary.find(m => m.id === modelId);
        return {
            name: model.name,
            inferenceTime: Math.random() * 50 + 10,
            memoryUsage: Math.random() * 2 + 1,
            powerConsumption: Math.random() * 15 + 5
        };
    });
    
    displayBenchmarkResults(results);
}

function displayBenchmarkResults(results) {
    const benchmarkResults = document.getElementById('benchmarkResults');
    if (!benchmarkResults) return;
    
    benchmarkResults.style.display = 'block';
    
    const ctx = document.getElementById('benchmarkChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: results.map(r => r.name),
            datasets: [{
                label: 'Inference Time (ms)',
                data: results.map(r => r.inferenceTime),
                backgroundColor: 'rgba(102, 126, 234, 0.8)'
            }, {
                label: 'Memory Usage (GB)',
                data: results.map(r => r.memoryUsage),
                backgroundColor: 'rgba(72, 187, 120, 0.8)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Device functions
async function scanForDevices() {
    updateStatus('processing', 'Scanning for devices...');
    
    // Simulate device discovery
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const mockDevices = [
        {
            id: 'jetson-xavier-001',
            name: 'Jetson Xavier Nano',
            status: 'detected',
            memory: '4GB',
            cudaCores: '128',
            tensorCores: '8'
        },
        {
            id: 'jetson-orin-001',
            name: 'Jetson Orin Nano',
            status: 'detected',
            memory: '8GB',
            cudaCores: '1024',
            tensorCores: '32'
        }
    ];
    
    connectedDevices = mockDevices;
    updateDevicesGrid();
    updateStatus('ready', 'Devices scanned');
}

function connectDevice(deviceId) {
    const device = connectedDevices.find(d => d.id === deviceId);
    if (device) {
        device.status = device.status === 'connected' ? 'detected' : 'connected';
        updateDevicesGrid();
    }
}

// Model import function for Models page
async function addNewModel() {
    if (!window.electronAPI) {
        alert('Electron API not available');
        return;
    }
    
    try {
        updateStatus('processing', 'Selecting model...');
        const fileInfo = await window.electronAPI.selectModelFile();
        
        if (fileInfo) {
            // Import model via backend to persist and validate
            try {
                const importResult = await window.electronAPI.importModel(fileInfo.path, fileInfo.name.replace(/\.[^/.]+$/, ""));
                if (importResult && importResult.success) {
                    // Refresh model library from backend
                    const modelsResult = await window.electronAPI.listModels();
                    if (modelsResult.success) {
                        modelLibrary = modelsResult.models || [];
                    }

                    updateModelsGrid();
                    updateModelSelector();
                    updateStatus('ready', 'Model added to library');
                    showSuccess(`Model "${importResult.model?.name || importResult.name || fileInfo.name}" imported successfully!`);
                } else {
                    throw new Error(importResult.error || 'Import failed');
                }
            } catch (err) {
                console.error('Backend import failed:', err);
                // Fallback to local in-memory add so user still sees the model
                const newModel = {
                    id: generateId(),
                    name: fileInfo.name.replace(/\.[^/.]+$/, ""),
                    type: fileInfo.extension.substring(1).toUpperCase(),
                    performanceGain: 'Not optimized',
                    memoryReduction: 'Not optimized',
                    targetDevice: 'Original',
                    timestamp: new Date().toISOString(),
                    originalPath: fileInfo.path,
                    optimizedPath: null,
                    isOriginal: true
                };
                modelLibrary.push(newModel);
                updateModelsGrid();
                updateModelSelector();
                updateStatus('ready', 'Model added to library (local)');
                showError('Failed to import model into backend. Model added locally.');
            }
        } else {
            updateStatus('ready', 'Ready');
        }
    } catch (error) {
        console.error('Error adding model:', error);
        showError('Failed to add model: ' + error.message);
        updateStatus('error', 'Failed to add model');
    }
}

// Refresh model library
async function refreshModelLibrary() {
    try {
        updateStatus('processing', 'Refreshing model library...');
        
        // Call the backend refresh function
        const refreshResult = await window.electronAPI.refreshModels();
        
        if (refreshResult.success) {
            modelLibrary = refreshResult.models || [];
            updateStatus('ready', `Found ${modelLibrary.length} models after refresh`);
            
            // Update all UI components
            updateDashboard();
            updateModelSelector();
            updateModelsGrid();
            
            showSuccess(`Model library refreshed: ${modelLibrary.length} models found`);
            console.log(`Model library refreshed: ${modelLibrary.length} models found`);
        } else {
            updateStatus('error', `Failed to refresh: ${refreshResult.error}`);
            showError(`Failed to refresh model library: ${refreshResult.error}`);
        }
        
    } catch (error) {
        console.error('Failed to refresh model library:', error);
        updateStatus('error', `Failed to refresh model library: ${error.message}`);
        showError(`Failed to refresh model library: ${error.message}`);
    }
}

// Model selection functions (for optimize page)
function updateModelSelector() {
    const modelSelect = document.getElementById('modelSelect');
    const emptyModelState = document.getElementById('emptyModelState');
    
    if (!modelSelect) return;
    
    // Clear existing options except the first one
    modelSelect.innerHTML = '<option value="">Choose from your uploaded models...</option>';
    
    if (modelLibrary.length === 0) {
        if (emptyModelState) emptyModelState.style.display = 'block';
        return;
    }
    
    if (emptyModelState) emptyModelState.style.display = 'none';
    
    // Populate with available models
    modelLibrary.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = `${model.name} (${model.type})`;
        modelSelect.appendChild(option);
    });
}

function handleModelSelection() {
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect || !modelSelect.value) {
        clearModelSelection();
        return;
    }
    
    const selectedModelId = modelSelect.value;
    const selectedModel = modelLibrary.find(model => model.id === selectedModelId);
    
    if (selectedModel) {
        setCurrentModel({
            path: selectedModel.local_path || selectedModel.originalPath || selectedModel.optimizedPath,
            name: selectedModel.name,
            size: selectedModel.size || 0,
            extension: selectedModel.type ? selectedModel.type.replace('.', '').toLowerCase() : (selectedModel.extension || '').replace('.', '').toLowerCase()
        });
    }
}

function clearModelSelection() {
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        modelSelect.value = '';
    }
    removeModel();
}

function setCurrentModel(fileInfo) {
    currentModel = fileInfo;
    
    // Update UI elements
    const selectedFileName = document.getElementById('selectedFileName');
    const selectedFileType = document.getElementById('selectedFileType');
    const selectedModelInfo = document.getElementById('selectedModelInfo');
    
    if (selectedFileName) selectedFileName.textContent = fileInfo.name;
    if (selectedFileType) selectedFileType.textContent = fileInfo.extension.toUpperCase();
    
    // Show selected model info
    if (selectedModelInfo) selectedModelInfo.style.display = 'flex';
    
    // Update status
    updateStatus('ready', 'Model Selected');
    updateOptimizeButton();
}

function removeModel() {
    currentModel = null;
    
    // Reset UI
    const selectedModelInfo = document.getElementById('selectedModelInfo');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    
    if (selectedModelInfo) selectedModelInfo.style.display = 'none';
    if (progressSection) progressSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
    
    // Update status
    updateStatus('ready', 'Ready');
    updateOptimizeButton();
}

// Load initial data
async function loadInitialData() {
    try {
        // Load system info to check capabilities
        const systemInfo = await window.electronAPI.getSystemInfo();
        console.log('System info:', systemInfo);
        
        // Load optimization history from backend
        const historyResult = await window.electronAPI.getOptimizationHistory();
        if (historyResult.success) {
            optimizationHistory = historyResult.history || [];
        }
        
        // Load model library from backend
        const modelsResult = await window.electronAPI.listModels();
        if (modelsResult.success) {
            modelLibrary = modelsResult.models || [];
            
            // If no models found, try refreshing to scan for existing files
            if (modelLibrary.length === 0) {
                console.log('No models found in library, scanning for existing files...');
                const refreshResult = await window.electronAPI.refreshModels();
                if (refreshResult.success) {
                    modelLibrary = refreshResult.models || [];
                    console.log(`Found ${modelLibrary.length} models after refresh`);
                }
            }
        }
        
        // Update UI with loaded data
        updateHistoryTable();
        updateDashboard();
        updateDashboardStats();
        updateModelSelector(); // Add this to update the model dropdown
        
    } catch (error) {
        console.error('Failed to load initial data:', error);
        // Start with empty data if backend fails
        modelLibrary = [];
        optimizationHistory = [];
    }
}

function updateOptimizeButton() {
    const optimizeBtn = document.getElementById('optimizeBtn');
    const deviceSelect = document.getElementById('deviceSelect');
    
    if (!optimizeBtn) return;
    
    const hasModel = currentModel !== null;
    const deviceSelected = deviceSelect ? deviceSelect.value : true;
    
    optimizeBtn.disabled = !hasModel || !deviceSelected || optimizationInProgress;
}

// Device configuration functions
function updateDeviceConfiguration() {
    const deviceSelect = document.getElementById('deviceSelect');
    const cudaConfig = document.getElementById('cudaConfig');
    const cpuConfig = document.getElementById('cpuConfig');
    
    if (!deviceSelect || !cudaConfig || !cpuConfig) return;
    
    const selectedDevice = deviceSelect.value;
    
    if (selectedDevice === 'cuda') {
        cudaConfig.style.display = 'block';
        cpuConfig.style.display = 'none';
    } else if (selectedDevice === 'cpu') {
        cudaConfig.style.display = 'none';
        cpuConfig.style.display = 'block';
    }
}

// Setup range sliders with value displays
function setupRangeSliders() {
    // CUDA sparsity target
    const cudaSparsityTarget = document.getElementById('cudaSparsityTarget');
    const cudaSparsityValue = document.getElementById('cudaSparsityValue');
    if (cudaSparsityTarget && cudaSparsityValue) {
        cudaSparsityTarget.addEventListener('input', (e) => {
            cudaSparsityValue.textContent = e.target.value + '%';
        });
    }
    
    // CPU channel pruning
    const cpuChannelPruning = document.getElementById('cpuChannelPruning');
    const cpuChannelPruningValue = document.getElementById('cpuChannelPruningValue');
    if (cpuChannelPruning && cpuChannelPruningValue) {
        cpuChannelPruning.addEventListener('input', (e) => {
            cpuChannelPruningValue.textContent = e.target.value + '%';
        });
    }
}

// Optimization functions
async function startOptimization() {
    if (!currentModel || optimizationInProgress) return;
    
    optimizationInProgress = true;
    updateOptimizeButton();
    updateStatus('processing', 'Optimizing...');
    
    // Show progress section
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const optimizeBtn = document.getElementById('optimizeBtn');
    
    if (progressSection) progressSection.style.display = 'block';
    if (resultsSection) resultsSection.style.display = 'none';
    
    // Update button state
    if (optimizeBtn) {
        const btnText = optimizeBtn.querySelector('.btn-text');
        const btnSpinner = optimizeBtn.querySelector('.btn-spinner');
        if (btnText) btnText.textContent = 'Optimizing...';
        if (btnSpinner) btnSpinner.style.display = 'block';
    }
    
    // Get configuration
    const deviceType = document.getElementById('deviceSelect')?.value || 'cuda';
    
    let config = {
        modelPath: currentModel.path,
        device: deviceType
    };
    
    // Debug logging
    console.log('Starting optimization with currentModel:', currentModel);
    console.log('Config modelPath:', config.modelPath);
    
    if (!config.modelPath) {
        throw new Error('Model path is undefined. Please select a model first.');
    }
    
    if (deviceType === 'cuda') {
        config = {
            ...config,
            // Quantization
            precision: document.getElementById('cudaPrecision')?.value || 'fp16',
            per_channel_quantization: document.getElementById('cudaPerChannelQuantization')?.checked || false,
            symmetric_quantization: document.getElementById('cudaSymmetricQuantization')?.checked || false,
            calibration_dataset_path: document.getElementById('cudaCalibrationDatasetPath')?.value || '',
            calibration_samples: parseInt(document.getElementById('cudaCalibrationSamples')?.value || '100'),
            kv_cache_quantization: document.getElementById('cudaKvCacheQuantization')?.value || 'off',
            outlier_retention: document.getElementById('cudaOutlierRetention')?.checked || false,
            // Pruning / Sparsity
            sparsity_pattern: document.getElementById('cudaSparsityPattern')?.value || '',
            sparsity_target: parseFloat(document.getElementById('cudaSparsityTarget')?.value || '0'),
            // Graph Tweaks
            graph_fusion: document.getElementById('cudaGraphFusion')?.checked ? 'on' : 'off',
            bn_folding: document.getElementById('cudaBnFolding')?.checked ? 'on' : 'off',
            constant_folding: document.getElementById('cudaConstantFolding')?.checked ? 'on' : 'off',
            // Engine / Runtime
            batch_size: parseInt(document.getElementById('cudaBatchSize')?.value || '1'),
            workspace_size: parseInt(document.getElementById('cudaWorkspaceSize')?.value || '1024'),
            tactics: document.getElementById('cudaTactics')?.checked ? 'on' : 'off',
            dynamic_shapes: document.getElementById('cudaDynamicShapes')?.value || ''
        };
    } else if (deviceType === 'cpu') {
        config = {
            ...config,
            // Quantization
            enable_quantization: document.getElementById('cpuEnableQuantization')?.checked ?? true,
            precision: document.getElementById('cpuPrecision')?.value || 'fp32',
            per_channel_quantization: document.getElementById('cpuPerChannelQuantization')?.checked || false,
            calibration_dataset_path: document.getElementById('cpuCalibrationDatasetPath')?.value || '',
            calibration_samples: parseInt(document.getElementById('cpuCalibrationSamples')?.value || '100'),
            // Pruning / Sparsity
            channel_pruning: parseFloat(document.getElementById('cpuChannelPruning')?.value || '0'),
            clustering: document.getElementById('cpuClustering')?.checked || false,
            // Graph Tweaks
            graph_fusion: document.getElementById('cpuGraphFusion')?.checked || false,
            constant_folding: document.getElementById('cpuConstantFolding')?.checked || false,
            bn_folding: document.getElementById('cpuBnFolding')?.checked || false,
            // Engine / Runtime
            batch_size: parseInt(document.getElementById('cpuBatchSize')?.value || '1'),
            num_threads: parseInt(document.getElementById('cpuNumThreads')?.value || '4'),
            intra_op_threads: parseInt(document.getElementById('cpuIntraOpThreads')?.value || '4'),
            inter_op_threads: parseInt(document.getElementById('cpuInterOpThreads')?.value || '2'),
            // Output Format
            preserve_format: document.getElementById('cpuPreserveFormat')?.checked ?? true
        };
    }
    
    try {
        // Start tracking optimization
        const optimizationStartTime = Date.now();
        
        // Simulate optimization progress
        await simulateOptimizationProgress();
        
        // Call actual optimization (when backend is ready)
        const result = await window.electronAPI.startOptimization(config);
        
        const optimizationDuration = (Date.now() - optimizationStartTime) / 1000;
        
        if (result.success) {
            showOptimizationResults(result);
            updateStatus('ready', 'Optimization Complete');
            
            // The backend now automatically saves to history, so we just need to update our local data
            // Refresh the history from backend to get the latest entry
            const historyResult = await window.electronAPI.getOptimizationHistory({ limit: 1 });
            if (historyResult.success && historyResult.history.length > 0) {
                // Add the latest optimization to our local history
                optimizationHistory.unshift(historyResult.history[0]);
            }
            
            // Create optimized model entry for local library
            const optimizedModel = {
                id: generateId(),
                name: currentModel.name.replace(/\.[^/.]+$/, "") + '_optimized',
                type: currentModel.extension.toUpperCase(),
                performanceGain: result.performanceGain,
                memoryReduction: result.memoryReduction,
                targetDevice: config.device === 'cuda' ? 'CUDA (GPU)' : 'CPU',
                timestamp: new Date().toISOString(),
                originalPath: currentModel.path,
                optimizedPath: result.optimizedPath || currentModel.path + '_optimized',
                isOriginal: false
            };
            
            modelLibrary.push(optimizedModel);
            
            // Update UI
            updateHistoryTable();
            updateDashboard();
            updateDashboardStats();
            
        } else {
            throw new Error(result.error || 'Optimization failed');
        }
    } catch (error) {
        console.error('Optimization error:', error);
        showError('Optimization failed: ' + error.message);
        updateStatus('error', 'Optimization Failed');
        
        // Backend already saves failed optimizations to history
        // Just refresh our local history to get the latest entry
        try {
            const historyResult = await window.electronAPI.getOptimizationHistory({ limit: 1 });
            if (historyResult.success && historyResult.history.length > 0) {
                optimizationHistory.unshift(historyResult.history[0]);
                updateHistoryTable();
            }
        } catch (historyError) {
            console.warn('Failed to update history after error:', historyError);
        }
    }
    
    // Reset button state
    optimizationInProgress = false;
    if (optimizeBtn) {
        const btnText = optimizeBtn.querySelector('.btn-text');
        const btnSpinner = optimizeBtn.querySelector('.btn-spinner');
        if (btnText) btnText.textContent = 'Start Optimization';
        if (btnSpinner) btnSpinner.style.display = 'none';
    }
    updateOptimizeButton();
}

async function simulateOptimizationProgress() {
    const stages = [
        { percent: 10, status: 'Loading model...' },
        { percent: 25, status: 'Analyzing architecture...' },
        { percent: 40, status: 'Applying tensor optimizations...' },
        { percent: 60, status: 'Memory bandwidth alignment...' },
        { percent: 80, status: 'Generating optimized model...' },
        { percent: 95, status: 'Validating results...' },
        { percent: 100, status: 'Complete!' }
    ];
    
    for (const stage of stages) {
        await new Promise(resolve => setTimeout(resolve, 400));
        updateProgress(stage.percent, stage.status);
    }
}

function updateProgress(percent, status) {
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const progressStatus = document.getElementById('progressStatus');
    
    if (progressFill) progressFill.style.width = `${percent}%`;
    if (progressPercent) progressPercent.textContent = `${percent}%`;
    if (progressStatus) progressStatus.textContent = status;
}

function showOptimizationResults(result) {
    // Store the last optimization result for export/report functions
    lastOptimizationResult = result;
    
    // Update result values
    const optimizationTime = document.getElementById('optimizationTime');
    const performanceGain = document.getElementById('performanceGain');
    const memoryReduction = document.getElementById('memoryReduction');
    const resultsSection = document.getElementById('resultsSection');
    
    if (optimizationTime) optimizationTime.textContent = `${result.duration?.toFixed(2) || result.optimizationTime || '--'}s`;
    if (performanceGain) performanceGain.textContent = result.performanceGain || '--';
    if (memoryReduction) memoryReduction.textContent = result.memoryReduction || '--';
    
    // Show results section
    if (resultsSection) resultsSection.style.display = 'block';
    
    // Create performance chart
    createPerformanceChart(result);
}

function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Inference Time', 'Memory Usage', 'Power Consumption'],
            datasets: [{
                label: 'Original',
                data: [100, 100, 100],
                backgroundColor: 'rgba(160, 174, 192, 0.8)',
                borderColor: 'rgba(160, 174, 192, 1)',
                borderWidth: 1
            }, {
                label: 'Optimized',
                data: [77, 82, 85],
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 120,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Performance Comparison'
                },
                legend: {
                    position: 'top'
                }
            }
        }
    });
}

// New page functions
function updateReportsPage() {
    // This could be expanded to populate report data
    console.log('Reports page loaded');
}

function updateDeploymentPage() {
    // This could be expanded to populate deployment options
    const modelSelect = document.querySelector('.deployment-select');
    if (modelSelect && modelLibrary.length > 0) {
        modelSelect.innerHTML = '<option>Choose from your model library...</option>' +
            modelLibrary.map(model => `<option value="${model.id}">${model.name}</option>`).join('');
    }
    
    const deviceSelect = document.querySelectorAll('.deployment-select')[1];
    if (deviceSelect && connectedDevices.length > 0) {
        deviceSelect.innerHTML = '<option>Select connected device...</option>' +
            connectedDevices.filter(device => device.status === 'connected')
                .map(device => `<option value="${device.id}">${device.name}</option>`).join('');
    }
}

// Utility functions
function updateStatus(type, text) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    if (statusDot) statusDot.className = `status-dot ${type}`;
    if (statusText) statusText.textContent = text;
}

function showError(message) {
    // You could implement a toast notification system here
    alert(message);
    updateStatus('error', 'Error');
}

function showSuccess(message) {
    // You could implement a toast notification system here
    alert(message);
    updateStatus('ready', 'Ready');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileExtension(filename) {
    return filename.slice((filename.lastIndexOf('.') - 1 >>> 0) + 2);
}

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function formatTimeAgo(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diffMs = now - time;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    return `${diffDays} days ago`;
}

// Model action functions
async function exportModel(modelId) {
    try {
        const model = modelLibrary.find(m => m.id === modelId);
        if (!model) {
            showError('Model not found');
            return;
        }

        updateStatus('processing', 'Selecting export location...');
        
        // Get the model file path
        const modelPath = model.local_path || model.path || model.originalPath;
        if (!modelPath) {
            showError('Model file path not found');
            updateStatus('ready', 'Ready');
            return;
        }

        // Open save dialog
        const savePath = await window.electronAPI.showSaveDialog({
            title: 'Export Model',
            defaultPath: `${model.name}.${model.type || 'model'}`,
            filters: [
                { name: 'Model Files', extensions: ['*'] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });

        if (savePath) {
            // Copy the model file to the selected location
            await window.electronAPI.copyFile(modelPath, savePath);
            updateStatus('ready', 'Model exported successfully');
            showSuccess(`Model "${model.name}" exported to: ${savePath}`);
        } else {
            updateStatus('ready', 'Export cancelled');
        }
        
    } catch (error) {
        console.error('Export error:', error);
        showError(`Failed to export model: ${error.message}`);
        updateStatus('error', 'Export failed');
    }
}

async function exportOptimizedModel(recordId) {
    try {
        updateStatus('processing', 'Finding optimized model...');
        
        // Find the optimization record
        const record = optimizationHistory.find(r => r.id === recordId || r.modelId === recordId);
        if (!record) {
            showError('Optimization record not found');
            updateStatus('ready', 'Ready');
            return;
        }

        // Get the optimized model path from the record
        const optimizedPath = record.results?.optimizedPath || record.optimizedPath;
        if (!optimizedPath) {
            showError('Optimized model file not found in record');
            updateStatus('ready', 'Ready');
            return;
        }

        updateStatus('processing', 'Selecting export location...');
        
        // Extract model name and device info for default filename
        const modelName = record.model_info?.name || record.modelName || 'optimized_model';
        const device = record.optimization_config?.device || record.targetDevice || 'unknown';
        const precision = record.optimization_config?.precision || 'optimized';
        const fileExtension = optimizedPath.split('.').pop() || 'model';
        
        // Create a descriptive filename
        const defaultFilename = `${modelName}_${device}_${precision}.${fileExtension}`;

        // Open save dialog
        const savePath = await window.electronAPI.showSaveDialog({
            title: 'Export Optimized Model',
            defaultPath: defaultFilename,
            filters: [
                { name: 'Model Files', extensions: [fileExtension] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });

        if (savePath) {
            // Copy the optimized model to the selected location
            await window.electronAPI.copyFile(optimizedPath, savePath);
            updateStatus('ready', 'Optimized model exported successfully');
            showSuccess(`Optimized model exported to: ${savePath}`);
        } else {
            updateStatus('ready', 'Export cancelled');
        }
        
    } catch (error) {
        console.error('Export optimized model error:', error);
        showError(`Failed to export optimized model: ${error.message}`);
        updateStatus('error', 'Export failed');
    }
}

// Global variable to store last optimization result
let lastOptimizationResult = null;

async function saveOptimizedModelToFile() {
    if (!lastOptimizationResult || !lastOptimizationResult.optimizedPath) {
        showError('No optimized model available to save. Please complete an optimization first.');
        return;
    }
    
    try {
        updateStatus('processing', 'Selecting save location...');
        
        // Get model info for default filename
        const modelName = currentModel?.name || 'model';
        const device = document.getElementById('deviceSelect')?.value || 'unknown';
        const precision = device === 'cpu' 
            ? (document.getElementById('cpuPrecision')?.value || 'optimized')
            : (document.getElementById('cudaPrecision')?.value || 'optimized');
        
        const fileExtension = lastOptimizationResult.optimizedPath.split('.').pop() || 'model';
        const defaultFilename = `${modelName}_${device}_${precision}.${fileExtension}`;
        
        // Open save dialog
        const savePath = await window.electronAPI.showSaveDialog({
            title: 'Save Optimized Model',
            defaultPath: defaultFilename,
            filters: [
                { name: 'Model Files', extensions: [fileExtension] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });
        
        if (savePath) {
            // Copy the optimized model to the selected location
            await window.electronAPI.copyFile(lastOptimizationResult.optimizedPath, savePath);
            updateStatus('ready', 'Model saved successfully');
            showSuccess(`Model saved to: ${savePath}`);
        } else {
            updateStatus('ready', 'Save cancelled');
        }
        
    } catch (error) {
        console.error('Save model error:', error);
        showError(`Failed to save model: ${error.message}`);
        updateStatus('error', 'Save failed');
    }
}

async function generateOptimizationReport() {
    if (!lastOptimizationResult) {
        showError('No optimization data available. Please complete an optimization first.');
        return;
    }
    
    try {
        updateStatus('processing', 'Generating report...');
        
        const modelName = currentModel?.name || 'model';
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
        const defaultFilename = `${modelName}_optimization_report_${timestamp}.txt`;
        
        // Open save dialog
        const savePath = await window.electronAPI.showSaveDialog({
            title: 'Save Optimization Report',
            defaultPath: defaultFilename,
            filters: [
                { name: 'Text Files', extensions: ['txt'] },
                { name: 'JSON Files', extensions: ['json'] },
                { name: 'All Files', extensions: ['*'] }
            ]
        });
        
        if (savePath) {
            // Generate report content
            const report = generateReportContent(lastOptimizationResult);
            
            // Write report to file
            const fs = require('fs');
            if (savePath.endsWith('.json')) {
                await window.electronAPI.writeFile(savePath, JSON.stringify(lastOptimizationResult, null, 2));
            } else {
                await window.electronAPI.writeFile(savePath, report);
            }
            
            updateStatus('ready', 'Report saved successfully');
            showSuccess(`Report saved to: ${savePath}`);
        } else {
            updateStatus('ready', 'Report generation cancelled');
        }
        
    } catch (error) {
        console.error('Generate report error:', error);
        showError(`Failed to generate report: ${error.message}`);
        updateStatus('error', 'Report generation failed');
    }
}

function generateReportContent(result) {
    const device = document.getElementById('deviceSelect')?.value || 'unknown';
    const modelName = currentModel?.name || 'Unknown Model';
    const timestamp = new Date().toLocaleString();
    
    return `
==============================================
    TSURUTUNE OPTIMIZATION REPORT
==============================================

Model: ${modelName}
Date: ${timestamp}
Target Device: ${device.toUpperCase()}

----------------------------------------------
OPTIMIZATION RESULTS
----------------------------------------------

Performance Gain:     ${result.performanceGain || 'N/A'}
Memory Reduction:     ${result.memoryReduction || 'N/A'}
Optimization Time:    ${result.duration?.toFixed(2) || 'N/A'}s

Original Size:        ${result.originalSize ? (result.originalSize / (1024 * 1024)).toFixed(2) + ' MB' : 'N/A'}
Optimized Size:       ${result.optimizedSize ? (result.optimizedSize / (1024 * 1024)).toFixed(2) + ' MB' : 'N/A'}

----------------------------------------------
OPTIMIZATION CONFIGURATION
----------------------------------------------

${result.optimization_info ? Object.entries(result.optimization_info)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n') : 'No configuration details available'}

----------------------------------------------
FILE PATHS
----------------------------------------------

Optimized Model: ${result.optimizedPath || 'N/A'}

==============================================
    Generated by TsuruTune v1.0.0
==============================================
`;
}

function optimizeExistingModel(modelId) {
    // Navigate to optimize page and pre-select this model
    navigateToPage('optimize');
    setTimeout(() => {
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.value = modelId;
            handleModelSelection();
        }
    }, 100);
}

function viewOptimizationDetails(modelId) {
    const model = modelLibrary.find(m => m.id === modelId);
    if (model) {
        const details = `Model: ${model.name}
Type: ${model.type}
Performance Gain: ${model.performanceGain}
Memory Reduction: ${model.memoryReduction}
Target Device: ${model.targetDevice}
Optimized: ${formatTimeAgo(model.timestamp)}`;
        
        alert('Optimization Details:\n\n' + details);
    }
}

function deleteModel(modelId) {
    const model = modelLibrary.find(m => m.id === modelId);
    if (model && confirm(`Are you sure you want to delete "${model.name}"?`)) {
        // Remove from model library
        const index = modelLibrary.findIndex(m => m.id === modelId);
        if (index > -1) {
            modelLibrary.splice(index, 1);
            updateModelsGrid();
            updateModelSelector(); // Update optimize page selector
            updateDashboard(); // Update dashboard counts
            showSuccess(`Model "${model.name}" deleted successfully!`);
        }
    }
}

// History functions
function updateHistoryTable() {
    const historyTableBody = document.getElementById('historyTableBody');
    const emptyHistoryState = document.getElementById('emptyHistoryState');
    const historyTable = document.querySelector('.history-table');
    
    if (!historyTableBody) return;
    
    if (optimizationHistory.length === 0) {
        if (historyTable) historyTable.style.display = 'none';
        if (emptyHistoryState) emptyHistoryState.style.display = 'block';
        return;
    }
    
    if (historyTable) historyTable.style.display = 'table';
    if (emptyHistoryState) emptyHistoryState.style.display = 'none';
    
    // Sort history by date (newest first)
    const sortedHistory = [...optimizationHistory].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    const historyHTML = sortedHistory.map(entry => {
        const date = new Date(entry.timestamp).toLocaleDateString();
        const time = new Date(entry.timestamp).toLocaleTimeString();
        
        // Handle both old format and new backend format
        const modelName = entry.model_info?.name || entry.modelName || 'Unknown Model';
        const modelId = entry.id || entry.modelId || 'unknown';
        const device = entry.optimization_config?.device || entry.targetDevice || 'unknown';
        const success = entry.results?.success || entry.status === 'success';
        const status = success ? 'success' : 'failed';
        const performanceGain = entry.results?.performance_gain || entry.performanceGain || '-';
        const memoryReduction = entry.results?.memory_reduction || entry.memoryReduction || '-';
        const duration = entry.results?.duration || entry.duration;
        const durationText = duration ? `${duration}s` : '-';
        
        return `
            <tr>
                <td>
                    <strong>${modelName}</strong>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">
                        ${modelId}
                    </div>
                </td>
                <td>${device.toUpperCase()}</td>
                <td>
                    <span class="history-status ${status}">
                        ${success ? 'Success' : 'Failed'}
                    </span>
                </td>
                <td>
                    <span class="performance-gain">
                        ${performanceGain}
                    </span>
                </td>
                <td>
                    <span class="memory-reduction">
                        ${memoryReduction}
                    </span>
                </td>
                <td>${durationText}</td>
                <td>
                    <div style="font-size: 0.85rem;">
                        ${date}<br>
                        <span style="color: var(--text-muted);">${time}</span>
                    </div>
                </td>
                <td>
                    <div class="table-actions">
                        <button class="btn btn-sm btn-secondary" onclick="viewParameters('${modelId}')">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="3"/>
                                <path d="M12 1v6m0 6v6m-6-6h6m6 0h6"/>
                            </svg>
                            Parameters
                        </button>
                        ${success ? 
                            `<button class="btn btn-sm btn-success" onclick="exportOptimizedModel('${modelId}')">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="7,10 12,15 17,10"/>
                                    <line x1="12" y1="15" x2="12" y2="3"/>
                                </svg>
                                Export
                            </button>` : 
                            ''
                        }
                        ${success ? 
                            `<button class="btn btn-sm btn-primary" onclick="rerunOptimizationFromHistory('${modelId}')">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M1 4v6h6"/>
                                    <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
                                </svg>
                                Rerun
                            </button>` : 
                            ''
                        }
                    </div>
                </td>
            </tr>
        `;
    }).join('');
    
    historyTableBody.innerHTML = historyHTML;
}

function addOptimizationHistory(entry) {
    optimizationHistory.push({
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        ...entry
    });
}

async function rerunOptimizationFromHistory(recordId) {
    try {
        // Show loading state
        updateStatus('processing', 'Rerunning optimization...');
        
        // Use backend to rerun the optimization
        const result = await window.electronAPI.rerunOptimization(recordId);
        
        if (result.success) {
            // Refresh history to get the new entry
            const historyResult = await window.electronAPI.getOptimizationHistory({ limit: 1 });
            if (historyResult.success && historyResult.history.length > 0) {
                optimizationHistory.unshift(historyResult.history[0]);
            }
            
            // Update UI
            updateHistoryTable();
            updateDashboard();
            updateDashboardStats();
            
            showSuccess('Optimization rerun completed successfully!');
            updateStatus('ready', 'Ready');
        } else {
            throw new Error(result.error || 'Rerun failed');
        }
    } catch (error) {
        console.error('Rerun failed:', error);
        showError('Failed to rerun optimization: ' + error.message);
        updateStatus('error', 'Rerun Failed');
    }
}

function viewParameters(entryId) {
    const entry = optimizationHistory.find(e => e.id === entryId);
    if (!entry) return;
    
    const modal = document.getElementById('parameterModal');
    const modalBody = document.getElementById('parameterModalBody');
    
    if (!modal || !modalBody) return;
    
    // Handle both old format and new backend format
    const modelName = entry.model_info?.name || entry.modelName || 'Unknown Model';
    const device = entry.optimization_config?.device || entry.targetDevice || 'unknown';
    const success = entry.results?.success || entry.status === 'success';
    const performanceGain = entry.results?.performance_gain || entry.performanceGain || 'N/A';
    const memoryReduction = entry.results?.memory_reduction || entry.memoryReduction || 'N/A';
    const duration = entry.results?.duration || entry.duration || 'N/A';
    const error = entry.results?.error || entry.error;
    const parameters = entry.optimization_config || entry.parameters || {};
    
    const parametersHTML = `
        <div class="parameter-section">
            <h4>Basic Configuration</h4>
            <div class="parameter-item">
                <span class="parameter-label">Model Name:</span>
                <span class="parameter-value">${modelName}</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Target Device:</span>
                <span class="parameter-value">${device.toUpperCase()}</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Optimization Date:</span>
                <span class="parameter-value">${new Date(entry.timestamp).toLocaleString()}</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Duration:</span>
                <span class="parameter-value">${duration}s</span>
            </div>
        </div>
        
        ${Object.keys(parameters).length > 0 ? `
        <div class="parameter-section">
            <h4>${device.toUpperCase()} Parameters</h4>
            ${Object.entries(parameters).map(([key, value]) => {
                if (key === 'device') return ''; // Skip device as it's already shown
                return `
                <div class="parameter-item">
                    <span class="parameter-label">${key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').toLowerCase()}:</span>
                    <span class="parameter-value">${typeof value === 'boolean' ? (value ? 'Enabled' : 'Disabled') : value}</span>
                </div>
                `;
            }).join('')}
        </div>
        ` : ''}
        
        <div class="parameter-section">
            <h4>Results</h4>
            <div class="parameter-item">
                <span class="parameter-label">Status:</span>
                <span class="parameter-value status-${success ? 'success' : 'failed'}">${success ? 'Success' : 'Failed'}</span>
            </div>
            ${success ? `
            <div class="parameter-item">
                <span class="parameter-label">Performance Gain:</span>
                <span class="parameter-value">${performanceGain}</span>
            </div>
            <div class="parameter-item">
                <span class="parameter-label">Memory Reduction:</span>
                <span class="parameter-value">${memoryReduction}</span>
            </div>
            ` : ''}
            ${error ? `
            <div class="parameter-item">
                <span class="parameter-label">Error:</span>
                <span class="parameter-value error-text">${error}</span>
            </div>
            ` : ''}
        </div>
    `;
    
    modalBody.innerHTML = parametersHTML;
    modal.style.display = 'flex';
}

function closeParameterModal() {
    const modal = document.getElementById('parameterModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function rerunOptimization(entryId) {
    const entry = optimizationHistory.find(e => e.id === entryId);
    if (!entry) return;
    
    // Find the model in the library
    const model = modelLibrary.find(m => m.id === entry.modelId);
    if (!model) {
        showError('Original model not found in library');
        return;
    }
    
    // Navigate to optimize page and set up the configuration
    navigateToPage('optimize');
    
    // Set the model in the selector
    setTimeout(() => {
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.value = entry.modelId;
            handleModelSelection();
        }
        
        // Set device configuration
        const deviceSelect = document.getElementById('deviceSelect');
        if (deviceSelect) {
            deviceSelect.value = entry.targetDevice;
            updateDeviceConfiguration();
        }
        
        // Set parameters if available
        if (entry.parameters) {
            Object.entries(entry.parameters).forEach(([key, value]) => {
                const element = document.getElementById(key);
                if (element) {
                    if (element.type === 'checkbox') {
                        element.checked = value;
                    } else if (element.type === 'range') {
                        element.value = value;
                        // Update the display value
                        const display = document.getElementById(key + 'Value');
                        if (display) display.textContent = value;
                    } else {
                        element.value = value;
                    }
                }
            });
        }
        
        showSuccess('Configuration restored from history. Ready to rerun optimization.');
    }, 100);
}

function clearOptimizationHistory() {
    if (confirm('Are you sure you want to clear all optimization history? This action cannot be undone.')) {
        optimizationHistory.length = 0;
        updateHistoryTable();
        updateDashboard(); // Update dashboard stats
        showSuccess('Optimization history cleared successfully.');
    }
}

function deviceDetails(deviceId) {
    const device = connectedDevices.find(d => d.id === deviceId);
    if (device) {
        alert(`Device Details:\n\nName: ${device.name}\nStatus: ${device.status}\nMemory: ${device.memory}\nCUDA Cores: ${device.cudaCores}\nTensor Cores: ${device.tensorCores}`);
    }
}

// ===========================
// Batch Optimization Functions
// ===========================

let batchOptimizationInProgress = false;
let batchResults = [];

// Initialize batch optimize page when navigating to it
function initializeBatchOptimizePage() {
    const batchModelSelect = document.getElementById('batchModelSelect');
    const batchOptimizeBtn = document.getElementById('batchOptimizeBtn');
    const batchDeviceSelect = document.getElementById('batchDeviceSelect');
    
    // Populate model select
    if (batchModelSelect) {
        batchModelSelect.innerHTML = '<option value="">Choose from your uploaded models...</option>';
        modelLibrary.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.name} (${model.type})`;
            batchModelSelect.appendChild(option);
        });
        
        batchModelSelect.addEventListener('change', updateBatchOptimizeButton);
    }
    
    if (batchOptimizeBtn) {
        batchOptimizeBtn.addEventListener('click', startBatchOptimization);
    }
    
    if (batchDeviceSelect) {
        batchDeviceSelect.addEventListener('change', updateBatchOptimizeButton);
    }
    
    // Add listeners to all checkboxes
    const checkboxes = ['batchFP32', 'batchFP16', 'batchBF16', 'batchINT8', 
                        'batchNoGraphOpt', 'batchWithGraphOpt', 
                        'batchPruning20', 'batchPruning40'];
    checkboxes.forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            checkbox.addEventListener('change', updateBatchSummary);
        }
    });
    
    updateBatchSummary();
}

function updateBatchOptimizeButton() {
    const batchModelSelect = document.getElementById('batchModelSelect');
    const batchOptimizeBtn = document.getElementById('batchOptimizeBtn');
    
    if (batchOptimizeBtn && batchModelSelect) {
        const hasModel = batchModelSelect.value !== '';
        const hasVariants = getSelectedBatchVariants().length > 0;
        batchOptimizeBtn.disabled = !hasModel || !hasVariants || batchOptimizationInProgress;
    }
}

function getSelectedBatchVariants() {
    const variants = [];
    const device = document.getElementById('batchDeviceSelect')?.value || 'cpu';
    
    // Precision variants
    const precisions = [];
    if (document.getElementById('batchFP32')?.checked) precisions.push('fp32');
    if (document.getElementById('batchFP16')?.checked) precisions.push('fp16');
    if (document.getElementById('batchBF16')?.checked) precisions.push('bf16');
    if (document.getElementById('batchINT8')?.checked) precisions.push('int8');
    
    // Graph optimization variants
    const graphOpts = [];
    if (document.getElementById('batchNoGraphOpt')?.checked) graphOpts.push(false);
    if (document.getElementById('batchWithGraphOpt')?.checked) graphOpts.push(true);
    
    // Pruning variants (CPU only)
    const pruning = [0];
    if (device === 'cpu') {
        if (document.getElementById('batchPruning20')?.checked) pruning.push(20);
        if (document.getElementById('batchPruning40')?.checked) pruning.push(40);
    }
    
    // Generate all combinations
    precisions.forEach(precision => {
        graphOpts.forEach(graphOpt => {
            pruning.forEach(pruningLevel => {
                variants.push({
                    name: `${precision.toUpperCase()}${graphOpt ? '+GraphOpt' : ''}${pruningLevel > 0 ? `+Prune${pruningLevel}%` : ''}`,
                    precision,
                    graphOpt,
                    pruningLevel
                });
            });
        });
    });
    
    return variants;
}

function updateBatchSummary() {
    const variants = getSelectedBatchVariants();
    const variantCount = document.getElementById('batchVariantCount');
    const estimatedTime = document.getElementById('batchEstimatedTime');
    
    if (variantCount) {
        variantCount.textContent = variants.length;
    }
    
    if (estimatedTime) {
        const minutes = Math.ceil(variants.length * 5); // Estimate 5 minutes per variant
        estimatedTime.textContent = `~${minutes} minutes`;
    }
    
    updateBatchOptimizeButton();
}

async function startBatchOptimization() {
    if (batchOptimizationInProgress) return;
    
    const batchModelSelect = document.getElementById('batchModelSelect');
    const selectedModelId = batchModelSelect.value;
    
    if (!selectedModelId) {
        showError('Please select a model first');
        return;
    }
    
    const model = modelLibrary.find(m => m.id === selectedModelId);
    if (!model) {
        showError('Selected model not found');
        return;
    }
    
    // Get the correct model path (try multiple properties for compatibility)
    const modelPath = model.local_path || model.path || model.originalPath;
    
    if (!modelPath) {
        showError('Model path not found. Please re-import the model.');
        return;
    }
    
    const variants = getSelectedBatchVariants();
    if (variants.length === 0) {
        showError('Please select at least one variant');
        return;
    }
    
    batchOptimizationInProgress = true;
    batchResults = [];
    
    const batchOptimizeBtn = document.getElementById('batchOptimizeBtn');
    const batchProgressSection = document.getElementById('batchProgressSection');
    const batchProgressBar = document.getElementById('batchProgressBar');
    const batchProgressText = document.getElementById('batchProgressText');
    const batchResultsList = document.getElementById('batchResultsList');
    
    // Show progress section
    if (batchProgressSection) batchProgressSection.style.display = 'block';
    if (batchResultsList) batchResultsList.innerHTML = '';
    
    // Update button state
    if (batchOptimizeBtn) {
        batchOptimizeBtn.querySelector('.btn-text').textContent = 'Optimizing...';
        batchOptimizeBtn.querySelector('.btn-spinner').style.display = 'inline-block';
        batchOptimizeBtn.disabled = true;
    }
    
    updateStatus('busy', 'Batch Optimization in Progress');
    
    const device = document.getElementById('batchDeviceSelect')?.value || 'cpu';
    
    // Process each variant
    for (let i = 0; i < variants.length; i++) {
        const variant = variants[i];
        
        // Update progress
        const progress = ((i) / variants.length) * 100;
        if (batchProgressBar) batchProgressBar.style.width = `${progress}%`;
        if (batchProgressText) batchProgressText.textContent = `Optimizing ${i + 1} of ${variants.length}: ${variant.name}`;
        
        // Build configuration
        const config = {
            modelPath: modelPath,
            device: device,
            precision: variant.precision,
            enable_quantization: variant.precision === 'int8',
            graph_fusion: variant.graphOpt,
            constant_folding: variant.graphOpt,
            bn_folding: variant.graphOpt,
            channel_pruning: variant.pruningLevel,
            batch_size: 1,
            num_threads: 4,
            intra_op_threads: 4,
            inter_op_threads: 2
        };
        
        // Add CPU-specific parameters
        if (device === 'cpu') {
            config.per_channel_quantization = variant.precision === 'int8';
            config.calibration_samples = 100;
        }
        
        try {
            const result = await window.electronAPI.startOptimization(config);
            
            batchResults.push({
                variant: variant.name,
                success: result.success,
                result: result
            });
            
            // Add result to list
            if (batchResultsList) {
                const resultItem = document.createElement('div');
                resultItem.className = `batch-result-item ${result.success ? 'success' : 'error'}`;
                resultItem.innerHTML = `
                    <div class="result-icon">
                        ${result.success ? '✓' : '✗'}
                    </div>
                    <div class="result-info">
                        <strong>${variant.name}</strong>
                        <p>${result.success ? `Performance: ${result.performanceGain}, Size: ${result.memoryReduction}` : `Error: ${result.error}`}</p>
                    </div>
                `;
                batchResultsList.appendChild(resultItem);
            }
            
        } catch (error) {
            console.error(`Failed to optimize ${variant.name}:`, error);
            batchResults.push({
                variant: variant.name,
                success: false,
                error: error.message
            });
        }
    }
    
    // Complete
    if (batchProgressBar) batchProgressBar.style.width = '100%';
    if (batchProgressText) batchProgressText.textContent = `Complete! Optimized ${variants.length} variants`;
    
    if (batchOptimizeBtn) {
        batchOptimizeBtn.querySelector('.btn-text').textContent = 'Start Batch Optimization';
        batchOptimizeBtn.querySelector('.btn-spinner').style.display = 'none';
        batchOptimizeBtn.disabled = false;
    }
    
    batchOptimizationInProgress = false;
    updateStatus('ready', 'Batch Optimization Complete');
    
    const successCount = batchResults.filter(r => r.success).length;
    showSuccess(`Batch optimization complete! ${successCount} of ${variants.length} variants created successfully.`);
    
    // Refresh history
    await loadOptimizationHistory();
}

function selectAllBatchPresets() {
    const checkboxes = ['batchFP32', 'batchFP16', 'batchBF16', 'batchINT8', 
                        'batchNoGraphOpt', 'batchWithGraphOpt', 
                        'batchPruning20', 'batchPruning40'];
    checkboxes.forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox) checkbox.checked = true;
    });
    updateBatchSummary();
}

function selectRecommendedBatchPresets() {
    // Clear all first
    clearAllBatchPresets();
    
    // Select recommended: FP32, FP16, INT8 with graph optimizations
    const recommended = ['batchFP32', 'batchFP16', 'batchINT8', 'batchWithGraphOpt'];
    recommended.forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox) checkbox.checked = true;
    });
    updateBatchSummary();
}

function clearAllBatchPresets() {
    const checkboxes = ['batchFP32', 'batchFP16', 'batchBF16', 'batchINT8', 
                        'batchNoGraphOpt', 'batchWithGraphOpt', 
                        'batchPruning20', 'batchPruning40'];
    checkboxes.forEach(id => {
        const checkbox = document.getElementById(id);
        if (checkbox) checkbox.checked = false;
    });
    updateBatchSummary();
}

// Export functions for potential use
window.TsuruTune = {
    navigateToPage,
    setCurrentModel,
    removeModel,
    startOptimization,
    updateStatus,
    scanForDevices,
    runBenchmark
};

// Make batch functions global
window.selectAllBatchPresets = selectAllBatchPresets;
window.selectRecommendedBatchPresets = selectRecommendedBatchPresets;
window.clearAllBatchPresets = clearAllBatchPresets;
