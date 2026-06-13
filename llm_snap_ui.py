<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM-SNAP Dashboard - LLM-BUBBLE</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Inter', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        
        :root {
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --bg-hover: #1a1a24;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --border: #1e1e2e;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        
        body {
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
        }
        
        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
            cursor: pointer;
            border: none;
        }
        
        .btn-primary {
            background: var(--accent);
            color: white;
        }
        
        .btn-primary:hover {
            box-shadow: 0 0 20px var(--accent-glow);
        }
        
        .btn-secondary {
            background: var(--bg-hover);
            color: var(--text);
            border: 1px solid var(--border);
        }
        
        .btn-secondary:hover {
            background: #252530;
        }
        
        .step-timeline {
            position: relative;
            padding-left: 30px;
        }
        
        .step-timeline::before {
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--border);
        }
        
        .step-item {
            position: relative;
            padding: 12px 16px;
            margin-bottom: 12px;
            background: var(--bg-hover);
            border-radius: 8px;
            border-left: 3px solid var(--accent);
        }
        
        .step-item::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 50%;
            transform: translateY(-50%);
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent);
            border: 2px solid var(--bg-dark);
        }
        
        .step-type-observe { border-left-color: #22c55e; }
        .step-type-reason { border-left-color: #6366f1; }
        .step-type-act { border-left-color: #f59e0b; }
        
        .status-badge {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .status-active { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
        .status-completed { background: rgba(99, 102, 241, 0.2); color: #6366f1; }
        
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-dark); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #333; }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
    </style>
</head>
<body class="p-6">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <header class="flex items-center justify-between mb-8">
            <div class="flex items-center gap-4">
                <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                    <svg class="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.018-4.016a11.944 11.944 0 014 4.082 11.944 11.944 0 01-4.082 4.018z"/>
                    </svg>
                </div>
                <div>
                    <h1 class="text-2xl font-bold">LLM-SNAP</h1>
                    <p class="text-sm text-gray-400">Bubble Transformer Experiment Tracker</p>
                </div>
            </div>
            <div class="flex items-center gap-3">
                <button onclick="refreshData()" class="btn btn-secondary">
                    <svg class="w-4 h-4 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                    </svg>
                    Refresh
                </button>
                <button onclick="startWatcher()" id="watcherBtn" class="btn btn-primary">
                    <span class="inline-block w-2 h-2 rounded-full bg-green-400 mr-2 pulse"></span>
                    Watcher Active
                </button>
            </div>
        </header>
        
        <!-- Stats Cards -->
        <div class="grid grid-cols-4 gap-4 mb-8">
            <div class="card p-5">
                <div class="text-sm text-gray-400 mb-1">Total Runs</div>
                <div class="text-3xl font-bold" id="totalRuns">-</div>
            </div>
            <div class="card p-5">
                <div class="text-sm text-gray-400 mb-1">Snapshots</div>
                <div class="text-3xl font-bold" id="totalSnapshots">-</div>
            </div>
            <div class="card p-5">
                <div class="text-sm text-gray-400 mb-1">Auto Captures</div>
                <div class="text-3xl font-bold" id="autoCaptures">-</div>
            </div>
            <div class="card p-5">
                <div class="text-sm text-gray-400 mb-1">Last Activity</div>
                <div class="text-lg font-semibold" id="lastActivity">-</div>
            </div>
        </div>
        
        <div class="grid grid-cols-3 gap-6">
            <!-- Runs List -->
            <div class="col-span-2">
                <div class="card p-5">
                    <div class="flex items-center justify-between mb-4">
                        <h2 class="text-lg font-semibold">Experiment Runs</h2>
                        <input type="text" id="searchRuns" placeholder="Search runs..." 
                               class="bg-dark border border-gray-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
                               onkeyup="filterRuns()">
                    </div>
                    <div id="runsList" class="space-y-2 max-h-96 overflow-y-auto">
                        <!-- Runs will be populated here -->
                    </div>
                </div>
            </div>
            
            <!-- Timeline Viewer -->
            <div class="col-span-1">
                <div class="card p-5">
                    <h2 class="text-lg font-semibold mb-4">Timeline</h2>
                    <div id="timeline" class="step-timeline max-h-96 overflow-y-auto">
                        <p class="text-gray-500 text-sm">Select a run to view timeline</p>
                    </div>
                </div>
                
                <!-- Quick Actions -->
                <div class="card p-5 mt-4">
                    <h3 class="text-sm font-semibold mb-3 text-gray-400">Quick Actions</h3>
                    <div class="space-y-2">
                        <button onclick="runExperiment()" class="btn btn-primary w-full">
                            <svg class="w-4 h-4 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.4v2.83a1 1 0 003.85.578l2.136-2.703a1 1 0 000-1.658l-2.136-2.703z"/>
                            </svg>
                            Run Experiment
                        </button>
                        <button onclick="openTerminal()" class="btn btn-secondary w-full">
                            <svg class="w-4 h-4 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0c3 0 6 2 6 5s-3 5-6 5m-1-9l-3 3 3 3"/>
                            </svg>
                            Open Terminal
                        </button>
                        <button onclick="toggleWatcher()" id="toggleWatcherBtn" class="btn btn-secondary w-full">
                            <svg class="w-4 h-4 inline mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
                            </svg>
                            Stop Watcher
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let allRuns = [];
        let selectedRun = null;
        
        // Fetch data from API
        async function refreshData() {
            try {
                const response = await fetch('/api/snapshots');
                const data = await response.json();
                
                document.getElementById('totalRuns').textContent = data.runs.length;
                document.getElementById('totalSnapshots').textContent = data.totalSnapshots;
                document.getElementById('autoCaptures').textContent = data.autoCaptures;
                document.getElementById('lastActivity').textContent = data.lastActivity || 'N/A';
                
                allRuns = data.runs;
                renderRuns(allRuns);
                
                if (selectedRun) {
                    showTimeline(selectedRun);
                }
            } catch (e) {
                console.error('Error fetching data:', e);
            }
        }
        
        function renderRuns(runs) {
            const container = document.getElementById('runsList');
            if (runs.length === 0) {
                container.innerHTML = '<p class="text-gray-500 text-sm">No runs found</p>';
                return;
            }
            
            container.innerHTML = runs.map((run, i) => `
                <div class="step-item cursor-pointer hover:bg-opacity-80 ${selectedRun === run.id ? 'ring-2 ring-indigo-500' : ''}"
                     onclick="selectRun('${run.id}')">
                    <div class="flex items-center justify-between">
                        <div>
                            <div class="font-semibold mono text-sm">${run.id}</div>
                            <div class="text-xs text-gray-400">${run.timestamp}</div>
                        </div>
                        <div class="text-right">
                            <span class="status-badge ${run.status === 'active' ? 'status-active' : 'status-completed'}">
                                ${run.stepCount} steps
                            </span>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        function selectRun(runId) {
            selectedRun = runId;
            renderRuns(allRuns);
            showTimeline(runId);
        }
        
        async function showTimeline(runId) {
            try {
                const response = await fetch('/api/timeline/' + runId);
                const steps = await response.json();
                
                const container = document.getElementById('timeline');
                
                if (steps.length === 0) {
                    container.innerHTML = '<p class="text-gray-500 text-sm">No snapshots for this run</p>';
                    return;
                }
                
                container.innerHTML = steps.map((step, i) => `
                    <div class="step-item step-type-${step.step_type}">
                        <div class="flex items-center justify-between mb-1">
                            <span class="text-xs font-mono text-gray-400">#${step.step_index}</span>
                            <span class="text-xs text-gray-500">${step.timestamp}</span>
                        </div>
                        <div class="font-medium">${step.step_type}</div>
                        ${step.messages ? `<div class="text-sm text-gray-400 mt-1 truncate">${step.messages[0]?.content || ''}</div>` : ''}
                    </div>
                `).join('');
            } catch (e) {
                console.error('Error loading timeline:', e);
            }
        }
        
        function filterRuns() {
            const query = document.getElementById('searchRuns').value.toLowerCase();
            const filtered = allRuns.filter(r => r.id.toLowerCase().includes(query));
            renderRuns(filtered);
        }
        
        function runExperiment() {
            window.open('/experiment', '_blank');
        }
        
        function openTerminal() {
            // Open PowerShell
            fetch('/api/terminal/open', { method: 'POST' });
        }
        
        function startWatcher() {
            fetch('/api/watcher/start', { method: 'POST' });
            document.getElementById('watcherBtn').innerHTML = '<span class="inline-block w-2 h-2 rounded-full bg-green-400 mr-2 pulse"></span>Watcher Active';
        }
        
        function toggleWatcher() {
            fetch('/api/watcher/toggle', { method: 'POST' })
                .then(() => refreshData());
        }
        
        // Auto refresh every 5 seconds
        setInterval(refreshData, 5000);
        
        // Initial load
        refreshData();
    </script>
</body>
</html>
