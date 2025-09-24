const { app, BrowserWindow, Menu, ipcMain, dialog } = require('electron');
app.commandLine.appendSwitch('gtk-version', '3');
// Add command line switches to help with X11 issues
app.commandLine.appendSwitch('disable-gpu-sandbox');
app.commandLine.appendSwitch('disable-software-rasterizer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow;

// Python backend communication functions
function executePythonCommand(command, args = [], config = null) {
  return new Promise((resolve, reject) => {
    const pythonPath = path.join(__dirname, '../../python/main.py');
    // Use the correct Python interpreter that has TensorFlow installed
    const pythonExecutable = '/opt/homebrew/bin/python3.11';
    const venvPath = path.join(__dirname, '../../.venv');
    const cmdArgs = [pythonPath, command, ...args];
    
    if (config) {
      cmdArgs.push('--config', JSON.stringify(config));
    }
    
    // Set environment to use virtual environment packages
    const env = {
      ...process.env,
      PYTHONPATH: [
        path.join(__dirname, '../../python'),
        path.join(venvPath, 'lib/python3.11/site-packages')
      ].join(':'),
      PATH: path.join(venvPath, 'bin') + ':' + (process.env.PATH || ''),
      TF_CPP_MIN_LOG_LEVEL: '1', // Reduce TensorFlow verbose logging
      VIRTUAL_ENV: venvPath
    };
    
    const pythonProcess = spawn(pythonExecutable, cmdArgs, { env });
    let stdoutData = '';
    let stderrData = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(stdoutData);
          resolve(result);
        } catch (e) {
          reject(new Error(`Failed to parse Python output: ${stdoutData}`));
        }
      } else {
        reject(new Error(`Python process failed with code ${code}: ${stderrData}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });
  });
}

function createWindow() {
  // Create the main application window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      // Add these to help with large data handling
      enableBlinkFeatures: 'CSSColorSchemeUARendering',
      experimentalFeatures: true
    },
    title: 'TsuruTune - Jetson Deep Learning Optimizer',
    titleBarStyle: 'default',
    show: false, // Don't show window until ready
    // Add frame buffer limits
    transparent: false,
    frame: true
  });

  // Load the HTML file
  mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
  
  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Open DevTools in development
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Import Model',
          accelerator: 'CmdOrCtrl+O',
          click: () => {
            // Send event to renderer to trigger file dialog
            mainWindow.webContents.send('menu-import-model');
          }
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'toggleFullscreen' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About TsuruTune',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About TsuruTune',
              message: 'TsuruTune v1.0.0',
              detail: 'Deep Learning Optimization for Jetson via Tensor Core and Memory Bandwidth Alignment\n\nDeveloped by Farrell Rafee Sudjatmiko\nITS Computer Engineering'
            });
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// App event handlers
app.whenReady().then(() => {
  createWindow();
  createMenu();
});

// Add error handling for X11 issues
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC handlers for communication with renderer process
ipcMain.handle('select-model-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Model Files', extensions: ['onnx', 'pt', 'pth', 'pb', 'h5', 'keras'] },
      { name: 'ONNX Models', extensions: ['onnx'] },
      { name: 'PyTorch Models', extensions: ['pt', 'pth'] },
      { name: 'TensorFlow / Keras Models', extensions: ['pb', 'h5', 'keras'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });

  if (!result.canceled && result.filePaths.length > 0) {
    const filePath = result.filePaths[0];
    const stats = fs.statSync(filePath);
    
    return {
      path: filePath,
      name: path.basename(filePath),
      size: stats.size,
      extension: path.extname(filePath)
    };
  }
  
  return null;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  try {
    const result = await dialog.showSaveDialog(mainWindow, options || {});
    return result.canceled ? null : result.filePath;
  } catch (error) {
    console.error('Show save dialog error:', error);
    return null;
  }
});

ipcMain.handle('copy-file', async (event, sourcePath, destPath) => {
  try {
    // Check if source file exists
    if (!fs.existsSync(sourcePath)) {
      throw new Error('Source file not found');
    }
    
    // Create destination directory if it doesn't exist
    const destDir = path.dirname(destPath);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    
    // Copy the file
    fs.copyFileSync(sourcePath, destPath);
    
    return { success: true };
  } catch (error) {
    console.error('Copy file error:', error);
    throw error;
  }
});

ipcMain.handle('start-optimization', async (event, config) => {
  try {
    console.log('Starting optimization with config:', config);
    const result = await executePythonCommand('optimize', [], config);
    return result;
  } catch (error) {
    console.error('Optimization failed:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// History management handlers
ipcMain.handle('get-optimization-history', async (event, filters = {}) => {
  try {
    const args = [];
    if (filters.limit) args.push('--limit', filters.limit.toString());
    if (filters.device) args.push('--device', filters.device);
    if (filters.status) args.push('--status', filters.status);
    
    const result = await executePythonCommand('history', args);
    return result;
  } catch (error) {
    console.error('Failed to get history:', error);
    return {
      success: false,
      error: error.message,
      history: [],
      total: 0
    };
  }
});

ipcMain.handle('get-history-statistics', async () => {
  try {
    const result = await executePythonCommand('stats');
    return result;
  } catch (error) {
    console.error('Failed to get statistics:', error);
    return {
      success: false,
      error: error.message,
      statistics: {}
    };
  }
});

ipcMain.handle('get-optimization-record', async (event, recordId) => {
  try {
    const result = await executePythonCommand('record', ['--record-id', recordId]);
    return result;
  } catch (error) {
    console.error('Failed to get record:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('delete-optimization-record', async (event, recordId) => {
  try {
    const result = await executePythonCommand('delete', ['--record-id', recordId]);
    return result;
  } catch (error) {
    console.error('Failed to delete record:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('rerun-optimization', async (event, recordId) => {
  try {
    const result = await executePythonCommand('rerun', ['--record-id', recordId]);
    return result;
  } catch (error) {
    console.error('Failed to rerun optimization:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('export-history', async (event, outputPath, format = 'json') => {
  try {
    const result = await executePythonCommand('export', [
      '--output', outputPath,
      '--format', format
    ]);
    return result;
  } catch (error) {
    console.error('Failed to export history:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('clear-optimization-history', async () => {
  try {
    const result = await executePythonCommand('clear');
    return result;
  } catch (error) {
    console.error('Failed to clear history:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

ipcMain.handle('get-system-info', async () => {
  try {
    const result = await executePythonCommand('system');
    return result;
  } catch (error) {
    console.error('Failed to get system info:', error);
    return {
      success: false,
      error: error.message,
      system: {}
    };
  }
});

ipcMain.handle('list-models', async () => {
  try {
    const result = await executePythonCommand('list');
    return result;
  } catch (error) {
    console.error('Failed to list models:', error);
    return {
      success: false,
      error: error.message,
      models: []
    };
  }
});

ipcMain.handle('refresh-models', async () => {
  try {
    const result = await executePythonCommand('refresh');
    return result;
  } catch (error) {
    console.error('Failed to refresh models:', error);
    return {
      success: false,
      error: error.message,
      models: []
    };
  }
});

ipcMain.handle('import-model', async (event, modelPath, modelName = null) => {
  try {
    const args = ['--model-path', modelPath];
    if (modelName) {
      args.push('--model-name', modelName);
    }
    
    const result = await executePythonCommand('import', args);
    return result;
  } catch (error) {
    console.error('Failed to import model:', error);
    return {
      success: false,
      error: error.message
    };
  }
});
