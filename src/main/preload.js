const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // File operations
  selectModelFile: () => ipcRenderer.invoke('select-model-file'),
  showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
  copyFile: (sourcePath, destPath) => ipcRenderer.invoke('copy-file', sourcePath, destPath),
  writeFile: (filePath, content) => ipcRenderer.invoke('write-file', filePath, content),
  
  // Optimization operations
  startOptimization: (config) => ipcRenderer.invoke('start-optimization', config),
  
  // History operations
  getOptimizationHistory: (filters) => ipcRenderer.invoke('get-optimization-history', filters),
  getHistoryStatistics: () => ipcRenderer.invoke('get-history-statistics'),
  getOptimizationRecord: (recordId) => ipcRenderer.invoke('get-optimization-record', recordId),
  deleteOptimizationRecord: (recordId) => ipcRenderer.invoke('delete-optimization-record', recordId),
  rerunOptimization: (recordId) => ipcRenderer.invoke('rerun-optimization', recordId),
  exportHistory: (outputPath, format) => ipcRenderer.invoke('export-history', outputPath, format),
  clearOptimizationHistory: () => ipcRenderer.invoke('clear-optimization-history'),
  
  // Model operations
  listModels: () => ipcRenderer.invoke('list-models'),
  refreshModels: () => ipcRenderer.invoke('refresh-models'),
  importModel: (modelPath, modelName) => ipcRenderer.invoke('import-model', modelPath, modelName),
  
  // System operations
  getSystemInfo: () => ipcRenderer.invoke('get-system-info'),
  
  // Menu events
  onMenuImportModel: (callback) => ipcRenderer.on('menu-import-model', callback),
  
  // Utility functions
  getPath: () => ipcRenderer.invoke('get-app-path'),
  
  // Remove listener
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});
