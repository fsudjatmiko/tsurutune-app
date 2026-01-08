# Deployment Module Examples

This directory contains examples for using the TsuruTune deployment module.

## Quick Start

### 1. Start the Server

```bash
# Start with default settings (port 5000)
python start_server.py

# Start on custom port
python start_server.py --port 8000

# Start in debug mode
python start_server.py --debug
```

### 2. Test the Server

```bash
# Run the test script
python test_deployment.py

# Test with custom URL
python test_deployment.py --url http://192.168.1.100:5000
```

### 3. Download a Model

```bash
# Using the test script
python test_deployment.py --download model_abc123 --output my_model.onnx

# Using curl
curl -O -J http://localhost:5000/api/models/model_abc123/download

# Using wget
wget --content-disposition http://localhost:5000/api/models/model_abc123/download
```

## Python Client Examples

### List All Models

```python
import requests

response = requests.get('http://localhost:5000/api/models')
data = response.json()

print(f"Found {data['count']} models:")
for model in data['models']:
    print(f"  - {model['name']} ({model['size_mb']} MB)")
    print(f"    Download: {model['download_url']}")
```

### Download a Model

```python
import requests

model_id = 'your_model_id'
url = f'http://localhost:5000/api/models/{model_id}/download'

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open('model.onnx', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete!")
```

### Get Model Information

```python
import requests

model_id = 'your_model_id'
response = requests.get(f'http://localhost:5000/api/models/{model_id}')

if response.status_code == 200:
    model = response.json()['model']
    print(f"Name: {model['name']}")
    print(f"Size: {model['size_mb']} MB")
    print(f"Type: {model['type']}")
    print(f"Hash: {model['hash']}")
```

## JavaScript/TypeScript Examples

### Fetch Models

```javascript
async function listModels() {
    const response = await fetch('http://localhost:5000/api/models');
    const data = await response.json();
    
    console.log(`Found ${data.count} models`);
    data.models.forEach(model => {
        console.log(`${model.name} - ${model.size_mb} MB`);
    });
}
```

### Download Model in Browser

```javascript
async function downloadModel(modelId) {
    const url = `http://localhost:5000/api/models/${modelId}/download`;
    const response = await fetch(url);
    const blob = await response.blob();
    
    // Trigger download
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = 'model.onnx';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(downloadUrl);
    a.remove();
}
```

### Get Server Statistics

```javascript
async function getStats() {
    const response = await fetch('http://localhost:5000/api/stats');
    const stats = await response.json();
    
    console.log(`Total models: ${stats.total_models}`);
    console.log(`Total size: ${stats.total_size_mb} MB`);
    console.log(`Original models: ${stats.original_models}`);
    console.log(`Optimized models: ${stats.optimized_models}`);
}
```

## cURL Examples

```bash
# List all models
curl http://localhost:5000/api/models

# Get model info
curl http://localhost:5000/api/models/model_abc123

# Download model
curl -O -J http://localhost:5000/api/models/model_abc123/download

# Get statistics
curl http://localhost:5000/api/stats

# Pretty print JSON
curl http://localhost:5000/api/models | python -m json.tool
```

## Integration Examples

### Mobile App (Flutter)

```dart
import 'package:http/http.dart' as http;
import 'dart:io';

Future<void> downloadModel(String serverUrl, String modelId) async {
  final url = '$serverUrl/api/models/$modelId/download';
  final response = await http.get(Uri.parse(url));
  
  if (response.statusCode == 200) {
    final file = File('/path/to/save/model.onnx');
    await file.writeAsBytes(response.bodyBytes);
    print('Model downloaded successfully');
  }
}
```

### Desktop App (Electron)

```javascript
const axios = require('axios');
const fs = require('fs');

async function downloadModel(serverUrl, modelId, savePath) {
    const url = `${serverUrl}/api/models/${modelId}/download`;
    const response = await axios.get(url, { responseType: 'stream' });
    
    const writer = fs.createWriteStream(savePath);
    response.data.pipe(writer);
    
    return new Promise((resolve, reject) => {
        writer.on('finish', resolve);
        writer.on('error', reject);
    });
}
```

### Web App (React)

```jsx
import React, { useState, useEffect } from 'react';

function ModelList({ serverUrl }) {
    const [models, setModels] = useState([]);
    
    useEffect(() => {
        fetch(`${serverUrl}/api/models`)
            .then(res => res.json())
            .then(data => setModels(data.models));
    }, [serverUrl]);
    
    const handleDownload = async (modelId) => {
        const url = `${serverUrl}/api/models/${modelId}/download`;
        const response = await fetch(url);
        const blob = await response.blob();
        
        const downloadUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = `model_${modelId}.onnx`;
        a.click();
        window.URL.revokeObjectURL(downloadUrl);
    };
    
    return (
        <div>
            {models.map(model => (
                <div key={model.id}>
                    <h3>{model.name}</h3>
                    <p>{model.size_mb} MB</p>
                    <button onClick={() => handleDownload(model.id)}>
                        Download
                    </button>
                </div>
            ))}
        </div>
    );
}
```

## Advanced Usage

### Programmatic Server Start

```python
from deployment import ModelServer
from model_manager import ModelManager

# Create server
model_manager = ModelManager()
server = ModelServer(
    model_manager=model_manager,
    host='0.0.0.0',
    port=5000
)

# Start in a thread (for integration into existing apps)
import threading
server_thread = threading.Thread(target=server.run, kwargs={'debug': False})
server_thread.daemon = True
server_thread.start()

print(f"Server running at {server.server_url}")
```

### Custom Integration

```python
from flask import Flask, jsonify
from deployment.model_server import ModelServer

# Integrate into existing Flask app
app = Flask(__name__)
model_server = ModelServer(host='0.0.0.0', port=5000)

# Add custom routes
@app.route('/api/custom')
def custom_endpoint():
    return jsonify({"message": "Custom endpoint"})

# Combine routes
# ... add model_server routes to your app

app.run()
```

## Security Notes

⚠️ **Important**: The basic server is designed for trusted networks. For production:

1. Add authentication
2. Use HTTPS
3. Implement rate limiting
4. Add logging
5. Use a production WSGI server

See the [deployment README](../deployment/README.md) for more details.
