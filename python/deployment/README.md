# TsuruTune Deployment Module

This module allows you to serve your trained AI models (both original and optimized) over HTTP, making them accessible for download and integration into other applications.

## Features

- 🌐 **HTTP API** - RESTful API for listing and downloading models
- 📱 **QR Code Access** - Generate QR codes for easy mobile access
- 🔒 **Network Support** - Works on local network (LAN) and can be exposed externally
- 📊 **Statistics** - View server stats and model information
- 🎨 **Web Interface** - Beautiful web UI with API documentation
- ⚡ **CORS Enabled** - Cross-origin requests supported for web apps

## Installation

Install the required dependencies:

```bash
pip install -r deployment/requirements.txt
```

Or install individually:

```bash
pip install Flask flask-cors qrcode Pillow
```

## Quick Start

### Option 1: Run Standalone Server

```bash
cd python
python -m deployment.model_server
```

Or with custom settings:

```bash
python -m deployment.model_server --host 0.0.0.0 --port 8000
```

### Option 2: Import and Use in Your Code

```python
from deployment import start_server, ModelServer
from model_manager import ModelManager

# Simple start
start_server()

# Or with custom configuration
model_manager = ModelManager()
server = ModelServer(
    model_manager=model_manager,
    host='0.0.0.0',
    port=5000
)
server.run()
```

## API Endpoints

### Base URL
`http://<your-ip>:<port>`

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface with documentation |
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Server statistics |
| GET | `/api/models` | List all models |
| GET | `/api/models/original` | List only original models |
| GET | `/api/models/optimized` | List only optimized models |
| GET | `/api/models/<id>` | Get model details |
| GET | `/api/models/<id>/download` | Download a model file |

## Usage Examples

### Python Client

```python
import requests

# List all models
response = requests.get('http://192.168.1.100:5000/api/models')
models = response.json()

print(f"Found {models['count']} models")
for model in models['models']:
    print(f"- {model['name']} ({model['size_mb']} MB)")

# Download a specific model
model_id = 'your_model_id'
response = requests.get(f'http://192.168.1.100:5000/api/models/{model_id}/download')

if response.status_code == 200:
    with open('downloaded_model.onnx', 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully!")
```

### JavaScript/TypeScript

```javascript
// Fetch models
async function getModels() {
    const response = await fetch('http://192.168.1.100:5000/api/models');
    const data = await response.json();
    console.log(data.models);
}

// Download model
async function downloadModel(modelId) {
    const response = await fetch(`http://192.168.1.100:5000/api/models/${modelId}/download`);
    const blob = await response.blob();
    
    // Trigger download in browser
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model.onnx';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
}
```

### cURL

```bash
# List models
curl http://192.168.1.100:5000/api/models

# Get model info
curl http://192.168.1.100:5000/api/models/model_abc123

# Download model
curl -O -J http://192.168.1.100:5000/api/models/model_abc123/download
```

## Network Access

### Same Network (LAN)
Anyone on the same WiFi/network can access the server using your local IP address. The server automatically detects and displays your local IP when started.

Example: `http://192.168.1.100:5000`

### External Network (Internet)

To allow access from outside your local network, you have several options:

#### 1. Port Forwarding (Permanent)
- Access your router's admin panel
- Set up port forwarding for port 5000 (or your chosen port)
- Forward to your machine's local IP
- Share your public IP with external users

#### 2. ngrok (Quick & Easy)
Perfect for temporary sharing or testing:

```bash
# Install ngrok from https://ngrok.com/download
# Then run:
ngrok http 5000
```

This gives you a public URL like: `https://abc123.ngrok.io`

#### 3. Cloud Deployment
Deploy to a cloud service:
- AWS EC2
- Google Cloud Platform
- Azure
- DigitalOcean
- Heroku

### Security Considerations

⚠️ **Important**: This server is designed for trusted networks. For production use:

1. Add authentication (API keys, JWT tokens)
2. Use HTTPS/SSL certificates
3. Implement rate limiting
4. Add access logs
5. Use a production WSGI server (gunicorn, waitress)

## Production Deployment

For production use, run with a production-grade server:

### Linux/Mac (with gunicorn)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 'deployment.model_server:create_app()'
```

### Windows (with waitress)

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 deployment.model_server:create_app
```

## Integration Examples

### Mobile App Integration

```dart
// Flutter example
import 'package:http/http.dart' as http;

Future<void> downloadModel(String serverUrl, String modelId) async {
  final url = '$serverUrl/api/models/$modelId/download';
  final response = await http.get(Uri.parse(url));
  
  if (response.statusCode == 200) {
    // Save to local storage
    final file = File('path/to/save/model.onnx');
    await file.writeAsBytes(response.bodyBytes);
  }
}
```

### Desktop App Integration

```python
# PyQt/Electron app example
import requests
from pathlib import Path

def download_model_to_app(server_url: str, model_id: str, save_path: Path):
    """Download model and save to application directory"""
    url = f"{server_url}/api/models/{model_id}/download"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False
```

## Troubleshooting

### Port Already in Use
```bash
# Use a different port
python -m deployment.model_server --port 8000
```

### Can't Access from Other Devices
- Check firewall settings
- Ensure devices are on the same network
- Verify the server is running on `0.0.0.0` not `127.0.0.1`

### Models Not Showing Up
- Check that models exist in the `models/` directory
- Verify `metadata.json` is properly formatted
- Check server logs for errors

## Configuration

Environment variables can be used:

```bash
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export FLASK_DEBUG=False
```

## License

Part of TsuruTune project.
