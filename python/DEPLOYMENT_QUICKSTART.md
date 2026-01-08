# 🚀 Quick Start Guide - Model Deployment

This guide will help you quickly set up and use the TsuruTune model deployment server.

## What is Model Deployment?

The deployment module allows you to:
- 📤 **Share models** over HTTP on your local network or internet
- 📥 **Download models** from any device (computer, phone, tablet)
- 🔗 **Integrate models** into your applications via REST API
- 📱 **QR Code access** for easy mobile connection

## Step-by-Step Setup

### Step 1: Install Dependencies

```bash
cd python
pip install Flask flask-cors qrcode Pillow
```

**Already done?** The dependencies are included in `requirements-cpu.txt`

### Step 2: Start the Server

```bash
# From the python directory
python start_server.py
```

**Custom port?**
```bash
python start_server.py --port 8000
```

### Step 3: Access the Server

The server will display:
```
🚀 TsuruTune Model Server Starting...
======================================================================

📡 Server URL: http://192.168.1.100:5000
🌐 Local IP: 192.168.1.100
🔌 Port: 5000

💡 Access the web interface at: http://192.168.1.100:5000
📱 Share this URL with others on your network!
```

**Open the URL in your browser** to see:
- Web interface with all features
- QR code for mobile access
- Complete API documentation
- List of available models

## Common Use Cases

### Use Case 1: Download Model to Another Computer

**On the same network:**
```bash
# From any computer on your network
curl -O -J http://192.168.1.100:5000/api/models/model_abc123/download
```

**Or use a browser:** Just click the download button in the web interface!

### Use Case 2: Integrate into Mobile App

```python
# Python example for mobile backend
import requests

def get_latest_model(server_url):
    response = requests.get(f'{server_url}/api/models')
    models = response.json()['models']
    
    # Get the latest optimized model
    optimized = [m for m in models if not m['is_original']]
    if optimized:
        latest = sorted(optimized, key=lambda x: x['created'], reverse=True)[0]
        return latest['download_url']
```

### Use Case 3: Share with Team Members

1. **Start the server** on your machine
2. **Share the URL** shown when starting (e.g., `http://192.168.1.100:5000`)
3. **Team members** can browse and download models from any device

### Use Case 4: Deploy to Remote Server

**For internet access**, you have options:

#### Option A: Using ngrok (Easiest)
```bash
# Install ngrok: https://ngrok.com/download
ngrok http 5000

# Share the generated URL (e.g., https://abc123.ngrok.io)
```

#### Option B: Cloud Deployment
Deploy to AWS, Google Cloud, Azure, or DigitalOcean for permanent access.

## Testing the Server

Run the test script to verify everything works:

```bash
python test_deployment.py
```

This will:
- ✅ Check server health
- ✅ List all models
- ✅ Show download URLs
- ✅ Display statistics

## API Quick Reference

| Endpoint | What it does |
|----------|-------------|
| `GET /` | Web interface |
| `GET /api/models` | List all models |
| `GET /api/models/<id>` | Get model details |
| `GET /api/models/<id>/download` | Download model |
| `GET /api/models/original` | List original models |
| `GET /api/models/optimized` | List optimized models |
| `GET /api/stats` | Server statistics |

## Example: Python Client

```python
import requests

# Connect to server
SERVER_URL = "http://192.168.1.100:5000"

# List all models
response = requests.get(f"{SERVER_URL}/api/models")
models = response.json()['models']

print(f"Found {len(models)} models:")
for model in models:
    print(f"  - {model['name']}: {model['download_url']}")

# Download a specific model
model_id = models[0]['id']
response = requests.get(f"{SERVER_URL}/api/models/{model_id}/download")

with open('my_model.onnx', 'wb') as f:
    f.write(response.content)
print("Model downloaded!")
```

## Example: JavaScript/Web

```javascript
// Fetch and display models
async function loadModels() {
    const response = await fetch('http://192.168.1.100:5000/api/models');
    const data = await response.json();
    
    console.log(`Found ${data.count} models`);
    data.models.forEach(model => {
        console.log(`${model.name} - ${model.size_mb}MB`);
    });
}

// Download a model in browser
async function downloadModel(modelId) {
    const url = `http://192.168.1.100:5000/api/models/${modelId}/download`;
    const a = document.createElement('a');
    a.href = url;
    a.download = 'model.onnx';
    a.click();
}
```

## Troubleshooting

### Problem: Can't connect to server

**Solution:**
1. Check if server is running: `python start_server.py`
2. Verify you're using the correct IP and port
3. Check firewall settings (allow port 5000)
4. Make sure you're on the same network

### Problem: No models showing up

**Solution:**
1. Import models using TsuruTune first
2. Check the `models/` directory exists
3. Verify `models/metadata.json` is valid
4. Run model optimization to create some models

### Problem: Can't access from other devices

**Solution:**
1. Server must be running on `0.0.0.0` (default)
2. Check Windows Firewall: allow Python/port 5000
3. Verify devices are on same network
4. Try using IP instead of localhost

### Problem: Want to use on the internet

**Solution:**
- Use **ngrok** for quick temporary access: `ngrok http 5000`
- Set up **port forwarding** on your router for permanent access
- Deploy to a **cloud service** (AWS, Azure, Google Cloud)

## Security Tips

⚠️ **For local network only:**
- Default setup is fine for trusted networks
- Good for home/office use

⚠️ **For internet access:**
- Add authentication (API keys, passwords)
- Use HTTPS/SSL certificates  
- Implement rate limiting
- Add access logs
- Consider using a reverse proxy (nginx)

## Next Steps

- 📖 Read full documentation: `python/deployment/README.md`
- 🧪 Try examples: `python/DEPLOYMENT_EXAMPLES.md`
- 🔧 Customize the server for your needs
- 🌐 Deploy to cloud for permanent access

## Need Help?

- Check the logs in the terminal where you started the server
- Run `python test_deployment.py` to diagnose issues
- See full documentation in `deployment/README.md`

---

**Happy deploying! 🎉**
