"""
Model Server - HTTP API for serving and downloading models
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import socket
import qrcode
import io
import base64

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, jsonify, send_file, request, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model_manager import ModelManager

class ModelServer:
    """Flask-based HTTP server for model serving and downloading"""
    
    def __init__(self, model_manager: ModelManager = None, host: str = '0.0.0.0', port: int = 5000):
        """
        Initialize the model server
        
        Args:
            model_manager: ModelManager instance (creates new one if None)
            host: Server host address (0.0.0.0 for all interfaces)
            port: Server port number
        """
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for cross-origin requests
        
        self.host = host
        self.port = port
        self.model_manager = model_manager or ModelManager()
        
        # Get local IP address
        self.local_ip = self._get_local_ip()
        self.server_url = f"http://{self.local_ip}:{self.port}"
        
        # Setup routes
        self._setup_routes()
        
    def _get_local_ip(self) -> str:
        """Get the local IP address of the machine"""
        try:
            # Create a socket connection to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"
    
    def _generate_qr_code(self, url: str) -> str:
        """Generate QR code for a URL and return as base64 string"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.route('/')
        def index():
            """Home page with API documentation"""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>TsuruTune Model Server</title>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                    }
                    .big-button {
                        display: block;
                        background: white;
                        color: #667eea;
                        text-decoration: none;
                        padding: 20px 30px;
                        border-radius: 8px;
                        font-size: 1.25rem;
                        font-weight: 600;
                        text-align: center;
                        margin: 20px 0;
                        transition: transform 0.2s, box-shadow 0.2s;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }
                    .big-button:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                    }
                    .big-button::before {
                        content: "📦 ";
                        font-size: 1.5rem;
                    }
                    .card {
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .endpoint {
                        background: #f8f9fa;
                        padding: 15px;
                        border-left: 4px solid #667eea;
                        margin: 10px 0;
                        border-radius: 4px;
                    }
                    .method {
                        display: inline-block;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-weight: bold;
                        font-size: 12px;
                        margin-right: 10px;
                    }
                    .get { background: #61affe; color: white; }
                    .post { background: #49cc90; color: white; }
                    .delete { background: #f93e3e; color: white; }
                    code {
                        background: #2d2d2d;
                        color: #f8f8f2;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-family: 'Courier New', monospace;
                    }
                    .url-box {
                        background: #e8f5e9;
                        padding: 15px;
                        border-radius: 8px;
                        margin: 10px 0;
                        text-align: center;
                    }
                    .qr-code {
                        margin: 20px 0;
                        text-align: center;
                    }
                    .qr-code img {
                        max-width: 200px;
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        padding: 10px;
                        background: white;
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 TsuruTune Model Server</h1>
                    <p>Serve and share your AI models across networks</p>
                </div>
                
                <a href="/models" class="big-button">View & Download Models</a>
                
                <div class="card">
                    <h2>📡 Server Information</h2>
                    <div class="url-box">
                        <strong>Server URL:</strong> <code>{{ server_url }}</code>
                    </div>
                    <p><strong>Local IP:</strong> {{ local_ip }}</p>
                    <p><strong>Port:</strong> {{ port }}</p>
                    <p><strong>Status:</strong> ✅ Running</p>
                    
                    <div class="qr-code">
                        <p><strong>Scan to access from mobile:</strong></p>
                        <img src="{{ qr_code }}" alt="QR Code">
                    </div>
                </div>
                
                <div class="card">
                    <h2>📚 API Endpoints</h2>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/</strong>
                        <p>This documentation page</p>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/api/models</strong>
                        <p>List all available models (original and optimized)</p>
                        <code>{{ server_url }}/api/models</code>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/api/models/&lt;model_id&gt;</strong>
                        <p>Get detailed information about a specific model</p>
                        <code>{{ server_url }}/api/models/model_abc123</code>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/api/models/&lt;model_id&gt;/download</strong>
                        <p>Download a model file</p>
                        <code>{{ server_url }}/api/models/model_abc123/download</code>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/api/models/original</strong>
                        <p>List only original (non-optimized) models</p>
                        <code>{{ server_url }}/api/models/original</code>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/api/models/optimized</strong>
                        <p>List only optimized models</p>
                        <code>{{ server_url }}/api/models/optimized</code>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/api/stats</strong>
                        <p>Get server statistics (total models, sizes, etc.)</p>
                        <code>{{ server_url }}/api/stats</code>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method get">GET</span>
                        <strong>/api/health</strong>
                        <p>Health check endpoint</p>
                        <code>{{ server_url }}/api/health</code>
                    </div>
                </div>
                
                <div class="card">
                    <h2>💡 Usage Examples</h2>
                    
                    <h3>Python</h3>
                    <pre><code>import requests

# List all models
response = requests.get('{{ server_url }}/api/models')
models = response.json()

# Download a model
model_id = 'your_model_id'
response = requests.get(f'{{ server_url }}/api/models/{model_id}/download')
with open('downloaded_model.onnx', 'wb') as f:
    f.write(response.content)</code></pre>
                    
                    <h3>JavaScript</h3>
                    <pre><code>// List all models
fetch('{{ server_url }}/api/models')
    .then(response => response.json())
    .then(data => console.log(data));

// Download a model
const modelId = 'your_model_id';
fetch(`{{ server_url }}/api/models/${modelId}/download`)
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'model.onnx';
        a.click();
    });</code></pre>
                    
                    <h3>cURL</h3>
                    <pre><code># List models
curl {{ server_url }}/api/models

# Download model
curl -O -J {{ server_url }}/api/models/your_model_id/download</code></pre>
                </div>
                
                <div class="card">
                    <h2>🌐 Network Access</h2>
                    <h3>Same Network (LAN)</h3>
                    <p>Use the server URL above: <code>{{ server_url }}</code></p>
                    <p>Anyone on the same WiFi/network can access this URL directly.</p>
                    
                    <h3>External Network (Internet)</h3>
                    <p>To allow access from outside your network, you need to:</p>
                    <ol>
                        <li>Set up port forwarding on your router (forward port {{ port }})</li>
                        <li>Use a service like ngrok for temporary tunneling</li>
                        <li>Deploy to a cloud server (AWS, Azure, Google Cloud, etc.)</li>
                    </ol>
                    
                    <h4>Quick External Access with ngrok:</h4>
                    <pre><code># Install ngrok: https://ngrok.com/download
ngrok http {{ port }}</code></pre>
                </div>
            </body>
            </html>
            """
            qr_code = self._generate_qr_code(self.server_url)
            return render_template_string(
                html,
                server_url=self.server_url,
                local_ip=self.local_ip,
                port=self.port,
                qr_code=qr_code
            )
        
        @self.app.route('/models')
        def models_page():
            """Models page with download buttons"""
            try:
                models = self.model_manager.list_models()
            except Exception:
                models = []
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>TsuruTune Models - Download</title>
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background: #f5f5f5;
                    }
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }
                    .header h1 {
                        margin: 0;
                        font-size: 2rem;
                    }
                    .header a {
                        color: white;
                        text-decoration: none;
                        background: rgba(255,255,255,0.2);
                        padding: 10px 20px;
                        border-radius: 5px;
                        transition: background 0.3s;
                    }
                    .header a:hover {
                        background: rgba(255,255,255,0.3);
                    }
                    .models-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }
                    .model-card {
                        background: white;
                        padding: 25px;
                        border-radius: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        transition: transform 0.2s, box-shadow 0.2s;
                    }
                    .model-card:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    }
                    .model-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: start;
                        margin-bottom: 15px;
                    }
                    .model-name {
                        font-size: 1.25rem;
                        font-weight: 600;
                        color: #333;
                        margin: 0 0 5px 0;
                    }
                    .model-badge {
                        padding: 4px 12px;
                        border-radius: 12px;
                        font-size: 0.75rem;
                        font-weight: 600;
                        text-transform: uppercase;
                    }
                    .badge-original {
                        background: #e3f2fd;
                        color: #1976d2;
                    }
                    .badge-optimized {
                        background: #e8f5e9;
                        color: #388e3c;
                    }
                    .model-info {
                        margin: 15px 0;
                        color: #666;
                        font-size: 0.9rem;
                    }
                    .model-info-row {
                        display: flex;
                        justify-content: space-between;
                        padding: 8px 0;
                        border-bottom: 1px solid #f0f0f0;
                    }
                    .model-info-row:last-child {
                        border-bottom: none;
                    }
                    .model-info-label {
                        font-weight: 500;
                        color: #888;
                    }
                    .model-info-value {
                        font-weight: 600;
                        color: #333;
                    }
                    .download-btn {
                        display: inline-block;
                        width: 100%;
                        padding: 12px 24px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        text-decoration: none;
                        border-radius: 6px;
                        font-weight: 600;
                        text-align: center;
                        transition: transform 0.2s, box-shadow 0.2s;
                        margin-top: 15px;
                        cursor: pointer;
                        border: none;
                        font-size: 1rem;
                    }
                    .download-btn:hover {
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
                    }
                    .download-btn:active {
                        transform: translateY(0);
                    }
                    .empty-state {
                        text-align: center;
                        padding: 60px 20px;
                        background: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .empty-state svg {
                        margin-bottom: 20px;
                        opacity: 0.5;
                    }
                    .empty-state h2 {
                        color: #333;
                        margin-bottom: 10px;
                    }
                    .empty-state p {
                        color: #666;
                    }
                    .stats-bar {
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 30px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        display: flex;
                        justify-content: space-around;
                        flex-wrap: wrap;
                        gap: 20px;
                    }
                    .stat-item {
                        text-align: center;
                    }
                    .stat-value {
                        font-size: 2rem;
                        font-weight: 700;
                        color: #667eea;
                    }
                    .stat-label {
                        color: #666;
                        font-size: 0.9rem;
                        margin-top: 5px;
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <div>
                        <h1>📦 Available Models</h1>
                        <p style="margin: 5px 0 0 0; opacity: 0.9;">Click to download models</p>
                    </div>
                    <a href="/">← Back to Docs</a>
                </div>
                
                {% if models %}
                <div class="stats-bar">
                    <div class="stat-item">
                        <div class="stat-value">{{ models|length }}</div>
                        <div class="stat-label">Total Models</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{{ original_count }}</div>
                        <div class="stat-label">Original Models</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{{ optimized_count }}</div>
                        <div class="stat-label">Optimized Models</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{{ total_size_mb }} MB</div>
                        <div class="stat-label">Total Size</div>
                    </div>
                </div>
                
                <div class="models-grid">
                    {% for model in models %}
                    <div class="model-card">
                        <div class="model-header">
                            <div>
                                <h3 class="model-name">{{ model.name or 'Unnamed Model' }}</h3>
                                <span class="model-badge {% if model.is_original %}badge-original{% else %}badge-optimized{% endif %}">
                                    {% if model.is_original %}Original{% else %}Optimized{% endif %}
                                </span>
                            </div>
                        </div>
                        
                        <div class="model-info">
                            <div class="model-info-row">
                                <span class="model-info-label">Size:</span>
                                <span class="model-info-value">{{ "%.2f"|format(model.size_mb or 0) }} MB</span>
                            </div>
                            <div class="model-info-row">
                                <span class="model-info-label">Type:</span>
                                <span class="model-info-value">{{ model.type or 'Unknown' }}</span>
                            </div>
                            <div class="model-info-row">
                                <span class="model-info-label">ID:</span>
                                <span class="model-info-value" style="font-size: 0.8rem; word-break: break-all;">{{ model.id }}</span>
                            </div>
                        </div>
                        
                        <a href="/api/models/{{ model.id }}/download" class="download-btn">
                            ⬇️ Download Model
                        </a>
                    </div>
                    {% endfor %}
                </div>
                {% else %}
                <div class="empty-state">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                    </svg>
                    <h2>No Models Available</h2>
                    <p>Import and optimize models in TsuruTune first, then they'll appear here for download.</p>
                </div>
                {% endif %}
            </body>
            </html>
            """
            
            original_count = sum(1 for m in models if m.get('is_original', False))
            optimized_count = len(models) - original_count
            total_size_mb = round(sum(m.get('size_mb', 0) for m in models), 2)
            
            return render_template_string(
                html,
                models=models,
                original_count=original_count,
                optimized_count=optimized_count,
                total_size_mb=total_size_mb
            )
        
        @self.app.route('/api/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "server_url": self.server_url,
                "version": "1.0.0"
            })
        
        @self.app.route('/api/stats')
        def stats():
            """Get server statistics"""
            models = self.model_manager.list_models()
            
            original_models = [m for m in models if m.get('is_original', False)]
            optimized_models = [m for m in models if not m.get('is_original', True)]
            
            total_size = sum(m.get('size', 0) for m in models)
            original_size = sum(m.get('size', 0) for m in original_models)
            optimized_size = sum(m.get('size', 0) for m in optimized_models)
            
            return jsonify({
                "total_models": len(models),
                "original_models": len(original_models),
                "optimized_models": len(optimized_models),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "original_size_mb": round(original_size / (1024 * 1024), 2),
                "optimized_size_mb": round(optimized_size / (1024 * 1024), 2),
                "server_url": self.server_url
            })
        
        @self.app.route('/api/models')
        def list_models():
            """List all available models"""
            try:
                models = self.model_manager.list_models()
                
                # Add download URLs to each model
                for model in models:
                    model_id = model.get('id')
                    if model_id:
                        model['download_url'] = f"{self.server_url}/api/models/{model_id}/download"
                        model['info_url'] = f"{self.server_url}/api/models/{model_id}"
                
                return jsonify({
                    "success": True,
                    "count": len(models),
                    "models": models,
                    "server_url": self.server_url
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/models/original')
        def list_original_models():
            """List only original (non-optimized) models"""
            try:
                all_models = self.model_manager.list_models()
                original_models = [m for m in all_models if m.get('is_original', False)]
                
                # Add download URLs
                for model in original_models:
                    model_id = model.get('id')
                    if model_id:
                        model['download_url'] = f"{self.server_url}/api/models/{model_id}/download"
                        model['info_url'] = f"{self.server_url}/api/models/{model_id}"
                
                return jsonify({
                    "success": True,
                    "count": len(original_models),
                    "models": original_models,
                    "server_url": self.server_url
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/models/optimized')
        def list_optimized_models():
            """List only optimized models"""
            try:
                all_models = self.model_manager.list_models()
                optimized_models = [m for m in all_models if not m.get('is_original', True)]
                
                # Add download URLs
                for model in optimized_models:
                    model_id = model.get('id')
                    if model_id:
                        model['download_url'] = f"{self.server_url}/api/models/{model_id}/download"
                        model['info_url'] = f"{self.server_url}/api/models/{model_id}"
                
                return jsonify({
                    "success": True,
                    "count": len(optimized_models),
                    "models": optimized_models,
                    "server_url": self.server_url
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/models/<model_id>')
        def get_model_info(model_id):
            """Get detailed information about a specific model"""
            try:
                model_info = self.model_manager.get_model_info(model_id)
                
                if not model_info:
                    return jsonify({
                        "success": False,
                        "error": "Model not found"
                    }), 404
                
                # Add download URL
                model_info['download_url'] = f"{self.server_url}/api/models/{model_id}/download"
                model_info['info_url'] = f"{self.server_url}/api/models/{model_id}"
                
                return jsonify({
                    "success": True,
                    "model": model_info
                })
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/api/models/<model_id>/download')
        def download_model(model_id):
            """Download a model file"""
            try:
                model_info = self.model_manager.get_model_info(model_id)
                
                if not model_info:
                    return jsonify({
                        "success": False,
                        "error": "Model not found"
                    }), 404
                
                model_path = model_info.get('local_path')
                
                if not model_path or not os.path.exists(model_path):
                    return jsonify({
                        "success": False,
                        "error": "Model file not found on disk"
                    }), 404
                
                # Get filename for download
                filename = os.path.basename(model_path)
                
                return send_file(
                    model_path,
                    as_attachment=True,
                    download_name=filename,
                    mimetype='application/octet-stream'
                )
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
    
    def run(self, debug: bool = False, threaded: bool = True):
        """
        Start the Flask server
        
        Args:
            debug: Enable debug mode
            threaded: Enable multi-threading
        """
        print("\n" + "="*70)
        print("[START] TsuruTune Model Server Starting...")
        print("="*70)
        print(f"\nServer URL: {self.server_url}")
        print(f"Local IP: {self.local_ip}")
        print(f"Port: {self.port}")
        print(f"\nAccess the web interface at: {self.server_url}")
        print(f"Share this URL with others on your network!")
        print("\n" + "="*70 + "\n")
        
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                debug=debug,
                threaded=threaded
            )
        except Exception as e:
            print(f"\n[ERROR] Error starting server: {str(e)}")
            raise


def start_server(host: str = '0.0.0.0', port: int = 5000, model_manager: ModelManager = None):
    """
    Convenience function to start the model server
    
    Args:
        host: Server host address
        port: Server port number
        model_manager: Optional ModelManager instance
    """
    server = ModelServer(model_manager=model_manager, host=host, port=port)
    server.run()


if __name__ == "__main__":
    # Allow running the server standalone
    import argparse
    
    parser = argparse.ArgumentParser(description='TsuruTune Model Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port number (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    server = ModelServer(host=args.host, port=args.port)
    server.run(debug=args.debug)
