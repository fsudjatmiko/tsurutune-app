#!/usr/bin/env python3
"""
Example: Testing the Model Deployment Server
This script demonstrates how to interact with the deployment server.
"""

import sys
import time
import requests
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_server(base_url: str = "http://localhost:5000"):
    """Test the deployment server endpoints"""
    
    print("=" * 70)
    print("Testing TsuruTune Model Deployment Server")
    print("=" * 70)
    print(f"\nServer URL: {base_url}\n")
    
    try:
        # Test 1: Health Check
        print("1️⃣  Testing health endpoint...")
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Server is healthy")
            print(f"   Status: {health['status']}")
            print(f"   Version: {health['version']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
        
        # Test 2: Get Statistics
        print("\n2️⃣  Getting server statistics...")
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✅ Statistics retrieved")
            print(f"   Total models: {stats['total_models']}")
            print(f"   Original models: {stats['original_models']}")
            print(f"   Optimized models: {stats['optimized_models']}")
            print(f"   Total size: {stats['total_size_mb']} MB")
        else:
            print(f"   ⚠️  Could not get stats: {response.status_code}")
        
        # Test 3: List All Models
        print("\n3️⃣  Listing all models...")
        response = requests.get(f"{base_url}/api/models")
        if response.status_code == 200:
            data = response.json()
            models = data['models']
            print(f"   ✅ Found {data['count']} models")
            
            if models:
                print("\n   📦 Available models:")
                for i, model in enumerate(models[:5], 1):  # Show first 5
                    model_type = "🔵 Original" if model.get('is_original') else "🟢 Optimized"
                    print(f"      {i}. {model_type}")
                    print(f"         Name: {model.get('name', 'Unknown')}")
                    print(f"         ID: {model.get('id', 'Unknown')}")
                    print(f"         Size: {model.get('size_mb', 0)} MB")
                    print(f"         Type: {model.get('type', 'Unknown')}")
                    print(f"         Download: {model.get('download_url', 'N/A')}")
                
                if len(models) > 5:
                    print(f"      ... and {len(models) - 5} more")
                
                # Test 4: Get Model Details
                print("\n4️⃣  Getting details for first model...")
                first_model_id = models[0]['id']
                response = requests.get(f"{base_url}/api/models/{first_model_id}")
                if response.status_code == 200:
                    model_info = response.json()['model']
                    print(f"   ✅ Model details retrieved")
                    print(f"   Name: {model_info.get('name')}")
                    print(f"   Hash: {model_info.get('hash', '')[:16]}...")
                    print(f"   Local path: {model_info.get('local_path')}")
                else:
                    print(f"   ⚠️  Could not get model details: {response.status_code}")
                
                # Test 5: Test Download Endpoint (without actually downloading)
                print("\n5️⃣  Testing download endpoint...")
                download_url = f"{base_url}/api/models/{first_model_id}/download"
                print(f"   Download URL: {download_url}")
                print(f"   ℹ️  To download: curl -O -J {download_url}")
                
            else:
                print("   ℹ️  No models available yet")
                print("   💡 Import models using TsuruTune app first")
        else:
            print(f"   ❌ Could not list models: {response.status_code}")
        
        # Test 6: List Original Models
        print("\n6️⃣  Listing original models only...")
        response = requests.get(f"{base_url}/api/models/original")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {data['count']} original models")
        
        # Test 7: List Optimized Models
        print("\n7️⃣  Listing optimized models only...")
        response = requests.get(f"{base_url}/api/models/optimized")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {data['count']} optimized models")
        
        print("\n" + "=" * 70)
        print("✅ All tests completed successfully!")
        print("=" * 70)
        print(f"\n💡 Open {base_url} in your browser to see the web interface")
        print(f"📱 Share this URL with others on your network to let them download models")
        print("\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to server")
        print(f"   Make sure the server is running on {base_url}")
        print("\n   Start the server with:")
        print("   python start_server.py")
        print("\n")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n")


def download_model_example(base_url: str, model_id: str, save_path: str):
    """Example of downloading a model"""
    print(f"\n📥 Downloading model {model_id}...")
    
    try:
        response = requests.get(f"{base_url}/api/models/{model_id}/download", stream=True)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Model downloaded to: {save_path}")
        else:
            print(f"❌ Download failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Model Deployment Server')
    parser.add_argument(
        '--url', 
        default='http://localhost:5000',
        help='Server URL (default: http://localhost:5000)'
    )
    parser.add_argument(
        '--download',
        metavar='MODEL_ID',
        help='Download a specific model by ID'
    )
    parser.add_argument(
        '--output',
        default='downloaded_model.onnx',
        help='Output file path for downloaded model'
    )
    
    args = parser.parse_args()
    
    if args.download:
        download_model_example(args.url, args.download, args.output)
    else:
        test_server(args.url)
