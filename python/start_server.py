#!/usr/bin/env python3
"""
Start TsuruTune Model Deployment Server
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from deployment import start_server

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TsuruTune Model Deployment Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with default settings (0.0.0.0:5000)
  python start_server.py
  
  # Start on custom port
  python start_server.py --port 8000
  
  # Start in debug mode
  python start_server.py --debug
  
  # Custom host and port
  python start_server.py --host 192.168.1.100 --port 8080
        """
    )
    
    parser.add_argument(
        '--host', 
        default='0.0.0.0', 
        help='Host address to bind to (default: 0.0.0.0 - all interfaces)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000, 
        help='Port number to listen on (default: 5000)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Run in debug mode (auto-reload on code changes)'
    )
    
    args = parser.parse_args()
    
    # Import here to avoid import errors before arg parsing
    from deployment.model_server import ModelServer
    
    try:
        server = ModelServer(host=args.host, port=args.port)
        server.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\n\n[STOP] Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        sys.exit(1)
