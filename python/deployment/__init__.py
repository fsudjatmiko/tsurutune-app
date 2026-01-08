"""
Deployment Module - Serves trained models via HTTP API
"""

from .model_server import ModelServer, start_server

__all__ = ['ModelServer', 'start_server']
