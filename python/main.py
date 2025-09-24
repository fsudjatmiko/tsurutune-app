#!/usr/bin/env python3
"""
TsuruTune Backend - Python optimization engine
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import traceback
import time
import argparse

# Suppress TensorFlow logging globally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add the python directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_manager import ModelManager
from optimizers.cuda_optimizer import CudaOptimizer
from optimizers.cpu_optimizer import CpuOptimizer
from history_manager import HistoryManager

try:
    from utils.logger import setup_logger
except ImportError:
    # Fallback logger setup
    import logging
    def setup_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

class TsuruTuneBackend:
    """Main backend class for TsuruTune optimization"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.cuda_optimizer = CudaOptimizer()
        self.cpu_optimizer = CpuOptimizer()
        self.history_manager = HistoryManager()
        
    def optimize_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a model based on the provided configuration
        
        Args:
            config: Optimization configuration containing:
                - modelPath: Path to the model file
                - device: 'cuda' or 'cpu'
                - precision: Model precision
                - batch_size: Batch size for optimization
                - Additional device-specific parameters
                
        Returns:
            Optimization results
        """
        start_time = time.time()
        result = {
            "success": False,
            "error": None,
            "optimizedPath": "",
            "performanceGain": "0%",
            "memoryReduction": "0%",
            "duration": 0,
            "originalSize": 0,
            "optimizedSize": 0
        }
        
        try:
            # Validate configuration
            if not config.get("modelPath"):
                raise ValueError("Model path is required")
            
            model_path = config["modelPath"]
            device = config.get("device", "cpu")
            
            # Check if model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Get original model size
            original_size = os.path.getsize(model_path)
            result["originalSize"] = original_size
            
            # Store model locally first if not already imported
            try:
                stored_model_info = self.model_manager.import_model(model_path)
                if not stored_model_info:
                    raise Exception("Failed to import model")
            except Exception as e:
                # Model might already be imported, continue with optimization
                self.logger.warning(f"Model import skipped: {str(e)}")
            
            # Perform optimization based on device
            if device.lower() == "cuda":
                optimization_result = self.cuda_optimizer.optimize(config)
            elif device.lower() == "cpu":
                optimization_result = self.cpu_optimizer.optimize(config)
            else:
                raise ValueError(f"Unsupported device: {device}")
            
            # Update result with optimization outcome
            result.update(optimization_result)
            
            # Calculate metrics if optimization was successful
            if result["success"] and result["optimizedPath"]:
                optimized_size = os.path.getsize(result["optimizedPath"])
                result["optimizedSize"] = optimized_size
                
                # Calculate size reduction percentage
                if original_size > 0:
                    size_reduction = ((original_size - optimized_size) / original_size) * 100
                    result["memoryReduction"] = f"{size_reduction:.1f}%"
            
            # Calculate duration
            result["duration"] = round(time.time() - start_time, 2)
            
            # Save to history
            try:
                record_id = self.history_manager.add_optimization_record(config, result)
                result["recordId"] = record_id
            except Exception as e:
                print(f"Warning: Failed to save optimization history: {str(e)}")
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = round(time.time() - start_time, 2)
            
            # Still try to save failed optimization to history
            try:
                record_id = self.history_manager.add_optimization_record(config, result)
                result["recordId"] = record_id
            except Exception as history_error:
                print(f"Warning: Failed to save failed optimization to history: {str(history_error)}")
        
        return result
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models"""
        try:
            models = self.model_manager.list_models()
            return {"success": True, "models": models}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def refresh_models(self) -> Dict[str, Any]:
        """Refresh and scan for new models in directories"""
        try:
            models = self.model_manager.refresh_models()
            return {"success": True, "models": models, "message": "Model library refreshed"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def import_model(self, model_path: str, model_name: str = None) -> Dict[str, Any]:
        """Import a model into the local storage"""
        try:
            result = self.model_manager.import_model(model_path, model_name)
            return {"success": True, "model": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_optimization_history(self, limit: int = None, 
                               device_filter: str = None, 
                               status_filter: str = None) -> Dict[str, Any]:
        """Get optimization history with optional filtering"""
        try:
            history = self.history_manager.get_history(
                limit=limit,
                device_filter=device_filter,
                status_filter=status_filter
            )
            
            return {
                "success": True,
                "history": history,
                "total": len(history)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "history": [],
                "total": 0
            }
    
    def get_history_statistics(self) -> Dict[str, Any]:
        """Get optimization history statistics"""
        try:
            stats = self.history_manager.get_statistics()
            return {
                "success": True,
                "statistics": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def get_optimization_record(self, record_id: str) -> Dict[str, Any]:
        """Get a specific optimization record by ID"""
        try:
            record = self.history_manager.get_record(record_id)
            if record:
                return {
                    "success": True,
                    "record": record
                }
            else:
                return {
                    "success": False,
                    "error": "Record not found"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_optimization_record(self, record_id: str) -> Dict[str, Any]:
        """Delete an optimization record"""
        try:
            success = self.history_manager.delete_record(record_id)
            return {
                "success": success,
                "message": "Record deleted successfully" if success else "Record not found"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def clear_optimization_history(self) -> Dict[str, Any]:
        """Clear all optimization history"""
        try:
            success = self.history_manager.clear_history()
            return {
                "success": success,
                "message": "History cleared successfully" if success else "Failed to clear history"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_history(self, output_path: str, format: str = "json") -> Dict[str, Any]:
        """Export optimization history to file"""
        try:
            success = self.history_manager.export_history(output_path, format)
            return {
                "success": success,
                "message": f"History exported to {output_path}" if success else "Export failed"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def rerun_optimization(self, record_id: str) -> Dict[str, Any]:
        """Rerun an optimization with the same parameters"""
        try:
            # Get the original record
            record = self.history_manager.get_record(record_id)
            if not record:
                return {
                    "success": False,
                    "error": "Original record not found"
                }
            
            # Extract original configuration
            original_config = record.get("optimization_config", {})
            model_path = record.get("model_info", {}).get("path", "")
            
            if not model_path:
                return {
                    "success": False,
                    "error": "Original model path not found in record"
                }
            
            # Reconstruct full configuration
            config = {
                "modelPath": model_path,
                **original_config
            }
            
            # Run optimization again
            return self.optimize_model(config)
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for optimization capabilities"""
        try:
            import platform
            import psutil
            
            # Check CUDA availability
            cuda_available = False
            cuda_version = None
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    cuda_version = torch.version.cuda
            except ImportError:
                pass
            
            # Get system stats
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            return {
                "success": True,
                "system": {
                    "platform": platform.system(),
                    "architecture": platform.machine(),
                    "python_version": platform.python_version(),
                    "cpu_count": cpu_count,
                    "memory_total": memory.total,
                    "memory_available": memory.available,
                    "cuda_available": cuda_available,
                    "cuda_version": cuda_version
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "system": {}
            }
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate optimization configuration"""
        required_fields = ["modelPath", "device"]
        for field in required_fields:
            if field not in config:
                return False
        return True

def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description="TsuruTune Model Optimization Backend")
    parser.add_argument("command", choices=[
        "optimize", "list", "import", "history", "stats", "record", 
        "delete", "clear", "export", "rerun", "system", "refresh"
    ])
    parser.add_argument("--config", type=str, help="JSON configuration file or string")
    parser.add_argument("--model-path", type=str, help="Path to model file")
    parser.add_argument("--model-name", type=str, help="Name for imported model")
    parser.add_argument("--record-id", type=str, help="Record ID for operations")
    parser.add_argument("--output", type=str, help="Output path for exports")
    parser.add_argument("--format", type=str, default="json", help="Export format (json/csv)")
    parser.add_argument("--limit", type=int, help="Limit number of results")
    parser.add_argument("--device", type=str, help="Filter by device (cuda/cpu)")
    parser.add_argument("--status", type=str, help="Filter by status (success/failed)")
    
    args = parser.parse_args()
    
    backend = TsuruTuneBackend()
    
    if args.command == "optimize":
        if not args.config:
            print(json.dumps({"success": False, "error": "Config required for optimization"}))
            return
        
        try:
            # Try to parse as JSON string first, then as file path
            try:
                config = json.loads(args.config)
            except json.JSONDecodeError:
                with open(args.config, 'r') as f:
                    config = json.load(f)
            
            result = backend.optimize_model(config)
            print(json.dumps(result))
            
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))
    
    elif args.command == "list":
        result = backend.list_models()
        print(json.dumps(result))
    
    elif args.command == "import":
        if not args.model_path:
            print(json.dumps({"success": False, "error": "Model path required"}))
            return
        
        result = backend.import_model(args.model_path, args.model_name)
        print(json.dumps(result))
    
    elif args.command == "history":
        result = backend.get_optimization_history(
            limit=args.limit,
            device_filter=args.device,
            status_filter=args.status
        )
        print(json.dumps(result))
    
    elif args.command == "stats":
        result = backend.get_history_statistics()
        print(json.dumps(result))
    
    elif args.command == "record":
        if not args.record_id:
            print(json.dumps({"success": False, "error": "Record ID required"}))
            return
        
        result = backend.get_optimization_record(args.record_id)
        print(json.dumps(result))
    
    elif args.command == "delete":
        if not args.record_id:
            print(json.dumps({"success": False, "error": "Record ID required"}))
            return
        
        result = backend.delete_optimization_record(args.record_id)
        print(json.dumps(result))
    
    elif args.command == "clear":
        result = backend.clear_optimization_history()
        print(json.dumps(result))
    
    elif args.command == "export":
        if not args.output:
            print(json.dumps({"success": False, "error": "Output path required"}))
            return
        
        result = backend.export_history(args.output, args.format)
        print(json.dumps(result))
    
    elif args.command == "rerun":
        if not args.record_id:
            print(json.dumps({"success": False, "error": "Record ID required"}))
            return
        
        result = backend.rerun_optimization(args.record_id)
        print(json.dumps(result))
    
    elif args.command == "system":
        result = backend.get_system_info()
        print(json.dumps(result))
    
    elif args.command == "refresh":
        result = backend.refresh_models()
        print(json.dumps(result))

if __name__ == "__main__":
    main()
