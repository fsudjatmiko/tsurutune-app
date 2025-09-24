"""
History Manager - Handles optimization history persistence
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class HistoryManager:
    """Manages optimization history storage and retrieval"""
    
    def __init__(self, history_file: str = None):
        if history_file is None:
            # Use history file in the project root
            self.history_file = Path(__file__).parent.parent / "data" / "optimization_history.json"
        else:
            self.history_file = Path(history_file)
        
        # Ensure data directory exists
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load optimization history from JSON file"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                return data.get("optimizations", [])
        except (json.JSONDecodeError, IOError, KeyError):
            # If file is corrupted or missing, start with empty history
            return []
    
    def _save_history(self):
        """Save optimization history to JSON file"""
        try:
            history_data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_optimizations": len(self.history),
                "optimizations": self.history
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
        except IOError as e:
            raise Exception(f"Failed to save history: {str(e)}")
    
    def add_optimization_record(self, config: Dict[str, Any], result: Dict[str, Any]) -> str:
        """
        Add a new optimization record to history
        
        Args:
            config: Optimization configuration used
            result: Optimization results
            
        Returns:
            Record ID
        """
        record_id = f"opt_{int(datetime.now().timestamp())}_{len(self.history)}"
        
        record = {
            "id": record_id,
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "path": config.get("modelPath", ""),
                "name": Path(config.get("modelPath", "")).stem,
                "size": result.get("originalSize", 0)
            },
            "optimization_config": {
                "device": config.get("device", "unknown"),
                "precision": config.get("precision", "unknown"),
                "batch_size": config.get("batch_size", 1),
                **self._extract_device_specific_config(config)
            },
            "results": {
                "success": result.get("success", False),
                "optimized_path": result.get("optimizedPath", ""),
                "performance_gain": result.get("performanceGain", "0%"),
                "memory_reduction": result.get("memoryReduction", "0%"),
                "duration": result.get("duration", 0),
                "original_size": result.get("originalSize", 0),
                "optimized_size": result.get("optimizedSize", 0),
                "error": result.get("error", None)
            },
            "metadata": {
                "tsurutune_version": "1.0.0",
                "optimization_engine": "TensorRT" if config.get("device") == "cuda" else "ONNX Runtime"
            }
        }
        
        self.history.append(record)
        self._save_history()
        
        return record_id
    
    def _extract_device_specific_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract device-specific configuration parameters"""
        device = config.get("device", "cpu")
        device_config = {}
        
        if device == "cuda":
            # CUDA/TensorRT specific parameters
            cuda_params = [
                "per_channel_quantization", "symmetric_quantization", 
                "calibration_dataset_path", "calibration_samples",
                "kv_cache_quantization", "outlier_retention",
                "sparsity_pattern", "sparsity_target",
                "graph_fusion", "bn_folding", "constant_folding",
                "workspace_size", "tactics", "dynamic_shapes"
            ]
            for param in cuda_params:
                if param in config:
                    device_config[param] = config[param]
        
        elif device == "cpu":
            # CPU specific parameters
            cpu_params = [
                "per_channel_quantization", "calibration_dataset_path", 
                "calibration_samples", "channel_pruning", "clustering",
                "graph_fusion", "constant_folding", "bn_folding",
                "num_threads", "intra_op_threads", "inter_op_threads"
            ]
            for param in cpu_params:
                if param in config:
                    device_config[param] = config[param]
        
        return device_config
    
    def get_history(self, limit: int = None, 
                   device_filter: str = None, 
                   status_filter: str = None) -> List[Dict[str, Any]]:
        """
        Get optimization history with optional filtering
        
        Args:
            limit: Maximum number of records to return
            device_filter: Filter by device type (cuda/cpu)
            status_filter: Filter by success status (success/failed)
            
        Returns:
            List of optimization records
        """
        filtered_history = self.history.copy()
        
        # Apply device filter
        if device_filter:
            filtered_history = [
                record for record in filtered_history
                if record.get("optimization_config", {}).get("device") == device_filter
            ]
        
        # Apply status filter
        if status_filter:
            if status_filter == "success":
                filtered_history = [
                    record for record in filtered_history
                    if record.get("results", {}).get("success") == True
                ]
            elif status_filter == "failed":
                filtered_history = [
                    record for record in filtered_history
                    if record.get("results", {}).get("success") == False
                ]
        
        # Sort by timestamp (newest first)
        filtered_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit:
            filtered_history = filtered_history[:limit]
        
        return filtered_history
    
    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific optimization record by ID"""
        for record in self.history:
            if record.get("id") == record_id:
                return record
        return None
    
    def delete_record(self, record_id: str) -> bool:
        """Delete an optimization record"""
        for i, record in enumerate(self.history):
            if record.get("id") == record_id:
                del self.history[i]
                self._save_history()
                return True
        return False
    
    def clear_history(self) -> bool:
        """Clear all optimization history"""
        try:
            self.history = []
            self._save_history()
            return True
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization history statistics"""
        if not self.history:
            return {
                "total_optimizations": 0,
                "successful_optimizations": 0,
                "failed_optimizations": 0,
                "success_rate": 0.0,
                "devices_used": {},
                "precisions_used": {},
                "average_performance_gain": 0.0,
                "average_memory_reduction": 0.0,
                "total_time_saved": 0.0
            }
        
        successful = [r for r in self.history if r.get("results", {}).get("success")]
        failed = [r for r in self.history if not r.get("results", {}).get("success")]
        
        # Count devices used
        devices_used = {}
        for record in self.history:
            device = record.get("optimization_config", {}).get("device", "unknown")
            devices_used[device] = devices_used.get(device, 0) + 1
        
        # Count precisions used
        precisions_used = {}
        for record in self.history:
            precision = record.get("optimization_config", {}).get("precision", "unknown")
            precisions_used[precision] = precisions_used.get(precision, 0) + 1
        
        # Calculate average performance gain
        performance_gains = []
        for record in successful:
            gain_str = record.get("results", {}).get("performance_gain", "0%")
            try:
                gain = float(gain_str.replace("%", "").replace("+", ""))
                performance_gains.append(gain)
            except:
                pass
        
        avg_performance_gain = sum(performance_gains) / len(performance_gains) if performance_gains else 0.0
        
        # Calculate average memory reduction
        memory_reductions = []
        for record in successful:
            reduction_str = record.get("results", {}).get("memory_reduction", "0%")
            try:
                reduction = float(reduction_str.replace("%", ""))
                memory_reductions.append(reduction)
            except:
                pass
        
        avg_memory_reduction = sum(memory_reductions) / len(memory_reductions) if memory_reductions else 0.0
        
        # Calculate total optimization time
        total_optimization_time = sum(
            record.get("results", {}).get("duration", 0) for record in self.history
        )
        
        return {
            "total_optimizations": len(self.history),
            "successful_optimizations": len(successful),
            "failed_optimizations": len(failed),
            "success_rate": (len(successful) / len(self.history)) * 100 if self.history else 0.0,
            "devices_used": devices_used,
            "precisions_used": precisions_used,
            "average_performance_gain": round(avg_performance_gain, 2),
            "average_memory_reduction": round(avg_memory_reduction, 2),
            "total_optimization_time": round(total_optimization_time, 2),
            "most_used_device": max(devices_used.items(), key=lambda x: x[1])[0] if devices_used else None,
            "most_used_precision": max(precisions_used.items(), key=lambda x: x[1])[0] if precisions_used else None
        }
    
    def export_history(self, output_path: str, format: str = "json") -> bool:
        """
        Export optimization history to file
        
        Args:
            output_path: Path to save the exported file
            format: Export format (json, csv)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump({
                        "export_timestamp": datetime.now().isoformat(),
                        "statistics": self.get_statistics(),
                        "optimizations": self.history
                    }, f, indent=2, default=str)
            
            elif format.lower() == "csv":
                import csv
                with open(output_path, 'w', newline='') as f:
                    if not self.history:
                        return True
                    
                    # Define CSV headers
                    headers = [
                        "timestamp", "model_name", "device", "precision", 
                        "success", "performance_gain", "memory_reduction", 
                        "duration", "error"
                    ]
                    
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    
                    for record in self.history:
                        writer.writerow({
                            "timestamp": record.get("timestamp", ""),
                            "model_name": record.get("model_info", {}).get("name", ""),
                            "device": record.get("optimization_config", {}).get("device", ""),
                            "precision": record.get("optimization_config", {}).get("precision", ""),
                            "success": record.get("results", {}).get("success", False),
                            "performance_gain": record.get("results", {}).get("performance_gain", ""),
                            "memory_reduction": record.get("results", {}).get("memory_reduction", ""),
                            "duration": record.get("results", {}).get("duration", 0),
                            "error": record.get("results", {}).get("error", "")
                        })
            
            else:
                return False
            
            return True
            
        except Exception as e:
            print(f"Export failed: {str(e)}")
            return False
    
    def get_recent_optimizations(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get optimizations from the last N days"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent = []
        for record in self.history:
            try:
                record_date = datetime.fromisoformat(record.get("timestamp", ""))
                if record_date >= cutoff_date:
                    recent.append(record)
            except:
                pass
        
        return sorted(recent, key=lambda x: x.get("timestamp", ""), reverse=True)
