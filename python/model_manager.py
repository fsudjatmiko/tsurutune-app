"""
Model Manager - Handles local model storage and management
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class ModelManager:
    """Manages local model storage and metadata"""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            # Use models directory in the project root
            self.base_path = Path(__file__).parent.parent / "models"
        else:
            self.base_path = Path(base_path)
        
        # Create directory structure
        self.original_models_path = self.base_path / "original"
        self.optimized_models_path = self.base_path / "optimized"
        self.metadata_path = self.base_path / "metadata.json"
        
        self._ensure_directories()
        self._load_metadata()
        self._scan_existing_models()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.original_models_path.mkdir(parents=True, exist_ok=True)
        self.optimized_models_path.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self):
        """Load model metadata from JSON file"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.metadata = {"models": {}}
        else:
            self.metadata = {"models": {}}
    
    def _scan_existing_models(self):
        """Scan for existing model files that aren't in metadata and add them"""
        supported_extensions = {'.onnx', '.pth', '.pt', '.h5', '.keras', '.pb', '.tflite', '.engine', '.trt'}
        
        # Scan original models directory
        if self.original_models_path.exists():
            for file_path in self.original_models_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    # Check if this file is already tracked
                    file_tracked = any(
                        model.get("local_path") == str(file_path) 
                        for model in self.metadata["models"].values()
                    )
                    
                    if not file_tracked:
                        try:
                            # Auto-import this existing model
                            model_name = file_path.stem
                            file_hash = self._get_file_hash(file_path)
                            model_id = f"{model_name}_{file_hash[:8]}"
                            
                            # Create model metadata for existing file
                            model_info = self._get_model_info(file_path)
                            model_metadata = {
                                "id": model_id,
                                "name": model_name,
                                "original_path": str(file_path),  # Same as local for existing files
                                "local_path": str(file_path),
                                "hash": file_hash,
                                "imported_at": datetime.now().isoformat(),
                                "is_original": True,
                                "type": model_info["extension"],
                                "auto_discovered": True,  # Mark as auto-discovered
                                **model_info
                            }
                            
                            self.metadata["models"][model_id] = model_metadata
                            
                        except Exception as e:
                            # Skip files that can't be processed
                            print(f"Warning: Could not auto-import {file_path}: {str(e)}")
                            continue
        
        # Scan optimized models directory
        if self.optimized_models_path.exists():
            for file_path in self.optimized_models_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    # Check if this file is already tracked
                    file_tracked = any(
                        model.get("local_path") == str(file_path) 
                        for model in self.metadata["models"].values()
                    )
                    
                    if not file_tracked:
                        try:
                            # Auto-import this existing optimized model
                            model_name = file_path.stem
                            file_hash = self._get_file_hash(file_path)
                            model_id = f"{model_name}_{file_hash[:8]}"
                            
                            # Create model metadata for existing optimized file
                            model_info = self._get_model_info(file_path)
                            model_metadata = {
                                "id": model_id,
                                "name": model_name,
                                "original_id": None,  # Unknown original
                                "local_path": str(file_path),
                                "hash": file_hash,
                                "created_at": datetime.now().isoformat(),
                                "is_original": False,
                                "optimization_config": {"auto_discovered": True},
                                "optimization_results": {"auto_discovered": True},
                                "type": model_info["extension"],
                                "auto_discovered": True,  # Mark as auto-discovered
                                **model_info
                            }
                            
                            self.metadata["models"][model_id] = model_metadata
                            
                        except Exception as e:
                            # Skip files that can't be processed
                            print(f"Warning: Could not auto-import optimized {file_path}: {str(e)}")
                            continue
        
        # Save metadata if any new models were discovered
        self._save_metadata()
    
    def _save_metadata(self):
        """Save model metadata to JSON file"""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except IOError as e:
            raise Exception(f"Failed to save metadata: {str(e)}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_model_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic model information"""
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": file_path.suffix.lower(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
    
    def import_model(self, source_path: str, model_name: str = None) -> Dict[str, Any]:
        """
        Import a model into local storage
        
        Args:
            source_path: Path to the source model file
            model_name: Optional custom name for the model
            
        Returns:
            Model metadata including local path and info
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source model file not found: {source_path}")
        
        # Generate model name if not provided
        if model_name is None:
            model_name = source_path.stem
        
        # Generate unique ID for the model
        file_hash = self._get_file_hash(source_path)
        model_id = f"{model_name}_{file_hash[:8]}"
        
        # Check if model already exists
        if model_id in self.metadata["models"]:
            existing_model = self.metadata["models"][model_id]
            return existing_model
        
        # Copy model to local storage
        local_filename = f"{model_id}{source_path.suffix}"
        local_path = self.original_models_path / local_filename
        
        try:
            shutil.copy2(source_path, local_path)
        except IOError as e:
            raise Exception(f"Failed to copy model file: {str(e)}")
        
        # Create model metadata
        model_info = self._get_model_info(local_path)
        model_metadata = {
            "id": model_id,
            "name": model_name,
            "original_path": str(source_path),
            "local_path": str(local_path),
            "hash": file_hash,
            "imported_at": datetime.now().isoformat(),
            "is_original": True,
            "type": model_info["extension"],
            **model_info
        }
        
        # Save metadata
        self.metadata["models"][model_id] = model_metadata
        self._save_metadata()
        
        return model_metadata
    
    def add_optimized_model(self, original_id: str, optimized_path: str, 
                          optimization_config: Dict[str, Any], 
                          optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add an optimized model to storage
        
        Args:
            original_id: ID of the original model
            optimized_path: Path to the optimized model file
            optimization_config: Configuration used for optimization
            optimization_results: Results from optimization
            
        Returns:
            Optimized model metadata
        """
        if original_id not in self.metadata["models"]:
            raise ValueError(f"Original model not found: {original_id}")
        
        original_model = self.metadata["models"][original_id]
        optimized_path = Path(optimized_path)
        
        # Generate optimized model ID
        device = optimization_config.get("device", "unknown")
        precision = optimization_config.get("precision", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        optimized_id = f"{original_id}_opt_{device}_{precision}_{timestamp}"
        
        # Copy to optimized models directory
        local_filename = f"{optimized_id}{optimized_path.suffix}"
        local_path = self.optimized_models_path / local_filename
        
        try:
            shutil.copy2(optimized_path, local_path)
        except IOError as e:
            raise Exception(f"Failed to copy optimized model: {str(e)}")
        
        # Create optimized model metadata
        model_info = self._get_model_info(local_path)
        optimized_metadata = {
            "id": optimized_id,
            "name": f"{original_model['name']}_optimized",
            "original_id": original_id,
            "local_path": str(local_path),
            "hash": self._get_file_hash(local_path),
            "created_at": datetime.now().isoformat(),
            "is_original": False,
            "optimization_config": optimization_config,
            "optimization_results": optimization_results,
            "type": model_info["extension"],
            **model_info
        }
        
        # Save metadata
        self.metadata["models"][optimized_id] = optimized_metadata
        self._save_metadata()
        
        return optimized_metadata
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models in storage"""
        return list(self.metadata["models"].values())
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by ID"""
        return self.metadata["models"].get(model_id)
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metadata by ID (alias for get_model)"""
        return self.get_model(model_id)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from storage"""
        if model_id not in self.metadata["models"]:
            return False
        
        model = self.metadata["models"][model_id]
        model_path = Path(model["local_path"])
        
        # Delete the file
        try:
            if model_path.exists():
                model_path.unlink()
        except OSError:
            pass  # File might already be deleted
        
        # Remove from metadata
        del self.metadata["models"][model_id]
        self._save_metadata()
        
        return True
    
    def get_original_models(self) -> List[Dict[str, Any]]:
        """Get only original (non-optimized) models"""
        return [model for model in self.metadata["models"].values() 
                if model.get("is_original", True)]
    
    def get_optimized_models(self, original_id: str = None) -> List[Dict[str, Any]]:
        """Get optimized models, optionally filtered by original model ID"""
        optimized = [model for model in self.metadata["models"].values() 
                    if not model.get("is_original", True)]
        
        if original_id:
            optimized = [model for model in optimized 
                        if model.get("original_id") == original_id]
        
        return optimized
    
    def refresh_models(self):
        """Manually refresh and scan for new models"""
        self._scan_existing_models()
        return self.list_models()
    
    def cleanup_orphaned_files(self):
        """Remove files that are not referenced in metadata"""
        # Get all referenced files
        referenced_files = set()
        for model in self.metadata["models"].values():
            referenced_files.add(Path(model["local_path"]))
        
        # Check original models directory
        for file_path in self.original_models_path.iterdir():
            if file_path.is_file() and file_path not in referenced_files:
                try:
                    file_path.unlink()
                except OSError:
                    pass
        
        # Check optimized models directory
        for file_path in self.optimized_models_path.iterdir():
            if file_path.is_file() and file_path not in referenced_files:
                try:
                    file_path.unlink()
                except OSError:
                    pass
