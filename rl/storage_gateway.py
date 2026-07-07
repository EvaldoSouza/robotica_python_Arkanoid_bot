import os
import pickle
from typing import Optional, Dict, Any

class StorageGateway:
    """Dependency injection interface for abstracting file system persistence."""
    def read_pickle(self, filepath: str) -> Optional[Dict[str, Any]]: 
        raise NotImplementedError
        
    def write_pickle(self, filepath: str, payload: Dict[str, Any]) -> None: 
        raise NotImplementedError
        
    def write_json(self, filepath: str, payload: Dict[str, Any]) -> None:
        raise NotImplementedError
        
    def exists(self, filepath: str) -> bool: 
        raise NotImplementedError


class LocalDiskStorage(StorageGateway):
    """Concrete implementation for persisting brain files to local disk."""
    def read_pickle(self, filepath: str) -> Optional[Dict[str, Any]]:
        if not self.exists(filepath):
            return None
            
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def write_pickle(self, filepath: str, payload: Dict[str, Any]) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    def write_json(self, filepath: str, payload: Dict[str, Any]) -> None:
        import json
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=4)

    def exists(self, filepath: str) -> bool:
        return os.path.exists(filepath)