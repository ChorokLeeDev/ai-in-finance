"""
Caching Utility for FK-Level Risk Attribution
==============================================

Simple caching for models, data, and experiment results.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional

# Cache directory
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _get_cache_path(key: str, ext: str = "pkl") -> Path:
    """Get cache file path for a given key."""
    # Sanitize key for filename
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return CACHE_DIR / f"{safe_key}.{ext}"


def cache_exists(key: str, ext: str = "pkl") -> bool:
    """Check if cache exists for a given key."""
    return _get_cache_path(key, ext).exists()


def save_cache(key: str, data: Any, ext: str = "pkl") -> Path:
    """
    Save data to cache.

    Args:
        key: Cache key (e.g., "model_sales-group_500")
        data: Data to cache
        ext: File extension ("pkl" for pickle, "json" for JSON)

    Returns:
        Path to cached file
    """
    path = _get_cache_path(key, ext)

    if ext == "json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(path, "wb") as f:
            pickle.dump(data, f)

    print(f"[CACHE] Saved: {path.name}")
    return path


def load_cache(key: str, ext: str = "pkl") -> Optional[Any]:
    """
    Load data from cache.

    Args:
        key: Cache key
        ext: File extension

    Returns:
        Cached data or None if not found
    """
    path = _get_cache_path(key, ext)

    if not path.exists():
        print(f"[CACHE] Not found: {key}")
        return None

    if ext == "json":
        with open(path, "r") as f:
            data = json.load(f)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)

    print(f"[CACHE] Loaded: {path.name}")
    return data


def clear_cache(pattern: str = "*") -> int:
    """
    Clear cache files matching pattern.

    Args:
        pattern: Glob pattern (default "*" clears all)

    Returns:
        Number of files deleted
    """
    count = 0
    for path in CACHE_DIR.glob(pattern):
        if path.is_file():
            path.unlink()
            count += 1
    print(f"[CACHE] Cleared {count} files")
    return count


def list_cache() -> list:
    """List all cached files."""
    files = sorted(CACHE_DIR.glob("*"))
    files = [f for f in files if f.is_file()]
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")
    return files


# Convenience functions for specific cache types

def cache_model(task: str, sample_size: int, models: list) -> Path:
    """Cache trained ensemble models."""
    key = f"model_{task}_{sample_size}"
    return save_cache(key, models)


def load_model(task: str, sample_size: int) -> Optional[list]:
    """Load cached ensemble models."""
    key = f"model_{task}_{sample_size}"
    return load_cache(key)


def cache_data(task: str, sample_size: int, data: dict) -> Path:
    """Cache processed data (X, y, feature_cols, fk_mapping)."""
    key = f"data_{task}_{sample_size}"
    return save_cache(key, data)


def load_data(task: str, sample_size: int) -> Optional[dict]:
    """Load cached data."""
    key = f"data_{task}_{sample_size}"
    return load_cache(key)


def cache_result(experiment: str, task: str, sample_size: int, result: dict) -> Path:
    """Cache experiment result."""
    key = f"result_{experiment}_{task}_{sample_size}"
    return save_cache(key, result, ext="json")


def load_result(experiment: str, task: str, sample_size: int) -> Optional[dict]:
    """Load cached experiment result."""
    key = f"result_{experiment}_{task}_{sample_size}"
    return load_cache(key, ext="json")


if __name__ == "__main__":
    # Test caching
    print("Testing cache utilities...")

    # Test basic save/load
    test_data = {"test": 123, "list": [1, 2, 3]}
    save_cache("test_key", test_data)
    loaded = load_cache("test_key")
    assert loaded == test_data, "Cache test failed!"

    # Test JSON
    save_cache("test_json", test_data, ext="json")
    loaded_json = load_cache("test_json", ext="json")
    assert loaded_json == test_data, "JSON cache test failed!"

    # List cache
    print("\nCached files:")
    list_cache()

    # Clear test files
    clear_cache("test_*")

    print("\nCache utilities OK!")
