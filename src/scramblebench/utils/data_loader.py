"""
Data loading utilities for ScrambleBench.

This module provides flexible data loading capabilities for various
benchmark datasets, supporting multiple formats and sources with
caching and validation.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import json
import csv
import logging
import hashlib
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from scramblebench.utils.config import Config


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    size: int
    format: str
    source: str
    columns: List[str]
    sample: Dict[str, Any]
    metadata: Dict[str, Any]


class DataLoader:
    """
    Flexible data loader for benchmark datasets.
    
    Supports loading from various sources and formats with caching,
    validation, and preprocessing capabilities.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config or Config()
        self.logger = logger or logging.getLogger("scramblebench.data_loader")
        
        # Cache management
        self.cache_dir = Path(self.config.get('data.cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = self.config.get('data.max_cache_size', 1000)
        
        # Data directory
        self.data_dir = Path(self.config.get('data.data_dir', 'data'))
        self.benchmarks_dir = Path(self.config.get('data.benchmarks_dir', 'data/benchmarks'))
        
        # Registered loaders
        self._loaders: Dict[str, Callable] = {}
        self._register_default_loaders()
    
    def _register_default_loaders(self) -> None:
        """Register default data loaders."""
        self.register_loader('json', self._load_json)
        self.register_loader('jsonl', self._load_jsonl)
        self.register_loader('csv', self._load_csv)
        
        if HAS_PANDAS:
            self.register_loader('parquet', self._load_parquet)
            self.register_loader('excel', self._load_excel)
        
        if HAS_DATASETS:
            self.register_loader('huggingface', self._load_huggingface)
    
    def register_loader(self, format_name: str, loader_func: Callable) -> None:
        """
        Register a custom data loader function.
        
        Args:
            format_name: Name of the format
            loader_func: Function that takes (path, **kwargs) and returns List[Dict]
        """
        self._loaders[format_name] = loader_func
        self.logger.debug(f"Registered loader for format: {format_name}")
    
    def load_dataset(
        self,
        dataset_name: str,
        format: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load a dataset from various sources.
        
        Args:
            dataset_name: Name or path of the dataset
            format: Format of the dataset (auto-detected if None)
            use_cache: Whether to use cached data
            **kwargs: Additional arguments for the loader
            
        Returns:
            List of data items as dictionaries
        """
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(dataset_name, **kwargs)
            if cached_data is not None:
                self.logger.info(f"Loaded {len(cached_data)} items from cache")
                return cached_data
        
        # Determine data source and format
        data_path, detected_format = self._resolve_dataset_path(dataset_name, format)
        
        # Load data using appropriate loader
        if detected_format not in self._loaders:
            raise ValueError(f"Unsupported format: {detected_format}")
        
        loader_func = self._loaders[detected_format]
        data = loader_func(data_path, **kwargs)
        
        # Validate data
        self._validate_data(data, dataset_name)
        
        # Cache data
        if use_cache:
            self._save_to_cache(dataset_name, data, **kwargs)
        
        self.logger.info(f"Loaded {len(data)} items from {dataset_name}")
        return data
    
    def _resolve_dataset_path(
        self,
        dataset_name: str,
        format: Optional[str]
    ) -> tuple[Path, str]:
        """
        Resolve dataset name to actual path and format.
        
        Args:
            dataset_name: Name or path of the dataset
            format: Explicit format (if provided)
            
        Returns:
            Tuple of (resolved_path, format)
        """
        # Check if it's a direct path
        path = Path(dataset_name)
        if path.exists():
            detected_format = format or self._detect_format(path)
            return path, detected_format
        
        # Check in benchmarks directory
        benchmarks_path = self.benchmarks_dir / dataset_name
        if benchmarks_path.exists():
            detected_format = format or self._detect_format(benchmarks_path)
            return benchmarks_path, detected_format
        
        # Check with various extensions
        if not path.suffix:
            for ext in ['.json', '.jsonl', '.csv', '.parquet']:
                test_path = self.benchmarks_dir / f"{dataset_name}{ext}"
                if test_path.exists():
                    detected_format = format or self._detect_format(test_path)
                    return test_path, detected_format
        
        # Check for HuggingFace dataset
        if HAS_DATASETS and (format == 'huggingface' or '/' in dataset_name):
            return Path(dataset_name), 'huggingface'
        
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    
    def _detect_format(self, path: Path) -> str:
        """Detect file format from extension."""
        suffix = path.suffix.lower()
        
        format_map = {
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.csv': 'csv',
            '.parquet': 'parquet',
            '.xlsx': 'excel',
            '.xls': 'excel'
        }
        
        return format_map.get(suffix, 'json')
    
    def _load_json(self, path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Check common keys that might contain the data
            for key in ['data', 'examples', 'instances', 'items']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If no list found, wrap the dict
            return [data]
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")
    
    def _load_jsonl(self, path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
        
        return data
    
    def _load_csv(self, path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        data = []
        
        with open(path, 'r', encoding='utf-8') as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Convert string representations of JSON back to objects
                processed_row = {}
                for key, value in row.items():
                    if value and value.startswith(('{', '[')):
                        try:
                            processed_row[key] = json.loads(value)
                        except json.JSONDecodeError:
                            processed_row[key] = value
                    else:
                        processed_row[key] = value
                data.append(processed_row)
        
        return data
    
    def _load_parquet(self, path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Load data from Parquet file (requires pandas)."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for Parquet support")
        
        df = pd.read_parquet(path)
        return df.to_dict('records')
    
    def _load_excel(self, path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Load data from Excel file (requires pandas)."""
        if not HAS_PANDAS:
            raise ImportError("pandas required for Excel support")
        
        sheet_name = kwargs.get('sheet_name', 0)
        df = pd.read_excel(path, sheet_name=sheet_name)
        return df.to_dict('records')
    
    def _load_huggingface(self, path: Path, **kwargs) -> List[Dict[str, Any]]:
        """Load data from HuggingFace datasets."""
        if not HAS_DATASETS:
            raise ImportError("datasets library required for HuggingFace support")
        
        dataset_name = str(path)
        split = kwargs.get('split', 'train')
        
        try:
            dataset = datasets.load_dataset(dataset_name, split=split)
            return list(dataset)
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace dataset {dataset_name}: {e}")
            raise
    
    def _validate_data(self, data: List[Dict[str, Any]], dataset_name: str) -> None:
        """Validate loaded data."""
        if not data:
            raise ValueError(f"No data loaded from {dataset_name}")
        
        if not isinstance(data, list):
            raise ValueError(f"Data must be a list, got {type(data)}")
        
        # Check that all items are dictionaries
        for i, item in enumerate(data[:10]):  # Check first 10 items
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dictionary: {type(item)}")
        
        self.logger.debug(f"Data validation passed for {dataset_name}")
    
    def _get_cache_key(self, dataset_name: str, **kwargs) -> str:
        """Generate cache key for dataset."""
        key_data = {
            'dataset_name': dataset_name,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_from_cache(
        self,
        dataset_name: str,
        **kwargs
    ) -> Optional[List[Dict[str, Any]]]:
        """Load data from cache if available."""
        cache_key = self._get_cache_key(dataset_name, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.logger.debug(f"Cache hit for {dataset_name}")
            return cached_data
        
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(
        self,
        dataset_name: str,
        data: List[Dict[str, Any]],
        **kwargs
    ) -> None:
        """Save data to cache."""
        cache_key = self._get_cache_key(dataset_name, **kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.debug(f"Data cached for {dataset_name}")
            
            # Cleanup old cache files if necessary
            self._cleanup_cache()
        
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")
    
    def _cleanup_cache(self) -> None:
        """Remove old cache files if cache size exceeds limit."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        if len(cache_files) > self.max_cache_size:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            files_to_remove = len(cache_files) - self.max_cache_size
            for cache_file in cache_files[:files_to_remove]:
                try:
                    cache_file.unlink()
                    self.logger.debug(f"Removed old cache file: {cache_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
    
    def get_dataset_info(
        self,
        dataset_name: str,
        format: Optional[str] = None
    ) -> DatasetInfo:
        """
        Get information about a dataset without loading all data.
        
        Args:
            dataset_name: Name of the dataset
            format: Format of the dataset
            
        Returns:
            DatasetInfo object with metadata
        """
        # Load a small sample to get info
        sample_data = self.load_dataset(dataset_name, format=format)
        
        if not sample_data:
            raise ValueError(f"No data found in {dataset_name}")
        
        # Get columns from first item
        columns = list(sample_data[0].keys()) if sample_data else []
        
        # Create sample (first item with potentially sensitive data masked)
        sample = self._create_safe_sample(sample_data[0]) if sample_data else {}
        
        return DatasetInfo(
            name=dataset_name,
            size=len(sample_data),
            format=format or 'auto',
            source=str(self._resolve_dataset_path(dataset_name, format)[0]),
            columns=columns,
            sample=sample,
            metadata={
                'first_10_columns': columns[:10],
                'estimated_size_mb': len(json.dumps(sample_data)) / (1024 * 1024)
            }
        )
    
    def _create_safe_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe sample with potentially sensitive data masked."""
        safe_sample = {}
        
        for key, value in item.items():
            if isinstance(value, str):
                # Truncate long strings
                if len(value) > 100:
                    safe_sample[key] = value[:100] + "..."
                else:
                    safe_sample[key] = value
            elif isinstance(value, (list, dict)):
                # Show structure but limit content
                safe_sample[key] = f"<{type(value).__name__} with {len(value)} items>"
            else:
                safe_sample[key] = value
        
        return safe_sample
    
    def list_datasets(self, directory: Optional[Path] = None) -> List[str]:
        """
        List available datasets in a directory.
        
        Args:
            directory: Directory to search (uses benchmarks_dir if None)
            
        Returns:
            List of dataset names
        """
        search_dir = directory or self.benchmarks_dir
        
        if not search_dir.exists():
            return []
        
        datasets = []
        
        for file_path in search_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in [
                '.json', '.jsonl', '.csv', '.parquet', '.xlsx', '.xls'
            ]:
                datasets.append(file_path.stem)
        
        return sorted(datasets)
    
    def convert_dataset(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_format: str,
        **kwargs
    ) -> None:
        """
        Convert a dataset from one format to another.
        
        Args:
            input_path: Path to input dataset
            output_path: Path to save converted dataset
            output_format: Target format
            **kwargs: Additional conversion options
        """
        # Load data
        data = self.load_dataset(str(input_path), use_cache=False)
        
        # Save in new format
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif output_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        elif output_format == 'csv':
            if not data:
                return
            
            fieldnames = list(data[0].keys())
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in data:
                    # Convert complex objects to JSON strings
                    csv_item = {}
                    for key, value in item.items():
                        if isinstance(value, (dict, list)):
                            csv_item[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            csv_item[key] = value
                    writer.writerow(csv_item)
        
        elif output_format == 'parquet' and HAS_PANDAS:
            df = pd.DataFrame(data)
            df.to_parquet(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        self.logger.info(f"Converted {len(data)} items to {output_path}")
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        self.logger.info(f"Cleared {len(cache_files)} cache files")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir),
            'max_cache_size': self.max_cache_size
        }