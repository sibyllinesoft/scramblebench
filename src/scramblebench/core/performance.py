"""
Performance monitoring and optimization utilities for ScrambleBench.

This module provides comprehensive performance monitoring, caching,
and optimization capabilities to enhance benchmark execution speed
and resource utilization.
"""

import time
import functools
import threading
import asyncio
import hashlib
import pickle
import json
from typing import Any, Dict, List, Optional, Callable, Union, TypeVar, Tuple
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import weakref
import sys

from scramblebench.core.exceptions import ResourceError, PerformanceError
from scramblebench.core.logging import get_logger


T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: Optional[int] = None
    memory_after: Optional[int] = None
    memory_peak: Optional[int] = None
    cpu_percent: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_used(self) -> Optional[int]:
        """Calculate memory used during operation."""
        if self.memory_before and self.memory_after:
            return self.memory_after - self.memory_before
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_peak': self.memory_peak,
            'memory_used': self.memory_used,
            'cpu_percent': self.cpu_percent,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'error_count': self.error_count,
            'metadata': self.metadata
        }


class PerformanceMonitor:
    """Thread-safe performance monitoring system."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.active_operations: Dict[str, float] = {}
        self.aggregated_stats: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.logger = get_logger(__name__)
        
        # Try to import psutil for memory monitoring
        try:
            import psutil
            self.process = psutil.Process()
            self.memory_monitoring = True
        except ImportError:
            self.memory_monitoring = False
            self.logger.warning("psutil not available, memory monitoring disabled")
    
    def start_operation(self, operation_name: str) -> str:
        """Start monitoring an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        with self._lock:
            self.active_operations[operation_id] = time.time()
        
        return operation_id
    
    def end_operation(
        self,
        operation_id: str,
        operation_name: str,
        error_count: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
        **metadata
    ) -> PerformanceMetrics:
        """End monitoring an operation and record metrics."""
        end_time = time.time()
        
        with self._lock:
            start_time = self.active_operations.pop(operation_id, end_time)
            
        duration = end_time - start_time
        
        # Get memory info if available
        memory_before = memory_after = memory_peak = cpu_percent = None
        if self.memory_monitoring:
            try:
                memory_info = self.process.memory_info()
                memory_after = memory_info.rss
                cpu_percent = self.process.cpu_percent()
            except Exception:
                pass
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            memory_after=memory_after,
            cpu_percent=cpu_percent,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            error_count=error_count,
            metadata=metadata
        )
        
        with self._lock:
            self.metrics_history.append(metrics)
            self.aggregated_stats[operation_name].append(duration)
        
        return metrics
    
    def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if operation_name:
                durations = self.aggregated_stats.get(operation_name, [])
                if not durations:
                    return {}
                
                return {
                    'operation_name': operation_name,
                    'total_calls': len(durations),
                    'total_time': sum(durations),
                    'avg_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations),
                    'recent_calls': len([m for m in self.metrics_history if m.operation_name == operation_name])
                }
            else:
                # Overall stats
                all_metrics = list(self.metrics_history)
                total_operations = len(all_metrics)
                
                if total_operations == 0:
                    return {'total_operations': 0}
                
                total_time = sum(m.duration for m in all_metrics)
                avg_time = total_time / total_operations
                
                return {
                    'total_operations': total_operations,
                    'total_time': total_time,
                    'avg_time': avg_time,
                    'operations_by_name': {
                        name: len(durations) for name, durations in self.aggregated_stats.items()
                    },
                    'active_operations': len(self.active_operations)
                }
    
    def clear_history(self) -> None:
        """Clear performance history."""
        with self._lock:
            self.metrics_history.clear()
            self.aggregated_stats.clear()


# Global performance monitor
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _global_monitor


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, maxsize: int = 1000, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order: deque = deque()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _evict_expired(self) -> None:
        """Remove expired entries if TTL is set."""
        if not self.ttl:
            return
        
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get value from cache. Returns (value, hit)."""
        with self._lock:
            self._evict_expired()
            
            if key in self.cache:
                # Move to end (most recently used)
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
                self.access_order.append(key)
                
                self.hits += 1
                return self.cache[key][0], True
            else:
                self.misses += 1
                return None, False
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self._lock:
            current_time = time.time()
            
            # If key already exists, update it
            if key in self.cache:
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
            
            # If cache is full, remove least recently used
            elif len(self.cache) >= self.maxsize:
                if self.access_order:
                    lru_key = self.access_order.popleft()
                    self.cache.pop(lru_key, None)
            
            self.cache[key] = (value, current_time)
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'ttl': self.ttl
            }


class CacheManager:
    """Manages multiple caches with different policies."""
    
    def __init__(self):
        self.caches: Dict[str, LRUCache] = {}
        self._lock = threading.Lock()
    
    def get_cache(
        self,
        name: str,
        maxsize: int = 1000,
        ttl: Optional[float] = None
    ) -> LRUCache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self.caches:
                self.caches[name] = LRUCache(maxsize=maxsize, ttl=ttl)
            return self.caches[name]
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self.caches.values():
                cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all caches."""
        with self._lock:
            return {name: cache.stats() for name, cache in self.caches.items()}


# Global cache manager
_global_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    return _global_cache_manager


def monitored(operation_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            monitor = get_performance_monitor()
            operation_id = monitor.start_operation(name)
            
            try:
                result = func(*args, **kwargs)
                monitor.end_operation(operation_id, name)
                return result
            except Exception as e:
                monitor.end_operation(operation_id, name, error_count=1)
                raise
        
        return wrapper
    return decorator


def cached(
    cache_name: str = "default",
    maxsize: int = 1000,
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Decorator to cache function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = get_cache_manager().get_cache(cache_name, maxsize, ttl)
        
        def default_key_func(*args, **kwargs) -> str:
            """Default key generation function."""
            # Create a stable key from arguments
            key_data = {
                'args': args,
                'kwargs': sorted(kwargs.items())
            }
            key_str = json.dumps(key_data, default=str, sort_keys=True)
            return hashlib.md5(key_str.encode()).hexdigest()
        
        actual_key_func = key_func or default_key_func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__name__}:{actual_key_func(*args, **kwargs)}"
            
            # Try to get from cache
            result, hit = cache.get(cache_key)
            
            if hit:
                return result
            
            # Not in cache, compute result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods to the function
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        
        return wrapper
    return decorator


class AsyncBatchProcessor:
    """Process items in batches asynchronously for better performance."""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_workers: int = 4,
        timeout: Optional[float] = None
    ):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.logger = get_logger(__name__)
    
    async def process_async(
        self,
        items: List[Any],
        processor: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items asynchronously in batches."""
        results = []
        total_items = len(items)
        processed = 0
        
        # Split items into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches concurrently
        async def process_batch(batch: List[Any]) -> List[Any]:
            batch_results = []
            for item in batch:
                try:
                    result = processor(item)
                    if asyncio.iscoroutine(result):
                        result = await result
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing item: {e}")
                    batch_results.append(None)
            return batch_results
        
        # Create tasks for all batches
        tasks = [process_batch(batch) for batch in batches]
        
        # Wait for completion with progress tracking
        for completed_task in asyncio.as_completed(tasks, timeout=self.timeout):
            try:
                batch_results = await completed_task
                results.extend(batch_results)
                processed += len(batch_results)
                
                if progress_callback:
                    progress_callback(processed, total_items)
                    
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
        
        return results
    
    def process_sync(
        self,
        items: List[Any],
        processor: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items synchronously using thread pool."""
        results = []
        total_items = len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all items
            future_to_item = {
                executor.submit(processor, item): item 
                for item in items
            }
            
            # Collect results as they complete
            for completed_future in as_completed(future_to_item, timeout=self.timeout):
                try:
                    result = completed_future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(len(results), total_items)
                        
                except Exception as e:
                    self.logger.error(f"Item processing error: {e}")
                    results.append(None)
        
        return results


class MemoryManager:
    """Manages memory usage and provides memory optimization."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._weak_refs: List[weakref.ref] = []
        
        try:
            import psutil
            self.process = psutil.Process()
            self.memory_monitoring = True
        except ImportError:
            self.memory_monitoring = False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        if not self.memory_monitoring:
            return {'available': False}
        
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            return {
                'available': True,
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': memory_percent,
                'pid': self.process.pid
            }
        except Exception as e:
            self.logger.error(f"Error getting memory info: {e}")
            return {'available': False, 'error': str(e)}
    
    def register_for_cleanup(self, obj: Any) -> None:
        """Register object for potential cleanup."""
        def cleanup_callback(ref):
            self._weak_refs.remove(ref)
        
        self._weak_refs.append(weakref.ref(obj, cleanup_callback))
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return collection stats."""
        import gc
        
        before_count = len(gc.get_objects())
        before_memory = self.get_memory_usage()
        
        collected = gc.collect()
        
        after_count = len(gc.get_objects())
        after_memory = self.get_memory_usage()
        
        return {
            'objects_before': before_count,
            'objects_after': after_count,
            'objects_collected': before_count - after_count,
            'gc_collected': collected,
            'memory_before': before_memory,
            'memory_after': after_memory
        }
    
    def check_memory_pressure(self, threshold_percent: float = 80.0) -> bool:
        """Check if memory usage is above threshold."""
        usage = self.get_memory_usage()
        if usage.get('available'):
            return usage.get('percent', 0) > threshold_percent
        return False


# Global memory manager
_global_memory_manager = MemoryManager()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager."""
    return _global_memory_manager


class PerformanceOptimizer:
    """Provides performance optimization recommendations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.monitor = get_performance_monitor()
        self.cache_manager = get_cache_manager()
        self.memory_manager = get_memory_manager()
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and provide recommendations."""
        stats = self.monitor.get_stats()
        cache_stats = self.cache_manager.get_all_stats()
        memory_stats = self.memory_manager.get_memory_usage()
        
        recommendations = []
        
        # Check cache hit rates
        for cache_name, cache_stat in cache_stats.items():
            hit_rate = cache_stat.get('hit_rate', 0)
            if hit_rate < 0.5:
                recommendations.append({
                    'type': 'cache',
                    'message': f"Low cache hit rate ({hit_rate:.1%}) for {cache_name}",
                    'suggestion': "Consider adjusting cache size or TTL"
                })
        
        # Check memory usage
        if memory_stats.get('available') and memory_stats.get('percent', 0) > 80:
            recommendations.append({
                'type': 'memory',
                'message': f"High memory usage ({memory_stats['percent']:.1f}%)",
                'suggestion': "Consider reducing batch sizes or clearing caches"
            })
        
        # Check slow operations
        operation_stats = self.monitor.aggregated_stats
        for op_name, durations in operation_stats.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                if avg_duration > 10.0:  # Operations taking > 10 seconds
                    recommendations.append({
                        'type': 'performance',
                        'message': f"Slow operation: {op_name} ({avg_duration:.1f}s avg)",
                        'suggestion': "Consider optimization or caching"
                    })
        
        return {
            'timestamp': time.time(),
            'overall_stats': stats,
            'cache_stats': cache_stats,
            'memory_stats': memory_stats,
            'recommendations': recommendations
        }
    
    def optimize_automatically(self) -> Dict[str, Any]:
        """Apply automatic optimizations."""
        optimizations = []
        
        # Force garbage collection if memory is high
        if self.memory_manager.check_memory_pressure():
            gc_stats = self.memory_manager.force_garbage_collection()
            optimizations.append({
                'type': 'memory',
                'action': 'garbage_collection',
                'result': gc_stats
            })
        
        # Clear low-performing caches
        cache_stats = self.cache_manager.get_all_stats()
        for cache_name, stats in cache_stats.items():
            if stats.get('hit_rate', 1.0) < 0.1 and stats.get('size', 0) > 100:
                cache = self.cache_manager.get_cache(cache_name)
                cache.clear()
                optimizations.append({
                    'type': 'cache',
                    'action': f'cleared_low_performing_cache_{cache_name}',
                    'result': f"Cleared cache with {stats['hit_rate']:.1%} hit rate"
                })
        
        return {
            'timestamp': time.time(),
            'optimizations_applied': optimizations
        }