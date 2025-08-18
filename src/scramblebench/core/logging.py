"""
Advanced logging system for ScrambleBench.

This module provides a comprehensive logging infrastructure with:
- Structured logging with JSON output
- Performance monitoring and metrics
- Error tracking and alerting
- Context-aware logging
- Multiple output formats and destinations
"""

import logging
import logging.handlers
import json
import time
import sys
import threading
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from enum import Enum
import traceback
import functools

from scramblebench.core.exceptions import ScrambleBenchError


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Logging output formats."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    COLORED = "colored"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    context: Dict[str, Any]
    performance: Dict[str, Any]
    error_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level,
            'logger_name': self.logger_name,
            'message': self.message,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            'context': self.context,
            'performance': self.performance,
            'error_info': self.error_info
        }


class PerformanceTracker:
    """Tracks performance metrics for logging."""
    
    def __init__(self):
        self._start_time = time.time()
        self._last_checkpoint = self._start_time
        self._checkpoints: List[Dict[str, Any]] = []
        self._memory_snapshots: List[int] = []
    
    def checkpoint(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Record a performance checkpoint."""
        current_time = time.time()
        elapsed = current_time - self._last_checkpoint
        total_elapsed = current_time - self._start_time
        
        checkpoint_data = {
            'name': name,
            'elapsed_since_last': elapsed,
            'total_elapsed': total_elapsed,
            'timestamp': current_time,
            'metadata': metadata or {}
        }
        
        # Try to get memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            checkpoint_data['memory_mb'] = memory_mb
            self._memory_snapshots.append(memory_mb)
        except ImportError:
            pass
        
        self._checkpoints.append(checkpoint_data)
        self._last_checkpoint = current_time
        
        return elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = time.time() - self._start_time
        
        summary = {
            'total_time': total_time,
            'checkpoint_count': len(self._checkpoints),
            'checkpoints': self._checkpoints
        }
        
        if self._memory_snapshots:
            summary.update({
                'memory_peak_mb': max(self._memory_snapshots),
                'memory_avg_mb': sum(self._memory_snapshots) / len(self._memory_snapshots),
                'memory_snapshots': self._memory_snapshots
            })
        
        return summary


class ContextualLogger:
    """Logger with automatic context management."""
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._context: Dict[str, Any] = {}
        self._performance_tracker = PerformanceTracker()
        self._local = threading.local()
    
    def add_context(self, **kwargs) -> None:
        """Add context information to all log messages."""
        self._context.update(kwargs)
    
    def remove_context(self, *keys) -> None:
        """Remove context keys."""
        for key in keys:
            self._context.pop(key, None)
    
    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()
    
    @contextmanager
    def context(self, **kwargs):
        """Temporary context manager."""
        old_context = self._context.copy()
        try:
            self.add_context(**kwargs)
            yield
        finally:
            self._context = old_context
    
    def _log_with_context(
        self,
        level: int,
        message: str,
        *args,
        exc_info: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log with automatic context injection."""
        # Get call frame information
        frame = sys._getframe(2)
        
        # Prepare extra context
        log_extra = {
            'context': {**self._context, **(extra or {}), **kwargs},
            'performance': self._performance_tracker.get_summary(),
            'module': frame.f_globals.get('__name__', 'unknown'),
            'function': frame.f_code.co_name,
            'line_number': frame.f_lineno
        }
        
        # Add error information if exception info is provided
        if exc_info:
            if exc_info is True:
                exc_info = sys.exc_info()
            
            if exc_info and exc_info[0]:
                error_info = {
                    'exception_type': exc_info[0].__name__,
                    'exception_message': str(exc_info[1]),
                    'traceback': traceback.format_exception(*exc_info)
                }
                
                # Add ScrambleBench exception details
                if isinstance(exc_info[1], ScrambleBenchError):
                    error_info.update(exc_info[1].to_dict())
                
                log_extra['error_info'] = error_info
        
        self._logger.log(level, message, *args, extra=log_extra, exc_info=exc_info)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self.error(message, *args, **kwargs)
    
    def performance_checkpoint(self, name: str, **metadata):
        """Record a performance checkpoint."""
        elapsed = self._performance_tracker.checkpoint(name, metadata)
        self.debug(f"Performance checkpoint: {name}", elapsed_time=elapsed, **metadata)
        return elapsed


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract our custom fields
        context = getattr(record, 'context', {})
        performance = getattr(record, 'performance', {})
        error_info = getattr(record, 'error_info', None)
        module = getattr(record, 'module', record.module if hasattr(record, 'module') else 'unknown')
        function = getattr(record, 'function', record.funcName)
        line_number = getattr(record, 'line_number', record.lineno)
        
        log_entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=module,
            function=function,
            line_number=line_number,
            thread_id=record.thread,
            process_id=record.process,
            context=context,
            performance=performance,
            error_info=error_info
        )
        
        return json.dumps(log_entry.to_dict(), ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get context if available
        context = getattr(record, 'context', {})
        context_str = f" | {json.dumps(context, default=str)}" if context else ""
        
        # Format message
        formatted = f"{color}[{timestamp}] {record.levelname:8} | {record.name} | {record.getMessage()}{context_str}{reset}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted


class MetricsCollector:
    """Collects logging metrics for monitoring."""
    
    def __init__(self):
        self._metrics: Dict[str, int] = {}
        self._errors: List[Dict[str, Any]] = []
        self._performance_data: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def record_log(self, record: logging.LogRecord):
        """Record a log entry for metrics."""
        with self._lock:
            # Count by level
            level_key = f"log_count_{record.levelname.lower()}"
            self._metrics[level_key] = self._metrics.get(level_key, 0) + 1
            
            # Count by logger
            logger_key = f"logger_count_{record.name}"
            self._metrics[logger_key] = self._metrics.get(logger_key, 0) + 1
            
            # Collect error details
            if record.levelno >= logging.ERROR:
                error_info = getattr(record, 'error_info', None)
                self._errors.append({
                    'timestamp': record.created,
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'error_info': error_info
                })
            
            # Collect performance data
            performance = getattr(record, 'performance', {})
            if performance:
                self._performance_data.append({
                    'timestamp': record.created,
                    'logger': record.name,
                    'performance': performance
                })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        with self._lock:
            return {
                'counts': self._metrics.copy(),
                'recent_errors': self._errors[-100:],  # Last 100 errors
                'performance_summary': self._get_performance_summary()
            }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self._performance_data:
            return {}
        
        # Analyze performance data
        total_times = [p['performance'].get('total_time', 0) for p in self._performance_data]
        
        return {
            'total_operations': len(self._performance_data),
            'avg_duration': sum(total_times) / len(total_times) if total_times else 0,
            'max_duration': max(total_times) if total_times else 0,
            'min_duration': min(total_times) if total_times else 0
        }


class LoggingSystem:
    """Main logging system coordinator."""
    
    def __init__(self):
        self._loggers: Dict[str, ContextualLogger] = {}
        self._metrics_collector = MetricsCollector()
        self._configured = False
    
    def configure(
        self,
        level: LogLevel = LogLevel.INFO,
        format_type: LogFormat = LogFormat.COLORED,
        console: bool = True,
        file_path: Optional[Union[str, Path]] = None,
        json_file_path: Optional[Union[str, Path]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_metrics: bool = True
    ):
        """Configure the logging system."""
        
        # Configure root logger
        root_logger = logging.getLogger('scramblebench')
        root_logger.setLevel(getattr(logging, level.value))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if format_type == LogFormat.JSON:
                console_handler.setFormatter(JSONFormatter())
            elif format_type == LogFormat.COLORED:
                console_handler.setFormatter(ColoredFormatter())
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(formatter)
            
            root_logger.addHandler(console_handler)
        
        # File handler
        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # JSON file handler
        if json_file_path:
            json_file_path = Path(json_file_path)
            json_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            json_handler = logging.handlers.RotatingFileHandler(
                json_file_path,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            json_handler.setFormatter(JSONFormatter())
            root_logger.addHandler(json_handler)
        
        # Metrics collection
        if enable_metrics:
            class MetricsHandler(logging.Handler):
                def __init__(self, collector):
                    super().__init__()
                    self.collector = collector
                
                def emit(self, record):
                    self.collector.record_log(record)
            
            metrics_handler = MetricsHandler(self._metrics_collector)
            root_logger.addHandler(metrics_handler)
        
        self._configured = True
    
    def get_logger(self, name: str) -> ContextualLogger:
        """Get or create a contextual logger."""
        if name not in self._loggers:
            logger = logging.getLogger(name)
            self._loggers[name] = ContextualLogger(logger)
        
        return self._loggers[name]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logging metrics."""
        return self._metrics_collector.get_metrics()
    
    def is_configured(self) -> bool:
        """Check if logging is configured."""
        return self._configured


# Global logging system instance
_logging_system = LoggingSystem()


def configure_logging(**kwargs):
    """Configure the global logging system."""
    _logging_system.configure(**kwargs)


def get_logger(name: str) -> ContextualLogger:
    """Get a contextual logger."""
    return _logging_system.get_logger(name)


def get_logging_metrics() -> Dict[str, Any]:
    """Get logging metrics."""
    return _logging_system.get_metrics()


def log_performance(func=None, *, logger_name: Optional[str] = None):
    """Decorator to log function performance."""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or f.__module__)
            
            with logger.context(function=f.__name__):
                start_time = time.time()
                logger.performance_checkpoint("function_start")
                
                try:
                    result = f(*args, **kwargs)
                    logger.performance_checkpoint("function_success")
                    return result
                except Exception as e:
                    logger.performance_checkpoint("function_error")
                    logger.exception(f"Function {f.__name__} failed", 
                                   function=f.__name__, 
                                   args_count=len(args),
                                   kwargs_keys=list(kwargs.keys()))
                    raise
                finally:
                    elapsed = time.time() - start_time
                    logger.info(f"Function {f.__name__} completed", 
                               duration=elapsed,
                               function=f.__name__)
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def log_operation(operation_name: str, logger_name: Optional[str] = None, **context):
    """Context manager to log operation performance."""
    logger = get_logger(logger_name or __name__)
    
    with logger.context(operation=operation_name, **context):
        start_time = time.time()
        logger.info(f"Starting operation: {operation_name}")
        logger.performance_checkpoint("operation_start")
        
        try:
            yield logger
            logger.performance_checkpoint("operation_success")
            logger.info(f"Operation completed: {operation_name}")
        except Exception as e:
            logger.performance_checkpoint("operation_error")
            logger.exception(f"Operation failed: {operation_name}")
            raise
        finally:
            elapsed = time.time() - start_time
            logger.info(f"Operation duration: {operation_name}", duration=elapsed)