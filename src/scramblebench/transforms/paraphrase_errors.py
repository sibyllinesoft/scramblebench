"""
Academic-grade error handling and logging for the paraphrase pipeline.

This module provides comprehensive error handling, logging, and monitoring
capabilities specifically designed for academic research requirements,
ensuring full traceability and reproducibility of paraphrase generation.
"""

import logging
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from scramblebench.core.exceptions import ValidationError, ModelError


class ParaphraseErrorSeverity(Enum):
    """Severity levels for paraphrase pipeline errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ACADEMIC_VIOLATION = "academic_violation"


class ParaphraseErrorCode(Enum):
    """Standardized error codes for paraphrase pipeline."""
    
    # Provider isolation errors (most critical for academic integrity)
    PROVIDER_ISOLATION_VIOLATION = "E001"
    PROVIDER_CONFIG_INVALID = "E002"
    PROVIDER_CONTAMINATION_RISK = "E003"
    
    # Quality control errors
    SEMANTIC_SIMILARITY_FAILED = "E101"
    SURFACE_DIVERGENCE_FAILED = "E102"
    QUALITY_THRESHOLD_NOT_MET = "E103"
    ACCEPTANCE_RATE_BELOW_TARGET = "E104"
    
    # Generation errors
    MODEL_ADAPTER_NOT_SET = "E201"
    GENERATION_FAILED = "E202"
    CANDIDATE_GENERATION_FAILED = "E203"
    PROMPT_INVALID = "E204"
    
    # Caching errors
    CACHE_READ_FAILED = "E301"
    CACHE_WRITE_FAILED = "E302"
    DATABASE_CONNECTION_FAILED = "E303"
    CACHE_CORRUPTION_DETECTED = "E304"
    
    # Configuration errors
    CONFIG_MISSING_REQUIRED_FIELD = "E401"
    CONFIG_INVALID_VALUE = "E402"
    CONFIG_VALIDATION_FAILED = "E403"
    
    # Academic standards violations
    INSUFFICIENT_SAMPLE_SIZE = "A001"
    REPRODUCIBILITY_COMPROMISED = "A002"
    METHODOLOGY_VIOLATION = "A003"


class ParaphraseError(Exception):
    """
    Custom exception for paraphrase pipeline with academic-grade error tracking.
    
    This exception provides detailed context, error codes, and recovery suggestions
    essential for academic research reproducibility and debugging.
    """
    
    def __init__(self,
                 message: str,
                 error_code: ParaphraseErrorCode,
                 severity: ParaphraseErrorSeverity = ParaphraseErrorSeverity.ERROR,
                 context: Optional[Dict[str, Any]] = None,
                 recovery_suggestion: Optional[str] = None,
                 academic_impact: Optional[str] = None):
        """
        Initialize paraphrase error.
        
        Args:
            message: Human-readable error message
            error_code: Standardized error code
            severity: Error severity level
            context: Additional context for debugging
            recovery_suggestion: Suggested recovery action
            academic_impact: Description of impact on academic integrity
        """
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        self.academic_impact = academic_impact
        self.timestamp = datetime.now().isoformat()
        
        # Capture stack trace for debugging
        self.stack_trace = traceback.format_stack()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and serialization."""
        return {
            "error_code": self.error_code.value,
            "message": str(self),
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "context": self.context,
            "recovery_suggestion": self.recovery_suggestion,
            "academic_impact": self.academic_impact,
            "stack_trace": self.stack_trace[-3:] if self.stack_trace else None  # Last 3 frames
        }


class AcademicLogger:
    """
    Academic-grade logger for paraphrase pipeline with full traceability.
    
    Provides structured logging with academic research requirements:
    - Full reproducibility tracking
    - Academic integrity monitoring
    - Comprehensive error documentation
    - Research audit trail maintenance
    """
    
    def __init__(self, 
                 name: str = "paraphrase_pipeline",
                 log_dir: Optional[Path] = None,
                 academic_mode: bool = True):
        """
        Initialize academic logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            academic_mode: Enable academic-specific logging features
        """
        self.name = name
        self.academic_mode = academic_mode
        
        # Set up log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path("logs/paraphrase_pipeline")
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if academic_mode else logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Set up handlers
        self._setup_handlers()
        
        # Academic tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.error_count = 0
        self.warning_count = 0
        self.academic_violations = []
        
        self.info("Academic logger initialized", extra={
            "session_id": self.session_id,
            "academic_mode": academic_mode,
            "log_dir": str(self.log_dir)
        })
    
    def _setup_handlers(self):
        """Set up logging handlers for academic requirements."""
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for detailed academic logging
        log_file = self.log_dir / f"paraphrase_pipeline_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter for academic traceability
        academic_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(academic_formatter)
        self.logger.addHandler(file_handler)
        
        # Separate handler for academic violations (most critical)
        if self.academic_mode:
            violation_file = self.log_dir / f"academic_violations_{self.session_id}.log"
            violation_handler = logging.FileHandler(violation_file)
            violation_handler.setLevel(logging.ERROR)
            violation_handler.addFilter(lambda record: 
                hasattr(record, 'academic_violation') and record.academic_violation)
            violation_handler.setFormatter(academic_formatter)
            self.logger.addHandler(violation_handler)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with academic context."""
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with academic context."""
        self.warning_count += 1
        extra = extra or {}
        extra['warning_count'] = self.warning_count
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, 
              error: Optional[Exception] = None,
              extra: Optional[Dict[str, Any]] = None):
        """Log error message with full academic traceability."""
        self.error_count += 1
        
        extra = extra or {}
        extra['error_count'] = self.error_count
        
        if error:
            extra['error_type'] = type(error).__name__
            if isinstance(error, ParaphraseError):
                extra.update(error.to_dict())
        
        self.logger.error(message, extra=extra, exc_info=error is not None)
    
    def academic_violation(self, 
                          message: str,
                          violation_type: str,
                          severity: ParaphraseErrorSeverity = ParaphraseErrorSeverity.ACADEMIC_VIOLATION,
                          context: Optional[Dict[str, Any]] = None):
        """Log academic integrity violation (most critical)."""
        violation_record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "violation_type": violation_type,
            "message": message,
            "severity": severity.value,
            "context": context or {}
        }
        
        self.academic_violations.append(violation_record)
        
        # Log with special academic violation marker
        self.logger.error(
            f"ACADEMIC VIOLATION: {message}",
            extra={
                "academic_violation": True,
                "violation_record": violation_record
            }
        )
    
    def log_paraphrase_event(self, 
                           event_type: str,
                           item_id: str,
                           success: bool,
                           details: Dict[str, Any]):
        """Log paraphrase-specific events for research audit trail."""
        event_record = {
            "event_type": event_type,
            "item_id": item_id,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            **details
        }
        
        if success:
            self.info(f"Paraphrase {event_type} successful: {item_id}", 
                     extra=event_record)
        else:
            self.error(f"Paraphrase {event_type} failed: {item_id}",
                      extra=event_record)
    
    def log_quality_metrics(self, metrics: Dict[str, float]):
        """Log quality metrics for academic analysis."""
        self.info("Quality metrics recorded", extra={
            "metrics_type": "quality_assessment",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary for academic reporting."""
        return {
            "session_id": self.session_id,
            "session_start": datetime.now().isoformat(),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "academic_violations": len(self.academic_violations),
            "violation_details": self.academic_violations,
            "academic_mode": self.academic_mode,
            "log_directory": str(self.log_dir)
        }
    
    def close_session(self):
        """Close logging session and generate final academic report."""
        summary = self.get_session_summary()
        
        self.info("Academic logging session complete", extra=summary)
        
        # Save session summary for academic records
        summary_file = self.log_dir / f"session_summary_{self.session_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate academic violation report if violations occurred
        if self.academic_violations:
            violation_report = self.log_dir / f"violation_report_{self.session_id}.json"
            with open(violation_report, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "total_violations": len(self.academic_violations),
                    "violations": self.academic_violations,
                    "impact_assessment": self._assess_academic_impact()
                }, f, indent=2)
    
    def _assess_academic_impact(self) -> Dict[str, Any]:
        """Assess the academic impact of violations."""
        if not self.academic_violations:
            return {"impact_level": "none", "description": "No violations detected"}
        
        critical_violations = [v for v in self.academic_violations 
                             if v["severity"] == ParaphraseErrorSeverity.ACADEMIC_VIOLATION.value]
        
        if critical_violations:
            return {
                "impact_level": "critical",
                "description": "Academic integrity compromised - results may not be usable",
                "critical_count": len(critical_violations),
                "recommendation": "Review methodology and regenerate clean results"
            }
        else:
            return {
                "impact_level": "moderate", 
                "description": "Some quality concerns but academic integrity maintained",
                "recommendation": "Address issues in future runs but current results usable with caveats"
            }


class ParaphraseErrorHandler:
    """
    Centralized error handling for the paraphrase pipeline.
    
    Provides recovery strategies, error classification, and academic
    compliance monitoring for all paraphrase pipeline operations.
    """
    
    def __init__(self, logger: AcademicLogger):
        """Initialize error handler with academic logger."""
        self.logger = logger
        self.error_history: List[ParaphraseError] = []
    
    def handle_provider_isolation_violation(self, 
                                          provider: str,
                                          context: Dict[str, Any]) -> ParaphraseError:
        """Handle critical provider isolation violations."""
        error = ParaphraseError(
            message=f"Provider '{provider}' used for both paraphrase generation and evaluation",
            error_code=ParaphraseErrorCode.PROVIDER_ISOLATION_VIOLATION,
            severity=ParaphraseErrorSeverity.ACADEMIC_VIOLATION,
            context=context,
            recovery_suggestion="Immediately configure separate providers for paraphrase and evaluation",
            academic_impact="Critical - results contaminated and unusable for academic publication"
        )
        
        self.logger.academic_violation(
            message=str(error),
            violation_type="provider_isolation",
            context=context
        )
        
        self.error_history.append(error)
        return error
    
    def handle_quality_failure(self,
                             item_id: str,
                             failure_type: str,
                             metrics: Dict[str, float]) -> ParaphraseError:
        """Handle quality control failures."""
        error_code_map = {
            "semantic": ParaphraseErrorCode.SEMANTIC_SIMILARITY_FAILED,
            "surface": ParaphraseErrorCode.SURFACE_DIVERGENCE_FAILED,
            "both": ParaphraseErrorCode.QUALITY_THRESHOLD_NOT_MET
        }
        
        error = ParaphraseError(
            message=f"Quality validation failed for item {item_id}: {failure_type}",
            error_code=error_code_map.get(failure_type, ParaphraseErrorCode.QUALITY_THRESHOLD_NOT_MET),
            severity=ParaphraseErrorSeverity.WARNING,
            context={"item_id": item_id, "metrics": metrics, "failure_type": failure_type},
            recovery_suggestion="Review paraphrase generation prompts or adjust quality thresholds"
        )
        
        self.logger.log_paraphrase_event(
            event_type="quality_validation",
            item_id=item_id,
            success=False,
            details={"failure_type": failure_type, "metrics": metrics}
        )
        
        self.error_history.append(error)
        return error
    
    def handle_generation_failure(self,
                                item_id: str,
                                cause: Exception) -> ParaphraseError:
        """Handle paraphrase generation failures."""
        error = ParaphraseError(
            message=f"Paraphrase generation failed for item {item_id}: {str(cause)}",
            error_code=ParaphraseErrorCode.GENERATION_FAILED,
            severity=ParaphraseErrorSeverity.ERROR,
            context={"item_id": item_id, "original_error": str(cause)},
            recovery_suggestion="Check model connectivity and retry with exponential backoff"
        )
        
        self.logger.error(
            f"Generation failure for {item_id}",
            error=cause,
            extra={"item_id": item_id}
        )
        
        self.error_history.append(error)
        return error
    
    def handle_cache_failure(self,
                           operation: str,
                           item_id: str,
                           cause: Exception) -> ParaphraseError:
        """Handle caching failures."""
        error = ParaphraseError(
            message=f"Cache {operation} failed for item {item_id}: {str(cause)}",
            error_code=ParaphraseErrorCode.CACHE_WRITE_FAILED if operation == "write" else ParaphraseErrorCode.CACHE_READ_FAILED,
            severity=ParaphraseErrorSeverity.ERROR,
            context={"item_id": item_id, "operation": operation, "original_error": str(cause)},
            recovery_suggestion="Check database connectivity and disk space, consider fallback to file cache"
        )
        
        self.logger.error(
            f"Cache {operation} failure for {item_id}",
            error=cause,
            extra={"item_id": item_id, "operation": operation}
        )
        
        self.error_history.append(error)
        return error
    
    def handle_acceptance_rate_below_target(self,
                                          actual_rate: float,
                                          target_rate: float = 0.95,
                                          context: Dict[str, Any] = None) -> ParaphraseError:
        """Handle acceptance rate below academic standards."""
        error = ParaphraseError(
            message=f"Acceptance rate ({actual_rate:.2%}) below academic target ({target_rate:.2%})",
            error_code=ParaphraseErrorCode.ACCEPTANCE_RATE_BELOW_TARGET,
            severity=ParaphraseErrorSeverity.ACADEMIC_VIOLATION,
            context=context or {},
            recovery_suggestion="Improve paraphrase generation quality or adjust validation thresholds",
            academic_impact="Results may not meet academic publication standards"
        )
        
        self.logger.academic_violation(
            message=str(error),
            violation_type="acceptance_rate_below_target",
            context={"actual_rate": actual_rate, "target_rate": target_rate}
        )
        
        self.error_history.append(error)
        return error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary for academic reporting."""
        error_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            # Count by error code
            code = error.error_code.value
            error_counts[code] = error_counts.get(code, 0) + 1
            
            # Count by severity
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_counts": error_counts,
            "severity_counts": severity_counts,
            "academic_violations": len([e for e in self.error_history 
                                      if e.severity == ParaphraseErrorSeverity.ACADEMIC_VIOLATION]),
            "most_recent_errors": [e.to_dict() for e in self.error_history[-5:]]  # Last 5 errors
        }


# Global instances for convenience
_global_logger: Optional[AcademicLogger] = None
_global_error_handler: Optional[ParaphraseErrorHandler] = None


def get_academic_logger(name: str = "paraphrase_pipeline") -> AcademicLogger:
    """Get or create global academic logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = AcademicLogger(name)
    return _global_logger


def get_error_handler() -> ParaphraseErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ParaphraseErrorHandler(get_academic_logger())
    return _global_error_handler


def setup_academic_logging(log_dir: Optional[Path] = None) -> AcademicLogger:
    """Set up academic-grade logging for paraphrase pipeline."""
    global _global_logger, _global_error_handler
    
    _global_logger = AcademicLogger("paraphrase_pipeline", log_dir, academic_mode=True)
    _global_error_handler = ParaphraseErrorHandler(_global_logger)
    
    return _global_logger