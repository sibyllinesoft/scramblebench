"""
Runtime validation and dependency checking for ScrambleBench.

This module provides comprehensive validation capabilities to ensure
that the environment is properly configured and all dependencies
are available before running benchmarks.
"""

import sys
import os
import subprocess
import importlib
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging

from scramblebench.core.exceptions import (
    ValidationError, ConfigurationError, DependencyError,
    ResourceError, handle_common_exceptions
)
from scramblebench.core.logging import get_logger


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"  # Will prevent operation
    WARNING = "warning"   # May cause issues
    INFO = "info"        # Good to know


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    details: Dict[str, Any]
    fix_suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'details': self.details,
            'fix_suggestion': self.fix_suggestion
        }


@dataclass
class ValidationResult:
    """Results of validation checks."""
    passed: bool
    issues: List[ValidationIssue]
    summary: Dict[str, Any]
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get critical issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get warnings."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return len(self.critical_issues) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passed': self.passed,
            'issues': [issue.to_dict() for issue in self.issues],
            'summary': self.summary,
            'critical_count': len(self.critical_issues),
            'warning_count': len(self.warnings)
        }


class SystemValidator:
    """Validates system requirements and environment."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.issues: List[ValidationIssue] = []
    
    def validate_python_version(self) -> None:
        """Validate Python version."""
        min_version = (3, 9)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="python",
                message=f"Python {'.'.join(map(str, min_version))}+ required, found {'.'.join(map(str, current_version))}",
                details={
                    'required': '.'.join(map(str, min_version)),
                    'current': '.'.join(map(str, current_version)),
                    'platform': platform.platform()
                },
                fix_suggestion="Upgrade Python to version 3.9 or later"
            ))
        else:
            self.logger.debug(f"Python version check passed: {'.'.join(map(str, current_version))}")
    
    def validate_platform(self) -> None:
        """Validate platform compatibility."""
        supported_platforms = ['Linux', 'Darwin', 'Windows']
        current_platform = platform.system()
        
        if current_platform not in supported_platforms:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="platform",
                message=f"Platform {current_platform} is not officially supported",
                details={
                    'current': current_platform,
                    'supported': supported_platforms
                },
                fix_suggestion="Use Linux, macOS, or Windows for best compatibility"
            ))
    
    def validate_memory(self) -> None:
        """Validate available memory."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            # Require at least 4GB RAM
            min_memory_gb = 4
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            if available_gb < min_memory_gb:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="memory",
                    message=f"Low available memory: {available_gb:.1f}GB (recommended: {min_memory_gb}GB+)",
                    details={
                        'available_gb': round(available_gb, 1),
                        'total_gb': round(total_gb, 1),
                        'required_gb': min_memory_gb
                    },
                    fix_suggestion="Close other applications or add more RAM"
                ))
            
        except ImportError:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="memory",
                message="Cannot check memory (psutil not available)",
                details={},
                fix_suggestion="Install psutil for memory monitoring: pip install psutil"
            ))
    
    def validate_disk_space(self, path: Path = None) -> None:
        """Validate available disk space."""
        if path is None:
            path = Path.cwd()
        
        try:
            import shutil
            free_space = shutil.disk_usage(path).free
            free_gb = free_space / (1024**3)
            
            # Require at least 5GB free space
            min_space_gb = 5
            
            if free_gb < min_space_gb:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="disk_space",
                    message=f"Low disk space: {free_gb:.1f}GB (recommended: {min_space_gb}GB+)",
                    details={
                        'available_gb': round(free_gb, 1),
                        'required_gb': min_space_gb,
                        'path': str(path)
                    },
                    fix_suggestion="Free up disk space or use a different location"
                ))
                
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="disk_space",
                message=f"Cannot check disk space: {str(e)}",
                details={'error': str(e)},
                fix_suggestion="Manually ensure sufficient disk space"
            ))


class DependencyValidator:
    """Validates Python dependencies and external tools."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.issues: List[ValidationIssue] = []
    
    def validate_required_packages(self) -> None:
        """Validate required Python packages."""
        required_packages = {
            'pydantic': '2.0.0',
            'click': '8.0.0',
            'rich': '13.0.0',
            'requests': '2.28.0',
            'pyyaml': '6.0.0',
            'numpy': '1.21.0'
        }
        
        for package, min_version in required_packages.items():
            try:
                module = importlib.import_module(package)
                
                # Try to get version
                version = None
                for attr in ['__version__', 'version', 'VERSION']:
                    if hasattr(module, attr):
                        version = getattr(module, attr)
                        break
                
                if version:
                    self.logger.debug(f"Package {package} version {version} found")
                else:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="dependencies",
                        message=f"Cannot determine version of {package}",
                        details={'package': package},
                        fix_suggestion="This may not affect functionality"
                    ))
                    
            except ImportError:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="dependencies",
                    message=f"Required package {package} not found",
                    details={'package': package, 'min_version': min_version},
                    fix_suggestion=f"Install package: pip install {package}>={min_version}"
                ))
    
    def validate_optional_packages(self) -> None:
        """Validate optional packages that enhance functionality."""
        optional_packages = {
            'psutil': 'System monitoring',
            'jupyter': 'Jupyter notebook support',
            'matplotlib': 'Plotting capabilities',
            'seaborn': 'Enhanced plotting',
            'pandas': 'Data analysis',
            'scipy': 'Scientific computing',
            'openai': 'OpenAI API support',
            'anthropic': 'Anthropic API support'
        }
        
        for package, description in optional_packages.items():
            try:
                importlib.import_module(package)
                self.logger.debug(f"Optional package {package} available")
            except ImportError:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="optional_dependencies",
                    message=f"Optional package {package} not available ({description})",
                    details={'package': package, 'description': description},
                    fix_suggestion=f"Install for enhanced functionality: pip install {package}"
                ))
    
    def validate_external_tools(self) -> None:
        """Validate external command-line tools."""
        tools = {
            'git': 'Version control',
            'curl': 'HTTP downloads'
        }
        
        for tool, description in tools.items():
            try:
                result = subprocess.run(
                    [tool, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.logger.debug(f"External tool {tool} available")
                else:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="external_tools",
                        message=f"External tool {tool} may not be working ({description})",
                        details={'tool': tool, 'description': description},
                        fix_suggestion=f"Install or fix {tool}"
                    ))
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="external_tools",
                    message=f"External tool {tool} not found ({description})",
                    details={'tool': tool, 'description': description},
                    fix_suggestion=f"Install {tool} for enhanced functionality"
                ))


class ConfigurationValidator:
    """Validates configuration settings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.issues: List[ValidationIssue] = []
    
    def validate_environment_variables(self) -> None:
        """Validate environment variables."""
        recommended_vars = {
            'OPENROUTER_API_KEY': 'OpenRouter API access',
            'OPENAI_API_KEY': 'OpenAI API access',
            'ANTHROPIC_API_KEY': 'Anthropic API access'
        }
        
        for var, description in recommended_vars.items():
            value = os.getenv(var)
            if not value:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="environment",
                    message=f"Environment variable {var} not set ({description})",
                    details={'variable': var, 'description': description},
                    fix_suggestion=f"Set {var} in your environment or .env file"
                ))
            elif len(value) < 10:  # Basic sanity check
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="environment",
                    message=f"Environment variable {var} seems too short",
                    details={'variable': var, 'length': len(value)},
                    fix_suggestion=f"Verify {var} value is correct"
                ))
    
    def validate_directories(self) -> None:
        """Validate required directories."""
        required_dirs = [
            'data',
            'data/benchmarks',
            'data/results',
            'data/cache',
            'logs'
        ]
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            
            if not path.exists():
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="directories",
                    message=f"Directory {dir_path} does not exist",
                    details={'path': str(path.absolute())},
                    fix_suggestion=f"Create directory: mkdir -p {dir_path}"
                ))
            elif not path.is_dir():
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="directories",
                    message=f"Path {dir_path} exists but is not a directory",
                    details={'path': str(path.absolute())},
                    fix_suggestion=f"Remove file and create directory: rm {dir_path} && mkdir -p {dir_path}"
                ))
            elif not os.access(path, os.W_OK):
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="directories",
                    message=f"Directory {dir_path} is not writable",
                    details={'path': str(path.absolute())},
                    fix_suggestion=f"Fix permissions: chmod u+w {dir_path}"
                ))


class NetworkValidator:
    """Validates network connectivity and API access."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self.issues: List[ValidationIssue] = []
    
    def validate_internet_connection(self) -> None:
        """Validate internet connectivity."""
        test_urls = [
            'https://api.openrouter.ai/api/v1/models',
            'https://api.openai.com/v1/models',
            'https://api.anthropic.com/v1/messages'
        ]
        
        import requests
        
        connection_working = False
        for url in test_urls:
            try:
                response = requests.head(url, timeout=10)
                if response.status_code in [200, 401, 403]:  # 401/403 means server is reachable
                    connection_working = True
                    break
            except Exception:
                continue
        
        if not connection_working:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="network",
                message="Cannot reach API endpoints (network connectivity issue)",
                details={'tested_urls': test_urls},
                fix_suggestion="Check internet connection and firewall settings"
            ))
    
    def validate_api_keys(self) -> None:
        """Validate API key functionality (without making actual calls)."""
        api_configs = {
            'OPENROUTER_API_KEY': {
                'url': 'https://openrouter.ai/api/v1/models',
                'header_format': 'Bearer {key}'
            },
            'OPENAI_API_KEY': {
                'url': 'https://api.openai.com/v1/models',
                'header_format': 'Bearer {key}'
            }
        }
        
        import requests
        
        for env_var, config in api_configs.items():
            api_key = os.getenv(env_var)
            if api_key:
                try:
                    headers = {
                        'Authorization': config['header_format'].format(key=api_key),
                        'User-Agent': 'ScrambleBench-Validator/1.0'
                    }
                    
                    response = requests.head(config['url'], headers=headers, timeout=10)
                    
                    if response.status_code == 401:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="api_keys",
                            message=f"API key {env_var} appears to be invalid",
                            details={'status_code': response.status_code},
                            fix_suggestion=f"Verify {env_var} is correct"
                        ))
                    elif response.status_code == 200:
                        self.logger.debug(f"API key {env_var} appears valid")
                    
                except Exception as e:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="api_keys",
                        message=f"Cannot validate {env_var}: {str(e)}",
                        details={'error': str(e)},
                        fix_suggestion="Check network connection and key validity"
                    ))


class ComprehensiveValidator:
    """Main validator that coordinates all validation checks."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        
        self.system_validator = SystemValidator(logger)
        self.dependency_validator = DependencyValidator(logger)
        self.config_validator = ConfigurationValidator(logger)
        self.network_validator = NetworkValidator(logger)
    
    @handle_common_exceptions
    def validate_all(
        self,
        skip_network: bool = False,
        skip_api_validation: bool = False
    ) -> ValidationResult:
        """
        Run all validation checks.
        
        Args:
            skip_network: Skip network connectivity checks
            skip_api_validation: Skip API key validation
            
        Returns:
            ValidationResult with all issues found
        """
        self.logger.info("Starting comprehensive validation...")
        
        all_issues = []
        
        # System validation
        self.logger.debug("Validating system requirements...")
        self.system_validator.validate_python_version()
        self.system_validator.validate_platform()
        self.system_validator.validate_memory()
        self.system_validator.validate_disk_space()
        all_issues.extend(self.system_validator.issues)
        
        # Dependency validation
        self.logger.debug("Validating dependencies...")
        self.dependency_validator.validate_required_packages()
        self.dependency_validator.validate_optional_packages()
        self.dependency_validator.validate_external_tools()
        all_issues.extend(self.dependency_validator.issues)
        
        # Configuration validation
        self.logger.debug("Validating configuration...")
        self.config_validator.validate_environment_variables()
        self.config_validator.validate_directories()
        all_issues.extend(self.config_validator.issues)
        
        # Network validation
        if not skip_network:
            self.logger.debug("Validating network connectivity...")
            self.network_validator.validate_internet_connection()
            
            if not skip_api_validation:
                self.network_validator.validate_api_keys()
            
            all_issues.extend(self.network_validator.issues)
        
        # Create summary
        critical_count = len([i for i in all_issues if i.severity == ValidationSeverity.CRITICAL])
        warning_count = len([i for i in all_issues if i.severity == ValidationSeverity.WARNING])
        info_count = len([i for i in all_issues if i.severity == ValidationSeverity.INFO])
        
        summary = {
            'total_checks': len(all_issues) if all_issues else 1,  # Avoid division by zero
            'critical_issues': critical_count,
            'warnings': warning_count,
            'info_messages': info_count,
            'passed': critical_count == 0
        }
        
        result = ValidationResult(
            passed=critical_count == 0,
            issues=all_issues,
            summary=summary
        )
        
        if result.passed:
            self.logger.info("[CHECK] All critical validation checks passed")
        else:
            self.logger.error(f"[X] Validation failed with {critical_count} critical issues")
        
        if warning_count > 0:
            self.logger.warning(f"[WARNING] {warning_count} warnings found")
        
        return result
    
    def create_validation_report(self, result: ValidationResult) -> str:
        """Create a formatted validation report."""
        lines = []
        lines.append("ScrambleBench Validation Report")
        lines.append("=" * 40)
        lines.append("")
        
        if result.passed:
            lines.append("[CHECK] STATUS: PASSED")
        else:
            lines.append("[X] STATUS: FAILED")
        
        lines.append("")
        lines.append(f"Summary:")
        lines.append(f"  Critical Issues: {result.summary['critical_issues']}")
        lines.append(f"  Warnings: {result.summary['warnings']}")
        lines.append(f"  Info Messages: {result.summary['info_messages']}")
        lines.append("")
        
        if result.issues:
            lines.append("Issues Found:")
            lines.append("-" * 20)
            
            for issue in result.issues:
                severity_icon = {
                    ValidationSeverity.CRITICAL: "[ALERT]",
                    ValidationSeverity.WARNING: "[WARNING]",
                    ValidationSeverity.INFO: "[INFO]"
                }[issue.severity]
                
                lines.append(f"{severity_icon} [{issue.severity.value.upper()}] {issue.category}: {issue.message}")
                if issue.fix_suggestion:
                    lines.append(f"   Fix: {issue.fix_suggestion}")
                lines.append("")
        
        return "\n".join(lines)


# Convenience function for quick validation
def validate_environment(
    skip_network: bool = False,
    skip_api_validation: bool = False,
    logger: Optional[logging.Logger] = None
) -> ValidationResult:
    """
    Quick environment validation.
    
    Args:
        skip_network: Skip network checks
        skip_api_validation: Skip API key validation  
        logger: Optional logger
        
    Returns:
        ValidationResult
    """
    validator = ComprehensiveValidator(logger)
    return validator.validate_all(skip_network, skip_api_validation)


def validate_and_report(
    skip_network: bool = False,
    skip_api_validation: bool = False,
    output_file: Optional[str] = None
) -> bool:
    """
    Validate environment and print report.
    
    Args:
        skip_network: Skip network checks
        skip_api_validation: Skip API key validation
        output_file: Optional file to save report
        
    Returns:
        True if validation passed
    """
    validator = ComprehensiveValidator()
    result = validator.validate_all(skip_network, skip_api_validation)
    
    report = validator.create_validation_report(result)
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
            f.write("\n\nJSON Report:\n")
            f.write(json.dumps(result.to_dict(), indent=2))
    
    return result.passed