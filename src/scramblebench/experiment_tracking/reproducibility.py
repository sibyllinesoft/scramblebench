"""
Reproducibility Validation and Environment Tracking

Comprehensive system for ensuring experiment reproducibility through
environment capture, validation, and replication package generation.
"""

import os
import sys
import platform
import subprocess
import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pkg_resources
import psutil
import git


@dataclass
class EnvironmentSnapshot:
    """Complete environment snapshot for reproducibility"""
    # System information
    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    cpu_count: int
    memory_total_gb: float
    
    # Python environment
    installed_packages: Dict[str, str]  # package -> version
    python_path: List[str]
    environment_variables: Dict[str, str]
    
    # Git information  
    git_commit_hash: str
    git_branch: str
    git_remote_url: str
    git_status: str
    uncommitted_changes: bool
    
    # Hardware information
    gpu_info: List[Dict[str, Any]]
    cuda_version: Optional[str] = None
    
    # Timestamps
    captured_at: datetime = field(default_factory=datetime.now)
    timezone: str = field(default_factory=lambda: str(datetime.now().astimezone().tzinfo))
    
    # Validation
    checksum: str = ""
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for validation"""
        # Create deterministic representation
        data = {
            'python_version': self.python_version,
            'packages': sorted(self.installed_packages.items()),
            'git_commit': self.git_commit_hash,
            'platform': f"{self.platform_system}_{self.platform_release}_{self.platform_machine}"
        }
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        data['captured_at'] = self.captured_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentSnapshot':
        """Create from dictionary"""
        if 'captured_at' in data and isinstance(data['captured_at'], str):
            data['captured_at'] = datetime.fromisoformat(data['captured_at'])
        return cls(**data)


@dataclass
class ReplicationPackage:
    """Complete package for experiment replication"""
    experiment_id: str
    experiment_name: str
    
    # Core files
    config_file: str
    requirements_file: str
    environment_snapshot: EnvironmentSnapshot
    
    # Code and data
    code_archive: str
    data_files: List[str]
    
    # Instructions
    setup_instructions: str
    execution_instructions: str
    validation_steps: List[str]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    version: str = "1.0"
    
    # Checksums for integrity
    file_checksums: Dict[str, str] = field(default_factory=dict)


class ReproducibilityValidator:
    """
    System for capturing, validating, and ensuring experiment reproducibility
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        include_sensitive_env_vars: bool = False
    ):
        """
        Initialize reproducibility validator
        
        Args:
            logger: Logger instance
            include_sensitive_env_vars: Whether to include sensitive environment variables
        """
        self.logger = logger or logging.getLogger(__name__)
        self.include_sensitive = include_sensitive_env_vars
        
        # Sensitive environment variable patterns to exclude
        self.sensitive_patterns = [
            'KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'PASS', 'PWD',
            'AUTH', 'CREDENTIAL', 'PRIVATE', 'API_KEY'
        ]
    
    async def capture_environment(self) -> Dict[str, Any]:
        """
        Capture complete environment snapshot
        
        Returns:
            Environment data dictionary
        """
        self.logger.info("Capturing environment snapshot")
        
        try:
            # System information
            system_info = self._capture_system_info()
            
            # Python environment
            python_info = self._capture_python_environment()
            
            # Git information
            git_info = await self.get_git_info()
            
            # Hardware information
            hardware_info = self._capture_hardware_info()
            
            # Create snapshot
            snapshot = EnvironmentSnapshot(
                # System
                python_version=system_info['python_version'],
                platform_system=system_info['platform_system'],
                platform_release=system_info['platform_release'],
                platform_machine=system_info['platform_machine'],
                cpu_count=system_info['cpu_count'],
                memory_total_gb=system_info['memory_total_gb'],
                
                # Python
                installed_packages=python_info['packages'],
                python_path=python_info['python_path'],
                environment_variables=python_info['env_vars'],
                
                # Git
                git_commit_hash=git_info['commit_hash'],
                git_branch=git_info['branch'],
                git_remote_url=git_info['remote_url'],
                git_status=git_info['status'],
                uncommitted_changes=git_info['uncommitted_changes'],
                
                # Hardware
                gpu_info=hardware_info['gpu_info'],
                cuda_version=hardware_info['cuda_version']
            )
            
            self.logger.info(f"Environment snapshot captured with checksum: {snapshot.checksum}")
            return snapshot.to_dict()
            
        except Exception as e:
            self.logger.error(f"Failed to capture environment: {e}")
            raise
    
    async def get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information"""
        try:
            # Try to find git repository
            repo_path = self._find_git_repository()
            if not repo_path:
                return {
                    'commit_hash': 'unknown',
                    'branch': 'unknown',
                    'remote_url': 'unknown',
                    'status': 'no_git_repo',
                    'uncommitted_changes': False
                }
            
            repo = git.Repo(repo_path)
            
            # Get current commit
            commit_hash = repo.head.commit.hexsha
            
            # Get current branch
            try:
                branch = repo.active_branch.name
            except TypeError:
                branch = "detached_head"
            
            # Get remote URL
            remote_url = "unknown"
            if repo.remotes:
                remote_url = repo.remotes.origin.url
            
            # Check for uncommitted changes
            uncommitted_changes = repo.is_dirty() or bool(repo.untracked_files)
            
            # Get status
            status = "clean"
            if repo.is_dirty():
                status = "modified"
            elif repo.untracked_files:
                status = "untracked_files"
            
            return {
                'commit_hash': commit_hash,
                'branch': branch,
                'remote_url': remote_url,
                'status': status,
                'uncommitted_changes': uncommitted_changes
            }
            
        except Exception as e:
            self.logger.warning(f"Could not get git info: {e}")
            return {
                'commit_hash': 'unknown',
                'branch': 'unknown', 
                'remote_url': 'unknown',
                'status': 'error',
                'uncommitted_changes': False
            }
    
    def _capture_system_info(self) -> Dict[str, Any]:
        """Capture system information"""
        memory_info = psutil.virtual_memory()
        
        return {
            'python_version': platform.python_version(),
            'platform_system': platform.system(),
            'platform_release': platform.release(),
            'platform_machine': platform.machine(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': memory_info.total / (1024**3)
        }
    
    def _capture_python_environment(self) -> Dict[str, Any]:
        """Capture Python environment details"""
        # Get installed packages
        packages = {}
        for dist in pkg_resources.working_set:
            packages[dist.project_name] = dist.version
        
        # Get Python path
        python_path = sys.path.copy()
        
        # Get environment variables (filtered)
        env_vars = {}
        for key, value in os.environ.items():
            if self.include_sensitive or not self._is_sensitive_env_var(key):
                env_vars[key] = value
        
        return {
            'packages': packages,
            'python_path': python_path,
            'env_vars': env_vars
        }
    
    def _capture_hardware_info(self) -> Dict[str, Any]:
        """Capture hardware information"""
        gpu_info = []
        cuda_version = None
        
        # Try to get GPU information
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'driver': gpu.driver
                })
        except ImportError:
            self.logger.debug("GPUtil not available - no GPU info captured")
        except Exception as e:
            self.logger.warning(f"Failed to get GPU info: {e}")
        
        # Try to get CUDA version
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Parse CUDA version from output
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        cuda_version = line.split('release ')[1].split(',')[0]
                        break
        except FileNotFoundError:
            self.logger.debug("nvcc not found - no CUDA version captured")
        except Exception as e:
            self.logger.warning(f"Failed to get CUDA version: {e}")
        
        return {
            'gpu_info': gpu_info,
            'cuda_version': cuda_version
        }
    
    def _is_sensitive_env_var(self, var_name: str) -> bool:
        """Check if environment variable might contain sensitive information"""
        var_upper = var_name.upper()
        return any(pattern in var_upper for pattern in self.sensitive_patterns)
    
    def _find_git_repository(self) -> Optional[str]:
        """Find the git repository root"""
        current_dir = Path.cwd()
        
        # Walk up the directory tree looking for .git
        for parent in [current_dir] + list(current_dir.parents):
            git_dir = parent / '.git'
            if git_dir.exists():
                return str(parent)
        
        return None
    
    async def validate_reproducibility(
        self,
        original_snapshot: EnvironmentSnapshot,
        tolerance: str = "strict"
    ) -> Dict[str, Any]:
        """
        Validate current environment against original snapshot
        
        Args:
            original_snapshot: Original environment snapshot
            tolerance: Validation tolerance ('strict', 'moderate', 'loose')
            
        Returns:
            Validation results
        """
        self.logger.info(f"Validating reproducibility with {tolerance} tolerance")
        
        current_env = await self.capture_environment()
        current_snapshot = EnvironmentSnapshot.from_dict(current_env)
        
        issues = []
        warnings = []
        
        # Python version check
        if original_snapshot.python_version != current_snapshot.python_version:
            if tolerance == "strict":
                issues.append(f"Python version mismatch: {original_snapshot.python_version} vs {current_snapshot.python_version}")
            else:
                warnings.append(f"Python version difference: {original_snapshot.python_version} vs {current_snapshot.python_version}")
        
        # Package version checks
        missing_packages = []
        version_mismatches = []
        
        for package, version in original_snapshot.installed_packages.items():
            if package not in current_snapshot.installed_packages:
                missing_packages.append(package)
            elif current_snapshot.installed_packages[package] != version:
                version_mismatches.append({
                    'package': package,
                    'original': version,
                    'current': current_snapshot.installed_packages[package]
                })
        
        if missing_packages:
            issues.append(f"Missing packages: {', '.join(missing_packages)}")
        
        if version_mismatches and tolerance == "strict":
            issues.append(f"Package version mismatches: {len(version_mismatches)} packages")
        elif version_mismatches:
            warnings.append(f"Package version differences: {len(version_mismatches)} packages")
        
        # Git commit check
        if (original_snapshot.git_commit_hash != current_snapshot.git_commit_hash and 
            tolerance == "strict"):
            issues.append(f"Git commit mismatch: {original_snapshot.git_commit_hash} vs {current_snapshot.git_commit_hash}")
        
        # Platform check
        if (original_snapshot.platform_system != current_snapshot.platform_system or
            original_snapshot.platform_machine != current_snapshot.platform_machine):
            if tolerance == "strict":
                issues.append("Platform architecture mismatch")
            else:
                warnings.append("Platform architecture difference")
        
        # Determine overall result
        is_reproducible = len(issues) == 0
        confidence = "high" if is_reproducible and len(warnings) == 0 else "medium" if is_reproducible else "low"
        
        return {
            'is_reproducible': is_reproducible,
            'confidence': confidence,
            'issues': issues,
            'warnings': warnings,
            'tolerance': tolerance,
            'validation_timestamp': datetime.now().isoformat(),
            'original_checksum': original_snapshot.checksum,
            'current_checksum': current_snapshot.checksum,
            'details': {
                'missing_packages': missing_packages,
                'version_mismatches': version_mismatches
            }
        }
    
    async def generate_replication_package(
        self,
        experiment_id: str,
        experiment_name: str,
        config_path: Path,
        output_dir: Path,
        include_data: bool = True
    ) -> ReplicationPackage:
        """
        Generate complete replication package
        
        Args:
            experiment_id: Experiment ID
            experiment_name: Experiment name
            config_path: Path to configuration file
            output_dir: Output directory for package
            include_data: Whether to include data files
            
        Returns:
            Replication package information
        """
        self.logger.info(f"Generating replication package for {experiment_id}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Capture current environment
        env_data = await self.capture_environment()
        env_snapshot = EnvironmentSnapshot.from_dict(env_data)
        
        # Generate requirements.txt
        requirements_file = output_dir / "requirements.txt"
        self._generate_requirements_file(env_snapshot, requirements_file)
        
        # Copy configuration file
        config_file = output_dir / "experiment_config.yaml"
        if config_path.exists():
            import shutil
            shutil.copy2(config_path, config_file)
        
        # Save environment snapshot
        env_file = output_dir / "environment_snapshot.json"
        with open(env_file, 'w') as f:
            json.dump(env_data, f, indent=2)
        
        # Generate setup instructions
        setup_instructions = self._generate_setup_instructions(env_snapshot)
        
        # Generate execution instructions
        execution_instructions = self._generate_execution_instructions()
        
        # Generate validation steps
        validation_steps = self._generate_validation_steps()
        
        # Create README
        readme_file = output_dir / "README.md"
        self._generate_readme(
            readme_file, experiment_name, setup_instructions,
            execution_instructions, validation_steps
        )
        
        # Calculate file checksums
        file_checksums = {}
        for file_path in output_dir.glob("*"):
            if file_path.is_file():
                file_checksums[file_path.name] = self._calculate_file_checksum(file_path)
        
        # Create replication package
        package = ReplicationPackage(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            config_file=str(config_file),
            requirements_file=str(requirements_file),
            environment_snapshot=env_snapshot,
            code_archive="",  # Would implement code archiving
            data_files=[],    # Would list data files
            setup_instructions=setup_instructions,
            execution_instructions=execution_instructions,
            validation_steps=validation_steps,
            file_checksums=file_checksums,
            created_by=os.getenv('USER', 'unknown')
        )
        
        # Save package metadata
        package_file = output_dir / "replication_package.json"
        with open(package_file, 'w') as f:
            json.dump(asdict(package), f, indent=2, default=str)
        
        self.logger.info(f"Replication package generated at {output_dir}")
        return package
    
    def _generate_requirements_file(
        self,
        env_snapshot: EnvironmentSnapshot,
        output_file: Path
    ) -> None:
        """Generate requirements.txt file"""
        with open(output_file, 'w') as f:
            f.write("# Generated requirements for experiment replication\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write(f"# Python version: {env_snapshot.python_version}\n\n")
            
            # Sort packages for consistency
            for package, version in sorted(env_snapshot.installed_packages.items()):
                f.write(f"{package}=={version}\n")
    
    def _generate_setup_instructions(self, env_snapshot: EnvironmentSnapshot) -> str:
        """Generate setup instructions"""
        return f"""# Experiment Setup Instructions

## Environment Requirements
- Python {env_snapshot.python_version}
- Platform: {env_snapshot.platform_system} {env_snapshot.platform_release}
- Architecture: {env_snapshot.platform_machine}

## Setup Steps
1. Create virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate   # Windows
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```
   python -c "import scramblebench; print('Installation successful')"
   ```

## Git Information
- Original commit: {env_snapshot.git_commit_hash}
- Branch: {env_snapshot.git_branch}
- Repository: {env_snapshot.git_remote_url}
"""
    
    def _generate_execution_instructions(self) -> str:
        """Generate execution instructions"""
        return """# Experiment Execution Instructions

## Running the Experiment
1. Ensure you have activated the virtual environment
2. Run the experiment:
   ```
   scramblebench run --config experiment_config.yaml
   ```

## Monitoring Progress
- Check experiment status: `scramblebench status`
- View logs: `tail -f experiment.log`
- Monitor resources: `htop` or similar

## Output Files
- Results will be saved to `results/` directory
- Logs will be saved to `logs/` directory
- Intermediate files in `cache/` directory
"""
    
    def _generate_validation_steps(self) -> List[str]:
        """Generate validation steps"""
        return [
            "Verify Python version matches original environment",
            "Check all required packages are installed with correct versions",
            "Confirm experiment configuration matches original",
            "Run experiment and compare results with original",
            "Validate statistical significance of results",
            "Check for any unexpected warnings or errors"
        ]
    
    def _generate_readme(
        self,
        readme_file: Path,
        experiment_name: str,
        setup_instructions: str,
        execution_instructions: str,
        validation_steps: List[str]
    ) -> None:
        """Generate comprehensive README file"""
        with open(readme_file, 'w') as f:
            f.write(f"""# {experiment_name} - Replication Package

This package contains everything needed to replicate the "{experiment_name}" experiment.

## Package Contents
- `experiment_config.yaml` - Experiment configuration
- `requirements.txt` - Python dependencies
- `environment_snapshot.json` - Complete environment details
- `replication_package.json` - Package metadata

{setup_instructions}

{execution_instructions}

## Validation Steps
""")
            for i, step in enumerate(validation_steps, 1):
                f.write(f"{i}. {step}\n")
            
            f.write(f"""
## Support
For questions or issues with replication, please:
1. Check that your environment matches the requirements
2. Verify all validation steps pass
3. Contact the original researchers

## Citation
If you use this experiment in your research, please cite the original work.

Generated: {datetime.now().isoformat()}
""")
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()