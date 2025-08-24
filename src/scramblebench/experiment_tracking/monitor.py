"""
Experiment Monitoring and Progress Tracking System

Real-time monitoring of experiment progress, resource utilization, and performance
with ETA calculations, alerting, and dashboard data generation.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import psutil
import json

from .database import DatabaseManager


@dataclass
class ProgressSnapshot:
    """Snapshot of experiment progress at a point in time"""
    timestamp: datetime
    stage: str
    progress: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    api_calls_count: int = 0
    cost_incurred: float = 0.0
    
    # Quality metrics
    error_rate: float = 0.0
    response_time_avg: float = 0.0
    success_rate: float = 1.0


@dataclass
class ResourceUsage:
    """Current resource usage metrics"""
    cpu_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_used_gb: float
    disk_available_gb: float
    network_sent_mb: float
    network_received_mb: float
    
    # Experiment-specific metrics
    api_calls_per_minute: float = 0.0
    cost_per_hour: float = 0.0
    active_connections: int = 0
    
    @property
    def memory_usage_percent(self) -> float:
        """Memory usage as percentage"""
        total = self.memory_used_gb + self.memory_available_gb
        return (self.memory_used_gb / total * 100) if total > 0 else 0.0
    
    @property  
    def disk_usage_percent(self) -> float:
        """Disk usage as percentage"""
        total = self.disk_used_gb + self.disk_available_gb
        return (self.disk_used_gb / total * 100) if total > 0 else 0.0


@dataclass
class PerformanceAlert:
    """Performance alert for experiment issues"""
    experiment_id: str
    alert_type: str  # 'performance', 'resource', 'error', 'quality'
    severity: str    # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


class ProgressTracker:
    """
    Tracks detailed progress for a single experiment
    """
    
    def __init__(
        self,
        experiment_id: str,
        db_manager: DatabaseManager,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize progress tracker for experiment
        
        Args:
            experiment_id: Experiment being tracked
            db_manager: Database manager for persistence
            logger: Logger instance
        """
        self.experiment_id = experiment_id
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Progress tracking
        self.start_time = datetime.now()
        self.current_stage = "initializing"
        self.progress = 0.0
        self.eta = None
        
        # History for ETA calculation  
        self.progress_history: deque = deque(maxlen=100)
        self.stage_start_times: Dict[str, datetime] = {}
        
        # Metrics tracking
        self.api_calls_count = 0
        self.total_cost = 0.0
        self.error_count = 0
        self.success_count = 0
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.last_update = datetime.now()
        
    async def initialize(self) -> None:
        """Initialize tracking for experiment start"""
        self.start_time = datetime.now()
        self.stage_start_times[self.current_stage] = self.start_time
        
        await self.update_progress(
            stage="initialized",
            progress=0.0,
            details={"started_at": self.start_time.isoformat()}
        )
        
        self.logger.info(f"Initialized progress tracking for {self.experiment_id}")
    
    async def update_progress(
        self,
        stage: str,
        progress: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update experiment progress
        
        Args:
            stage: Current execution stage
            progress: Progress as float 0.0 to 1.0
            details: Additional stage details
        """
        now = datetime.now()
        
        # Update stage if changed
        if stage != self.current_stage:
            self.stage_start_times[stage] = now
            self.current_stage = stage
        
        self.progress = max(0.0, min(1.0, progress))
        
        # Create progress snapshot
        snapshot = ProgressSnapshot(
            timestamp=now,
            stage=stage,
            progress=self.progress,
            details=details or {},
            api_calls_count=self.api_calls_count,
            cost_incurred=self.total_cost,
            error_rate=self._calculate_error_rate(),
            response_time_avg=self._calculate_avg_response_time(),
            success_rate=self._calculate_success_rate()
        )
        
        # Add system metrics
        try:
            process = psutil.Process()
            snapshot.cpu_percent = process.cpu_percent()
            snapshot.memory_mb = process.memory_info().rss / 1024 / 1024
        except Exception as e:
            self.logger.warning(f"Failed to get system metrics: {e}")
        
        # Store in history
        self.progress_history.append(snapshot)
        
        # Calculate ETA
        self.eta = self._calculate_eta()
        
        # Save to database
        await self._save_progress_to_db(snapshot)
        
        # Log significant progress updates
        if progress == 0.0 or progress == 1.0 or progress - getattr(self, '_last_logged_progress', 0) >= 0.1:
            self.logger.info(f"Experiment {self.experiment_id}: {stage} - {progress*100:.1f}% complete")
            self._last_logged_progress = progress
    
    def record_api_call(
        self,
        response_time: float,
        cost: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Record an API call for metrics tracking
        
        Args:
            response_time: Response time in seconds
            cost: Cost of the API call
            success: Whether the call succeeded
            error: Error message if failed
        """
        self.api_calls_count += 1
        self.total_cost += cost
        self.response_times.append(response_time)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            if error:
                self.logger.warning(f"API call failed: {error}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current progress and performance metrics"""
        return {
            'experiment_id': self.experiment_id,
            'stage': self.current_stage,
            'progress': self.progress,
            'eta': self.eta.isoformat() if self.eta else None,
            'elapsed_time': str(datetime.now() - self.start_time),
            'api_calls': self.api_calls_count,
            'total_cost': self.total_cost,
            'success_rate': self._calculate_success_rate(),
            'error_rate': self._calculate_error_rate(),
            'avg_response_time': self._calculate_avg_response_time()
        }
    
    def _calculate_eta(self) -> Optional[datetime]:
        """Calculate estimated time to completion"""
        if self.progress <= 0.0:
            return None
        
        if len(self.progress_history) < 2:
            return None
        
        # Calculate progress rate from recent history
        recent_snapshots = list(self.progress_history)[-10:]  # Last 10 snapshots
        if len(recent_snapshots) < 2:
            return None
        
        time_span = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds()
        if time_span <= 0:
            return None
        
        progress_change = recent_snapshots[-1].progress - recent_snapshots[0].progress
        if progress_change <= 0:
            return None
        
        progress_rate = progress_change / time_span  # Progress per second
        remaining_progress = 1.0 - self.progress
        
        if progress_rate > 0:
            eta_seconds = remaining_progress / progress_rate
            return datetime.now() + timedelta(seconds=eta_seconds)
        
        return None
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        total_calls = self.success_count + self.error_count
        return self.error_count / total_calls if total_calls > 0 else 0.0
    
    def _calculate_success_rate(self) -> float:
        """Calculate current success rate"""
        total_calls = self.success_count + self.error_count
        return self.success_count / total_calls if total_calls > 0 else 1.0
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    async def _save_progress_to_db(self, snapshot: ProgressSnapshot) -> None:
        """Save progress snapshot to database"""
        try:
            await self.db_manager.save_progress_snapshot(
                self.experiment_id, snapshot
            )
        except Exception as e:
            self.logger.error(f"Failed to save progress to database: {e}")


class ResourceMonitor:
    """
    System-wide resource monitoring for all experiments
    """
    
    def __init__(
        self,
        update_interval: int = 60,  # seconds
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize resource monitor
        
        Args:
            update_interval: Seconds between resource updates
            logger: Logger instance
        """
        self.update_interval = update_interval
        self.logger = logger or logging.getLogger(__name__)
        
        # Resource tracking
        self.current_usage = None
        self.usage_history: deque = deque(maxlen=1440)  # 24 hours at 1min intervals
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 85.0,
            'memory_percent': 90.0,
            'disk_percent': 95.0,
            'api_calls_per_minute': 1000,
            'cost_per_hour': 100.0
        }
        
        self.alerts: List[PerformanceAlert] = []
        self._monitoring_task = None
        self._start_time = datetime.now()
    
    async def start_monitoring(self) -> None:
        """Start background resource monitoring"""
        if self._monitoring_task:
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started resource monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        self.logger.info("Stopped resource monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Collect resource metrics
                usage = self._collect_resource_metrics()
                self.current_usage = usage
                self.usage_history.append((datetime.now(), usage))
                
                # Check for alerts
                await self._check_alerts(usage)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    def _collect_resource_metrics(self) -> ResourceUsage:
        """Collect current resource usage metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            return ResourceUsage(
                cpu_percent=cpu_percent,
                memory_used_gb=(memory.total - memory.available) / 1024**3,
                memory_available_gb=memory.available / 1024**3,
                disk_used_gb=disk.used / 1024**3,
                disk_available_gb=disk.free / 1024**3,
                network_sent_mb=network.bytes_sent / 1024**2,
                network_received_mb=network.bytes_recv / 1024**2
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
            return ResourceUsage(0, 0, 0, 0, 0, 0, 0)
    
    async def _check_alerts(self, usage: ResourceUsage) -> None:
        """Check for performance alerts"""
        alerts_triggered = []
        
        # CPU alert
        if usage.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts_triggered.append(PerformanceAlert(
                experiment_id="system",
                alert_type="resource",
                severity="high" if usage.cpu_percent > 95 else "medium",
                message=f"High CPU usage: {usage.cpu_percent:.1f}%",
                timestamp=datetime.now(),
                details={'cpu_percent': usage.cpu_percent}
            ))
        
        # Memory alert
        if usage.memory_usage_percent > self.alert_thresholds['memory_percent']:
            alerts_triggered.append(PerformanceAlert(
                experiment_id="system",
                alert_type="resource",
                severity="critical" if usage.memory_usage_percent > 98 else "high",
                message=f"High memory usage: {usage.memory_usage_percent:.1f}%",
                timestamp=datetime.now(),
                details={'memory_percent': usage.memory_usage_percent}
            ))
        
        # Disk alert
        if usage.disk_usage_percent > self.alert_thresholds['disk_percent']:
            alerts_triggered.append(PerformanceAlert(
                experiment_id="system",
                alert_type="resource", 
                severity="critical",
                message=f"High disk usage: {usage.disk_usage_percent:.1f}%",
                timestamp=datetime.now(),
                details={'disk_percent': usage.disk_usage_percent}
            ))
        
        # Add new alerts
        self.alerts.extend(alerts_triggered)
        
        # Log critical alerts
        for alert in alerts_triggered:
            if alert.severity == "critical":
                self.logger.critical(alert.message)
            elif alert.severity == "high":
                self.logger.error(alert.message)
            else:
                self.logger.warning(alert.message)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        if not self.current_usage:
            return {}
        
        uptime = datetime.now() - self._start_time
        
        return {
            'cpu_percent': self.current_usage.cpu_percent,
            'memory_usage_percent': self.current_usage.memory_usage_percent,
            'disk_usage_percent': self.current_usage.disk_usage_percent,
            'network_sent_mb': self.current_usage.network_sent_mb,
            'network_received_mb': self.current_usage.network_received_mb,
            'uptime_seconds': uptime.total_seconds(),
            'active_alerts': len([a for a in self.alerts if not a.acknowledged]),
            'last_update': datetime.now().isoformat()
        }
    
    def get_usage_history(self, hours: int = 24) -> List[Tuple[datetime, ResourceUsage]]:
        """Get resource usage history"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [(timestamp, usage) for timestamp, usage in self.usage_history
                if timestamp >= cutoff]


class ExperimentMonitor:
    """
    Main monitoring system coordinating progress tracking and resource monitoring
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize experiment monitor
        
        Args:
            db_manager: Database manager for persistence
            logger: Logger instance
        """
        self.db_manager = db_manager
        self.logger = logger or logging.getLogger(__name__)
        
        # Component managers
        self.progress_trackers: Dict[str, ProgressTracker] = {}
        self.resource_monitor = ResourceMonitor(logger=self.logger)
        
        # Monitoring state
        self._monitoring_active = False
    
    async def start_monitoring(self) -> None:
        """Start all monitoring systems"""
        if self._monitoring_active:
            return
        
        await self.resource_monitor.start_monitoring()
        self._monitoring_active = True
        
        self.logger.info("Started experiment monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop all monitoring systems"""
        if not self._monitoring_active:
            return
        
        await self.resource_monitor.stop_monitoring()
        self._monitoring_active = False
        
        self.logger.info("Stopped experiment monitoring")
    
    def create_progress_tracker(self, experiment_id: str) -> ProgressTracker:
        """Create progress tracker for experiment"""
        if experiment_id in self.progress_trackers:
            return self.progress_trackers[experiment_id]
        
        tracker = ProgressTracker(
            experiment_id=experiment_id,
            db_manager=self.db_manager,
            logger=self.logger
        )
        
        self.progress_trackers[experiment_id] = tracker
        return tracker
    
    async def remove_progress_tracker(self, experiment_id: str) -> None:
        """Remove progress tracker when experiment completes"""
        if experiment_id in self.progress_trackers:
            del self.progress_trackers[experiment_id]
            self.logger.info(f"Removed progress tracker for {experiment_id}")
    
    async def get_current_progress(self, experiment_id: str) -> Dict[str, Any]:
        """Get current progress for experiment"""
        if experiment_id not in self.progress_trackers:
            return {}
        
        tracker = self.progress_trackers[experiment_id]
        return tracker.get_current_metrics()
    
    async def update_experiment_metrics(self, experiment_id: str) -> None:
        """Update metrics for a running experiment"""
        if experiment_id not in self.progress_trackers:
            return
        
        # This would be called periodically to update metrics
        # The actual implementation would depend on how we integrate
        # with the evaluation runner to get real-time metrics
        pass
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring data for dashboard"""
        # Current experiments
        experiments = {}
        for exp_id, tracker in self.progress_trackers.items():
            experiments[exp_id] = tracker.get_current_metrics()
        
        # System resources
        system_metrics = self.resource_monitor.get_current_metrics()
        
        # Recent alerts
        recent_alerts = [
            {
                'experiment_id': alert.experiment_id,
                'type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'acknowledged': alert.acknowledged
            }
            for alert in self.resource_monitor.alerts[-10:]  # Last 10 alerts
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'experiments': experiments,
            'system_metrics': system_metrics,
            'alerts': recent_alerts,
            'monitoring_status': 'active' if self._monitoring_active else 'inactive'
        }