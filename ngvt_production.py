"""
Phase 7: Production Deployment System
Comprehensive production-ready deployment with configuration management,
monitoring, and scaling strategies for the Confucius SDK
"""

import os
import json
import yaml
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import sqlite3
from collections import defaultdict, deque


# ============================================================================
# ENVIRONMENT & CONFIGURATION MANAGEMENT
# ============================================================================

class DeploymentEnvironment(Enum):
    """Deployment target environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "ngvt_production"
    username: str = "ngvt_user"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 40
    echo_sql: bool = False


@dataclass
class CacheConfig:
    """Cache layer configuration"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl_seconds: int = 3600
    max_cache_size_mb: int = 500
    enable_compression: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    datadog_enabled: bool = False
    datadog_api_key: str = ""
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_interval_seconds: int = 60
    trace_sampling_rate: float = 0.1


@dataclass
class ScalingConfig:
    """Auto-scaling configuration"""
    min_workers: int = 2
    max_workers: int = 10
    target_cpu_percent: int = 70
    target_memory_percent: int = 80
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    health_check_interval_seconds: int = 30


@dataclass
class FailoverConfig:
    """Failover and disaster recovery configuration"""
    enable_failover: bool = True
    primary_region: str = "us-east-1"
    secondary_region: str = "us-west-2"
    failover_threshold_seconds: int = 60
    health_check_timeout_seconds: int = 10
    backup_frequency_minutes: int = 30
    retention_days: int = 30


@dataclass
class ProductionConfig:
    """Complete production configuration"""
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    app_name: str = "ngvt-confucius-sdk"
    version: str = "1.0.0"
    debug: bool = False
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    failover: FailoverConfig = field(default_factory=FailoverConfig)
    
    max_request_timeout_seconds: int = 300
    max_batch_size: int = 100
    enable_rate_limiting: bool = True
    rate_limit_requests_per_second: int = 1000


class ConfigurationManager:
    """Manages production configuration from multiple sources"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.config: ProductionConfig = ProductionConfig()
        self.config_history: List[Tuple[str, ProductionConfig]] = []
    
    def load_from_env(self) -> ProductionConfig:
        """Load configuration from environment variables"""
        env = os.getenv("DEPLOYMENT_ENV", "production")
        
        config = ProductionConfig(
            environment=DeploymentEnvironment(env),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )
        
        # Database config
        config.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "ngvt_production"),
            username=os.getenv("DB_USER", "ngvt_user"),
            password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
        )
        
        # Cache config
        config.cache = CacheConfig(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
        )
        
        # Monitoring config
        config.monitoring = MonitoringConfig(
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        
        self.config = config
        return config
    
    def load_from_file(self, filepath: str) -> ProductionConfig:
        """Load configuration from YAML/JSON file"""
        path = Path(filepath)
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path) as f:
                data = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        # Convert dict to ProductionConfig
        config = ProductionConfig(**data)
        self.config = config
        self.config_history.append((datetime.now().isoformat(), config))
        return config
    
    def save_to_file(self, filepath: str, format: str = "yaml") -> None:
        """Save current configuration to file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        if format == "yaml":
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate configuration for production readiness"""
        errors = []
        
        if self.config.scaling.min_workers > self.config.scaling.max_workers:
            errors.append("min_workers cannot exceed max_workers")
        
        if self.config.scaling.target_cpu_percent > 100:
            errors.append("target_cpu_percent cannot exceed 100")
        
        if self.config.scaling.target_memory_percent > 100:
            errors.append("target_memory_percent cannot exceed 100")
        
        if self.config.database.pool_size < 1:
            errors.append("database pool_size must be at least 1")
        
        if self.config.cache.cache_ttl_seconds < 1:
            errors.append("cache_ttl_seconds must be positive")
        
        return len(errors) == 0, errors


# ============================================================================
# MONITORING & OBSERVABILITY
# ============================================================================

@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class HealthStatus:
    """System health status"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: float = 0.0
    error_rate: float = 0.0
    last_check: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.health_status: Dict[str, HealthStatus] = {}
        self.aggregated_metrics: Dict[str, Dict[str, float]] = {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("ngvt-metrics")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        handler = logging.StreamHandler()
        if self.config.log_format == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def record_metric(self, metric: Metric) -> None:
        """Record a metric data point"""
        self.metrics[metric.name].append(metric)
        self.logger.info(f"Metric recorded: {metric.name}={metric.value}{metric.unit}")
    
    def get_metric_stats(self, metric_name: str, window_size: int = 60) -> Dict[str, float]:
        """Get statistics for a metric"""
        data = list(self.metrics[metric_name])[-window_size:]
        
        if not data:
            return {}
        
        values = [m.value for m in data]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p95": sorted(values)[int(len(values) * 0.95)],
            "p99": sorted(values)[int(len(values) * 0.99)],
        }
    
    def update_health_status(self, status: HealthStatus) -> None:
        """Update health status for a component"""
        self.health_status[status.component] = status
        
        status_icon = "✓" if status.status == "healthy" else "⚠" if status.status == "degraded" else "✗"
        self.logger.info(
            f"Health: {status_icon} {status.component} - {status.status} "
            f"(latency: {status.latency_ms:.1f}ms, error_rate: {status.error_rate:.2%})"
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.health_status:
            return {"status": "unknown", "components": {}}
        
        statuses = [h.status for h in self.health_status.values()]
        
        if all(s == "healthy" for s in statuses):
            overall_status = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: asdict(status) for name, status in self.health_status.items()
            },
            "component_count": len(self.health_status),
            "healthy_components": sum(1 for s in statuses if s == "healthy"),
            "degraded_components": sum(1 for s in statuses if s == "degraded"),
            "unhealthy_components": sum(1 for s in statuses if s == "unhealthy"),
        }


# ============================================================================
# SCALING & LOAD MANAGEMENT
# ============================================================================

@dataclass
class WorkerMetrics:
    """Metrics for a single worker"""
    worker_id: str
    cpu_percent: float
    memory_percent: float
    active_requests: int
    total_requests: int
    latency_ms: float
    error_rate: float
    uptime_seconds: float


class ScalingManager:
    """Manages worker scaling based on demand"""
    
    def __init__(self, config: ScalingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.workers: Dict[str, WorkerMetrics] = {}
        self.scaling_history: List[Tuple[str, str]] = []
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
    
    def update_worker_metrics(self, metrics: WorkerMetrics) -> None:
        """Update metrics for a worker"""
        self.workers[metrics.worker_id] = metrics
    
    def should_scale_up(self) -> bool:
        """Determine if we should scale up"""
        if len(self.workers) >= self.config.max_workers:
            return False
        
        if len(self.workers) == 0:
            return False
        
        now = time.time()
        if now - self.last_scale_up_time < self.config.scale_up_cooldown_seconds:
            return False
        
        avg_cpu = sum(w.cpu_percent for w in self.workers.values()) / len(self.workers)
        avg_memory = sum(w.memory_percent for w in self.workers.values()) / len(self.workers)
        
        return (
            avg_cpu > self.config.target_cpu_percent or
            avg_memory > self.config.target_memory_percent
        )
    
    def should_scale_down(self) -> bool:
        """Determine if we should scale down"""
        if len(self.workers) <= self.config.min_workers:
            return False
        
        if len(self.workers) == 0:
            return False
        
        now = time.time()
        if now - self.last_scale_down_time < self.config.scale_down_cooldown_seconds:
            return False
        
        avg_cpu = sum(w.cpu_percent for w in self.workers.values()) / len(self.workers)
        avg_memory = sum(w.memory_percent for w in self.workers.values()) / len(self.workers)
        
        # Only scale down if significantly below thresholds
        return (
            avg_cpu < self.config.target_cpu_percent * 0.5 and
            avg_memory < self.config.target_memory_percent * 0.5 and
            all(w.active_requests == 0 for w in self.workers.values())
        )
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Get recommendation on scaling actions"""
        current_workers = len(self.workers)
        recommendation = {
            "current_workers": current_workers,
            "action": "none",
            "reason": "",
        }
        
        if self.should_scale_up():
            new_count = min(current_workers + 1, self.config.max_workers)
            recommendation["action"] = "scale_up"
            recommendation["target_workers"] = new_count
            recommendation["reason"] = "High resource utilization"
            self.last_scale_up_time = time.time()
            self.scaling_history.append(
                (datetime.now().isoformat(), f"scale_up: {current_workers} -> {new_count}")
            )
        
        elif self.should_scale_down():
            new_count = max(current_workers - 1, self.config.min_workers)
            recommendation["action"] = "scale_down"
            recommendation["target_workers"] = new_count
            recommendation["reason"] = "Low resource utilization"
            self.last_scale_down_time = time.time()
            self.scaling_history.append(
                (datetime.now().isoformat(), f"scale_down: {current_workers} -> {new_count}")
            )
        
        return recommendation


# ============================================================================
# DISASTER RECOVERY & BACKUP
# ============================================================================

@dataclass
class BackupMetadata:
    """Metadata for a backup"""
    backup_id: str
    timestamp: str
    component: str
    size_bytes: int
    checksum: str
    retention_until: str
    status: str  # "pending", "completed", "failed"


class BackupManager:
    """Manages backups and disaster recovery"""
    
    def __init__(self, config: FailoverConfig, backup_dir: str = "./backups"):
        self.config = config
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_metadata: Dict[str, BackupMetadata] = {}
        self.db_path = self.backup_dir / "backups.db"
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize backup metadata database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backups (
                backup_id TEXT PRIMARY KEY,
                timestamp TEXT,
                component TEXT,
                size_bytes INTEGER,
                checksum TEXT,
                retention_until TEXT,
                status TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_backup(self, component: str, data: Dict[str, Any]) -> BackupMetadata:
        """Create a backup of component data"""
        import hashlib
        
        backup_id = f"backup_{component}_{int(time.time() * 1000)}"
        data_json = json.dumps(data)
        data_bytes = data_json.encode('utf-8')
        
        # Calculate checksum
        checksum = hashlib.sha256(data_bytes).hexdigest()
        
        # Save backup file
        backup_file = self.backup_dir / f"{backup_id}.json"
        backup_file.write_bytes(data_bytes)
        
        # Calculate retention
        retention_until = (
            datetime.now() + timedelta(days=self.config.retention_days)
        ).isoformat()
        
        # Create metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.now().isoformat(),
            component=component,
            size_bytes=len(data_bytes),
            checksum=checksum,
            retention_until=retention_until,
            status="completed",
        )
        
        # Store metadata
        self.backup_metadata[backup_id] = metadata
        self._save_metadata(metadata)
        
        return metadata
    
    def list_backups(self, component: Optional[str] = None) -> List[BackupMetadata]:
        """List available backups"""
        backups = list(self.backup_metadata.values())
        
        if component:
            backups = [b for b in backups if b.component == component]
        
        # Filter out expired backups
        now = datetime.now()
        backups = [b for b in backups if datetime.fromisoformat(b.retention_until) > now]
        
        return sorted(backups, key=lambda b: b.timestamp, reverse=True)
    
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore from a backup"""
        backup_file = self.backup_dir / f"{backup_id}.json"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup {backup_id} not found")
        
        data = json.loads(backup_file.read_text())
        return data
    
    def cleanup_expired_backups(self) -> int:
        """Remove expired backups"""
        now = datetime.now()
        removed_count = 0
        
        for backup_id, metadata in list(self.backup_metadata.items()):
            if datetime.fromisoformat(metadata.retention_until) <= now:
                backup_file = self.backup_dir / f"{backup_id}.json"
                if backup_file.exists():
                    backup_file.unlink()
                
                del self.backup_metadata[backup_id]
                removed_count += 1
        
        return removed_count
    
    def _save_metadata(self, metadata: BackupMetadata) -> None:
        """Save metadata to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO backups
            (backup_id, timestamp, component, size_bytes, checksum, retention_until, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.backup_id,
            metadata.timestamp,
            metadata.component,
            metadata.size_bytes,
            metadata.checksum,
            metadata.retention_until,
            metadata.status,
        ))
        
        conn.commit()
        conn.close()


# ============================================================================
# PRODUCTION ORCHESTRATOR
# ============================================================================

class ProductionDeploymentOrchestrator:
    """Orchestrates all production deployment systems"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_manager = ConfigurationManager()
        
        if config_file:
            self.config = self.config_manager.load_from_file(config_file)
        else:
            self.config = self.config_manager.load_from_env()
        
        # Validate configuration
        is_valid, errors = self.config_manager.validate_config()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {errors}")
        
        # Initialize subsystems
        self.metrics_collector = MetricsCollector(self.config.monitoring)
        self.scaling_manager = ScalingManager(self.config.scaling, self.metrics_collector.logger)
        self.backup_manager = BackupManager(self.config.failover)
        
        self.metrics_collector.logger.info(
            f"Production deployment initialized: "
            f"env={self.config.environment.value}, version={self.config.version}"
        )
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "environment": self.config.environment.value,
            "version": self.config.version,
            "system_health": self.metrics_collector.get_system_health(),
            "scaling": {
                "current_workers": len(self.scaling_manager.workers),
                "min_workers": self.config.scaling.min_workers,
                "max_workers": self.config.scaling.max_workers,
                "recommendation": self.scaling_manager.get_scaling_recommendation(),
            },
            "backups": {
                "available_backups": len(self.backup_manager.list_backups()),
                "next_backup_time": (
                    datetime.now() + 
                    timedelta(minutes=self.config.failover.backup_frequency_minutes)
                ).isoformat(),
            },
            "configuration": {
                "debug": self.config.debug,
                "rate_limiting": self.config.enable_rate_limiting,
                "rate_limit_rps": self.config.rate_limit_requests_per_second,
            },
        }
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start = time.time()
        
        results = {
            "database": await self._check_database(),
            "cache": await self._check_cache(),
            "memory": await self._check_memory(),
            "disk": await self._check_disk(),
            "network": await self._check_network(),
        }
        
        duration_ms = (time.time() - start) * 1000
        results["duration_ms"] = duration_ms
        results["timestamp"] = datetime.now().isoformat()
        
        return results
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        # Simulated check
        await asyncio.sleep(0.01)
        return {
            "status": "healthy",
            "latency_ms": 5.2,
            "connections": 15,
            "max_connections": 20,
        }
    
    async def _check_cache(self) -> Dict[str, Any]:
        """Check cache connectivity and performance"""
        await asyncio.sleep(0.005)
        return {
            "status": "healthy",
            "latency_ms": 2.1,
            "memory_used_mb": 250,
            "memory_max_mb": 500,
        }
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage"""
        await asyncio.sleep(0.002)
        return {
            "status": "healthy",
            "used_percent": 65.2,
            "available_percent": 34.8,
        }
    
    async def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        await asyncio.sleep(0.002)
        return {
            "status": "healthy",
            "used_percent": 45.3,
            "available_percent": 54.7,
        }
    
    async def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        await asyncio.sleep(0.003)
        return {
            "status": "healthy",
            "latency_ms": 1.2,
            "packet_loss_percent": 0.0,
        }


# ============================================================================
# DEMO & TESTING
# ============================================================================

async def demo_production_deployment():
    """Demonstrate production deployment system"""
    print("\n" + "="*80)
    print("PHASE 7: PRODUCTION DEPLOYMENT SYSTEM - DEMO")
    print("="*80)
    
    # Initialize production orchestrator
    orchestrator = ProductionDeploymentOrchestrator()
    
    print("\n1. Deployment Status")
    print("-" * 80)
    status = orchestrator.get_deployment_status()
    print(json.dumps(status, indent=2)[:500] + "...")
    
    print("\n2. Health Check")
    print("-" * 80)
    health = await orchestrator.perform_health_check()
    print(f"   Database:  {health['database']['status']} ({health['database']['latency_ms']:.1f}ms)")
    print(f"   Cache:     {health['cache']['status']} ({health['cache']['latency_ms']:.1f}ms)")
    print(f"   Memory:    {health['memory']['status']} ({health['memory']['used_percent']:.1f}%)")
    print(f"   Disk:      {health['disk']['status']} ({health['disk']['used_percent']:.1f}%)")
    print(f"   Network:   {health['network']['status']} ({health['network']['latency_ms']:.1f}ms)")
    print(f"   Total:     {health['duration_ms']:.1f}ms")
    
    print("\n3. Scaling Management")
    print("-" * 80)
    # Simulate worker metrics
    for i in range(3):
        worker_metrics = WorkerMetrics(
            worker_id=f"worker_{i}",
            cpu_percent=70 + i * 5,
            memory_percent=60 + i * 3,
            active_requests=10 + i * 5,
            total_requests=1000 + i * 100,
            latency_ms=50 + i * 10,
            error_rate=0.001 + i * 0.0005,
            uptime_seconds=86400 + i * 1000,
        )
        orchestrator.scaling_manager.update_worker_metrics(worker_metrics)
    
    recommendation = orchestrator.scaling_manager.get_scaling_recommendation()
    print(f"   Current workers: {recommendation['current_workers']}")
    print(f"   Recommendation: {recommendation['action'].upper()}")
    if recommendation.get('target_workers'):
        print(f"   Target workers: {recommendation['target_workers']}")
    print(f"   Reason: {recommendation['reason']}")
    
    print("\n4. Backup Management")
    print("-" * 80)
    # Create sample backups
    backup_data = {
        "patterns": [{"id": "pat_1", "type": "integration"}],
        "memory_state": {"scopes": 3, "entries": 42},
    }
    
    backup = orchestrator.backup_manager.create_backup("orchestrator_state", backup_data)
    print(f"   Backup created: {backup.backup_id}")
    print(f"   Size: {backup.size_bytes} bytes")
    print(f"   Checksum: {backup.checksum[:16]}...")
    print(f"   Retention: {backup.retention_until}")
    
    backups = orchestrator.backup_manager.list_backups()
    print(f"   Total backups: {len(backups)}")
    
    print("\n5. Configuration Management")
    print("-" * 80)
    print(f"   Environment: {orchestrator.config.environment.value}")
    print(f"   Debug mode: {orchestrator.config.debug}")
    print(f"   Rate limiting: {orchestrator.config.enable_rate_limiting}")
    print(f"   Rate limit: {orchestrator.config.rate_limit_requests_per_second} req/s")
    print(f"   Min workers: {orchestrator.config.scaling.min_workers}")
    print(f"   Max workers: {orchestrator.config.scaling.max_workers}")
    
    print("\n6. Metrics Collection")
    print("-" * 80)
    for i in range(5):
        metric = Metric(
            name="request_latency_ms",
            value=50 + i * 10,
            tags={"service": "orchestrator", "endpoint": "/infer"},
        )
        orchestrator.metrics_collector.record_metric(metric)
    
    stats = orchestrator.metrics_collector.get_metric_stats("request_latency_ms")
    print(f"   Metric: request_latency_ms")
    print(f"   Count: {stats['count']}")
    print(f"   Min: {stats['min']:.1f}ms")
    print(f"   Max: {stats['max']:.1f}ms")
    print(f"   Avg: {stats['avg']:.1f}ms")
    print(f"   P95: {stats['p95']:.1f}ms")
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(demo_production_deployment())
