"""
Phase 8: Advanced Extension Framework
Custom extension development, profiling tools, and plugin marketplace
for the Confucius SDK
"""

import json
import asyncio
import time
import inspect
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import importlib.util


# ============================================================================
# EXTENSION DEVELOPMENT FRAMEWORK
# ============================================================================

class ExtensionCategory(Enum):
    """Categories for extensions"""
    DATA_PROCESSING = "data_processing"
    INFERENCE = "inference"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    INTEGRATION = "integration"
    CUSTOM = "custom"


@dataclass
class ExtensionMetadata:
    """Metadata for an extension"""
    name: str
    version: str
    author: str
    description: str
    category: ExtensionCategory
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    min_sdk_version: str = "1.0.0"
    max_sdk_version: str = "2.0.0"
    enabled: bool = True
    compatibility_score: float = 1.0


@dataclass
class ExtensionMetrics:
    """Performance metrics for an extension"""
    name: str
    calls: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')
    max_duration_ms: float = 0.0
    errors: int = 0
    last_error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def avg_duration_ms(self) -> float:
        """Get average duration"""
        return self.total_duration_ms / max(self.calls, 1)


class AdvancedExtension(ABC):
    """Base class for advanced extensions"""
    
    def __init__(self, metadata: ExtensionMetadata):
        self.metadata = metadata
        self.metrics = ExtensionMetrics(name=metadata.name)
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize extension"""
        pass
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute extension logic"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown extension"""
        pass
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure extension with parameters"""
        # Validate config against schema
        for key, value in config.items():
            if key not in self.metadata.config_schema:
                return False
        
        self.config = config
        return True
    
    def get_metrics(self) -> ExtensionMetrics:
        """Get performance metrics"""
        return self.metrics


# ============================================================================
# CUSTOM EXTENSION TEMPLATES
# ============================================================================

class DataProcessingExtension(AdvancedExtension):
    """Template for data processing extensions"""
    
    async def initialize(self) -> None:
        """Initialize data processor"""
        pass
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data"""
        return await self._process(input_data)
    
    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method for custom processing"""
        return data
    
    async def shutdown(self) -> None:
        """Cleanup"""
        pass


class InferenceExtension(AdvancedExtension):
    """Template for inference extensions"""
    
    async def initialize(self) -> None:
        """Initialize inference engine"""
        pass
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference"""
        return await self._infer(input_data)
    
    async def _infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method for custom inference"""
        return {"result": None}
    
    async def shutdown(self) -> None:
        """Cleanup"""
        pass


class MonitoringExtension(AdvancedExtension):
    """Template for monitoring extensions"""
    
    async def initialize(self) -> None:
        """Initialize monitoring"""
        pass
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor system"""
        return await self._monitor(input_data)
    
    async def _monitor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method for custom monitoring"""
        return {"metrics": {}}
    
    async def shutdown(self) -> None:
        """Cleanup"""
        pass


# ============================================================================
# EXTENSION PROFILING & ANALYSIS
# ============================================================================

@dataclass
class ProfileSample:
    """Single profile sample"""
    timestamp: str
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    input_size_bytes: int
    output_size_bytes: int


class ExtensionProfiler:
    """Profiles extension performance"""
    
    def __init__(self):
        self.profiles: Dict[str, List[ProfileSample]] = {}
        self.active_profiles: Dict[str, float] = {}
    
    def start_profile(self, extension_name: str) -> None:
        """Start profiling an extension"""
        self.active_profiles[extension_name] = time.time()
    
    async def end_profile(
        self,
        extension_name: str,
        input_size_bytes: int = 0,
        output_size_bytes: int = 0,
    ) -> Optional[ProfileSample]:
        """End profiling and record sample"""
        if extension_name not in self.active_profiles:
            return None
        
        duration_ms = (time.time() - self.active_profiles[extension_name]) * 1000
        
        # Simulated memory tracking
        import random
        memory_before = 100 + random.uniform(-10, 10)
        memory_after = memory_before + random.uniform(-5, 20)
        memory_peak = max(memory_before, memory_after) + random.uniform(0, 10)
        
        sample = ProfileSample(
            timestamp=datetime.now().isoformat(),
            duration_ms=duration_ms,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_peak_mb=memory_peak,
            input_size_bytes=input_size_bytes,
            output_size_bytes=output_size_bytes,
        )
        
        if extension_name not in self.profiles:
            self.profiles[extension_name] = []
        
        self.profiles[extension_name].append(sample)
        del self.active_profiles[extension_name]
        
        return sample
    
    def get_profile_stats(self, extension_name: str, limit: int = 100) -> Dict[str, Any]:
        """Get profiling statistics"""
        if extension_name not in self.profiles:
            return {}
        
        samples = self.profiles[extension_name][-limit:]
        if not samples:
            return {}
        
        durations = [s.duration_ms for s in samples]
        memories = [s.memory_peak_mb for s in samples]
        
        return {
            "extension": extension_name,
            "sample_count": len(samples),
            "duration": {
                "min_ms": min(durations),
                "max_ms": max(durations),
                "avg_ms": sum(durations) / len(durations),
                "median_ms": sorted(durations)[len(durations) // 2],
            },
            "memory": {
                "min_mb": min(memories),
                "max_mb": max(memories),
                "avg_mb": sum(memories) / len(memories),
                "peak_mb": max(memories),
            },
            "throughput": {
                "calls_per_second": len(samples) / (max(s.duration_ms for s in samples) / 1000 + 0.001),
                "avg_input_kb": sum(s.input_size_bytes for s in samples) / len(samples) / 1024,
                "avg_output_kb": sum(s.output_size_bytes for s in samples) / len(samples) / 1024,
            },
        }


# ============================================================================
# PLUGIN MARKETPLACE
# ============================================================================

@dataclass
class PluginListing:
    """Plugin in the marketplace"""
    id: str
    name: str
    version: str
    author: str
    category: ExtensionCategory
    description: str
    downloads: int = 0
    rating: float = 0.0
    reviews: int = 0
    dependencies: List[str] = field(default_factory=list)
    repository_url: str = ""
    license: str = "MIT"
    verified: bool = False
    install_script: str = ""


class PluginMarketplace:
    """Manages plugin discovery and installation"""
    
    def __init__(self, marketplace_dir: str = "./plugins"):
        self.marketplace_dir = Path(marketplace_dir)
        self.marketplace_dir.mkdir(exist_ok=True)
        self.plugins: Dict[str, PluginListing] = {}
        self.installed_plugins: Dict[str, str] = {}  # name -> version
        self._load_marketplace()
    
    def _load_marketplace(self) -> None:
        """Load marketplace metadata"""
        marketplace_file = self.marketplace_dir / "marketplace.json"
        
        if marketplace_file.exists():
            with open(marketplace_file) as f:
                data = json.load(f)
                for plugin_data in data.get("plugins", []):
                    plugin = PluginListing(**plugin_data)
                    self.plugins[plugin.id] = plugin
    
    def search_plugins(
        self,
        query: Optional[str] = None,
        category: Optional[ExtensionCategory] = None,
        min_rating: float = 0.0,
        verified_only: bool = False,
    ) -> List[PluginListing]:
        """Search for plugins"""
        results = list(self.plugins.values())
        
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.name.lower() or query_lower in p.description.lower()
            ]
        
        if category:
            results = [p for p in results if p.category == category]
        
        if min_rating > 0:
            results = [p for p in results if p.rating >= min_rating]
        
        if verified_only:
            results = [p for p in results if p.verified]
        
        # Sort by rating and downloads
        results.sort(key=lambda p: (p.rating, p.downloads), reverse=True)
        
        return results
    
    async def install_plugin(self, plugin_id: str) -> bool:
        """Install a plugin"""
        if plugin_id not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_id]
        
        # Check dependencies
        for dep in plugin.dependencies:
            if dep not in self.installed_plugins:
                return False  # Dependency not installed
        
        # Simulate installation
        await asyncio.sleep(0.1)
        
        self.installed_plugins[plugin.name] = plugin.version
        plugin.downloads += 1
        
        return True
    
    async def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin"""
        if plugin_name not in self.installed_plugins:
            return False
        
        del self.installed_plugins[plugin_name]
        return True
    
    def get_installed_plugins(self) -> Dict[str, str]:
        """Get list of installed plugins"""
        return self.installed_plugins.copy()
    
    def rate_plugin(self, plugin_id: str, rating: float, review: str = "") -> bool:
        """Rate a plugin"""
        if plugin_id not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_id]
        
        # Update rating (simple average)
        plugin.rating = (plugin.rating * plugin.reviews + rating) / (plugin.reviews + 1)
        plugin.reviews += 1
        
        return True


# ============================================================================
# EXTENSION REGISTRY & LOADER
# ============================================================================

class AdvancedExtensionRegistry:
    """Advanced registry for managing extensions"""
    
    def __init__(self):
        self.extensions: Dict[str, AdvancedExtension] = {}
        self.profiler = ExtensionProfiler()
        self.marketplace = PluginMarketplace()
        self.metrics: Dict[str, ExtensionMetrics] = {}
    
    def register_extension(self, extension: AdvancedExtension) -> bool:
        """Register a new extension"""
        if extension.metadata.name in self.extensions:
            return False
        
        self.extensions[extension.metadata.name] = extension
        self.metrics[extension.metadata.name] = extension.metrics
        return True
    
    async def execute_extension(
        self,
        extension_name: str,
        input_data: Dict[str, Any],
        profile: bool = False,
    ) -> Dict[str, Any]:
        """Execute an extension with profiling"""
        if extension_name not in self.extensions:
            return {"error": f"Extension {extension_name} not found"}
        
        extension = self.extensions[extension_name]
        
        if profile:
            self.profiler.start_profile(extension_name)
        
        start_time = time.time()
        
        try:
            result = await extension.execute(input_data)
            
            duration_ms = (time.time() - start_time) * 1000
            extension.metrics.calls += 1
            extension.metrics.total_duration_ms += duration_ms
            extension.metrics.min_duration_ms = min(
                extension.metrics.min_duration_ms, duration_ms
            )
            extension.metrics.max_duration_ms = max(
                extension.metrics.max_duration_ms, duration_ms
            )
            
            if profile:
                input_size = len(json.dumps(input_data).encode())
                output_size = len(json.dumps(result).encode())
                await self.profiler.end_profile(
                    extension_name,
                    input_size_bytes=input_size,
                    output_size_bytes=output_size,
                )
            
            return result
        
        except Exception as e:
            extension.metrics.errors += 1
            extension.metrics.last_error = str(e)
            return {"error": str(e)}
    
    def load_extension_from_file(self, filepath: str) -> Optional[AdvancedExtension]:
        """Load extension from Python file"""
        spec = importlib.util.spec_from_file_location("extension_module", filepath)
        if spec.loader is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find AdvancedExtension subclass
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                issubclass(obj, AdvancedExtension) and
                obj is not AdvancedExtension):
                return obj(ExtensionMetadata(name="unknown", version="1.0.0", author="unknown", description=""))
        
        return None
    
    def get_extension_info(self, extension_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an extension"""
        if extension_name not in self.extensions:
            return None
        
        extension = self.extensions[extension_name]
        metrics = self.metrics[extension_name]
        
        return {
            "metadata": asdict(extension.metadata),
            "metrics": asdict(metrics),
            "config": extension.config,
        }
    
    def list_extensions(self) -> List[Dict[str, Any]]:
        """List all registered extensions"""
        return [
            self.get_extension_info(name)
            for name in self.extensions.keys()
        ]


# ============================================================================
# EXAMPLE CUSTOM EXTENSIONS
# ============================================================================

class TextNormalizationExtension(DataProcessingExtension):
    """Example: Text normalization extension"""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="text_normalization",
            version="1.0.0",
            author="confucius-sdk",
            description="Normalizes text input (lowercase, trim, etc.)",
            category=ExtensionCategory.DATA_PROCESSING,
            config_schema={
                "lowercase": {"type": "boolean", "default": True},
                "trim_whitespace": {"type": "boolean", "default": True},
            },
        )
        super().__init__(metadata)
    
    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize text"""
        text = data.get("text", "")
        
        if self.config.get("lowercase", True):
            text = text.lower()
        
        if self.config.get("trim_whitespace", True):
            text = " ".join(text.split())
        
        return {"normalized_text": text}


class LatencyMonitoringExtension(MonitoringExtension):
    """Example: Latency monitoring extension"""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="latency_monitoring",
            version="1.0.0",
            author="confucius-sdk",
            description="Monitors and reports latency metrics",
            category=ExtensionCategory.MONITORING,
            config_schema={
                "threshold_ms": {"type": "number", "default": 100},
            },
        )
        super().__init__(metadata)
    
    async def _monitor(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor latency"""
        latency_ms = data.get("latency_ms", 0)
        threshold = self.config.get("threshold_ms", 100)
        
        return {
            "latency_ms": latency_ms,
            "status": "slow" if latency_ms > threshold else "fast",
            "within_threshold": latency_ms <= threshold,
        }


# ============================================================================
# DEMO & TESTING
# ============================================================================

async def demo_advanced_framework():
    """Demonstrate advanced extension framework"""
    print("\n" + "="*80)
    print("PHASE 8: ADVANCED EXTENSION FRAMEWORK - DEMO")
    print("="*80)
    
    # Initialize registry
    registry = AdvancedExtensionRegistry()
    
    print("\n1. Custom Extension Registration")
    print("-" * 80)
    
    # Register custom extensions
    text_ext = TextNormalizationExtension()
    text_ext.configure({"lowercase": True, "trim_whitespace": True})
    registry.register_extension(text_ext)
    
    monitor_ext = LatencyMonitoringExtension()
    monitor_ext.configure({"threshold_ms": 100})
    registry.register_extension(monitor_ext)
    
    print(f"   Registered extensions: {list(registry.extensions.keys())}")
    
    print("\n2. Extension Execution & Profiling")
    print("-" * 80)
    
    # Execute text extension with profiling
    result = await registry.execute_extension(
        "text_normalization",
        {"text": "  HELLO  WORLD  "},
        profile=True,
    )
    print(f"   Text normalization result: {result}")
    
    # Execute monitor extension
    result = await registry.execute_extension(
        "latency_monitoring",
        {"latency_ms": 75},
        profile=True,
    )
    print(f"   Latency monitoring result: {result}")
    
    print("\n3. Extension Performance Analysis")
    print("-" * 80)
    
    # Get profiling stats
    for ext_name in ["text_normalization", "latency_monitoring"]:
        stats = registry.profiler.get_profile_stats(ext_name)
        if stats:
            print(f"   {ext_name}:")
            print(f"     Duration: {stats['duration']['avg_ms']:.2f}ms (avg)")
            print(f"     Memory: {stats['memory']['avg_mb']:.1f}MB (avg)")
            print(f"     Throughput: {stats['throughput']['calls_per_second']:.1f} calls/sec")
    
    print("\n4. Extension Metrics")
    print("-" * 80)
    
    for ext_name, ext in registry.extensions.items():
        metrics = ext.get_metrics()
        print(f"   {ext_name}:")
        print(f"     Calls: {metrics.calls}")
        print(f"     Avg latency: {metrics.avg_duration_ms():.2f}ms")
        print(f"     Errors: {metrics.errors}")
    
    print("\n5. Plugin Marketplace")
    print("-" * 80)
    
    marketplace = registry.marketplace
    
    # Add sample plugins to marketplace
    sample_plugins = [
        PluginListing(
            id="json_validator",
            name="JSON Validator",
            version="1.2.0",
            author="community",
            category=ExtensionCategory.DATA_PROCESSING,
            description="Validates JSON data structures",
            rating=4.8,
            reviews=42,
            downloads=1500,
            verified=True,
        ),
        PluginListing(
            id="gpu_acceleration",
            name="GPU Acceleration",
            version="2.0.0",
            author="nvidia-partner",
            category=ExtensionCategory.OPTIMIZATION,
            description="Accelerates inference on NVIDIA GPUs",
            rating=4.9,
            reviews=156,
            downloads=3200,
            verified=True,
        ),
    ]
    
    for plugin in sample_plugins:
        marketplace.plugins[plugin.id] = plugin
    
    # Search plugins
    results = marketplace.search_plugins(category=ExtensionCategory.DATA_PROCESSING)
    print(f"   Data processing plugins: {len(results)}")
    
    results = marketplace.search_plugins(verified_only=True)
    print(f"   Verified plugins: {len(results)}")
    
    for plugin in marketplace.search_plugins(min_rating=4.5)[:3]:
        print(f"   - {plugin.name} (v{plugin.version}) - ⭐ {plugin.rating:.1f} ({plugin.reviews} reviews)")
    
    print("\n6. Plugin Installation")
    print("-" * 80)
    
    # Install a plugin
    success = await marketplace.install_plugin("json_validator")
    print(f"   JSON Validator installed: {success}")
    print(f"   Installed plugins: {marketplace.get_installed_plugins()}")
    
    print("\n7. Extension Registry Information")
    print("-" * 80)
    
    # Get all extension info
    extensions_info = registry.list_extensions()
    print(f"   Total extensions: {len(extensions_info)}")
    
    for ext_info in extensions_info:
        if ext_info:
            print(f"   - {ext_info['metadata']['name']} (v{ext_info['metadata']['version']})")
            print(f"     Category: {ext_info['metadata']['category']}")
            print(f"     Calls: {ext_info['metrics']['calls']}")
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(demo_advanced_framework())
