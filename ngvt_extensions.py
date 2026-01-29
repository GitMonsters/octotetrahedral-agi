"""
NGVT Extension Interface System
Formal extension protocol with hooks and tool chain integration
Enables modular architecture for runtime extension loading and management
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path


class ExtensionPhase(Enum):
    """Lifecycle phases where extensions can hook"""
    PRE_PROMPT = "pre_prompt"         # Before prompt composition
    POST_PROMPT = "post_prompt"       # After prompt composition
    PRE_INFERENCE = "pre_inference"   # Before LLM call
    POST_INFERENCE = "post_inference" # After LLM call
    PRE_ACTION = "pre_action"         # Before action execution
    POST_ACTION = "post_action"       # After action execution
    PRE_SESSION = "pre_session"       # Session initialization
    POST_SESSION = "post_session"     # Session cleanup


class ExtensionPriority(Enum):
    """Execution priority for extensions"""
    CRITICAL = 0    # Must run, errors stop execution
    HIGH = 1        # Should run, errors are logged
    NORMAL = 2      # Standard execution
    LOW = 3         # Optional, best effort
    DEFERRED = 4    # Can run asynchronously


@dataclass
class ExtensionMetadata:
    """Metadata about an extension"""
    name: str
    version: str
    author: str
    description: str
    capabilities: List[str]          # What it can do (e.g., "inference", "logging")
    required_config: List[str]       # Required configuration keys
    compatible_phases: List[ExtensionPhase]
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HookContext:
    """Context passed to extension hooks"""
    phase: ExtensionPhase
    extension_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """Result returned by extension hook"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    modified: bool = False  # Did this hook modify input data?


class ExtensionInterface(ABC):
    """
    Base interface for all NGVT extensions
    Defines the protocol for extension lifecycle and hooks
    """
    
    def __init__(self, metadata: ExtensionMetadata, config: Dict[str, Any] = None):
        self.metadata = metadata
        self.config = config or {}
        self.execution_stats = {
            "total_hooks": 0,
            "successful_hooks": 0,
            "failed_hooks": 0,
            "total_execution_time_ms": 0.0,
        }
    
    async def initialize(self) -> None:
        """Initialize extension - called once at startup"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup extension - called once at shutdown"""
        pass
    
    async def on_phase(self, context: HookContext) -> HookResult:
        """
        Main hook called at specific phases
        Override in subclasses to handle specific phases
        """
        return HookResult(success=True)
    
    async def handle_pre_prompt(self, context: HookContext) -> HookResult:
        """Hook before prompt composition"""
        return HookResult(success=True)
    
    async def handle_post_prompt(self, context: HookContext) -> HookResult:
        """Hook after prompt composition"""
        return HookResult(success=True)
    
    async def handle_pre_inference(self, context: HookContext) -> HookResult:
        """Hook before LLM inference"""
        return HookResult(success=True)
    
    async def handle_post_inference(self, context: HookContext) -> HookResult:
        """Hook after LLM inference"""
        return HookResult(success=True)
    
    async def handle_pre_action(self, context: HookContext) -> HookResult:
        """Hook before action execution"""
        return HookResult(success=True)
    
    async def handle_post_action(self, context: HookContext) -> HookResult:
        """Hook after action execution"""
        return HookResult(success=True)
    
    async def handle_pre_session(self, context: HookContext) -> HookResult:
        """Hook at session start"""
        return HookResult(success=True)
    
    async def handle_post_session(self, context: HookContext) -> HookResult:
        """Hook at session end"""
        return HookResult(success=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extension statistics"""
        success_rate = (
            self.execution_stats["successful_hooks"] / max(1, self.execution_stats["total_hooks"])
        )
        avg_time = (
            self.execution_stats["total_execution_time_ms"] / max(1, self.execution_stats["total_hooks"])
        )
        
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "enabled": self.metadata.enabled,
            **self.execution_stats,
            "success_rate": success_rate,
            "avg_execution_time_ms": avg_time,
        }


class LoggingExtension(ExtensionInterface):
    """Built-in extension for logging all phases"""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="LoggingExtension",
            version="1.0.0",
            author="NGVT",
            description="Logs all orchestration phases for debugging",
            capabilities=["logging", "debugging"],
            required_config=[],
            compatible_phases=list(ExtensionPhase),
        )
        super().__init__(metadata)
        self.logs = []
    
    async def initialize(self) -> None:
        """Initialize logging"""
        self.logs = []
    
    async def on_phase(self, context: HookContext) -> HookResult:
        """Log the phase"""
        start_time = time.time()
        
        try:
            log_entry = {
                "timestamp": context.timestamp,
                "phase": context.phase.value,
                "extension": context.extension_name,
            }
            self.logs.append(log_entry)
            
            execution_time = (time.time() - start_time) * 1000
            
            return HookResult(
                success=True,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            return HookResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class MetricsExtension(ExtensionInterface):
    """Built-in extension for collecting metrics"""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="MetricsExtension",
            version="1.0.0",
            author="NGVT",
            description="Collects performance metrics across orchestration",
            capabilities=["metrics", "monitoring"],
            required_config=[],
            compatible_phases=list(ExtensionPhase),
        )
        super().__init__(metadata)
        self.metrics = {}
    
    async def initialize(self) -> None:
        """Initialize metrics collection"""
        self.metrics = {
            "prompt_composition_time_ms": 0,
            "inference_time_ms": 0,
            "action_execution_time_ms": 0,
            "total_session_time_ms": 0,
        }
    
    async def on_phase(self, context: HookContext) -> HookResult:
        """Collect metrics for phase"""
        start_time = time.time()
        
        try:
            phase_name = context.phase.value
            
            # Track phase execution
            if phase_name not in self.metrics:
                self.metrics[phase_name] = 0
            
            # Simulated processing
            await asyncio.sleep(0.001)
            
            execution_time = (time.time() - start_time) * 1000
            
            return HookResult(
                success=True,
                data={
                    "phase": phase_name,
                    "execution_time_ms": execution_time,
                },
                execution_time_ms=execution_time,
            )
        except Exception as e:
            return HookResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class CacheExtension(ExtensionInterface):
    """Built-in extension for caching results"""
    
    def __init__(self):
        metadata = ExtensionMetadata(
            name="CacheExtension",
            version="1.0.0",
            author="NGVT",
            description="Caches inference results for repeated queries",
            capabilities=["caching", "optimization"],
            required_config=[],
            compatible_phases=[
                ExtensionPhase.PRE_INFERENCE,
                ExtensionPhase.POST_INFERENCE,
            ],
        )
        super().__init__(metadata)
        self.cache = {}
    
    async def initialize(self) -> None:
        """Initialize cache"""
        self.cache = {}
    
    async def handle_pre_inference(self, context: HookContext) -> HookResult:
        """Check cache before inference"""
        start_time = time.time()
        
        prompt = context.data.get("prompt", "")
        cache_key = self._make_cache_key(prompt)
        
        if cache_key in self.cache:
            context.data["cached_result"] = self.cache[cache_key]
            
            return HookResult(
                success=True,
                data={"cache_hit": True, "result": self.cache[cache_key]},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        
        return HookResult(
            success=True,
            data={"cache_hit": False},
            execution_time_ms=(time.time() - start_time) * 1000,
        )
    
    async def handle_post_inference(self, context: HookContext) -> HookResult:
        """Cache inference result"""
        start_time = time.time()
        
        prompt = context.data.get("prompt", "")
        result = context.data.get("result")
        
        if prompt and result:
            cache_key = self._make_cache_key(prompt)
            self.cache[cache_key] = result
        
        return HookResult(
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000,
        )
    
    @staticmethod
    def _make_cache_key(prompt: str) -> str:
        """Create cache key from prompt"""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()


class ExtensionRegistry:
    """Manages extension lifecycle and hook execution"""
    
    def __init__(self):
        self.extensions: Dict[str, ExtensionInterface] = {}
        self.hooks_by_phase: Dict[ExtensionPhase, List[str]] = {
            phase: [] for phase in ExtensionPhase
        }
    
    async def register(self, extension: ExtensionInterface) -> None:
        """Register an extension"""
        self.extensions[extension.metadata.name] = extension
        
        # Register for compatible phases
        for phase in extension.metadata.compatible_phases:
            if extension.metadata.name not in self.hooks_by_phase[phase]:
                self.hooks_by_phase[phase].append(extension.metadata.name)
        
        # Initialize extension
        await extension.initialize()
    
    async def unregister(self, extension_name: str) -> None:
        """Unregister an extension"""
        if extension_name in self.extensions:
            ext = self.extensions[extension_name]
            await ext.cleanup()
            
            del self.extensions[extension_name]
            
            # Remove from all phase hooks
            for phase_hooks in self.hooks_by_phase.values():
                if extension_name in phase_hooks:
                    phase_hooks.remove(extension_name)
    
    async def call_phase(
        self,
        phase: ExtensionPhase,
        context: HookContext,
    ) -> List[HookResult]:
        """Call all extensions registered for a phase"""
        results = []
        
        for extension_name in self.hooks_by_phase[phase]:
            if extension_name not in self.extensions:
                continue
            
            extension = self.extensions[extension_name]
            
            if not extension.metadata.enabled:
                continue
            
            try:
                start_time = time.time()
                
                # Call appropriate handler based on phase
                if phase == ExtensionPhase.PRE_PROMPT:
                    result = await extension.handle_pre_prompt(context)
                elif phase == ExtensionPhase.POST_PROMPT:
                    result = await extension.handle_post_prompt(context)
                elif phase == ExtensionPhase.PRE_INFERENCE:
                    result = await extension.handle_pre_inference(context)
                elif phase == ExtensionPhase.POST_INFERENCE:
                    result = await extension.handle_post_inference(context)
                elif phase == ExtensionPhase.PRE_ACTION:
                    result = await extension.handle_pre_action(context)
                elif phase == ExtensionPhase.POST_ACTION:
                    result = await extension.handle_post_action(context)
                elif phase == ExtensionPhase.PRE_SESSION:
                    result = await extension.handle_pre_session(context)
                elif phase == ExtensionPhase.POST_SESSION:
                    result = await extension.handle_post_session(context)
                else:
                    result = await extension.on_phase(context)
                
                # Update extension stats
                extension.execution_stats["total_hooks"] += 1
                if result.success:
                    extension.execution_stats["successful_hooks"] += 1
                else:
                    extension.execution_stats["failed_hooks"] += 1
                extension.execution_stats["total_execution_time_ms"] += result.execution_time_ms
                
                results.append(result)
            
            except Exception as e:
                result = HookResult(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
                extension.execution_stats["total_hooks"] += 1
                extension.execution_stats["failed_hooks"] += 1
                results.append(result)
        
        return results
    
    def get_extension(self, name: str) -> Optional[ExtensionInterface]:
        """Get extension by name"""
        return self.extensions.get(name)
    
    def list_extensions(self) -> List[ExtensionMetadata]:
        """List all registered extensions"""
        return [ext.metadata for ext in self.extensions.values()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all extensions"""
        return {
            name: ext.get_stats()
            for name, ext in self.extensions.items()
        }


class ExtensionToolChain:
    """Builds toolchains from registered extensions"""
    
    def __init__(self, registry: ExtensionRegistry):
        self.registry = registry
        self.toolchains: Dict[str, List[str]] = {}
    
    def create_toolchain(
        self,
        name: str,
        extension_names: List[str],
    ) -> None:
        """Create a named toolchain"""
        # Verify all extensions exist
        for ext_name in extension_names:
            if ext_name not in self.registry.extensions:
                raise ValueError(f"Extension '{ext_name}' not found in registry")
        
        self.toolchains[name] = extension_names
    
    async def execute_toolchain(
        self,
        name: str,
        phase: ExtensionPhase,
        context: HookContext,
    ) -> List[HookResult]:
        """Execute a toolchain"""
        if name not in self.toolchains:
            raise ValueError(f"Toolchain '{name}' not found")
        
        results = []
        
        for ext_name in self.toolchains[name]:
            ext = self.registry.get_extension(ext_name)
            if ext and ext.metadata.enabled:
                result = await self.registry.call_phase(phase, context)
                results.extend(result)
        
        return results
    
    def list_toolchains(self) -> Dict[str, List[str]]:
        """List all toolchains"""
        return self.toolchains.copy()


async def demo_extension_system():
    """Demo extension system"""
    print("\n" + "="*70)
    print("Extension Interface System Demo")
    print("="*70 + "\n")
    
    # Create registry
    registry = ExtensionRegistry()
    
    # Register built-in extensions
    print("[*] Registering built-in extensions...")
    await registry.register(LoggingExtension())
    await registry.register(MetricsExtension())
    await registry.register(CacheExtension())
    
    print(f"[✓] Registered {len(registry.extensions)} extensions\n")
    
    # List registered extensions
    print("[*] Registered Extensions:")
    for metadata in registry.list_extensions():
        print(f"  - {metadata.name} v{metadata.version}: {metadata.description}")
    
    # Create toolchain
    print("\n[*] Creating extension toolchain...")
    toolchain = ExtensionToolChain(registry)
    toolchain.create_toolchain(
        "full_pipeline",
        ["LoggingExtension", "MetricsExtension", "CacheExtension"]
    )
    print("[✓] Toolchain 'full_pipeline' created\n")
    
    # Simulate orchestration phases
    print("[*] Simulating orchestration phases...\n")
    
    phases_to_test = [
        ExtensionPhase.PRE_SESSION,
        ExtensionPhase.PRE_PROMPT,
        ExtensionPhase.POST_PROMPT,
        ExtensionPhase.PRE_INFERENCE,
        ExtensionPhase.POST_INFERENCE,
        ExtensionPhase.PRE_ACTION,
        ExtensionPhase.POST_ACTION,
        ExtensionPhase.POST_SESSION,
    ]
    
    for phase in phases_to_test:
        context = HookContext(
            phase=phase,
            extension_name="DemoExtension",
            data={
                "prompt": "Test prompt for inference",
                "result": {"output": "Test result"},
            }
        )
        
        results = await registry.call_phase(phase, context)
        
        successful = sum(1 for r in results if r.success)
        print(f"[{phase.value:20s}] {successful}/{len(results)} extensions successful")
    
    # Show statistics
    print("\n" + "-"*70)
    print("Extension Statistics")
    print("-"*70 + "\n")
    
    stats = registry.get_stats()
    for ext_name, ext_stats in stats.items():
        print(f"{ext_name}:")
        print(f"  Total hooks: {ext_stats['total_hooks']}")
        print(f"  Success rate: {ext_stats['success_rate']:.1%}")
        print(f"  Avg execution: {ext_stats['avg_execution_time_ms']:.2f}ms")
    
    # Show cache state
    print("\n[*] Cache Extension state:")
    cache_ext = registry.get_extension("CacheExtension")
    print(f"  Cache entries: {len(cache_ext.cache)}")
    
    # Show logging
    print("\n[*] Logging Extension events:")
    log_ext = registry.get_extension("LoggingExtension")
    for i, log in enumerate(log_ext.logs[:5], 1):
        print(f"  {i}. {log['phase']}")
    if len(log_ext.logs) > 5:
        print(f"  ... and {len(log_ext.logs) - 5} more")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(demo_extension_system())
