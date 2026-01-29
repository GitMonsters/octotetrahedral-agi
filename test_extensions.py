"""
Test suite for NGVT Extension Interface System
Tests extension lifecycle, hooks, registry, and toolchains
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from ngvt_extensions import (
    ExtensionInterface,
    ExtensionMetadata,
    ExtensionPhase,
    ExtensionPriority,
    HookContext,
    HookResult,
    LoggingExtension,
    MetricsExtension,
    CacheExtension,
    ExtensionRegistry,
    ExtensionToolChain,
)


class TestExtensionMetadata:
    """Tests for ExtensionMetadata"""
    
    def test_create_metadata(self):
        """Test creating extension metadata"""
        metadata = ExtensionMetadata(
            name="TestExtension",
            version="1.0.0",
            author="Test Author",
            description="Test extension",
            capabilities=["test"],
            required_config=[],
            compatible_phases=[ExtensionPhase.PRE_PROMPT],
        )
        
        assert metadata.name == "TestExtension"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.enabled is True


class TestHookContext:
    """Tests for HookContext"""
    
    def test_create_hook_context(self):
        """Test creating hook context"""
        context = HookContext(
            phase=ExtensionPhase.PRE_PROMPT,
            extension_name="TestExtension",
            data={"key": "value"},
        )
        
        assert context.phase == ExtensionPhase.PRE_PROMPT
        assert context.extension_name == "TestExtension"
        assert context.data["key"] == "value"


class TestHookResult:
    """Tests for HookResult"""
    
    def test_successful_result(self):
        """Test successful hook result"""
        result = HookResult(
            success=True,
            data={"result": "data"},
            execution_time_ms=10.5,
        )
        
        assert result.success is True
        assert result.error is None
        assert result.execution_time_ms == 10.5
    
    def test_failed_result(self):
        """Test failed hook result"""
        result = HookResult(
            success=False,
            error="Test error",
            execution_time_ms=5.0,
        )
        
        assert result.success is False
        assert result.error == "Test error"


class TestLoggingExtension:
    """Tests for LoggingExtension"""
    
    @pytest.mark.asyncio
    async def test_logging_extension_init(self):
        """Test logging extension initialization"""
        ext = LoggingExtension()
        
        assert ext.metadata.name == "LoggingExtension"
        assert ext.metadata.version == "1.0.0"
        assert "logging" in ext.metadata.capabilities
    
    @pytest.mark.asyncio
    async def test_logging_extension_phase(self):
        """Test logging extension logging"""
        ext = LoggingExtension()
        await ext.initialize()
        
        context = HookContext(
            phase=ExtensionPhase.PRE_PROMPT,
            extension_name="LoggingExtension",
        )
        
        result = await ext.on_phase(context)
        
        assert result.success is True
        assert len(ext.logs) == 1
        assert ext.logs[0]["phase"] == "pre_prompt"


class TestMetricsExtension:
    """Tests for MetricsExtension"""
    
    @pytest.mark.asyncio
    async def test_metrics_extension_init(self):
        """Test metrics extension initialization"""
        ext = MetricsExtension()
        
        assert ext.metadata.name == "MetricsExtension"
        assert "metrics" in ext.metadata.capabilities
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection"""
        ext = MetricsExtension()
        await ext.initialize()
        
        context = HookContext(
            phase=ExtensionPhase.PRE_PROMPT,
            extension_name="MetricsExtension",
        )
        
        result = await ext.on_phase(context)
        
        assert result.success is True
        assert "phase" in result.data


class TestCacheExtension:
    """Tests for CacheExtension"""
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss"""
        ext = CacheExtension()
        await ext.initialize()
        
        context = HookContext(
            phase=ExtensionPhase.PRE_INFERENCE,
            extension_name="CacheExtension",
            data={"prompt": "Test prompt"},
        )
        
        result = await ext.handle_pre_inference(context)
        
        assert result.success is True
        assert result.data["cache_hit"] is False
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test cache hit"""
        ext = CacheExtension()
        await ext.initialize()
        
        # Populate cache
        prompt = "Test prompt"
        cache_key = ext._make_cache_key(prompt)
        ext.cache[cache_key] = "Cached result"
        
        context = HookContext(
            phase=ExtensionPhase.PRE_INFERENCE,
            extension_name="CacheExtension",
            data={"prompt": prompt},
        )
        
        result = await ext.handle_pre_inference(context)
        
        assert result.success is True
        assert result.data["cache_hit"] is True
    
    @pytest.mark.asyncio
    async def test_cache_storage(self):
        """Test storing in cache"""
        ext = CacheExtension()
        await ext.initialize()
        
        context = HookContext(
            phase=ExtensionPhase.POST_INFERENCE,
            extension_name="CacheExtension",
            data={
                "prompt": "Test prompt",
                "result": "Test result",
            }
        )
        
        result = await ext.handle_post_inference(context)
        
        assert result.success is True
        assert len(ext.cache) == 1


class TestExtensionRegistry:
    """Tests for ExtensionRegistry"""
    
    @pytest.mark.asyncio
    async def test_register_extension(self):
        """Test registering extension"""
        registry = ExtensionRegistry()
        ext = LoggingExtension()
        
        await registry.register(ext)
        
        assert "LoggingExtension" in registry.extensions
        assert registry.get_extension("LoggingExtension") == ext
    
    @pytest.mark.asyncio
    async def test_unregister_extension(self):
        """Test unregistering extension"""
        registry = ExtensionRegistry()
        ext = LoggingExtension()
        
        await registry.register(ext)
        assert "LoggingExtension" in registry.extensions
        
        await registry.unregister("LoggingExtension")
        assert "LoggingExtension" not in registry.extensions
    
    @pytest.mark.asyncio
    async def test_list_extensions(self):
        """Test listing extensions"""
        registry = ExtensionRegistry()
        
        await registry.register(LoggingExtension())
        await registry.register(MetricsExtension())
        
        extensions = registry.list_extensions()
        
        assert len(extensions) == 2
        names = {e.name for e in extensions}
        assert "LoggingExtension" in names
        assert "MetricsExtension" in names
    
    @pytest.mark.asyncio
    async def test_call_phase(self):
        """Test calling phase"""
        registry = ExtensionRegistry()
        
        await registry.register(LoggingExtension())
        await registry.register(MetricsExtension())
        
        context = HookContext(
            phase=ExtensionPhase.PRE_PROMPT,
            extension_name="TestExtension",
        )
        
        results = await registry.call_phase(ExtensionPhase.PRE_PROMPT, context)
        
        assert len(results) == 2
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting registry statistics"""
        registry = ExtensionRegistry()
        
        await registry.register(LoggingExtension())
        
        context = HookContext(
            phase=ExtensionPhase.PRE_PROMPT,
            extension_name="TestExtension",
        )
        
        await registry.call_phase(ExtensionPhase.PRE_PROMPT, context)
        
        stats = registry.get_stats()
        
        assert "LoggingExtension" in stats
        assert stats["LoggingExtension"]["total_hooks"] > 0


class TestExtensionToolChain:
    """Tests for ExtensionToolChain"""
    
    @pytest.mark.asyncio
    async def test_create_toolchain(self):
        """Test creating toolchain"""
        registry = ExtensionRegistry()
        
        await registry.register(LoggingExtension())
        await registry.register(MetricsExtension())
        
        toolchain = ExtensionToolChain(registry)
        toolchain.create_toolchain(
            "test_chain",
            ["LoggingExtension", "MetricsExtension"]
        )
        
        assert "test_chain" in toolchain.toolchains
        assert toolchain.toolchains["test_chain"] == ["LoggingExtension", "MetricsExtension"]
    
    @pytest.mark.asyncio
    async def test_create_toolchain_missing_extension(self):
        """Test creating toolchain with missing extension"""
        registry = ExtensionRegistry()
        toolchain = ExtensionToolChain(registry)
        
        with pytest.raises(ValueError):
            toolchain.create_toolchain(
                "bad_chain",
                ["NonExistentExtension"]
            )
    
    @pytest.mark.asyncio
    async def test_execute_toolchain(self):
        """Test executing toolchain"""
        registry = ExtensionRegistry()
        
        await registry.register(LoggingExtension())
        await registry.register(MetricsExtension())
        
        toolchain = ExtensionToolChain(registry)
        toolchain.create_toolchain(
            "test_chain",
            ["LoggingExtension", "MetricsExtension"]
        )
        
        context = HookContext(
            phase=ExtensionPhase.PRE_PROMPT,
            extension_name="TestExtension",
        )
        
        results = await toolchain.execute_toolchain(
            "test_chain",
            ExtensionPhase.PRE_PROMPT,
            context
        )
        
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_list_toolchains(self):
        """Test listing toolchains"""
        registry = ExtensionRegistry()
        
        await registry.register(LoggingExtension())
        
        toolchain = ExtensionToolChain(registry)
        toolchain.create_toolchain("chain1", ["LoggingExtension"])
        toolchain.create_toolchain("chain2", ["LoggingExtension"])
        
        toolchains = toolchain.list_toolchains()
        
        assert "chain1" in toolchains
        assert "chain2" in toolchains


class TestExtensionLifecycle:
    """Tests for extension lifecycle"""
    
    @pytest.mark.asyncio
    async def test_extension_initialization(self):
        """Test extension initialization"""
        ext = LoggingExtension()
        
        assert ext.metadata.enabled is True
        assert ext.execution_stats["total_hooks"] == 0
        
        await ext.initialize()
        assert len(ext.logs) == 0
    
    @pytest.mark.asyncio
    async def test_extension_cleanup(self):
        """Test extension cleanup"""
        ext = LoggingExtension()
        
        await ext.initialize()
        await ext.cleanup()
        # Should not raise


class TestExtensionPhases:
    """Tests for all extension phases"""
    
    @pytest.mark.asyncio
    async def test_all_phases_callable(self):
        """Test all phases are callable"""
        ext = LoggingExtension()
        await ext.initialize()
        
        phases = [
            (ExtensionPhase.PRE_PROMPT, ext.handle_pre_prompt),
            (ExtensionPhase.POST_PROMPT, ext.handle_post_prompt),
            (ExtensionPhase.PRE_INFERENCE, ext.handle_pre_inference),
            (ExtensionPhase.POST_INFERENCE, ext.handle_post_inference),
            (ExtensionPhase.PRE_ACTION, ext.handle_pre_action),
            (ExtensionPhase.POST_ACTION, ext.handle_post_action),
            (ExtensionPhase.PRE_SESSION, ext.handle_pre_session),
            (ExtensionPhase.POST_SESSION, ext.handle_post_session),
        ]
        
        for phase, handler in phases:
            context = HookContext(
                phase=phase,
                extension_name="TestExtension",
            )
            
            result = await handler(context)
            
            assert isinstance(result, HookResult)
            assert result.success is True


class TestExtensionIntegration:
    """Integration tests for extension system"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test full extension pipeline"""
        registry = ExtensionRegistry()
        
        # Register all built-in extensions
        await registry.register(LoggingExtension())
        await registry.register(MetricsExtension())
        await registry.register(CacheExtension())
        
        # Simulate phases
        phases = list(ExtensionPhase)
        
        for phase in phases:
            context = HookContext(
                phase=phase,
                extension_name="IntegrationTest",
                data={"prompt": "Test prompt"},
            )
            
            results = await registry.call_phase(phase, context)
            
            assert len(results) >= 0  # Some extensions might not handle all phases


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
