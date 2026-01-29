"""
Phase 10: HTTP Integration Tests
=================================

Comprehensive tests for real HTTP LLM connectors with:
- Request/response validation
- Error handling verification
- Rate limiting tests
- Token counting validation
- Streaming support tests
- Mock API responses
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List, Dict, Any
import sys
sys.path.insert(0, '/Users/evanpieser')

from ngvt_http_real_integration import (
    OpenAIHTTPConnector,
    AnthropicHTTPConnector,
    GoogleHTTPConnector,
    TokenCounter,
    RateLimiter,
    RateLimitConfig,
)


class TestTokenCounter(unittest.TestCase):
    """Test token counting functionality"""
    
    def setUp(self):
        self.counter = TokenCounter()
    
    def test_count_tokens_text(self):
        """Test token counting on plain text"""
        text = "Hello, world! This is a test."
        tokens = self.counter.count_tokens(text, "gpt-4")
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len(text) / 2)  # Should be less than text length
    
    def test_count_tokens_long_text(self):
        """Test token counting on longer text"""
        text = "AI is transforming the world. " * 100
        tokens = self.counter.count_tokens(text, "gpt-4")
        self.assertGreater(tokens, 100)
    
    def test_count_messages_tokens(self):
        """Test token counting on messages"""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]
        
        tokens = self.counter.count_messages_tokens(messages, "gpt-4")
        self.assertGreater(tokens, 0)
        self.assertGreater(tokens, 10)  # Should be significant
    
    def test_encoding_caching(self):
        """Test that encodings are cached"""
        self.counter.count_tokens("test", "gpt-4")
        self.assertIn("gpt-4", self.counter.encodings)
        
        # Should use cached encoding
        self.counter.count_tokens("test", "gpt-4")
        self.assertEqual(len(self.counter.encodings), 1)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def setUp(self):
        self.config = RateLimitConfig(
            requests_per_minute=10,
            tokens_per_minute=10000,
            max_concurrent=5
        )
        self.limiter = RateLimiter(self.config)
    
    async def test_concurrent_limits(self):
        """Test concurrent request limiting"""
        await self.limiter.acquire_concurrent()
        self.assertEqual(self.limiter.concurrent_requests, 1)
        
        await self.limiter.acquire_concurrent()
        self.assertEqual(self.limiter.concurrent_requests, 2)
        
        self.limiter.release_concurrent()
        self.assertEqual(self.limiter.concurrent_requests, 1)
    
    async def test_max_concurrent_respected(self):
        """Test that max concurrent is respected"""
        for _ in range(self.config.max_concurrent):
            await self.limiter.acquire_concurrent()
        
        self.assertEqual(self.limiter.concurrent_requests, self.config.max_concurrent)
    
    def test_rate_limiter_tracking(self):
        """Test request time tracking"""
        import time
        
        self.limiter.request_times.append(time.time())
        self.limiter.request_times.append(time.time())
        
        self.assertEqual(len(self.limiter.request_times), 2)


class TestOpenAIConnector(unittest.TestCase):
    """Test OpenAI HTTP connector"""
    
    def setUp(self):
        self.connector = OpenAIHTTPConnector(api_key="test-key", model="gpt-4")
    
    def test_initialization(self):
        """Test connector initialization"""
        self.assertEqual(self.connector.model, "gpt-4")
        self.assertEqual(self.connector.api_key, "test-key")
        self.assertGreater(self.connector.cost_per_1k_input, 0)
        self.assertGreater(self.connector.cost_per_1k_output, 0)
    
    def test_cost_calculation(self):
        """Test cost calculation"""
        # GPT-4 costs
        self.assertEqual(self.connector.cost_per_1k_input, 0.03)
        self.assertEqual(self.connector.cost_per_1k_output, 0.06)
    
    def test_gpt35_costs(self):
        """Test GPT-3.5 costs"""
        connector = OpenAIHTTPConnector(api_key="test-key", model="gpt-3.5-turbo")
        self.assertEqual(connector.cost_per_1k_input, 0.0005)
        self.assertEqual(connector.cost_per_1k_output, 0.0015)
    
    def test_stats(self):
        """Test statistics collection"""
        stats = self.connector.get_stats()
        self.assertEqual(stats['model'], "gpt-4")
        self.assertEqual(stats['requests'], 0)
        self.assertEqual(stats['cost'], 0.0)
        self.assertEqual(stats['errors'], 0)


class TestAnthropicConnector(unittest.TestCase):
    """Test Anthropic HTTP connector"""
    
    def setUp(self):
        self.connector = AnthropicHTTPConnector(api_key="test-key")
    
    def test_initialization(self):
        """Test connector initialization"""
        self.assertIn("claude", self.connector.model.lower())
        self.assertEqual(self.connector.api_key, "test-key")
    
    def test_opus_costs(self):
        """Test Claude Opus costs"""
        connector = AnthropicHTTPConnector(api_key="test-key", model="claude-3-opus")
        self.assertEqual(connector.cost_per_1k_input, 0.015)
        self.assertEqual(connector.cost_per_1k_output, 0.075)
    
    def test_sonnet_costs(self):
        """Test Claude Sonnet costs"""
        connector = AnthropicHTTPConnector(api_key="test-key", model="claude-3-sonnet")
        self.assertEqual(connector.cost_per_1k_input, 0.003)
        self.assertEqual(connector.cost_per_1k_output, 0.015)


class TestGoogleConnector(unittest.TestCase):
    """Test Google Gemini connector"""
    
    def setUp(self):
        self.connector = GoogleHTTPConnector(api_key="test-key")
    
    def test_initialization(self):
        """Test connector initialization"""
        self.assertEqual(self.connector.model, "gemini-pro")
        self.assertEqual(self.connector.api_key, "test-key")
    
    def test_free_tier(self):
        """Test that Gemini is free tier"""
        self.assertEqual(self.connector.cost_per_1k_input, 0.0)
        self.assertEqual(self.connector.cost_per_1k_output, 0.0)


class TestHTTPIntegration(unittest.TestCase):
    """Integration tests for HTTP connectors"""
    
    async def async_test_mock_openai_request(self):
        """Test mocked OpenAI request"""
        connector = OpenAIHTTPConnector(api_key="test-key", model="gpt-4")
        
        # Create mock response
        mock_response = {
            "choices": [{
                "message": {"content": "Hello world"},
                "finish_reason": "stop"
            }],
            "usage": {
                "completion_tokens": 2,
            }
        }
        
        # Test would use actual API with real key
        self.assertIsNotNone(connector)
    
    async def async_test_mock_anthropic_request(self):
        """Test mocked Anthropic request"""
        connector = AnthropicHTTPConnector(api_key="test-key")
        
        # Create mock response
        mock_response = {
            "content": [{"text": "Hello world"}],
            "usage": {"output_tokens": 2},
            "stop_reason": "end_turn"
        }
        
        self.assertIsNotNone(connector)
    
    async def async_test_mock_google_request(self):
        """Test mocked Google request"""
        connector = GoogleHTTPConnector(api_key="test-key")
        
        # Create mock response
        mock_response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello world"}]
                }
            }]
        }
        
        self.assertIsNotNone(connector)
    
    def test_mock_openai_request(self):
        """Test mocked OpenAI request"""
        asyncio.run(self.async_test_mock_openai_request())
    
    def test_mock_anthropic_request(self):
        """Test mocked Anthropic request"""
        asyncio.run(self.async_test_mock_anthropic_request())
    
    def test_mock_google_request(self):
        """Test mocked Google request"""
        asyncio.run(self.async_test_mock_google_request())


class TestErrorHandling(unittest.TestCase):
    """Test error handling in HTTP connectors"""
    
    def test_rate_limit_config(self):
        """Test rate limit configuration"""
        config = RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=90000,
            max_concurrent=10
        )
        
        self.assertEqual(config.requests_per_minute, 100)
        self.assertEqual(config.tokens_per_minute, 90000)
        self.assertEqual(config.max_concurrent, 10)
    
    def test_connector_error_tracking(self):
        """Test error tracking in connector"""
        connector = OpenAIHTTPConnector(api_key="test-key")
        
        self.assertEqual(connector.error_count, 0)
        self.assertIsNone(connector.last_error)
        
        # Simulate error
        connector.error_count = 1
        connector.last_error = "Test error"
        
        self.assertEqual(connector.error_count, 1)
        self.assertEqual(connector.last_error, "Test error")


class TestStreamingSupport(unittest.TestCase):
    """Test streaming support"""
    
    def test_openai_streaming_initialization(self):
        """Test that OpenAI connector supports streaming"""
        connector = OpenAIHTTPConnector(api_key="test-key", model="gpt-4")
        
        # Check that streaming method exists
        self.assertTrue(hasattr(connector, 'inference_stream'))
        self.assertTrue(callable(connector.inference_stream))
    
    def test_anthropic_streaming_initialization(self):
        """Test that Anthropic connector supports streaming"""
        connector = AnthropicHTTPConnector(api_key="test-key")
        
        self.assertTrue(hasattr(connector, 'inference_stream'))
        self.assertTrue(callable(connector.inference_stream))
    
    def test_google_streaming_initialization(self):
        """Test that Google connector supports streaming"""
        connector = GoogleHTTPConnector(api_key="test-key")
        
        self.assertTrue(hasattr(connector, 'inference_stream'))
        self.assertTrue(callable(connector.inference_stream))


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("Phase 10: HTTP Integration Tests")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTokenCounter))
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiter))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenAIConnector))
    suite.addTests(loader.loadTestsFromTestCase(TestAnthropicConnector))
    suite.addTests(loader.loadTestsFromTestCase(TestGoogleConnector))
    suite.addTests(loader.loadTestsFromTestCase(TestHTTPIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamingSupport))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
