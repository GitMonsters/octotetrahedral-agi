"""
Phase 10: Real HTTP Integration for LLM APIs
=============================================

Replaces mock implementations with actual HTTP clients for:
- OpenAI API
- Anthropic Claude API
- Google Gemini API
- Local LLM endpoints

Features:
- Real async HTTP with aiohttp
- Token counting and cost tracking
- Streaming responses
- Comprehensive error handling
- Rate limiting and retry logic
- Request/response logging
- SSL/TLS support
"""

import asyncio
import aiohttp
import json
import logging
import time
from typing import Optional, List, Dict, Any, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import tiktoken
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# TOKEN COUNTING UTILITIES
# ============================================================================

class TokenCounter:
    """Handles token counting for different models"""
    
    def __init__(self):
        self.encodings: Dict[str, tiktoken.Encoding] = {}
    
    def get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get or create token encoding for model"""
        if model not in self.encodings:
            try:
                if "gpt" in model.lower():
                    encoding_name = "cl100k_base"  # GPT-3.5, GPT-4
                elif "claude" in model.lower():
                    encoding_name = "cl100k_base"  # Claude uses similar
                else:
                    encoding_name = "cl100k_base"  # Default
                
                self.encodings[model] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Could not load encoding for {model}: {e}")
                # Fallback: rough estimation (4 chars per token)
                return None
        
        return self.encodings[model]
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text"""
        encoding = self.get_encoding(model)
        
        if encoding:
            try:
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Error counting tokens: {e}")
        
        # Fallback: rough estimation
        return len(text) // 4
    
    def count_messages_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """Count tokens in message list"""
        total = 0
        
        for message in messages:
            total += self.count_tokens(message.get("content", ""), model)
            # Add overhead for message formatting
            total += 4
        
        # Add completion tokens overhead
        total += 2
        
        return total


# ============================================================================
# RATE LIMITING
# ============================================================================

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 100
    tokens_per_minute: int = 90000
    max_concurrent: int = 10


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times: List[float] = []
        self.token_times: List[Tuple[float, int]] = []
        self.concurrent_requests = 0
    
    async def wait_if_needed(self, token_count: int = 0) -> None:
        """Wait if rate limits would be exceeded"""
        now = time.time()
        minute_ago = now - 60
        
        # Check request rate
        self.request_times = [t for t in self.request_times if t > minute_ago]
        if len(self.request_times) >= self.config.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit: waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Check token rate
        if token_count > 0:
            self.token_times = [(t, c) for t, c in self.token_times if t > minute_ago]
            total_tokens = sum(c for _, c in self.token_times)
            
            if total_tokens + token_count > self.config.tokens_per_minute:
                sleep_time = 5  # Conservative backoff
                logger.warning(f"Token rate limit: waiting {sleep_time}s")
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(now)
        if token_count > 0:
            self.token_times.append((now, token_count))
    
    async def acquire_concurrent(self) -> None:
        """Acquire slot for concurrent request"""
        while self.concurrent_requests >= self.config.max_concurrent:
            await asyncio.sleep(0.1)
        self.concurrent_requests += 1
    
    def release_concurrent(self) -> None:
        """Release concurrent request slot"""
        self.concurrent_requests = max(0, self.concurrent_requests - 1)


# ============================================================================
# HTTP CLIENT BASE CLASS
# ============================================================================

class HTTPLLMConnector(ABC):
    """Base class for real HTTP LLM connectors"""
    
    def __init__(self, api_key: str, model: str, rate_limit_config: Optional[RateLimitConfig] = None):
        self.api_key = api_key
        self.model = model
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.token_counter = TokenCounter()
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.total_cost = 0.0
        self.error_count = 0
        self.last_error: Optional[str] = None
    
    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json_data: Optional[Dict] = None,
        timeout: int = 30,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """Make HTTP request with retries"""
        
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        for attempt in range(retry_count):
            try:
                await self.rate_limiter.wait_if_needed()
                await self.rate_limiter.acquire_concurrent()
                
                try:
                    async with self.session.request(
                        method,
                        url,
                        headers=headers,
                        json=json_data,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data
                        elif response.status == 429:  # Rate limited
                            retry_after = int(response.headers.get("Retry-After", 60))
                            logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            error_text = await response.text()
                            raise RuntimeError(f"HTTP {response.status}: {error_text}")
                finally:
                    self.rate_limiter.release_concurrent()
                
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{retry_count})")
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                logger.error(f"Request failed (attempt {attempt + 1}/{retry_count}): {e}")
                
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        raise RuntimeError("Request failed after all retries")
    
    @abstractmethod
    async def inference(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Run inference"""
        pass
    
    @abstractmethod
    async def inference_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """Stream inference"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics"""
        return {
            'model': self.model,
            'requests': self.request_count,
            'cost': self.total_cost,
            'errors': self.error_count,
            'last_error': self.last_error,
        }


# ============================================================================
# OPENAI CONNECTOR
# ============================================================================

class OpenAIHTTPConnector(HTTPLLMConnector):
    """Real OpenAI API connector"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        self.base_url = "https://api.openai.com/v1"
        self.cost_per_1k_input = 0.03 if "gpt-4" in model else 0.0005
        self.cost_per_1k_output = 0.06 if "gpt-4" in model else 0.0015
    
    async def inference(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """Call OpenAI API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Count input tokens
        input_tokens = self.token_counter.count_messages_tokens(messages, self.model)
        
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        # Make request
        start_time = time.time()
        response_data = await self._make_request(
            "POST",
            f"{self.base_url}/chat/completions",
            headers,
            json_data=request_data
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response
        content = response_data["choices"][0]["message"]["content"]
        output_tokens = response_data["usage"]["completion_tokens"]
        
        # Calculate cost
        cost = (
            (input_tokens / 1000) * self.cost_per_1k_input +
            (output_tokens / 1000) * self.cost_per_1k_output
        )
        
        self.request_count += 1
        self.total_cost += cost
        
        return {
            "content": content,
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
            "finish_reason": response_data["choices"][0]["finish_reason"],
        }
    
    async def inference_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream from OpenAI API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        request_data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=request_data,
        ) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        pass


# ============================================================================
# ANTHROPIC CONNECTOR
# ============================================================================

class AnthropicHTTPConnector(HTTPLLMConnector):
    """Real Anthropic Claude API connector"""
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        super().__init__(api_key, model)
        self.base_url = "https://api.anthropic.com/v1"
        self.cost_per_1k_input = 0.015 if "opus" in model else 0.003
        self.cost_per_1k_output = 0.075 if "opus" in model else 0.015
    
    async def inference(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """Call Anthropic API"""
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        # Count input tokens
        input_tokens = self.token_counter.count_messages_tokens(messages, self.model)
        
        request_data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Make request
        start_time = time.time()
        response_data = await self._make_request(
            "POST",
            f"{self.base_url}/messages",
            headers,
            json_data=request_data
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response
        content = response_data["content"][0]["text"]
        output_tokens = response_data["usage"]["output_tokens"]
        
        # Calculate cost
        cost = (
            (input_tokens / 1000) * self.cost_per_1k_input +
            (output_tokens / 1000) * self.cost_per_1k_output
        )
        
        self.request_count += 1
        self.total_cost += cost
        
        return {
            "content": content,
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
            "finish_reason": response_data["stop_reason"],
        }
    
    async def inference_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream from Anthropic API"""
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        request_data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=request_data,
        ) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if "delta" in data and "text" in data["delta"]:
                            yield data["delta"]["text"]
                    except json.JSONDecodeError:
                        pass


# ============================================================================
# GOOGLE GEMINI CONNECTOR
# ============================================================================

class GoogleHTTPConnector(HTTPLLMConnector):
    """Real Google Gemini API connector"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        super().__init__(api_key, model)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.cost_per_1k_input = 0.0  # Free tier
        self.cost_per_1k_output = 0.0
    
    async def inference(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """Call Google Gemini API"""
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            contents.append({
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}]
            })
        
        # Count input tokens
        input_tokens = self.token_counter.count_messages_tokens(messages, self.model)
        
        request_data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        # Make request
        start_time = time.time()
        response_data = await self._make_request(
            "POST",
            f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
            headers,
            json_data=request_data
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract response
        content = response_data["candidates"][0]["content"]["parts"][0]["text"]
        output_tokens = len(content) // 4  # Rough estimate
        
        self.request_count += 1
        
        return {
            "content": content,
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": 0.0,
            "latency_ms": latency_ms,
            "finish_reason": "STOP",
        }
    
    async def inference_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream from Google API"""
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Convert messages
        contents = []
        for msg in messages:
            contents.append({
                "role": "model" if msg["role"] == "assistant" else "user",
                "parts": [{"text": msg["content"]}]
            })
        
        request_data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        async with self.session.post(
            f"{self.base_url}/models/{self.model}:streamGenerateContent?key={self.api_key}",
            headers=headers,
            json=request_data,
        ) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "candidates" in data:
                            delta = data["candidates"][0].get("content", {}).get("parts", [{}])[0]
                            if "text" in delta:
                                yield delta["text"]
                    except json.JSONDecodeError:
                        pass


# ============================================================================
# DEMO AND TESTING
# ============================================================================

async def demo_real_http_integration():
    """Demonstrate real HTTP integration"""
    print("=" * 70)
    print("Phase 10: Real HTTP Integration for LLM APIs")
    print("=" * 70)
    
    # Example with mock credentials (won't work without real API keys)
    print("\n1. OpenAI Connector Example:")
    print("-" * 70)
    
    try:
        async with OpenAIHTTPConnector(api_key="sk-test", model="gpt-4") as connector:
            print(f"   ✓ Initialized OpenAI connector")
            print(f"   • Model: {connector.model}")
            print(f"   • Cost per 1K input: ${connector.cost_per_1k_input}")
            print(f"   • Cost per 1K output: ${connector.cost_per_1k_output}")
            print(f"   • Rate limiter: {connector.rate_limiter.config.requests_per_minute} req/min")
    except Exception as e:
        print(f"   ℹ Demo mode - would need real API key for actual calls")
    
    print("\n2. Anthropic Connector Example:")
    print("-" * 70)
    
    try:
        async with AnthropicHTTPConnector(api_key="sk-test") as connector:
            print(f"   ✓ Initialized Anthropic connector")
            print(f"   • Model: {connector.model}")
            print(f"   • Cost per 1K input: ${connector.cost_per_1k_input}")
            print(f"   • Cost per 1K output: ${connector.cost_per_1k_output}")
    except Exception as e:
        print(f"   ℹ Demo mode - would need real API key for actual calls")
    
    print("\n3. Google Connector Example:")
    print("-" * 70)
    
    try:
        async with GoogleHTTPConnector(api_key="test-key") as connector:
            print(f"   ✓ Initialized Google Gemini connector")
            print(f"   • Model: {connector.model}")
            print(f"   • Free tier: Yes (no cost)")
    except Exception as e:
        print(f"   ℹ Demo mode - would need real API key for actual calls")
    
    print("\n4. Token Counting Example:")
    print("-" * 70)
    
    counter = TokenCounter()
    test_text = "What is artificial intelligence? AI is the simulation of human intelligence."
    gpt4_tokens = counter.count_tokens(test_text, "gpt-4")
    claude_tokens = counter.count_tokens(test_text, "claude-3-opus")
    
    print(f"   Text: {test_text}")
    print(f"   GPT-4 tokens: {gpt4_tokens}")
    print(f"   Claude tokens: {claude_tokens}")
    
    print("\n5. Rate Limiting Example:")
    print("-" * 70)
    
    config = RateLimitConfig(requests_per_minute=5, max_concurrent=2)
    limiter = RateLimiter(config)
    
    print(f"   Config: {config.requests_per_minute} req/min, {config.max_concurrent} concurrent")
    print(f"   ✓ Rate limiter initialized and ready")
    
    print("\n" + "=" * 70)
    print("Phase 10: Real HTTP Integration Framework Ready ✓")
    print("=" * 70)
    print("\nNEXT STEPS:")
    print("  1. Set environment variables:")
    print("     export OPENAI_API_KEY='sk-...'")
    print("     export ANTHROPIC_API_KEY='sk-ant-...'")
    print("     export GOOGLE_API_KEY='...'")
    print("\n  2. Use real connectors:")
    print("     async with OpenAIHTTPConnector(api_key) as connector:")
    print("         response = await connector.inference(messages)")
    print("\n  3. Stream responses:")
    print("     async with OpenAIHTTPConnector(api_key) as connector:")
    print("         async for chunk in connector.inference_stream(messages):")
    print("             print(chunk, end='', flush=True)")


if __name__ == "__main__":
    asyncio.run(demo_real_http_integration())
