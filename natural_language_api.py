#!/usr/bin/env python3
"""
Natural Language API Implementation
Complete working example for natural language prompting with OctoTetrahedral AGI
"""

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import torch
import time
from datetime import datetime

app = FastAPI(title="OctoTetrahedral AGI - Natural Language API")

# ============================================================================
# Models & Request/Response Types
# ============================================================================

class PromptRequest(BaseModel):
    """Natural language prompt request"""
    prompt: str
    mode: str = "answer"  # answer, code, creative, technical
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # user, assistant, system
    content: str

class ChatRequest(BaseModel):
    """Chat request"""
    messages: List[ChatMessage]
    system_prompt: Optional[str] = None
    max_length: int = 200

class CommandRequest(BaseModel):
    """Command request"""
    command: str  # e.g., "analyze", "summarize", "translate"
    input_text: str
    options: Optional[Dict] = None

# ============================================================================
# Authentication
# ============================================================================

VALID_API_KEY = "qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ"

def verify_auth(authorization: str = Header(None)) -> bool:
    """Verify API key from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    if "Bearer " not in authorization:
        raise HTTPException(status_code=401, detail="Invalid Authorization format")
    
    token = authorization.split("Bearer ")[-1]
    if token != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# ============================================================================
# Natural Language Processor
# ============================================================================

class NaturalLanguageProcessor:
    """Handle natural language processing"""
    
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Simulate model - in production, load actual model
        self.model = None
    
    def process_prompt(
        self, 
        prompt: str, 
        mode: str = "answer",
        max_length: int = 200,
        temperature: float = 0.7
    ) -> Dict:
        """Process natural language prompt"""
        
        start_time = time.time()
        
        # Simulate different modes
        response = self._generate_response(prompt, mode, max_length)
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "prompt": prompt,
            "response": response,
            "mode": mode,
            "latency_ms": round(latency, 2),
            "device": self.device,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_response(self, prompt: str, mode: str, max_length: int) -> str:
        """Generate response based on mode"""
        
        if mode == "code":
            return self._generate_code(prompt)
        elif mode == "creative":
            return self._generate_creative(prompt)
        elif mode == "technical":
            return self._generate_technical(prompt)
        else:  # answer
            return self._generate_answer(prompt)
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate factual answer"""
        return f"Response to: {prompt}\n\nThis is a generated response based on your question. In production, this would use the actual model to generate meaningful responses."
    
    def _generate_code(self, prompt: str) -> str:
        """Generate code"""
        return f"""# Generated code for: {prompt}

def solution():
    # Implementation here
    pass
"""
    
    def _generate_creative(self, prompt: str) -> str:
        """Generate creative response"""
        return f"Creative response to: {prompt}\n\nThis is a creatively generated response that aims to inspire and engage..."
    
    def _generate_technical(self, prompt: str) -> str:
        """Generate technical explanation"""
        return f"Technical explanation for: {prompt}\n\nFrom a technical perspective, this involves several key concepts..."

# Initialize processor
processor = NaturalLanguageProcessor()

# ============================================================================
# Natural Language Endpoints
# ============================================================================

@app.post("/prompt")
async def handle_prompt(
    request: PromptRequest,
    authorization: str = Header(None)
):
    """
    Natural language prompt endpoint
    
    Example:
    {
        "prompt": "What is machine learning?",
        "mode": "answer",
        "max_length": 200
    }
    """
    verify_auth(authorization)
    
    try:
        result = processor.process_prompt(
            prompt=request.prompt,
            mode=request.mode,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def handle_chat(
    request: ChatRequest,
    authorization: str = Header(None)
):
    """
    Conversational chat endpoint
    
    Example:
    {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What can you help with?"}
        ]
    }
    """
    verify_auth(authorization)
    
    try:
        # Build conversation context
        conversation = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in request.messages
        ])
        
        # Generate response
        result = processor.process_prompt(
            prompt=conversation,
            mode="answer",
            max_length=request.max_length
        )
        
        return {
            "success": True,
            "data": {
                "response": result["response"],
                "latency_ms": result["latency_ms"],
                "device": result["device"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/command")
async def handle_command(
    request: CommandRequest,
    authorization: str = Header(None)
):
    """
    Execute natural language commands
    
    Supported commands:
    - analyze: Analyze the input text
    - summarize: Create a summary
    - translate: Translate to another language
    - expand: Expand the content
    - simplify: Simplify the content
    
    Example:
    {
        "command": "summarize",
        "input_text": "Long text here...",
        "options": {"length": "short"}
    }
    """
    verify_auth(authorization)
    
    try:
        command = request.command.lower()
        
        if command == "summarize":
            prompt = f"Summarize this: {request.input_text}"
            mode = "answer"
        elif command == "analyze":
            prompt = f"Analyze this: {request.input_text}"
            mode = "technical"
        elif command == "translate":
            target_lang = request.options.get("target_language", "Spanish") if request.options else "Spanish"
            prompt = f"Translate to {target_lang}: {request.input_text}"
            mode = "answer"
        elif command == "expand":
            prompt = f"Expand on this: {request.input_text}"
            mode = "creative"
        elif command == "simplify":
            prompt = f"Simplify this: {request.input_text}"
            mode = "answer"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown command: {command}")
        
        result = processor.process_prompt(prompt=prompt, mode=mode)
        
        return {
            "success": True,
            "command": command,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(
    request: dict,
    authorization: str = Header(None)
):
    """
    Simple ask endpoint - just ask a question
    
    Example:
    {
        "question": "How does photosynthesis work?"
    }
    """
    verify_auth(authorization)
    
    try:
        question = request.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="question field required")
        
        result = processor.process_prompt(prompt=question, mode="answer")
        
        return {
            "success": True,
            "question": question,
            "answer": result["response"],
            "latency_ms": result["latency_ms"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Health Checks
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "OctoTetrahedralModel",
        "device": processor.device,
        "features": ["prompt", "chat", "command", "ask"]
    }

@app.get("/stats")
async def stats():
    """Get API statistics"""
    return {
        "status": "operational",
        "device": processor.device,
        "endpoints": {
            "/prompt": "Send natural language prompts",
            "/chat": "Have conversations",
            "/command": "Execute commands",
            "/ask": "Ask questions",
            "/health": "Health check",
            "/stats": "Statistics"
        }
    }

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting OctoTetrahedral AGI - Natural Language API")
    print("📍 API Key: qU62MH7IOkLzFDUHCVJoRlrc41nzzNNa8-Hhnm2YwVQ")
    print("🌐 Base URL: http://localhost:8000")
    print("📚 Endpoints:")
    print("  - POST /prompt - Send natural language prompts")
    print("  - POST /chat - Conversational chat")
    print("  - POST /command - Execute commands")
    print("  - POST /ask - Ask questions")
    print("  - GET /health - Health check")
    print("  - GET /stats - API statistics")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
