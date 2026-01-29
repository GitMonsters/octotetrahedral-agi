"""
NGVT Enhanced Server with Compound Learning & Integrations
Combines standard inference with meta-learning and multi-model orchestration
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ngvt_compound_learning import (
    CompoundLearningEngine, 
    CompoundIntegrationEngine, 
    LearningExperience
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NGVT Compound Learning & Integration Server",
    description="Advanced NGVT with meta-learning and multi-model orchestration",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global engines
learning_engine = CompoundLearningEngine(max_patterns=10000)
integration_engine = CompoundIntegrationEngine()

# Request/Response Models
class CompoundInferenceRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    use_learning: bool = Field(default=True, description="Use compound learning")
    request_id: Optional[str] = None

class IntegrationPathRequest(BaseModel):
    path_id: str = Field(..., description="Integration path identifier")
    input_data: Dict[str, Any] = Field(..., description="Input data for workflow")

class ModelRegistrationRequest(BaseModel):
    model_id: str = Field(..., description="Unique model identifier")
    model_type: str = Field(..., description="Type of model (nlp, vision, speech, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict)

class CompoundInferenceResponse(BaseModel):
    request_id: str
    prompt: str
    response: str
    tokens_generated: int
    latency_ms: float
    model: str = "NGVT-Compound"
    
    # Compound learning fields
    used_learned_pattern: bool = False
    pattern_id: Optional[str] = None
    pattern_accuracy: Optional[float] = None
    learning_confidence: Optional[float] = None
    transfer_learning_applied: bool = False
    
    timestamp: str

# Global metrics
start_time = time.time()
request_count = 0
learning_updates = 0

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - start_time
    }

@app.post("/inference/compound")
async def compound_inference(request: CompoundInferenceRequest):
    """Run inference with compound learning"""
    global request_count
    
    request_id = request.request_id or str(uuid.uuid4())
    request_count += 1
    
    inference_start = time.time()
    
    try:
        # Step 1: Check if we have learned about this query
        learning_prediction = None
        used_pattern = False
        transfer_applied = False
        pattern_id = None
        pattern_accuracy = None
        learning_confidence = 0.0
        
        if request.use_learning:
            learning_prediction = learning_engine.predict_with_learning(request.prompt)
            used_pattern = learning_prediction.get('has_learned_pattern', False)
            transfer_applied = learning_prediction.get('similar_pattern_found', False)
            pattern_id = learning_prediction.get('pattern_id')
            pattern_accuracy = learning_prediction.get('predicted_accuracy')
            learning_confidence = learning_prediction.get('confidence', 0.0)
        
        # Step 2: Simulate inference
        tokens_to_generate = min(request.max_tokens, 100)
        await asyncio.sleep((tokens_to_generate / 45.0) * 0.8)  # 80% of expected time
        
        inference_latency = (time.time() - inference_start) * 1000
        
        # Generate response
        response_text = f"Response [pattern:{pattern_id or 'none'}, tokens:{tokens_to_generate}]"
        
        # Step 3: Record experience for learning
        experience = LearningExperience(
            query=request.prompt,
            response=response_text,
            latency_ms=inference_latency,
            success=True,
            timestamp=datetime.now().isoformat(),
            cache_hit=used_pattern,
            tokens_generated=tokens_to_generate,
            confidence=learning_confidence,
            metadata={
                'used_pattern': used_pattern,
                'transfer_applied': transfer_applied,
                'pattern_id': pattern_id,
            }
        )
        
        learning_engine.record_experience(experience)
        
        return CompoundInferenceResponse(
            request_id=request_id,
            prompt=request.prompt,
            response=response_text,
            tokens_generated=tokens_to_generate,
            latency_ms=inference_latency,
            model="NGVT-Compound",
            used_learned_pattern=used_pattern,
            pattern_id=pattern_id,
            pattern_accuracy=pattern_accuracy,
            learning_confidence=learning_confidence,
            transfer_learning_applied=transfer_applied,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/cycle")
async def run_learning_cycle():
    """Run a compound learning cycle"""
    global learning_updates
    
    cycle_result = learning_engine.compound_learning_cycle()
    learning_updates += 1
    
    return {
        'cycle_number': cycle_result['cycle_number'],
        'patterns_discovered': cycle_result['patterns_discovered'],
        'total_patterns': cycle_result['total_patterns'],
        'transfer_efficiency': cycle_result['transfer_efficiency'],
        'cumulative_accuracy': cycle_result['cumulative_accuracy'],
        'timestamp': cycle_result['timestamp'],
    }

@app.get("/learning/stats")
async def get_learning_stats():
    """Get compound learning statistics"""
    return learning_engine.get_learning_stats()

@app.post("/integration/path/execute")
async def execute_integration_path(request: IntegrationPathRequest):
    """Execute a compound integration path (workflow)"""
    result = integration_engine.execute_integration_path(
        request.path_id,
        request.input_data
    )
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result

@app.post("/integration/model/register")
async def register_model(request: ModelRegistrationRequest):
    """Register a new model for compound integration"""
    integration_engine.register_model(
        request.model_id,
        request.model_type,
        request.config
    )
    
    return {
        'status': 'registered',
        'model_id': request.model_id,
        'model_type': request.model_type,
        'timestamp': datetime.now().isoformat(),
    }

@app.post("/integration/path/define")
async def define_integration_path(body: Dict[str, Any]):
    """Define a compound integration path"""
    path_id = body.get('path_id')
    model_sequence = body.get('model_sequence', [])
    
    if not path_id or not model_sequence:
        raise HTTPException(status_code=400, detail="path_id and model_sequence required")
    
    integration_engine.define_integration_path(path_id, model_sequence)
    
    return {
        'status': 'defined',
        'path_id': path_id,
        'model_sequence': model_sequence,
        'timestamp': datetime.now().isoformat(),
    }

@app.get("/integration/suggestions")
async def get_integration_suggestions():
    """Get suggested integration paths"""
    suggestions = integration_engine.suggest_integration_paths()
    
    return {
        'suggestions': suggestions,
        'timestamp': datetime.now().isoformat(),
    }

@app.get("/integration/stats")
async def get_integration_stats():
    """Get compound integration statistics"""
    return integration_engine.get_integration_stats()

@app.get("/metrics/compound")
async def get_compound_metrics():
    """Get comprehensive compound learning and integration metrics"""
    uptime = time.time() - start_time
    
    return {
        'uptime_seconds': uptime,
        'total_requests': request_count,
        'learning_cycles': learning_engine.knowledge_base['total_learning_cycles'],
        'learning_stats': learning_engine.get_learning_stats(),
        'integration_stats': integration_engine.get_integration_stats(),
        'combined_performance': {
            'avg_requests_per_cycle': request_count / learning_engine.knowledge_base['total_learning_cycles'] if learning_engine.knowledge_base['total_learning_cycles'] > 0 else 0,
            'knowledge_transfer_rate': (learning_engine.knowledge_base['transfer_efficiency'] * 100),
            'system_intelligence_score': (
                learning_engine.knowledge_base['cumulative_accuracy'] * 0.5 +
                integration_engine.get_integration_stats().get('success_rate', 0) * 0.5
            )
        },
        'timestamp': datetime.now().isoformat(),
    }

@app.get("/info")
async def get_info():
    """Get system information"""
    return {
        'name': 'NGVT Compound Learning & Integration System',
        'version': '2.0.0',
        'description': 'Advanced NGVT with meta-learning and multi-model orchestration',
        'capabilities': [
            'Compound Inference with Meta-Learning',
            'Pattern Discovery & Knowledge Accumulation',
            'Transfer Learning (Cross-Query)',
            'Multi-Model Integration (Compound Workflows)',
            'Model Compatibility Analysis',
            'Integration Path Orchestration',
            'Real-time Learning Cycles',
        ],
        'endpoints': {
            'health': 'GET /health',
            'compound_inference': 'POST /inference/compound',
            'learning_cycle': 'POST /learning/cycle',
            'learning_stats': 'GET /learning/stats',
            'integration_execute': 'POST /integration/path/execute',
            'integration_register_model': 'POST /integration/model/register',
            'integration_define_path': 'POST /integration/path/define',
            'integration_suggestions': 'GET /integration/suggestions',
            'integration_stats': 'GET /integration/stats',
            'compound_metrics': 'GET /metrics/compound',
            'info': 'GET /info',
        }
    }

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add request timing header"""
    start_time_req = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time_req
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8082
    logger.info(f"Starting NGVT Compound Learning & Integration Server on port {port}")
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1, log_level="info")
