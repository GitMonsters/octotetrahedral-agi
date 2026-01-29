"""
NGVT Unified Orchestrator Loop
Confucius SDK-inspired orchestration system for coordinated multi-model inference
Manages iteration, memory composition, extension routing, and execution control
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import httpx
import re

from ngvt_memory import NGVTHierarchicalMemory, MemoryScopeType
from ngvt_notes import PatternNoteStore, PatternType


class ActionType(Enum):
    """Types of actions the orchestrator can route"""
    INFERENCE = "inference"
    MODEL_REGISTER = "model_register"
    RETRIEVE_PATTERN = "retrieve_pattern"
    STORE_PATTERN = "store_pattern"
    EVALUATE = "evaluate"
    TERMINATE = "terminate"


@dataclass
class Task:
    """Represents a task to be orchestrated"""
    id: str
    title: str
    description: str
    max_iterations: int = 10
    timeout_seconds: int = 300
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Action parsed from LLM output"""
    type: ActionType
    params: Dict[str, Any]
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Observation:
    """Result from executing an action"""
    action_type: ActionType
    success: bool
    result: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""
    max_iterations: int = 10
    max_context_tokens: int = 8000
    temperature: float = 0.7
    timeout_seconds: int = 300
    memory_composition_strategy: str = "hierarchical"  # hierarchical or flat
    extension_timeout_ms: int = 5000
    verbose: bool = True


class Extension(ABC):
    """Base class for orchestrator extensions"""
    
    def __init__(self, name: str, config: OrchestratorConfig):
        self.name = name
        self.config = config
        self.execution_count = 0
        self.total_latency_ms = 0.0
    
    @abstractmethod
    async def can_handle(self, action: Action) -> bool:
        """Check if this extension can handle the action"""
        pass
    
    @abstractmethod
    async def execute(self, action: Action) -> Observation:
        """Execute the action and return observation"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        avg_latency = (self.total_latency_ms / self.execution_count 
                      if self.execution_count > 0 else 0)
        return {
            "name": self.name,
            "execution_count": self.execution_count,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": avg_latency,
        }


class InferenceExtension(Extension):
    """Routes inference requests to NGVT servers"""
    
    def __init__(self, config: OrchestratorConfig, 
                 server_port: int = 8082,
                 server_host: str = "127.0.0.1"):
        super().__init__("InferenceExtension", config)
        self.server_port = server_port
        self.server_host = server_host
        self.server_url = f"http://{server_host}:{server_port}"
    
    async def can_handle(self, action: Action) -> bool:
        return action.type == ActionType.INFERENCE
    
    async def execute(self, action: Action) -> Observation:
        start_time = time.time()
        try:
            prompt = action.params.get("prompt", "")
            model = action.params.get("model", "claude")
            temperature = action.params.get("temperature", self.config.temperature)
            max_tokens = action.params.get("max_tokens", 1000)
            
            async with httpx.AsyncClient(timeout=self.config.extension_timeout_ms / 1000) as client:
                response = await client.post(
                    f"{self.server_url}/infer",
                    json={
                        "prompt": prompt,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                self.execution_count += 1
                self.total_latency_ms += latency_ms
                
                return Observation(
                    action_type=ActionType.INFERENCE,
                    success=True,
                    result=result,
                    latency_ms=latency_ms,
                )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_latency_ms += latency_ms
            return Observation(
                action_type=ActionType.INFERENCE,
                success=False,
                result=None,
                error=str(e),
                latency_ms=latency_ms,
            )


class PatternExtension(Extension):
    """Routes pattern retrieval and storage operations"""
    
    def __init__(self, config: OrchestratorConfig, note_store: PatternNoteStore):
        super().__init__("PatternExtension", config)
        self.note_store = note_store
    
    async def can_handle(self, action: Action) -> bool:
        return action.type in (ActionType.RETRIEVE_PATTERN, ActionType.STORE_PATTERN)
    
    async def execute(self, action: Action) -> Observation:
        start_time = time.time()
        try:
            result = None
            if action.type == ActionType.RETRIEVE_PATTERN:
                # Retrieve patterns based on query
                query = action.params.get("query", "")
                pattern_type = action.params.get("pattern_type")
                limit = action.params.get("limit", 5)
                
                # Try different retrieval methods
                if query:
                    patterns = self.note_store.search_similar(query, top_k=limit)
                elif pattern_type:
                    all_patterns = self.note_store.search_by_type(PatternType(pattern_type))
                    patterns = all_patterns[:limit]
                else:
                    patterns = list(self.note_store.index.values())[:limit]
                
                result = [asdict(p) for p in patterns]
            
            elif action.type == ActionType.STORE_PATTERN:
                # Store new pattern from params
                from ngvt_notes import PatternNote
                pattern_note = PatternNote(
                    id=action.params.get("id", ""),
                    title=action.params.get("title", ""),
                    pattern_type=PatternType(action.params.get("pattern_type", "integration_pattern")),
                    problem=action.params.get("problem", ""),
                    solution=action.params.get("solution", ""),
                    keywords=action.params.get("keywords", []),
                    effectiveness=action.params.get("effectiveness", 0.5),
                    examples=action.params.get("examples", []),
                )
                pattern_id = self.note_store.add_pattern(pattern_note)
                result = {"stored": True, "pattern_id": pattern_id}
            
            latency_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_latency_ms += latency_ms
            
            return Observation(
                action_type=action.type,
                success=True,
                result=result,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_latency_ms += latency_ms
            return Observation(
                action_type=action.type,
                success=False,
                result=None,
                error=str(e),
                latency_ms=latency_ms,
            )


class EvaluationExtension(Extension):
    """Routes evaluation and completion checks"""
    
    def __init__(self, config: OrchestratorConfig):
        super().__init__("EvaluationExtension", config)
    
    async def can_handle(self, action: Action) -> bool:
        return action.type == ActionType.EVALUATE
    
    async def execute(self, action: Action) -> Observation:
        start_time = time.time()
        try:
            eval_type = action.params.get("type", "quality")
            data_to_eval = action.params.get("data", {})
            
            # Simple evaluation logic
            if eval_type == "quality":
                # Check if result contains sufficient information
                quality_score = len(str(data_to_eval)) / 100.0
                quality_score = min(quality_score, 1.0)
                result = {
                    "quality_score": quality_score,
                    "passed": quality_score > 0.5,
                }
            elif eval_type == "completeness":
                # Check if task objectives are met
                objectives = action.params.get("objectives", [])
                completed = action.params.get("completed_count", 0)
                completeness = completed / len(objectives) if objectives else 0
                result = {
                    "completeness": completeness,
                    "objectives_completed": completed,
                    "objectives_total": len(objectives),
                    "passed": completeness >= 0.8,
                }
            else:
                result = {"status": "unknown_eval_type"}
            
            latency_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_latency_ms += latency_ms
            
            return Observation(
                action_type=ActionType.EVALUATE,
                success=True,
                result=result,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.execution_count += 1
            self.total_latency_ms += latency_ms
            return Observation(
                action_type=ActionType.EVALUATE,
                success=False,
                result=None,
                error=str(e),
                latency_ms=latency_ms,
            )


class ExtensionRegistry:
    """Manages and routes to extensions"""
    
    def __init__(self):
        self.extensions: Dict[str, Extension] = {}
    
    def register(self, extension: Extension) -> None:
        """Register an extension"""
        self.extensions[extension.name] = extension
    
    async def route(self, action: Action) -> Optional[Extension]:
        """Find extension that can handle action"""
        for ext in self.extensions.values():
            if await ext.can_handle(action):
                return ext
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all extensions"""
        return {
            name: ext.get_stats()
            for name, ext in self.extensions.items()
        }


class NGVTUnifiedOrchestrator:
    """
    Unified orchestration loop for coordinated NGVT inference
    Manages iteration, memory, extensions, and execution control
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.memory = NGVTHierarchicalMemory()
        self.note_store = PatternNoteStore(storage_dir="./ngvt_patterns")
        self.extensions = ExtensionRegistry()
        self.session_context: Dict[str, Any] = {}
        self.iteration_count = 0
        self.start_time = 0.0
        self.actions_history: List[Action] = []
        self.observations_history: List[Observation] = []
        
        # Register default extensions
        self._register_default_extensions()
    
    def _register_default_extensions(self) -> None:
        """Register built-in extensions"""
        self.extensions.register(InferenceExtension(self.config))
        self.extensions.register(PatternExtension(self.config, self.note_store))
        self.extensions.register(EvaluationExtension(self.config))
    
    async def run_session(self, task: Task) -> Dict[str, Any]:
        """
        Main orchestration loop - execute until completion or max iterations
        """
        self.iteration_count = 0
        self.start_time = time.time()
        self.session_context = {"task": task}
        
        # Initialize memory for this session
        self.memory.initialize_session()
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"Starting orchestration session for task: {task.title}")
            print(f"Max iterations: {task.max_iterations}")
            print(f"{'='*70}\n")
        
        try:
            for iteration in range(task.max_iterations):
                self.iteration_count = iteration + 1
                
                # Check timeout
                elapsed = time.time() - self.start_time
                if elapsed > task.timeout_seconds:
                    if self.config.verbose:
                        print(f"[!] Timeout reached ({elapsed:.1f}s)")
                    break
                
                if self.config.verbose:
                    print(f"--- Iteration {self.iteration_count} ---")
                
                # 1. Compose prompt with memory
                prompt = self._compose_prompt(task)
                
                # 2. Parse actions (simulated LLM call)
                actions = self._parse_actions(prompt)
                
                if not actions:
                    if self.config.verbose:
                        print("[*] No actions generated, terminating")
                    break
                
                # 3. Execute actions
                completed = False
                for action in actions:
                    ext = await self.extensions.route(action)
                    
                    if ext is None:
                        if self.config.verbose:
                            print(f"[!] No extension found for action: {action.type.value}")
                        continue
                    
                    # Execute action
                    observation = await ext.execute(action)
                    self.observations_history.append(observation)
                    
                    # Record in memory
                    self.memory.record_experience(
                        scope=MemoryScopeType.RUNNABLE,
                        content={
                            "action_type": action.type.value,
                            "observation": observation.result if observation.success else observation.error,
                        }
                    )
                    
                    if self.config.verbose:
                        status = "✓" if observation.success else "✗"
                        print(f"  {status} {action.type.value} ({observation.latency_ms:.1f}ms)")
                    
                    # Check for termination
                    if action.type == ActionType.TERMINATE:
                        completed = True
                        break
                
                if completed:
                    break
                
                # 4. Brief pause between iterations
                await asyncio.sleep(0.1)
            
            # Extract and return artifacts
            elapsed = time.time() - self.start_time
            artifacts = self._extract_artifacts(elapsed)
            
            if self.config.verbose:
                print(f"\n{'='*70}")
                print(f"Session completed in {elapsed:.2f}s ({self.iteration_count} iterations)")
                print(f"{'='*70}\n")
            
            return artifacts
        
        except Exception as e:
            if self.config.verbose:
                print(f"[ERROR] {str(e)}")
            raise
    
    def _compose_prompt(self, task: Task) -> str:
        """
        Compose final prompt respecting context limits
        Uses hierarchical memory and recent patterns
        """
        components = []
        
        # System prompt
        components.append(self._get_system_prompt())
        
        # Task context
        components.append(f"Task: {task.title}\n{task.description}")
        
        # Hierarchical memory composition
        memory_composition = self.memory.compose_for_prompt()
        if memory_composition:
            components.append(f"Memory Context:\n{memory_composition}")
        
        # Relevant patterns from notes
        if self.note_store.index:
            patterns_summary = self._summarize_patterns(task.description)
            if patterns_summary:
                components.append(f"Relevant Patterns:\n{patterns_summary}")
        
        # Available actions
        components.append(self._get_actions_description())
        
        # Session stats
        components.append(self._get_session_stats())
        
        prompt = "\n\n".join(components)
        
        # Compress if exceeds context limit
        if len(prompt) > self.config.max_context_tokens * 4:  # Rough token estimate
            prompt = self._compress_prompt(prompt)
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """System prompt for the orchestrator"""
        return """You are an intelligent orchestration system managing multiple LLM inference engines.
Your goal is to:
1. Understand the task requirements
2. Determine what actions are needed (inference, pattern retrieval, evaluation)
3. Execute actions iteratively until the task is complete
4. Learn from patterns and improve over iterations

Available action types:
- inference: Call LLM for reasoning/generation
- retrieve_pattern: Get relevant patterns from memory
- evaluate: Assess progress toward task completion
- terminate: End the session when task is complete

Respond with a single JSON action object or array."""
    
    def _get_actions_description(self) -> str:
        """Describe available actions"""
        return """Available Actions:
- {"type": "inference", "params": {"prompt": "...", "model": "claude", "max_tokens": 1000}}
- {"type": "retrieve_pattern", "params": {"query": "...", "limit": 5}}
- {"type": "evaluate", "params": {"type": "quality", "data": {...}}}
- {"type": "terminate", "params": {}}"""
    
    def _get_session_stats(self) -> str:
        """Get current session statistics"""
        elapsed = time.time() - self.start_time
        return f"""Session Stats:
- Iteration: {self.iteration_count}
- Elapsed: {elapsed:.1f}s
- Total actions: {len(self.actions_history)}
- Successful actions: {sum(1 for o in self.observations_history if o.success)}"""
    
    def _summarize_patterns(self, task_description: str) -> str:
        """Retrieve and summarize relevant patterns"""
        relevant_patterns = self.note_store.search_similar(task_description, top_k=3)
        
        if not relevant_patterns:
            return ""
        
        lines = []
        for pattern in relevant_patterns:
            lines.append(f"- {pattern.title} (effectiveness: {pattern.effectiveness:.0%})")
            lines.append(f"  Problem: {pattern.problem[:100]}...")
            lines.append(f"  Solution: {pattern.solution[:100]}...")
        
        return "\n".join(lines)
    
    def _parse_actions(self, prompt: str) -> List[Action]:
        """
        Parse actions from prompt (simulated LLM response)
        In real implementation, would call actual LLM
        """
        actions = []
        
        # Simulated action determination based on prompt content
        if "inference" in prompt.lower() or not self.actions_history:
            # First, try inference
            actions.append(Action(
                type=ActionType.INFERENCE,
                params={
                    "prompt": f"Task-specific inference needed. Describe approach.",
                    "model": "claude",
                    "max_tokens": 500,
                },
                reasoning="Initial inference to understand task"
            ))
        
        if "pattern" in prompt.lower() and len(self.actions_history) > 0:
            # Retrieve patterns
            task = self.session_context.get("task")
            task_desc = task.description if (task and hasattr(task, 'description')) else ""
            if task_desc:
                actions.append(Action(
                    type=ActionType.RETRIEVE_PATTERN,
                    params={
                        "query": task_desc,
                        "limit": 3,
                    },
                    reasoning="Retrieve relevant patterns from memory"
                ))
        
        # Add evaluation periodically
        if self.iteration_count % 2 == 0 and self.iteration_count > 0:
            actions.append(Action(
                type=ActionType.EVALUATE,
                params={
                    "type": "quality",
                    "data": {"observations": len(self.observations_history)},
                },
                reasoning="Check progress toward task completion"
            ))
        
        # Terminate after a few iterations
        if self.iteration_count >= 3:
            actions.append(Action(
                type=ActionType.TERMINATE,
                params={},
                reasoning="Task completed, terminating session"
            ))
        
        self.actions_history.extend(actions)
        return actions
    
    def _compress_prompt(self, prompt: str) -> str:
        """Compress prompt by removing less important sections"""
        lines = prompt.split("\n")
        # Keep system prompt, task, and recent memory
        compressed = lines[:20] + lines[-15:]
        return "\n".join(compressed)
    
    def _extract_artifacts(self, elapsed_time: float) -> Dict[str, Any]:
        """Extract final artifacts from session"""
        successful_observations = [o for o in self.observations_history if o.success]
        failed_observations = [o for o in self.observations_history if not o.success]
        
        task = self.session_context.get('task')
        task_id = task.id if (task and hasattr(task, 'id')) else 'unknown'
        task_title = task.title if (task and hasattr(task, 'title')) else 'unknown'
        task_desc = task.description if (task and hasattr(task, 'description')) else ''
        
        # Store successful patterns in note system
        if successful_observations:
            from ngvt_notes import PatternNote
            # Record as a successful integration pattern
            pattern = PatternNote(
                id=f"session_{int(self.start_time)}",
                title=f"Successful Session: {task_title}",
                pattern_type=PatternType.INTEGRATION_PATTERN,
                problem=task_desc,
                solution=f"Completed {len(self.actions_history)} actions in {self.iteration_count} iterations",
                keywords=[a.type.value for a in self.actions_history],
                effectiveness=len(successful_observations) / max(len(self.observations_history), 1),
                examples=[{"action_type": o.action_type.value, "result": o.result, "latency_ms": o.latency_ms} 
                         for o in successful_observations[:3]],
            )
            self.note_store.add_pattern(pattern)
        
        return {
            "task_id": task_id,
            "task_title": task_title,
            "status": "completed",
            "iterations": self.iteration_count,
            "elapsed_seconds": elapsed_time,
            "total_actions": len(self.actions_history),
            "successful_actions": len(successful_observations),
            "failed_actions": len(failed_observations),
            "extension_stats": self.extensions.get_stats(),
            "memory_stats": {
                "scopes": len(self.memory.scopes),
                "total_entries": sum(len(scope.entries) for scope in self.memory.scopes.values()),
            },
            "actions": [asdict(a) for a in self.actions_history],
            "observations": [asdict(o) for o in self.observations_history],
        }


async def demo_orchestrator():
    """Demo orchestrator with example task"""
    
    config = OrchestratorConfig(
        max_iterations=5,
        max_context_tokens=4000,
        verbose=True,
    )
    
    orchestrator = NGVTUnifiedOrchestrator(config)
    
    task = Task(
        id="demo_001",
        title="Multi-Model Inference Orchestration",
        description="Coordinate inference across multiple NGVT servers using memory and patterns",
        max_iterations=5,
    )
    
    artifacts = await orchestrator.run_session(task)
    
    print("\n" + "="*70)
    print("FINAL ARTIFACTS")
    print("="*70)
    print(json.dumps(artifacts, indent=2, default=str))
    
    return artifacts


if __name__ == "__main__":
    asyncio.run(demo_orchestrator())
