"""
NGVT Hierarchical Memory System
Inspired by Confucius SDK's memory architecture
Supports infinite context through hierarchical scoping and compression
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from collections import OrderedDict
import hashlib


class MemoryScopeType(Enum):
    """Memory scope levels in hierarchy"""
    SESSION = "session"      # Lifetime patterns & global insights
    ENTRY = "entry"          # Per-integration-path summaries
    RUNNABLE = "runnable"    # Per-execution details


@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    id: str
    scope: MemoryScopeType
    timestamp: str
    content: Dict[str, Any]
    summary: Optional[str] = None
    compressed: bool = False
    importance_score: float = 0.5
    retention_priority: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['scope'] = self.scope.value  # Convert enum to string
        # Convert any remaining enums in data to strings
        if isinstance(d.get('data'), dict):
            d['data'] = self._serialize_data(d['data'])
        return d
    
    @staticmethod
    def _serialize_data(data: Any) -> Any:
        """Recursively convert enums to their values"""
        from enum import Enum
        if isinstance(data, Enum):
            return data.value
        elif isinstance(data, dict):
            return {k: MemoryEntry._serialize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [MemoryEntry._serialize_data(item) for item in data]
        else:
            return data
    
    def size_bytes(self) -> int:
        """Estimate size in bytes"""
        return len(json.dumps(self.to_dict()).encode('utf-8'))


@dataclass
class MemoryScope:
    """Represents a single memory scope with entries"""
    scope_type: MemoryScopeType
    entries: Dict[str, MemoryEntry] = field(default_factory=OrderedDict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_size_bytes: int = 1_000_000  # 1MB per scope by default
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add entry to scope"""
        self.entries[entry.id] = entry
        self._prune_if_needed()
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve entry by ID"""
        return self.entries.get(entry_id)
    
    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries in insertion order"""
        return list(self.entries.values())
    
    def total_size_bytes(self) -> int:
        """Calculate total size of all entries"""
        return sum(entry.size_bytes() for entry in self.entries.values())
    
    def _prune_if_needed(self) -> None:
        """Remove low-priority entries if size exceeded"""
        if self.total_size_bytes() > self.max_size_bytes:
            # Sort by importance/retention priority
            entries_list = list(self.entries.items())
            entries_list.sort(
                key=lambda x: x[1].importance_score * x[1].retention_priority,
                reverse=True
            )
            
            # Keep only top entries
            target_size = int(self.max_size_bytes * 0.8)
            current_size = 0
            keep_ids = set()
            
            for entry_id, entry in entries_list:
                if current_size + entry.size_bytes() <= target_size:
                    keep_ids.add(entry_id)
                    current_size += entry.size_bytes()
            
            # Remove others
            to_remove = [eid for eid in self.entries if eid not in keep_ids]
            for eid in to_remove:
                del self.entries[eid]
    
    def compose_for_prompt(self, max_tokens: int = 2000) -> str:
        """Compose scope entries into prompt text"""
        lines = []
        tokens = 0
        
        for entry in self.get_all_entries():
            entry_text = self._format_entry(entry)
            entry_tokens = len(entry_text.split())
            
            if tokens + entry_tokens <= max_tokens:
                lines.append(entry_text)
                tokens += entry_tokens
            else:
                break
        
        return "\n".join(lines)
    
    def _format_entry(self, entry: MemoryEntry) -> str:
        """Format entry for inclusion in prompt"""
        scope_label = entry.scope.value.upper()
        content_str = json.dumps(entry.content, indent=2)[:500]  # Truncate
        
        if entry.summary:
            return f"[{scope_label}] {entry.summary}"
        else:
            return f"[{scope_label}] {content_str}..."


class NGVTHierarchicalMemory:
    """
    Hierarchical memory system for NGVT
    Organizes information into session, entry, and runnable scopes
    Supports automatic compression for context management
    """
    
    def __init__(
        self,
        max_context_tokens: int = 32_000,
        compression_threshold: float = 0.8,
        session_scope_weight: float = 0.4,
        entry_scope_weight: float = 0.35,
        runnable_scope_weight: float = 0.25,
    ):
        """
        Initialize hierarchical memory
        
        Args:
            max_context_tokens: Maximum tokens for prompt
            compression_threshold: When to trigger compression (0.8 = 80%)
            session_scope_weight: Weight for session scope (global)
            entry_scope_weight: Weight for entry scope (per-path)
            runnable_scope_weight: Weight for runnable scope (per-execution)
        """
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold
        
        # Scope weights
        self.weights = {
            MemoryScopeType.SESSION: session_scope_weight,
            MemoryScopeType.ENTRY: entry_scope_weight,
            MemoryScopeType.RUNNABLE: runnable_scope_weight,
        }
        
        # Initialize scopes
        self.scopes: Dict[MemoryScopeType, MemoryScope] = {
            scope_type: MemoryScope(scope_type)
            for scope_type in MemoryScopeType
        }
        
        # Metadata
        self.session_id = self._generate_id()
        self.created_at = datetime.now()
        self.compression_count = 0
        self.compression_history: List[Dict[str, Any]] = []
    
    def initialize_session(self) -> None:
        """Initialize a new session"""
        self.session_id = self._generate_id()
        self.created_at = datetime.now()
        self.scopes[MemoryScopeType.RUNNABLE].entries.clear()
        
        # Keep session and entry scopes (persistent across tasks)
    
    def record_experience(
        self,
        scope: MemoryScopeType,
        content: Dict[str, Any],
        importance: float = 0.5,
        summary: Optional[str] = None,
    ) -> str:
        """
        Record an experience in memory
        
        Args:
            scope: Which scope to record in
            content: Content to record
            importance: Importance score (0-1)
            summary: Optional summary for compression
        
        Returns:
            Entry ID
        """
        entry_id = self._generate_id()
        entry = MemoryEntry(
            id=entry_id,
            scope=scope,
            timestamp=datetime.now().isoformat(),
            content=content,
            summary=summary,
            importance_score=importance,
        )
        
        self.scopes[scope].add_entry(entry)
        
        # Check if compression needed
        if self._needs_compression():
            self.compress_memory()
        
        return entry_id
    
    def update_with_observation(self, observation: Dict[str, Any]) -> None:
        """Update memory with LLM observation"""
        self.record_experience(
            scope=MemoryScopeType.RUNNABLE,
            content=observation,
            importance=observation.get("importance", 0.5),
            summary=observation.get("summary"),
        )
    
    def record_pattern(
        self,
        pattern_name: str,
        pattern_details: Dict[str, Any],
        effectiveness: float = 0.5,
    ) -> str:
        """Record a discovered pattern at session level"""
        content = {
            "pattern_name": pattern_name,
            "details": pattern_details,
            "effectiveness": effectiveness,
            "discovered_at": datetime.now().isoformat(),
        }
        
        return self.record_experience(
            scope=MemoryScopeType.SESSION,
            content=content,
            importance=effectiveness,
            summary=f"Pattern: {pattern_name} (effectiveness={effectiveness:.2f})",
        )
    
    def record_integration_summary(
        self,
        integration_path: str,
        metrics: Dict[str, Any],
    ) -> str:
        """Record integration path summary at entry level"""
        content = {
            "integration_path": integration_path,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        importance = metrics.get("success_rate", 0.5)
        
        return self.record_experience(
            scope=MemoryScopeType.ENTRY,
            content=content,
            importance=importance,
            summary=f"Integration: {integration_path} (success={importance:.2f})",
        )
    
    def compose_for_prompt(self, max_tokens: Optional[int] = None) -> str:
        """
        Compose memory scopes into prompt text
        Respects scope weights and max token limit
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        # Allocate tokens per scope based on weights
        allocations = {}
        remaining = max_tokens
        
        for scope_type in [MemoryScopeType.SESSION, MemoryScopeType.ENTRY, MemoryScopeType.RUNNABLE]:
            allocation = int(max_tokens * self.weights[scope_type])
            allocations[scope_type] = min(allocation, remaining)
            remaining -= allocations[scope_type]
        
        # Compose each scope
        prompt_parts = [
            "=== SESSION MEMORY (Global Patterns) ===",
            self.scopes[MemoryScopeType.SESSION].compose_for_prompt(allocations[MemoryScopeType.SESSION]),
            "",
            "=== ENTRY MEMORY (Integration Summaries) ===",
            self.scopes[MemoryScopeType.ENTRY].compose_for_prompt(allocations[MemoryScopeType.ENTRY]),
            "",
            "=== RUNNABLE MEMORY (Recent Executions) ===",
            self.scopes[MemoryScopeType.RUNNABLE].compose_for_prompt(allocations[MemoryScopeType.RUNNABLE]),
        ]
        
        return "\n".join(prompt_parts)
    
    def _needs_compression(self) -> bool:
        """Check if compression is needed"""
        total_tokens = self._estimate_total_tokens()
        threshold = int(self.max_context_tokens * self.compression_threshold)
        return total_tokens >= threshold
    
    def _estimate_total_tokens(self) -> int:
        """Estimate total tokens in memory"""
        total = 0
        for scope in self.scopes.values():
            prompt_text = scope.compose_for_prompt(99_999)
            total += len(prompt_text.split())
        return total
    
    def compress_memory(self) -> None:
        """
        Compress memory by summarizing old entries
        Keeps recent entries, summarizes older ones
        """
        compression_start = time.time()
        
        # Compress SESSION scope (least frequently accessed)
        self._compress_scope(MemoryScopeType.SESSION, ratio=0.7)
        
        # Compress ENTRY scope
        self._compress_scope(MemoryScopeType.ENTRY, ratio=0.5)
        
        # RUNNABLE scope stays mostly uncompressed (recent data)
        self._compress_scope(MemoryScopeType.RUNNABLE, ratio=0.2)
        
        compression_time = time.time() - compression_start
        self.compression_count += 1
        
        self.compression_history.append({
            "timestamp": datetime.now().isoformat(),
            "compression_count": self.compression_count,
            "compression_time_ms": compression_time * 1000,
            "tokens_before": self._estimate_total_tokens(),
        })
    
    def _compress_scope(self, scope_type: MemoryScopeType, ratio: float) -> None:
        """
        Compress a specific scope by summarizing entries
        
        Args:
            scope_type: Which scope to compress
            ratio: How many entries to compress (0.5 = compress 50%)
        """
        scope = self.scopes[scope_type]
        entries = scope.get_all_entries()
        
        # Determine split point
        split_idx = int(len(entries) * (1 - ratio))
        
        if split_idx <= 0:
            return  # Not enough entries to compress
        
        to_compress = entries[:split_idx]
        to_keep = entries[split_idx:]
        
        # Create summary of compressed entries
        if to_compress:
            summary_content = self._create_summary(to_compress)
            summary_entry = MemoryEntry(
                id=self._generate_id(),
                scope=scope_type,
                timestamp=datetime.now().isoformat(),
                content=summary_content,
                summary=f"Compressed {len(to_compress)} entries",
                compressed=True,
                importance_score=0.7,
                retention_priority=2.0,  # Higher priority to keep
            )
            
            # Clear old entries and add summary
            scope.entries.clear()
            scope.add_entry(summary_entry)
            
            # Re-add kept entries
            for entry in to_keep:
                scope.add_entry(entry)
    
    def _create_summary(self, entries: List[MemoryEntry]) -> Dict[str, Any]:
        """Create summary of multiple entries"""
        patterns = []
        metrics = {}
        
        for entry in entries:
            if "pattern_name" in entry.content:
                patterns.append(entry.content["pattern_name"])
            
            # Aggregate metrics
            if "metrics" in entry.content:
                for key, value in entry.content["metrics"].items():
                    if isinstance(value, (int, float)):
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(value)
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        for key, values in metrics.items():
            aggregate_metrics[f"{key}_avg"] = sum(values) / len(values)
            aggregate_metrics[f"{key}_max"] = max(values)
            aggregate_metrics[f"{key}_min"] = min(values)
        
        return {
            "type": "compressed_summary",
            "entry_count": len(entries),
            "unique_patterns": list(set(patterns)),
            "aggregate_metrics": aggregate_metrics,
            "time_span": {
                "start": entries[0].timestamp if entries else None,
                "end": entries[-1].timestamp if entries else None,
            }
        }
    
    def get_session_patterns(self) -> List[Dict[str, Any]]:
        """Get all discovered patterns in session"""
        patterns = []
        for entry in self.scopes[MemoryScopeType.SESSION].get_all_entries():
            if "pattern_name" in entry.content:
                patterns.append(entry.content)
        return patterns
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
            "total_tokens": self._estimate_total_tokens(),
            "max_tokens": self.max_context_tokens,
            "utilization_percent": (self._estimate_total_tokens() / self.max_context_tokens) * 100,
            "compression_count": self.compression_count,
            "scope_stats": {
                scope_type.value: {
                    "entries": len(self.scopes[scope_type].entries),
                    "size_bytes": self.scopes[scope_type].total_size_bytes(),
                    "weight": self.weights[scope_type],
                }
                for scope_type in MemoryScopeType
            },
            "compression_history": self.compression_history[-10:],  # Last 10
        }
    
    def export_memory(self) -> Dict[str, Any]:
        """Export entire memory state"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "scopes": {
                scope_type.value: [
                    entry.to_dict()
                    for entry in self.scopes[scope_type].get_all_entries()
                ]
                for scope_type in MemoryScopeType
            },
            "stats": self.get_memory_stats(),
        }
    
    def import_memory(self, memory_state: Dict[str, Any]) -> None:
        """Import memory from previous session"""
        self.session_id = memory_state.get("session_id", self._generate_id())
        
        for scope_type_str, entries_data in memory_state.get("scopes", {}).items():
            scope_type = MemoryScopeType(scope_type_str)
            scope = self.scopes[scope_type]
            
            for entry_data in entries_data:
                entry = MemoryEntry(
                    id=entry_data["id"],
                    scope=scope_type,
                    timestamp=entry_data["timestamp"],
                    content=entry_data["content"],
                    summary=entry_data.get("summary"),
                    compressed=entry_data.get("compressed", False),
                    importance_score=entry_data.get("importance_score", 0.5),
                    retention_priority=entry_data.get("retention_priority", 1.0),
                )
                scope.add_entry(entry)
    
    @staticmethod
    def _generate_id() -> str:
        """Generate unique ID"""
        return hashlib.md5(
            f"{time.time()}{hash(id(object()))}".encode()
        ).hexdigest()[:16]


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("NGVT HIERARCHICAL MEMORY SYSTEM - DEMO")
    print("="*80)
    
    # Initialize memory
    memory = NGVTHierarchicalMemory(max_context_tokens=8000)
    memory.initialize_session()
    
    print(f"\n✓ Created memory system (session: {memory.session_id[:8]}...)")
    
    # Record session-level patterns
    print("\n1. Recording global patterns...")
    memory.record_pattern(
        "query_similarity",
        {"threshold": 0.85, "method": "cosine"},
        effectiveness=0.92
    )
    memory.record_pattern(
        "model_selection",
        {"strategy": "performance-based"},
        effectiveness=0.88
    )
    
    # Record entry-level summaries
    print("2. Recording integration summaries...")
    memory.record_integration_summary(
        "nlp_vision_pipeline",
        {"success_rate": 0.95, "latency_ms": 150, "throughput": 50}
    )
    
    # Record runnable-level details
    print("3. Recording execution details...")
    for i in range(5):
        memory.update_with_observation({
            "request_id": f"req_{i}",
            "status": "success",
            "latency_ms": 100 + i*10,
            "importance": 0.8,
        })
    
    # Get stats
    stats = memory.get_memory_stats()
    print(f"\n4. Memory Statistics:")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Utilization: {stats['utilization_percent']:.1f}%")
    print(f"   Entries: SESSION={stats['scope_stats']['session']['entries']}, "
          f"ENTRY={stats['scope_stats']['entry']['entries']}, "
          f"RUNNABLE={stats['scope_stats']['runnable']['entries']}")
    
    # Compose for prompt
    print(f"\n5. Memory for prompt:")
    prompt = memory.compose_for_prompt()
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE")
    print("="*80)
