"""
NGVT Pattern Note-Taking System
Cross-session persistent learning through structured pattern storage
Inspired by Confucius SDK's note-taking mechanism
"""

import json
import os
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib


class PatternType(Enum):
    """Types of patterns that can be stored"""
    NLP_PATTERN = "nlp_pattern"
    INTEGRATION_PATTERN = "integration_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    ERROR_PATTERN = "error_pattern"
    MODEL_PATTERN = "model_pattern"


@dataclass
class PatternNote:
    """Structured pattern note for persistent storage"""
    id: str
    title: str
    pattern_type: PatternType
    problem: str
    solution: str
    keywords: List[str] = field(default_factory=list)
    effectiveness: float = 0.5
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['pattern_type'] = self.pattern_type.value
        # Recursively serialize any remaining enums
        d = self._serialize_dict(d)
        return d
    
    @staticmethod
    def _serialize_dict(obj: Any) -> Any:
        """Recursively convert enums to their values"""
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: PatternNote._serialize_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [PatternNote._serialize_dict(item) for item in obj]
        else:
            return obj
    
    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        lines = [
            f"# {self.title}",
            f"**ID:** `{self.id}`",
            f"**Type:** {self.pattern_type.value}",
            f"**Effectiveness:** {self.effectiveness:.2%}",
            f"**Usage Count:** {self.usage_count}",
            f"**Created:** {self.created_at}",
            f"**Last Updated:** {self.last_updated}",
            f"",
            f"## Problem",
            self.problem,
            f"",
            f"## Solution",
            self.solution,
            f"",
            f"## Keywords",
            ", ".join(f"`{kw}`" for kw in self.keywords) if self.keywords else "None",
            f"",
        ]
        
        if self.examples:
            lines.extend([
                f"## Examples",
                f"",
            ])
            for i, ex in enumerate(self.examples, 1):
                lines.append(f"### Example {i}")
                lines.append(f"```json")
                lines.append(json.dumps(ex, indent=2))
                lines.append(f"```")
                lines.append("")
        
        if self.metadata:
            lines.extend([
                f"## Metadata",
                f"```json",
                json.dumps(self.metadata, indent=2),
                f"```",
            ])
        
        return "\n".join(lines)


class PatternNoteStore:
    """
    Persistent store for pattern notes
    Supports file-based and database storage
    """
    
    def __init__(self, storage_dir: str = "/tmp/ngvt_patterns"):
        """
        Initialize pattern note store
        
        Args:
            storage_dir: Directory for storing pattern notes
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Index for fast retrieval
        self.index: Dict[str, PatternNote] = {}
        self.keyword_index: Dict[str, List[str]] = {}  # keyword -> pattern_ids
        self.type_index: Dict[PatternType, List[str]] = {
            pt: [] for pt in PatternType
        }
        
        # Load existing patterns
        self._load_all()
    
    def add_pattern(self, note: PatternNote) -> str:
        """Add pattern to store"""
        # Generate ID if not provided
        if not note.id:
            note.id = self._generate_id()
        
        # Update timestamps
        note.last_updated = datetime.now().isoformat()
        
        # Add to indexes
        self.index[note.id] = note
        self.type_index[note.pattern_type].append(note.id)
        
        for keyword in note.keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            self.keyword_index[keyword].append(note.id)
        
        # Persist to disk
        self._save_pattern(note)
        
        return note.id
    
    def get_pattern(self, pattern_id: str) -> Optional[PatternNote]:
        """Retrieve pattern by ID"""
        pattern = self.index.get(pattern_id)
        if pattern:
            pattern.usage_count += 1
            self._save_pattern(pattern)
        return pattern
    
    def search_by_keyword(self, keyword: str) -> List[PatternNote]:
        """Search patterns by keyword"""
        pattern_ids = self.keyword_index.get(keyword, [])
        return [self.index[pid] for pid in pattern_ids if pid in self.index]
    
    def search_by_type(self, pattern_type: PatternType) -> List[PatternNote]:
        """Get all patterns of a type"""
        pattern_ids = self.type_index.get(pattern_type, [])
        return [self.index[pid] for pid in pattern_ids if pid in self.index]
    
    def search_similar(self, query: str, top_k: int = 5,
                       spatial_embedding: Optional[List[float]] = None,
                       spatial_weight: float = 0.3) -> List[PatternNote]:
        """
        Find patterns similar to query using dual-index retrieval.
        
        Combines:
        1. Semantic matching: TF-IDF-weighted word overlap (upgraded from raw overlap)
        2. Spatial matching: cosine similarity over embedding vectors (if provided)
        
        Inspired by OpenClaw's Spatial RAG hybrid retrieval.
        
        Args:
            query: Text query
            top_k: Number of results
            spatial_embedding: Optional embedding vector for spatial similarity
            spatial_weight: Weight for spatial vs semantic score [0,1]
        """
        query_words = set(query.lower().split())
        
        # Build document frequency for TF-IDF weighting
        doc_freq: Dict[str, int] = {}
        for pattern in self.index.values():
            pattern_words = set(pattern.keywords) | set(pattern.title.lower().split())
            for w in pattern_words:
                doc_freq[w] = doc_freq.get(w, 0) + 1
        
        num_docs = max(len(self.index), 1)
        
        scores = []
        for pattern_id, pattern in self.index.items():
            pattern_words = set(pattern.keywords) | set(pattern.title.lower().split())
            
            # TF-IDF weighted semantic overlap
            overlap_words = query_words & pattern_words
            if not overlap_words and spatial_embedding is None:
                continue
            
            semantic_score = 0.0
            for w in overlap_words:
                idf = math.log(num_docs / (doc_freq.get(w, 1) + 1)) + 1
                semantic_score += idf
            
            # Normalize by query size
            if query_words:
                semantic_score /= len(query_words)
            
            # Spatial similarity (cosine) if embeddings available
            spatial_score = 0.0
            if spatial_embedding is not None:
                pattern_emb = pattern.metadata.get('embedding')
                if pattern_emb is not None:
                    # Cosine similarity
                    a = spatial_embedding
                    b = pattern_emb
                    dot = sum(x * y for x, y in zip(a, b))
                    norm_a = math.sqrt(sum(x * x for x in a) + 1e-8)
                    norm_b = math.sqrt(sum(x * x for x in b) + 1e-8)
                    spatial_score = dot / (norm_a * norm_b)
            
            # Combined score
            if spatial_embedding is not None:
                final_score = (1 - spatial_weight) * semantic_score + spatial_weight * spatial_score
            else:
                final_score = semantic_score
            
            if final_score > 0:
                scores.append((pattern_id, final_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.index[pid] for pid, _ in scores[:top_k]]
    
    def get_top_patterns(self, limit: int = 10) -> List[PatternNote]:
        """Get top patterns by effectiveness"""
        patterns = sorted(
            self.index.values(),
            key=lambda p: (p.effectiveness, p.usage_count),
            reverse=True
        )
        return patterns[:limit]
    
    def export_as_markdown(self, output_file: Optional[str] = None) -> str:
        """Export all patterns as Markdown"""
        lines = [
            "# NGVT Pattern Knowledge Base",
            f"Generated: {datetime.now().isoformat()}",
            f"Total Patterns: {len(self.index)}",
            "",
        ]
        
        # Group by type
        for pattern_type in PatternType:
            patterns = self.search_by_type(pattern_type)
            if patterns:
                lines.append(f"## {pattern_type.value.replace('_', ' ').title()}")
                lines.append(f"({len(patterns)} patterns)")
                lines.append("")
                
                for pattern in patterns:
                    lines.append(f"### {pattern.title}")
                    lines.append(f"- Effectiveness: {pattern.effectiveness:.2%}")
                    lines.append(f"- Used: {pattern.usage_count} times")
                    lines.append("")
        
        content = "\n".join(lines)
        
        if output_file:
            Path(output_file).write_text(content)
        
        return content
    
    def _load_all(self) -> None:
        """Load all patterns from disk"""
        pattern_files = list(self.storage_dir.glob("pattern_*.json"))
        
        for file_path in pattern_files:
            try:
                data = json.loads(file_path.read_text())
                note = PatternNote(
                    id=data['id'],
                    title=data['title'],
                    pattern_type=PatternType(data['pattern_type']),
                    problem=data['problem'],
                    solution=data['solution'],
                    keywords=data.get('keywords', []),
                    effectiveness=data.get('effectiveness', 0.5),
                    examples=data.get('examples', []),
                    metadata=data.get('metadata', {}),
                    created_at=data.get('created_at'),
                    last_updated=data.get('last_updated'),
                    usage_count=data.get('usage_count', 0),
                )
                
                # Re-index
                self.index[note.id] = note
                self.type_index[note.pattern_type].append(note.id)
                for keyword in note.keywords:
                    if keyword not in self.keyword_index:
                        self.keyword_index[keyword] = []
                    self.keyword_index[keyword].append(note.id)
            except Exception as e:
                print(f"Error loading pattern from {file_path}: {e}")
    
    def _save_pattern(self, note: PatternNote) -> None:
        """Save pattern to disk"""
        filename = self.storage_dir / f"pattern_{note.id}.json"
        filename.write_text(json.dumps(note.to_dict(), indent=2))
    
    @staticmethod
    def _generate_id() -> str:
        """Generate unique ID"""
        return hashlib.md5(
            f"{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]


class NoteTaker:
    """
    Async note-taking agent
    Extracts patterns from learning trajectories and stores them
    """
    
    def __init__(self, store: PatternNoteStore):
        """Initialize NoteTaker with storage backend"""
        self.store = store
    
    def extract_patterns_from_trajectory(
        self,
        trajectory: Dict[str, Any],
    ) -> List[PatternNote]:
        """
        Extract patterns from a learning trajectory
        
        Args:
            trajectory: Learning experience with input/output/result
        
        Returns:
            List of extracted pattern notes
        """
        patterns = []
        
        # Extract NLP patterns
        if "query" in trajectory and "response" in trajectory:
            patterns.extend(self._extract_nlp_patterns(trajectory))
        
        # Extract integration patterns
        if "integration_path" in trajectory:
            patterns.extend(self._extract_integration_patterns(trajectory))
        
        # Extract performance patterns
        if "latency_ms" in trajectory or "throughput" in trajectory:
            patterns.extend(self._extract_performance_patterns(trajectory))
        
        # Extract error patterns
        if "error" in trajectory or "exception" in trajectory:
            patterns.extend(self._extract_error_patterns(trajectory))
        
        return patterns
    
    def _extract_nlp_patterns(self, trajectory: Dict[str, Any]) -> List[PatternNote]:
        """Extract NLP-related patterns"""
        patterns = []
        
        # Example: Query similarity pattern
        if "similar_queries" in trajectory:
            note = PatternNote(
                id="",
                title="Query Similarity Detection",
                pattern_type=PatternType.NLP_PATTERN,
                problem="How to detect similar user queries",
                solution="Use embedding-based similarity with cosine distance",
                keywords=["query", "similarity", "embedding", "clustering"],
                effectiveness=trajectory.get("similarity_score", 0.85),
                examples=trajectory.get("similar_queries", [])[:3],
                metadata={
                    "threshold": trajectory.get("similarity_threshold", 0.85),
                    "method": "cosine_distance",
                },
            )
            patterns.append(note)
        
        return patterns
    
    def _extract_integration_patterns(self, trajectory: Dict[str, Any]) -> List[PatternNote]:
        """Extract integration/multi-model patterns"""
        patterns = []
        
        path = trajectory.get("integration_path", "")
        if path:
            note = PatternNote(
                id="",
                title=f"Integration Pattern: {path}",
                pattern_type=PatternType.INTEGRATION_PATTERN,
                problem=f"How to execute {path} workflow",
                solution=f"Execute models in sequence: {path}",
                keywords=["integration", "workflow", "models"],
                effectiveness=trajectory.get("success_rate", 0.8),
                metadata={
                    "path": path,
                    "models": trajectory.get("models", []),
                    "execution_time_ms": trajectory.get("execution_time_ms", 0),
                }
            )
            patterns.append(note)
        
        return patterns
    
    def _extract_performance_patterns(self, trajectory: Dict[str, Any]) -> List[PatternNote]:
        """Extract performance optimization patterns"""
        patterns = []
        
        if trajectory.get("latency_ms", 0) < 100:
            note = PatternNote(
                id="",
                title="Low-Latency Inference",
                pattern_type=PatternType.PERFORMANCE_PATTERN,
                problem="How to achieve sub-100ms inference",
                solution="Use caching and model optimization",
                keywords=["performance", "latency", "optimization"],
                effectiveness=0.95,
                metadata={
                    "latency_ms": trajectory.get("latency_ms"),
                    "throughput_req_s": trajectory.get("throughput", 0),
                    "optimization_technique": trajectory.get("technique", "unknown"),
                }
            )
            patterns.append(note)
        
        return patterns
    
    def _extract_error_patterns(self, trajectory: Dict[str, Any]) -> List[PatternNote]:
        """Extract error handling patterns"""
        patterns = []
        
        error = trajectory.get("error") or trajectory.get("exception")
        if error:
            note = PatternNote(
                id="",
                title=f"Error Handling: {type(error).__name__}",
                pattern_type=PatternType.ERROR_PATTERN,
                problem=f"Handle {type(error).__name__}",
                solution=trajectory.get("resolution", "Implement retry logic"),
                keywords=["error", "handling", "recovery"],
                effectiveness=trajectory.get("recovery_success_rate", 0.7),
                metadata={
                    "error_type": str(type(error)),
                    "resolution": trajectory.get("resolution"),
                    "recovery_attempts": trajectory.get("retry_count", 0),
                }
            )
            patterns.append(note)
        
        return patterns


# Demo and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("NGVT PATTERN NOTE-TAKING SYSTEM - DEMO")
    print("="*80)
    
    # Initialize store
    store = PatternNoteStore()
    print(f"\n✓ Created pattern store: {store.storage_dir}")
    
    # Create NoteTaker
    note_taker = NoteTaker(store)
    
    # Extract patterns from sample trajectory
    trajectory = {
        "query": "What is machine learning?",
        "response": "ML is...",
        "similar_queries": [
            {"query": "Tell me about ML", "similarity": 0.92},
            {"query": "Define ML", "similarity": 0.88},
        ],
        "similarity_score": 0.90,
        "similarity_threshold": 0.85,
        "integration_path": "nlp_base->vision_base",
        "models": ["nlp_base", "vision_base"],
        "execution_time_ms": 85,
        "success_rate": 0.95,
        "latency_ms": 85,
        "throughput": 50,
        "technique": "caching",
    }
    
    print("\n1. Extracting patterns from trajectory...")
    patterns = note_taker.extract_patterns_from_trajectory(trajectory)
    print(f"   Found {len(patterns)} patterns")
    
    print("\n2. Adding patterns to store...")
    for pattern in patterns:
        pattern_id = store.add_pattern(pattern)
        print(f"   ✓ {pattern.title} (ID: {pattern_id})")
    
    print("\n3. Searching patterns...")
    nlp_patterns = store.search_by_type(PatternType.NLP_PATTERN)
    print(f"   NLP Patterns: {len(nlp_patterns)}")
    
    similar = store.search_similar("query similarity embedding")
    print(f"   Similar to 'query similarity': {len(similar)} patterns")
    
    print("\n4. Top patterns by effectiveness...")
    top = store.get_top_patterns(limit=3)
    for p in top:
        print(f"   - {p.title} ({p.effectiveness:.2%})")
    
    print("\n5. Exporting to Markdown...")
    output_file = "/tmp/ngvt_patterns.md"
    store.export_as_markdown(output_file)
    print(f"   ✓ Exported to {output_file}")
    
    print("\n" + "="*80)
    print("✓ DEMO COMPLETE")
    print("="*80)
