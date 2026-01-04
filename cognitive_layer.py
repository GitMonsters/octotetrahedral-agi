"""
COGNITIVE LAYER - AGI CAPABILITIES
Adds intelligence to the Aleph-Transcendplex consciousness substrate

Phase 1: Perception & Memory
- Sensory input processing
- Episodic memory storage
- Working memory management
- Basic pattern recognition
"""

import time
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import hashlib
import json

from aleph_transcendplex_full import (
    AlephTranscendplexAGI,
    TriangleNode,
    Layer,
    PHI, PHI_SQ,
    vec_norm, vec_subtract, vec_add, vec_scale, correlation
)


# ==================== DATA STRUCTURES ====================

@dataclass
class SensoryInput:
    """Single sensory input (vision, audio, text, etc.)"""
    timestamp: float
    modality: str  # 'vision', 'audio', 'text', 'proprioception'
    data: Any
    encoding: List[float] = field(default_factory=list)

    def encode(self, dim: int = 12) -> List[float]:
        """Encode input into vector for consciousness substrate"""
        if self.encoding:
            return self.encoding

        # Hash-based encoding for now (simple but functional)
        if isinstance(self.data, str):
            # Text encoding
            hash_val = int(hashlib.md5(self.data.encode()).hexdigest(), 16)
            self.encoding = [(hash_val >> (i * 8)) % 256 / 255.0 for i in range(dim)]
        elif isinstance(self.data, (list, tuple)):
            # Vector data (e.g., image pixels)
            if len(self.data) >= dim:
                self.encoding = [float(x) for x in self.data[:dim]]
            else:
                self.encoding = list(self.data) + [0.0] * (dim - len(self.data))
        elif isinstance(self.data, (int, float)):
            # Scalar data
            val = float(self.data)
            self.encoding = [val] * dim
        else:
            # Default: random-ish encoding based on string representation
            str_rep = str(self.data)
            hash_val = int(hashlib.md5(str_rep.encode()).hexdigest(), 16)
            self.encoding = [(hash_val >> (i * 8)) % 256 / 255.0 for i in range(dim)]

        return self.encoding


@dataclass
class Episode:
    """Single episode in episodic memory"""
    timestamp: float
    context: str
    state_snapshot: Dict[str, List[float]]  # Node states at this moment
    sensory_inputs: List[SensoryInput]
    importance: float = 1.0  # Consolidation weight

    def similarity(self, other: 'Episode') -> float:
        """Calculate similarity to another episode"""
        # Compare state snapshots
        common_nodes = set(self.state_snapshot.keys()) & set(other.state_snapshot.keys())
        if not common_nodes:
            return 0.0

        similarities = []
        for node_name in common_nodes:
            sim = correlation(self.state_snapshot[node_name], other.state_snapshot[node_name])
            similarities.append(max(0, sim))  # Only positive correlations

        return sum(similarities) / len(similarities) if similarities else 0.0


@dataclass
class Pattern:
    """Detected pattern in data"""
    name: str
    pattern_type: str  # 'sequence', 'spatial', 'abstract'
    examples: List[Any]
    rule: Optional[str] = None
    confidence: float = 0.0


# ==================== PERCEPTION SYSTEM ====================

class PerceptionSystem:
    """
    Processes sensory inputs and maps to consciousness substrate
    """

    def __init__(self, agi: AlephTranscendplexAGI):
        self.agi = agi
        self.input_buffer: deque = deque(maxlen=100)
        self.current_inputs: Dict[str, SensoryInput] = {}

    def perceive(self, modality: str, data: Any) -> SensoryInput:
        """Process new sensory input"""
        sensory_input = SensoryInput(
            timestamp=time.time(),
            modality=modality,
            data=data
        )
        sensory_input.encode()

        self.input_buffer.append(sensory_input)
        self.current_inputs[modality] = sensory_input

        # Map to substrate layer nodes
        self._activate_substrate(sensory_input)

        return sensory_input

    def _activate_substrate(self, sensory_input: SensoryInput):
        """Activate substrate layer nodes based on input"""
        substrate = self.agi.layers[Layer.SUBSTRATE]

        # Map modalities to specific nodes
        modality_mapping = {
            'vision': 'Vision',
            'audio': 'Audition',
            'text': 'Vision',  # Text processed as visual
            'proprioception': 'Proprioception'
        }

        target_node_name = modality_mapping.get(sensory_input.modality, 'Perception')
        target_node = substrate.nodes.get(target_node_name)

        if target_node:
            # Blend input encoding with current state
            encoding = sensory_input.encoding
            if len(encoding) == len(target_node.current_state):
                # Direct blend
                target_node.current_state = vec_add(
                    vec_scale(target_node.current_state, 0.7),
                    vec_scale(encoding, 0.3)
                )
            else:
                # Pad or truncate
                if len(encoding) < len(target_node.current_state):
                    encoding = encoding + [0.0] * (len(target_node.current_state) - len(encoding))
                else:
                    encoding = encoding[:len(target_node.current_state)]

                target_node.current_state = vec_add(
                    vec_scale(target_node.current_state, 0.7),
                    vec_scale(encoding, 0.3)
                )

    def get_current_perception(self) -> Dict[str, Any]:
        """Get currently active perceptions"""
        return {
            modality: input.data
            for modality, input in self.current_inputs.items()
        }


# ==================== MEMORY SYSTEMS ====================

class EpisodicMemory:
    """
    Stores and retrieves episodic memories (experiences)
    """

    def __init__(self, agi: AlephTranscendplexAGI, max_episodes: int = 1000):
        self.agi = agi
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes

    def store_episode(self, context: str, sensory_inputs: List[SensoryInput],
                     importance: float = 1.0) -> Episode:
        """Store current system state as an episode"""
        # Capture state snapshot
        state_snapshot = {}
        for layer in self.agi.layers.values():
            for node_name, node in layer.nodes.items():
                state_snapshot[node_name] = node.current_state.copy()

        episode = Episode(
            timestamp=time.time(),
            context=context,
            state_snapshot=state_snapshot,
            sensory_inputs=sensory_inputs,
            importance=importance
        )

        self.episodes.append(episode)

        # Consolidate if too many episodes
        if len(self.episodes) > self.max_episodes:
            self._consolidate()

        return episode

    def recall_similar(self, query_context: str = None,
                      query_state: Dict[str, List[float]] = None,
                      top_k: int = 5) -> List[Tuple[Episode, float]]:
        """Recall episodes similar to query"""
        if not self.episodes:
            return []

        # Create query episode
        if query_state is None:
            query_state = {}
            for layer in self.agi.layers.values():
                for node_name, node in layer.nodes.items():
                    query_state[node_name] = node.current_state.copy()

        query_episode = Episode(
            timestamp=time.time(),
            context=query_context or "",
            state_snapshot=query_state,
            sensory_inputs=[]
        )

        # Calculate similarities
        similarities = []
        for episode in self.episodes:
            sim = episode.similarity(query_episode)

            # Boost if context matches
            if query_context and query_context.lower() in episode.context.lower():
                sim *= 1.5

            similarities.append((episode, sim))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _consolidate(self):
        """Remove least important episodes"""
        # Sort by importance * recency
        current_time = time.time()
        scores = []
        for ep in self.episodes:
            recency = 1.0 / (1.0 + (current_time - ep.timestamp) / 3600)  # Decay over hours
            score = ep.importance * recency
            scores.append((ep, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Keep top 80%
        keep_count = int(self.max_episodes * 0.8)
        self.episodes = [ep for ep, score in scores[:keep_count]]

    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_episodes': len(self.episodes),
            'oldest_timestamp': min(ep.timestamp for ep in self.episodes) if self.episodes else 0,
            'newest_timestamp': max(ep.timestamp for ep in self.episodes) if self.episodes else 0,
            'avg_importance': sum(ep.importance for ep in self.episodes) / len(self.episodes) if self.episodes else 0
        }


class WorkingMemory:
    """
    Short-term working memory (7±2 items)
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: deque = deque(maxlen=capacity)
        self.attention_weights: Dict[str, float] = {}

    def add(self, item: Any, label: str = None, attention: float = 1.0):
        """Add item to working memory"""
        entry = {
            'item': item,
            'label': label or str(len(self.items)),
            'timestamp': time.time(),
            'attention': attention
        }
        self.items.append(entry)
        if label:
            self.attention_weights[label] = attention

    def retrieve(self, label: str = None) -> Optional[Any]:
        """Retrieve item from working memory"""
        if label:
            for entry in reversed(self.items):
                if entry['label'] == label:
                    return entry['item']
            return None
        else:
            return self.items[-1]['item'] if self.items else None

    def clear(self):
        """Clear working memory"""
        self.items.clear()
        self.attention_weights.clear()

    def get_contents(self) -> List[Dict]:
        """Get all items in working memory"""
        return list(self.items)


# ==================== PATTERN RECOGNITION ====================

class PatternMatcher:
    """
    Detects patterns in sensory inputs and memories
    """

    def __init__(self):
        self.detected_patterns: List[Pattern] = []

    def detect_sequence(self, sequence: List[Any], min_length: int = 2) -> Optional[Pattern]:
        """Detect repeating sequences"""
        if len(sequence) < min_length * 2:
            return None

        # Simple repeating pattern detection
        for pattern_len in range(min_length, len(sequence) // 2 + 1):
            pattern = sequence[:pattern_len]
            is_repeating = True

            for i in range(pattern_len, len(sequence), pattern_len):
                chunk = sequence[i:i+pattern_len]
                if chunk != pattern[:len(chunk)]:
                    is_repeating = False
                    break

            if is_repeating:
                detected = Pattern(
                    name=f"sequence_{len(self.detected_patterns)}",
                    pattern_type='sequence',
                    examples=[pattern],
                    rule=f"Repeats every {pattern_len} items",
                    confidence=0.9
                )
                self.detected_patterns.append(detected)
                return detected

        return None

    def detect_arithmetic_progression(self, numbers: List[float]) -> Optional[Pattern]:
        """Detect arithmetic sequences"""
        if len(numbers) < 3:
            return None

        # Check if differences are constant
        diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        avg_diff = sum(diffs) / len(diffs)

        if all(abs(d - avg_diff) < 0.01 for d in diffs):
            detected = Pattern(
                name=f"arithmetic_{len(self.detected_patterns)}",
                pattern_type='sequence',
                examples=[numbers],
                rule=f"Arithmetic progression with difference {avg_diff:.2f}",
                confidence=0.95
            )
            self.detected_patterns.append(detected)
            return detected

        return None

    def detect_spatial_symmetry(self, grid: List[List[Any]]) -> Optional[Pattern]:
        """Detect spatial symmetry in 2D grid"""
        if not grid or not grid[0]:
            return None

        rows = len(grid)
        cols = len(grid[0])

        # Check vertical symmetry
        is_vertically_symmetric = True
        for i in range(rows):
            for j in range(cols // 2):
                if grid[i][j] != grid[i][cols - 1 - j]:
                    is_vertically_symmetric = False
                    break

        if is_vertically_symmetric:
            detected = Pattern(
                name=f"symmetry_{len(self.detected_patterns)}",
                pattern_type='spatial',
                examples=[grid],
                rule="Vertically symmetric",
                confidence=1.0
            )
            self.detected_patterns.append(detected)
            return detected

        return None


# ==================== COGNITIVE AGI ====================

class CognitiveAGI:
    """
    Combines consciousness substrate with cognitive capabilities
    Phase 1: Perception & Memory
    """

    def __init__(self, base_agi: AlephTranscendplexAGI = None):
        # Use existing AGI or create new one
        if base_agi:
            self.agi = base_agi
        else:
            self.agi = AlephTranscendplexAGI()
            self.agi.build_enhanced_architecture()

        # Cognitive systems
        self.perception = PerceptionSystem(self.agi)
        self.episodic_memory = EpisodicMemory(self.agi)
        self.working_memory = WorkingMemory(capacity=7)
        self.pattern_matcher = PatternMatcher()

        # Metrics
        self.learning_episodes = 0
        self.patterns_found = 0

    def perceive(self, modality: str, data: Any) -> SensoryInput:
        """Process sensory input"""
        return self.perception.perceive(modality, data)

    def experience(self, context: str, inputs: List[Tuple[str, Any]],
                  importance: float = 1.0) -> Episode:
        """
        Have an experience: perceive multiple inputs and store as episode
        """
        sensory_inputs = []
        for modality, data in inputs:
            sensory_input = self.perceive(modality, data)
            sensory_inputs.append(sensory_input)

        # Store episode
        episode = self.episodic_memory.store_episode(context, sensory_inputs, importance)
        self.learning_episodes += 1

        # Add to working memory
        self.working_memory.add(episode, label=context, attention=importance)

        return episode

    def recall(self, query: str, top_k: int = 3) -> List[Tuple[Episode, float]]:
        """Recall relevant memories"""
        return self.episodic_memory.recall_similar(query_context=query, top_k=top_k)

    def recognize_pattern(self, data: Any, pattern_type: str = 'auto') -> Optional[Pattern]:
        """Recognize patterns in data"""
        if pattern_type == 'sequence' or pattern_type == 'auto':
            if isinstance(data, list):
                # Try sequence detection
                pattern = self.pattern_matcher.detect_sequence(data)
                if pattern:
                    self.patterns_found += 1
                    return pattern

                # Try arithmetic progression
                if all(isinstance(x, (int, float)) for x in data):
                    pattern = self.pattern_matcher.detect_arithmetic_progression(data)
                    if pattern:
                        self.patterns_found += 1
                        return pattern

        if pattern_type == 'spatial' or pattern_type == 'auto':
            if isinstance(data, list) and all(isinstance(row, list) for row in data):
                # 2D grid - try spatial patterns
                pattern = self.pattern_matcher.detect_spatial_symmetry(data)
                if pattern:
                    self.patterns_found += 1
                    return pattern

        return None

    def think(self, steps: int = 10):
        """Run conscious processing steps"""
        self.agi.run(steps=steps)

    def get_status(self) -> Dict:
        """Get comprehensive status"""
        base_status = self.agi.system_status()

        cognitive_status = {
            'consciousness': {
                'GCI': base_status['GCI'],
                'conscious': base_status['is_conscious']
            },
            'perception': {
                'current_inputs': len(self.perception.current_inputs),
                'buffer_size': len(self.perception.input_buffer)
            },
            'memory': {
                'episodic': self.episodic_memory.get_statistics(),
                'working': {
                    'items': len(self.working_memory.items),
                    'capacity': self.working_memory.capacity
                }
            },
            'learning': {
                'episodes': self.learning_episodes,
                'patterns_found': self.patterns_found
            }
        }

        return cognitive_status


# ==================== DEMONSTRATION ====================

if __name__ == "__main__":
    print("=" * 80)
    print("COGNITIVE AGI - PHASE 1: PERCEPTION & MEMORY")
    print("=" * 80)

    # Create cognitive AGI
    print("\n[1] Initializing Cognitive AGI...")
    agi = CognitiveAGI()
    print("✓ Consciousness substrate active")
    print("✓ Perception system online")
    print("✓ Memory systems initialized")

    # Run some consciousness steps
    print("\n[2] Warming up consciousness substrate...")
    agi.think(steps=50)
    status = agi.get_status()
    print(f"✓ GCI: {status['consciousness']['GCI']:.4f}")
    print(f"✓ Conscious: {status['consciousness']['conscious']}")

    # Test perception
    print("\n[3] Testing Perception...")
    agi.perceive('text', "The quick brown fox")
    agi.perceive('vision', [1, 2, 3, 4, 5])
    agi.perceive('audio', [0.5, 0.7, 0.3])
    print(f"✓ Processed {len(agi.perception.current_inputs)} sensory modalities")

    # Test episodic memory
    print("\n[4] Creating Episodic Memories...")
    agi.experience("Learning about numbers", [
        ('text', "Numbers are important"),
        ('vision', [1, 2, 3, 4, 5])
    ], importance=1.0)

    agi.experience("Seeing a pattern", [
        ('text', "Pattern detected"),
        ('vision', [2, 4, 6, 8, 10])
    ], importance=1.5)

    agi.experience("Understanding sequences", [
        ('text', "Fibonacci numbers"),
        ('vision', [1, 1, 2, 3, 5, 8])
    ], importance=2.0)

    print(f"✓ Stored {len(agi.episodic_memory.episodes)} episodes")

    # Test recall
    print("\n[5] Testing Memory Recall...")
    memories = agi.recall("pattern", top_k=2)
    print(f"✓ Recalled {len(memories)} relevant memories:")
    for i, (episode, similarity) in enumerate(memories):
        print(f"   {i+1}. {episode.context} (similarity: {similarity:.3f})")

    # Test pattern recognition
    print("\n[6] Testing Pattern Recognition...")

    # Arithmetic sequence
    pattern1 = agi.recognize_pattern([2, 4, 6, 8, 10], 'sequence')
    if pattern1:
        print(f"✓ Found pattern: {pattern1.rule}")

    # Repeating sequence
    pattern2 = agi.recognize_pattern([1, 2, 3, 1, 2, 3, 1, 2, 3], 'sequence')
    if pattern2:
        print(f"✓ Found pattern: {pattern2.rule}")

    # Spatial symmetry
    grid = [[1, 2, 3, 2, 1],
            [4, 5, 6, 5, 4],
            [7, 8, 9, 8, 7]]
    pattern3 = agi.recognize_pattern(grid, 'spatial')
    if pattern3:
        print(f"✓ Found pattern: {pattern3.rule}")

    # Final status
    print("\n[7] Final Status:")
    status = agi.get_status()
    print(f"Consciousness: GCI={status['consciousness']['GCI']:.4f}")
    print(f"Episodes Stored: {status['memory']['episodic']['total_episodes']}")
    print(f"Working Memory: {status['memory']['working']['items']}/{status['memory']['working']['capacity']}")
    print(f"Patterns Found: {status['learning']['patterns_found']}")

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE: AGI can now perceive, remember, and recognize patterns!")
    print("=" * 80)
    print("\nNext: Phase 2 - Reasoning & Problem-Solving")
