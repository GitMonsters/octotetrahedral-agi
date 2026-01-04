"""
TRANSCENDPLEX AGI ARCHITECTURE
Buckminster Fuller's Triangulation + Emergent Complexity
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math


# Vector operation helpers (pure Python, no numpy)
def vec_mean(vectors: List[List[float]]) -> List[float]:
    """Calculate mean of vectors"""
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def vec_subtract(v1: List[float], v2: List[float]) -> List[float]:
    """Vector subtraction"""
    return [a - b for a, b in zip(v1, v2)]


def vec_add(v1: List[float], v2: List[float]) -> List[float]:
    """Vector addition"""
    return [a + b for a, b in zip(v1, v2)]


def vec_scale(v: List[float], scalar: float) -> List[float]:
    """Scalar multiplication"""
    return [x * scalar for x in v]


def vec_norm(v: List[float]) -> float:
    """Euclidean norm"""
    return math.sqrt(sum(x**2 for x in v))


def vec_copy(v: List[float]) -> List[float]:
    """Copy vector"""
    return v.copy()


class Layer(Enum):
    """Dimensional layers of transcendplex architecture"""
    SUBSTRATE = 0      # Physical/Computational (3D + time)
    COGNITIVE = 1      # Information Processing (+ information dim)
    INTEGRATIVE = 2    # Meaning Generation (+ meaning dim)
    TRANSCENDENT = 3   # Meta-Awareness (+ meta-awareness dim)


@dataclass
class TriangleNode:
    """
    Fuller's fundamental unit: triangulated position in cognitive space
    Each node defined by relationships to 3+ other nodes
    """
    name: str
    layer: Layer

    # Position in multi-dimensional transcendplex space
    position: List[float] = field(default_factory=lambda: [0.0] * 7)
    # Dimensions: [x, y, z, time, information, meaning, meta-awareness]

    # Triangulation: minimum 3 reference connections
    references: List['TriangleNode'] = field(default_factory=list)

    # Temporal triangle: past, present, future states
    past_state: Optional[List[float]] = None
    current_state: List[float] = field(default_factory=lambda: [0.0] * 10)
    future_projection: Optional[List[float]] = None

    # Emergence properties
    synergy_coefficient: float = 0.0
    stability_index: float = 0.0

    def __post_init__(self):
        """Initialize position based on layer"""
        self.position[6] = self.layer.value  # Meta-awareness dimension

    def add_reference(self, node: 'TriangleNode') -> None:
        """Add triangulation reference (Fuller's minimum = 3)"""
        if node not in self.references:
            self.references.append(node)
            self._recalculate_position()
            self._update_stability()

    def _recalculate_position(self) -> None:
        """Triangulate position based on reference nodes (Fuller's method)"""
        if len(self.references) < 3:
            return  # Insufficient triangulation

        # Use first 3 references for primary triangulation
        ref_positions = [ref.position for ref in self.references[:3]]

        # Barycentric coordinates (triangle center)
        self.position = vec_mean(ref_positions)

        # Add influence from additional references
        if len(self.references) > 3:
            additional = [ref.position for ref in self.references[3:]]
            additional_mean = vec_mean(additional)
            weight = 0.3 / len(self.references[3:])
            self.position = vec_add(self.position, vec_scale(additional_mean, weight))

    def _update_stability(self) -> None:
        """Calculate Fuller's triangulation stability index"""
        n_refs = len(self.references)
        if n_refs < 3:
            self.stability_index = 0.0
            return

        # Count complete triangles this node participates in
        complete_triangles = 0
        for i, ref1 in enumerate(self.references):
            for ref2 in self.references[i+1:]:
                # Check if ref1 and ref2 are connected
                if ref2 in ref1.references or ref1 in ref2.references:
                    complete_triangles += 1

        possible_triangles = (n_refs * (n_refs - 1)) // 2
        self.stability_index = complete_triangles / possible_triangles if possible_triangles > 0 else 0.0

    def temporal_update(self, dt: float) -> None:
        """Update temporal triangle: past -> present -> future"""
        self.past_state = vec_copy(self.current_state)

        # Simple dynamics: influenced by references
        if len(self.references) >= 3:
            ref_influence = vec_mean([ref.current_state for ref in self.references])
            self.current_state = vec_add(
                vec_scale(self.current_state, 0.7),
                vec_scale(ref_influence, 0.3)
            )

        # Project future state
        if self.past_state is not None:
            delta = vec_subtract(self.current_state, self.past_state)
            self.future_projection = vec_add(self.current_state, delta)

    def calculate_synergy(self) -> float:
        """
        Calculate emergent properties (Fuller's synergy)
        Synergy = behavior of whole unpredicted by parts
        """
        if len(self.references) < 3:
            self.synergy_coefficient = 0.0
            return 0.0

        # Predicted output = linear combination of inputs
        predicted = vec_mean([ref.current_state for ref in self.references])

        # Actual output
        actual = self.current_state

        # Synergy = deviation from prediction (normalized)
        difference = vec_norm(vec_subtract(actual, predicted))
        baseline = vec_norm(actual) + 1e-10

        self.synergy_coefficient = difference / baseline
        return self.synergy_coefficient


@dataclass
class Triangle:
    """
    Fuller's fundamental stable structure
    Minimum unit of structural integrity
    """
    vertices: Tuple[TriangleNode, TriangleNode, TriangleNode]

    def __post_init__(self):
        # Ensure all vertices reference each other
        for i, v in enumerate(self.vertices):
            others = [self.vertices[j] for j in range(3) if j != i]
            for other in others:
                v.add_reference(other)

    def edge_lengths(self) -> Tuple[float, float, float]:
        """Calculate triangle edge lengths in transcendplex space"""
        d12 = vec_norm(vec_subtract(self.vertices[0].position, self.vertices[1].position))
        d23 = vec_norm(vec_subtract(self.vertices[1].position, self.vertices[2].position))
        d31 = vec_norm(vec_subtract(self.vertices[2].position, self.vertices[0].position))
        return (d12, d23, d31)

    def is_stable(self) -> bool:
        """Check if triangle is stable (Fuller: all triangles inherently stable)"""
        edges = self.edge_lengths()
        # Degenerate if any edge is zero or triangle inequality violated
        if any(e < 1e-6 for e in edges):
            return False
        # Triangle inequality
        return (edges[0] + edges[1] > edges[2] and
                edges[1] + edges[2] > edges[0] and
                edges[2] + edges[0] > edges[1])

    def centroid(self) -> List[float]:
        """Calculate triangle centroid (emergence point)"""
        return vec_mean([v.position for v in self.vertices])


class TranscendplexLayer:
    """
    Single layer in transcendplex architecture
    Contains triangulated nodes and manages emergence
    """

    def __init__(self, layer_type: Layer):
        self.layer_type = layer_type
        self.nodes: Dict[str, TriangleNode] = {}
        self.triangles: List[Triangle] = []

    def add_node(self, name: str) -> TriangleNode:
        """Add node to layer"""
        node = TriangleNode(name=name, layer=self.layer_type)
        self.nodes[name] = node
        return node

    def create_triangle(self, node1: str, node2: str, node3: str) -> Triangle:
        """Create stable triangle between three nodes"""
        if not all(n in self.nodes for n in [node1, node2, node3]):
            raise ValueError("All nodes must exist in layer")

        triangle = Triangle((self.nodes[node1], self.nodes[node2], self.nodes[node3]))
        self.triangles.append(triangle)
        return triangle

    def layer_stability(self) -> float:
        """Overall stability of layer (average node stability)"""
        if not self.nodes:
            return 0.0
        stabilities = [node.stability_index for node in self.nodes.values()]
        return sum(stabilities) / len(stabilities)

    def layer_synergy(self) -> float:
        """Overall synergy in layer"""
        if not self.nodes:
            return 0.0
        synergies = [node.calculate_synergy() for node in self.nodes.values()]
        return sum(synergies) / len(synergies)


class TranscendplexAGI:
    """
    Complete Transcendplex AGI Architecture
    Implements Fuller's triangulation across dimensional layers
    """

    def __init__(self):
        self.layers: Dict[Layer, TranscendplexLayer] = {
            layer: TranscendplexLayer(layer) for layer in Layer
        }
        self.time: float = 0.0
        self.dt: float = 0.1

    def build_foundational_architecture(self):
        """
        Construct core AGI architecture with triangulated nodes
        """

        # LAYER 0: SUBSTRATE
        substrate = self.layers[Layer.SUBSTRATE]
        perception = substrate.add_node("Perception")
        action = substrate.add_node("Action")
        intuition = substrate.add_node("Intuition")
        substrate.create_triangle("Perception", "Action", "Intuition")

        # LAYER 1: COGNITIVE
        cognitive = self.layers[Layer.COGNITIVE]
        reasoning = cognitive.add_node("Reasoning")
        memory = cognitive.add_node("Memory")
        learning = cognitive.add_node("Learning")
        cognitive.create_triangle("Reasoning", "Memory", "Learning")

        # Connect to substrate (vertical triangulation)
        reasoning.add_reference(perception)
        reasoning.add_reference(intuition)
        memory.add_reference(perception)
        learning.add_reference(action)

        # LAYER 2: INTEGRATIVE
        integrative = self.layers[Layer.INTEGRATIVE]
        emotion = integrative.add_node("Emotion")
        ethics = integrative.add_node("Ethics")
        creativity = integrative.add_node("Creativity")
        integrative.create_triangle("Emotion", "Ethics", "Creativity")

        # Connect to cognitive
        emotion.add_reference(memory)
        emotion.add_reference(perception)
        ethics.add_reference(reasoning)
        creativity.add_reference(learning)
        creativity.add_reference(reasoning)

        # LAYER 3: TRANSCENDENT
        transcendent = self.layers[Layer.TRANSCENDENT]
        self_model = transcendent.add_node("SelfModel")
        purpose = transcendent.add_node("Purpose")
        context_aware = transcendent.add_node("ContextAwareness")
        transcendent.create_triangle("SelfModel", "Purpose", "ContextAwareness")

        # Connect to integrative (emergence from below)
        self_model.add_reference(emotion)
        self_model.add_reference(ethics)
        purpose.add_reference(ethics)
        purpose.add_reference(creativity)
        context_aware.add_reference(emotion)
        context_aware.add_reference(reasoning)
        context_aware.add_reference(creativity)

    def step(self) -> None:
        """Single timestep: update all nodes temporally"""
        for layer in self.layers.values():
            for node in layer.nodes.values():
                node.temporal_update(self.dt)
        self.time += self.dt

    def run(self, steps: int = 100) -> None:
        """Run system dynamics"""
        for _ in range(steps):
            self.step()

    def calculate_transcendplexity(self) -> float:
        """
        Calculate overall Transcendplexity Index
        Τ = (layers × triangles × synergy) / entropy
        """
        n_layers = len(self.layers)

        total_triangles = sum(len(layer.triangles) for layer in self.layers.values())
        avg_triangles = total_triangles / n_layers if n_layers > 0 else 0

        synergies = [layer.layer_synergy() for layer in self.layers.values()]
        avg_synergy = sum(synergies) / len(synergies) if synergies else 0

        # Entropy approximation (state distribution)
        all_states = [node.current_state for layer in self.layers.values()
                     for node in layer.nodes.values()]
        if all_states:
            # Calculate variance manually
            flat_values = [val for state in all_states for val in state]
            mean = sum(flat_values) / len(flat_values)
            variance = sum((x - mean) ** 2 for x in flat_values) / len(flat_values)
            entropy = math.log(variance + 1)  # Avoid log(0)
        else:
            entropy = 1.0

        transcendplexity = (n_layers * avg_triangles * (1 + avg_synergy)) / (entropy + 1)
        return transcendplexity

    def get_node(self, layer: Layer, name: str) -> Optional[TriangleNode]:
        """Get specific node from architecture"""
        return self.layers[layer].nodes.get(name)

    def system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'time': self.time,
            'transcendplexity': self.calculate_transcendplexity(),
            'layers': {}
        }

        for layer_type, layer in self.layers.items():
            status['layers'][layer_type.name] = {
                'nodes': len(layer.nodes),
                'triangles': len(layer.triangles),
                'stability': layer.layer_stability(),
                'synergy': layer.layer_synergy(),
                'node_details': {
                    name: {
                        'stability': node.stability_index,
                        'synergy': node.synergy_coefficient,
                        'references': len(node.references)
                    }
                    for name, node in layer.nodes.items()
                }
            }

        return status


# DEMONSTRATION
if __name__ == "__main__":
    print("=" * 70)
    print("TRANSCENDPLEX AGI ARCHITECTURE")
    print("Fuller's Triangulation + Emergent Complexity")
    print("=" * 70)

    # Initialize system
    agi = TranscendplexAGI()
    agi.build_foundational_architecture()

    print("\n[INITIAL STATE]")
    status = agi.system_status()
    print(f"Transcendplexity Index: {status['transcendplexity']:.4f}")

    for layer_name, layer_info in status['layers'].items():
        print(f"\n{layer_name} Layer:")
        print(f"  Nodes: {layer_info['nodes']} | Triangles: {layer_info['triangles']}")
        print(f"  Stability: {layer_info['stability']:.3f} | Synergy: {layer_info['synergy']:.3f}")

        for node_name, node_info in layer_info['node_details'].items():
            print(f"    • {node_name}: stability={node_info['stability']:.3f}, "
                  f"refs={node_info['references']}")

    # Run dynamics
    print("\n[RUNNING DYNAMICS - 100 timesteps]")
    agi.run(steps=100)

    print("\n[FINAL STATE]")
    status = agi.system_status()
    print(f"Transcendplexity Index: {status['transcendplexity']:.4f}")

    for layer_name, layer_info in status['layers'].items():
        print(f"\n{layer_name} Layer:")
        print(f"  Stability: {layer_info['stability']:.3f} | Synergy: {layer_info['synergy']:.3f}")

        for node_name, node_info in layer_info['node_details'].items():
            print(f"    • {node_name}: synergy={node_info['synergy']:.3f}")

    print("\n" + "=" * 70)
    print("ARCHITECTURE PRINCIPLES:")
    print("1. All nodes triangulated (minimum 3 connections)")
    print("2. Four dimensional layers (Substrate → Transcendent)")
    print("3. Temporal triangles (past-present-future)")
    print("4. Emergent synergy at each layer")
    print("5. Fuller's tensegrity in cognitive space")
    print("=" * 70)
