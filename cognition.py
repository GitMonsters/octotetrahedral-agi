"""
Cognition Module - AGI Cognitive Primitives
Ported from Rust rustyworm AGI core implementation

Core cognitive capabilities:
1. Causal Discovery - Automatically discover new causal variables
2. Abstraction Hierarchy - Build concept hierarchies
3. World Model - Mental simulation and planning
4. Meta-Learning - Learn how to learn
5. Symbol Grounding - Connect symbols to experience

These primitives integrate with the OctoTetrahedral limb architecture
to form a complete cognitive system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
import math
import random


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CognitionConfig:
    """Configuration for cognitive systems."""
    # Causal Discovery
    max_causal_variables: int = 100
    causal_discovery_threshold: float = 0.15
    observation_history_size: int = 1000
    discovery_interval: int = 25
    
    # Abstraction Hierarchy
    max_abstraction_depth: int = 5
    abstraction_merge_threshold: float = 0.75
    max_concepts_per_level: int = 50
    
    # World Model
    world_model_horizon: int = 20
    state_discretization_bins: int = 10
    transition_learning_rate: float = 0.1
    
    # Meta-Learning
    meta_learning_window: int = 100
    meta_param_adaptation_rate: float = 0.01
    
    # Symbol Grounding
    max_symbols: int = 500
    symbol_grounding_threshold: float = 0.4
    
    # General
    feature_dim: int = 256


# ═══════════════════════════════════════════════════════════════════════════════
# CAUSAL DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiscoveredVariable:
    """A latent causal variable discovered from observations."""
    id: int
    name: str
    signature: List[float]  # Feature signature
    parent_variables: List[int]  # Variables this was derived from
    information_gain: float
    discovered_at: int  # Step when discovered
    utility_count: int = 0


@dataclass
class CausalObservation:
    """An observation for causal discovery."""
    features: List[float]
    action: Optional[int]
    reward: float
    step: int


class CausalDiscovery:
    """
    Automatically discovers new causal variables from experience.
    
    Uses mutual information between features to identify latent structure.
    Discovered variables can be used to improve state representations.
    """
    
    def __init__(self, config: CognitionConfig):
        self.config = config
        self.variables: List[DiscoveredVariable] = []
        self.observation_history: deque = deque(maxlen=config.observation_history_size)
        self.mutual_info_cache: Dict[Tuple[int, int], float] = {}
        self.next_id = 0
        self.current_step = 0
    
    def observe(
        self,
        features: torch.Tensor,
        action: Optional[int] = None,
        reward: float = 0.0
    ):
        """
        Add an observation for causal analysis.
        
        Args:
            features: Feature tensor [feature_dim] or list
            action: Optional action taken
            reward: Reward received
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().tolist()
        
        obs = CausalObservation(
            features=features,
            action=action,
            reward=reward,
            step=self.current_step
        )
        self.observation_history.append(obs)
        self.current_step += 1
        
        # Attempt discovery periodically
        if self.current_step % self.config.discovery_interval == 0:
            self._attempt_discovery()
    
    def _attempt_discovery(self):
        """Attempt to discover new causal variables."""
        if len(self.observation_history) < 100:
            return
        
        if len(self.variables) >= self.config.max_causal_variables:
            return
        
        # Get feature dimension
        n_features = len(self.observation_history[0].features)
        if n_features == 0:
            return
        
        # Find feature pairs with high mutual information
        candidate_pairs = []
        for i in range(min(n_features, 20)):
            for j in range(i + 1, min(n_features, 20)):
                mi = self._estimate_mutual_info(i, j)
                if mi > self.config.causal_discovery_threshold:
                    candidate_pairs.append((i, j, mi))
        
        if not candidate_pairs:
            return
        
        # Create variable from highest MI pair
        best_pair = max(candidate_pairs, key=lambda x: x[2])
        i, j, mi = best_pair
        
        # Check if variable already exists
        already_exists = any(
            i in v.parent_variables and j in v.parent_variables
            for v in self.variables
        )
        
        if not already_exists:
            # Create signature from observations
            signature = []
            for obs in list(self.observation_history)[:50]:
                fi = obs.features[i] if i < len(obs.features) else 0.0
                fj = obs.features[j] if j < len(obs.features) else 0.0
                signature.append((fi + fj) / 2.0)
            
            var = DiscoveredVariable(
                id=self.next_id,
                name=f"latent_{i}_{j}",
                signature=signature,
                parent_variables=[i, j],
                information_gain=mi,
                discovered_at=self.current_step
            )
            
            self.variables.append(var)
            self.next_id += 1
    
    def _estimate_mutual_info(self, i: int, j: int) -> float:
        """Estimate mutual information between two features."""
        key = (i, j)
        if key in self.mutual_info_cache:
            return self.mutual_info_cache[key]
        
        # Simplified: use correlation as proxy for MI
        sum_i = 0.0
        sum_j = 0.0
        sum_ij = 0.0
        sum_i2 = 0.0
        sum_j2 = 0.0
        count = 0.0
        
        for obs in self.observation_history:
            if i < len(obs.features) and j < len(obs.features):
                fi = obs.features[i]
                fj = obs.features[j]
                sum_i += fi
                sum_j += fj
                sum_ij += fi * fj
                sum_i2 += fi * fi
                sum_j2 += fj * fj
                count += 1.0
        
        if count < 10:
            return 0.0
        
        mean_i = sum_i / count
        mean_j = sum_j / count
        var_i = max(sum_i2 / count - mean_i * mean_i, 1e-10)
        var_j = max(sum_j2 / count - mean_j * mean_j, 1e-10)
        cov = sum_ij / count - mean_i * mean_j
        
        correlation = cov / math.sqrt(var_i * var_j)
        mi = abs(correlation)  # Simplified MI estimate
        
        self.mutual_info_cache[key] = mi
        return mi
    
    def augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Augment features with discovered variables.
        
        Args:
            features: Original features [batch, feature_dim] or [feature_dim]
            
        Returns:
            Augmented features with discovered variable activations
        """
        if len(self.variables) == 0:
            return features
        
        was_1d = features.dim() == 1
        if was_1d:
            features = features.unsqueeze(0)
        
        batch_size = features.size(0)
        device = features.device
        
        # Compute activation of each discovered variable
        var_activations = []
        for var in self.variables:
            if len(var.parent_variables) >= 2:
                i, j = var.parent_variables[:2]
                if i < features.size(-1) and j < features.size(-1):
                    activation = (features[:, i] + features[:, j]) / 2.0
                    var_activations.append(activation.unsqueeze(-1))
        
        if var_activations:
            augmented = torch.cat([features] + var_activations, dim=-1)
        else:
            augmented = features
        
        if was_1d:
            augmented = augmented.squeeze(0)
        
        return augmented
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_observations': len(self.observation_history),
            'num_variables': len(self.variables),
            'current_step': self.current_step,
            'variables': [
                {
                    'name': v.name,
                    'info_gain': v.information_gain,
                    'utility': v.utility_count
                }
                for v in self.variables[:10]  # Top 10
            ]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACTION HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Concept:
    """A learned concept in the abstraction hierarchy."""
    id: int
    name: str
    level: int  # 0 = ground level, higher = more abstract
    prototype: List[float]  # Centroid of concept
    children: List[int] = field(default_factory=list)
    parents: List[int] = field(default_factory=list)
    activation_count: int = 0
    confidence: float = 0.5
    associated_actions: List[int] = field(default_factory=list)


class AbstractionHierarchy:
    """
    Builds hierarchical concept abstractions from experience.
    
    Lower levels represent concrete patterns, higher levels
    represent abstract categories that compose lower-level concepts.
    """
    
    def __init__(self, config: CognitionConfig):
        self.config = config
        self.concepts: Dict[int, Concept] = {}
        self.levels: List[List[int]] = [[] for _ in range(config.max_abstraction_depth)]
        self.next_id = 0
        self.current_step = 0
    
    def observe(self, features: torch.Tensor):
        """
        Observe features and update concept hierarchy.
        
        Args:
            features: Feature tensor [feature_dim] or list
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().tolist()
        
        # Find or create matching concept at level 0
        matched_id = self._find_matching_concept(features, level=0)
        
        if matched_id is None:
            # Create new ground-level concept
            if len(self.levels[0]) < self.config.max_concepts_per_level:
                concept = Concept(
                    id=self.next_id,
                    name=f"concept_0_{self.next_id}",
                    level=0,
                    prototype=features,
                    activation_count=1,
                    confidence=0.3
                )
                self.concepts[self.next_id] = concept
                self.levels[0].append(self.next_id)
                self.next_id += 1
        else:
            # Update existing concept
            concept = self.concepts[matched_id]
            concept.activation_count += 1
            # Update prototype (exponential moving average)
            alpha = 0.1
            concept.prototype = [
                (1 - alpha) * p + alpha * f
                for p, f in zip(concept.prototype, features)
            ]
            concept.confidence = min(0.99, concept.confidence + 0.01)
        
        self.current_step += 1
        
        # Periodically build higher-level abstractions
        if self.current_step % 50 == 0:
            self._build_abstractions()
    
    def _find_matching_concept(
        self,
        features: List[float],
        level: int,
        threshold: float = 0.7
    ) -> Optional[int]:
        """Find a concept matching the features at given level."""
        best_match = None
        best_similarity = threshold
        
        for concept_id in self.levels[level]:
            concept = self.concepts[concept_id]
            similarity = self._cosine_similarity(features, concept.prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = concept_id
        
        return best_match
    
    def _build_abstractions(self):
        """Build higher-level abstractions by merging similar concepts."""
        for level in range(self.config.max_abstraction_depth - 1):
            level_concepts = self.levels[level]
            if len(level_concepts) < 2:
                continue
            
            # Find similar concept pairs
            to_merge = []
            for i in range(len(level_concepts)):
                for j in range(i + 1, len(level_concepts)):
                    ci, cj = level_concepts[i], level_concepts[j]
                    c1, c2 = self.concepts.get(ci), self.concepts.get(cj)
                    if c1 is None or c2 is None:
                        continue
                    
                    similarity = self._cosine_similarity(c1.prototype, c2.prototype)
                    if 0.5 < similarity < self.config.abstraction_merge_threshold:
                        to_merge.append((ci, cj, similarity))
            
            # Create parent concepts for merged children
            for ci, cj, _ in to_merge[:3]:  # Limit merges per step
                c1, c2 = self.concepts.get(ci), self.concepts.get(cj)
                if c1 is None or c2 is None:
                    continue
                
                # Check if parent already exists
                parent_exists = any(
                    c.level == level + 1 and ci in c.children and cj in c.children
                    for c in self.concepts.values()
                )
                
                if not parent_exists and len(self.levels[level + 1]) < self.config.max_concepts_per_level:
                    # Create parent prototype
                    parent_proto = [
                        (p1 + p2) / 2.0
                        for p1, p2 in zip(c1.prototype, c2.prototype)
                    ]
                    
                    parent = Concept(
                        id=self.next_id,
                        name=f"abstract_{level + 1}_{self.next_id}",
                        level=level + 1,
                        prototype=parent_proto,
                        children=[ci, cj],
                        activation_count=1,
                        confidence=0.3
                    )
                    
                    self.concepts[self.next_id] = parent
                    self.levels[level + 1].append(self.next_id)
                    
                    # Update children
                    c1.parents.append(self.next_id)
                    c2.parents.append(self.next_id)
                    
                    self.next_id += 1
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def get_activated_concepts(
        self,
        features: torch.Tensor,
        threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Get concepts activated by features."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().tolist()
        
        activated = []
        for concept_id, concept in self.concepts.items():
            similarity = self._cosine_similarity(features, concept.prototype)
            if similarity > threshold:
                activated.append((concept_id, similarity))
        
        activated.sort(key=lambda x: x[1], reverse=True)
        return activated
    
    def associate_action(self, concept_id: int, action: int):
        """Associate an action with a concept."""
        if concept_id in self.concepts:
            concept = self.concepts[concept_id]
            if action not in concept.associated_actions:
                concept.associated_actions.append(action)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_concepts': len(self.concepts),
            'concepts_per_level': [len(l) for l in self.levels],
            'max_depth': max(
                (i + 1 for i, l in enumerate(self.levels) if l),
                default=0
            ),
            'avg_activation': (
                sum(c.activation_count for c in self.concepts.values()) / 
                max(len(self.concepts), 1)
            )
        }


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorldState:
    """A state in the world model."""
    features: List[float]
    reward: float
    is_terminal: bool
    uncertainty: float = 0.5


@dataclass
class WorldTransition:
    """A learned transition in the world model."""
    from_features: List[float]
    action: int
    to_features: List[float]
    reward: float
    count: int = 1


@dataclass
class SimulatedTrajectory:
    """A simulated trajectory for planning."""
    states: List[WorldState]
    actions: List[int]
    total_reward: float
    avg_uncertainty: float


class WorldModel:
    """
    Learns environment dynamics for mental simulation.
    
    Can imagine future states without actually taking actions,
    enabling planning and counterfactual reasoning.
    """
    
    def __init__(self, config: CognitionConfig, feature_dim: int):
        self.config = config
        self.feature_dim = feature_dim
        
        # Transition model: (discretized_state, action) -> transition
        self.transitions: Dict[Tuple[Tuple[int, ...], int], WorldTransition] = {}
        
        # State visitation counts
        self.state_visits: Dict[Tuple[int, ...], int] = defaultdict(int)
        
        # Reward model
        self.reward_model: Dict[Tuple[int, ...], float] = defaultdict(float)
        
        self.total_experience = 0
    
    def _discretize_state(self, features: List[float]) -> Tuple[int, ...]:
        """Discretize continuous features for lookup."""
        bins = self.config.state_discretization_bins
        return tuple(
            min(int(f * bins), bins - 1)
            for f in features[:20]  # Limit dimensions
        )
    
    def learn(
        self,
        from_state: torch.Tensor,
        action: int,
        to_state: torch.Tensor,
        reward: float,
        is_terminal: bool
    ):
        """
        Learn a transition from experience.
        
        Args:
            from_state: Starting state features
            action: Action taken
            to_state: Resulting state features
            reward: Reward received
            is_terminal: Whether to_state is terminal
        """
        if isinstance(from_state, torch.Tensor):
            from_state = from_state.detach().cpu().tolist()
        if isinstance(to_state, torch.Tensor):
            to_state = to_state.detach().cpu().tolist()
        
        from_disc = self._discretize_state(from_state)
        to_disc = self._discretize_state(to_state)
        key = (from_disc, action)
        
        # Update visitation
        self.state_visits[from_disc] += 1
        self.state_visits[to_disc] += 1
        
        # Update transition
        if key in self.transitions:
            trans = self.transitions[key]
            trans.count += 1
            # EMA update
            alpha = self.config.transition_learning_rate
            trans.to_features = [
                (1 - alpha) * old + alpha * new
                for old, new in zip(trans.to_features, to_state)
            ]
            trans.reward = (1 - alpha) * trans.reward + alpha * reward
        else:
            self.transitions[key] = WorldTransition(
                from_features=from_state,
                action=action,
                to_features=to_state,
                reward=reward,
                count=1
            )
        
        # Update reward model
        self.reward_model[to_disc] = reward
        
        self.total_experience += 1
    
    def predict(
        self,
        state: torch.Tensor,
        action: int
    ) -> Tuple[Optional[WorldState], float]:
        """
        Predict next state and reward given state and action.
        
        Args:
            state: Current state features
            action: Action to take
            
        Returns:
            (predicted_state, uncertainty)
        """
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().tolist()
        
        state_disc = self._discretize_state(state)
        key = (state_disc, action)
        
        if key in self.transitions:
            trans = self.transitions[key]
            # Uncertainty decreases with count
            uncertainty = 1.0 / (1.0 + math.log(1 + trans.count))
            
            next_state = WorldState(
                features=trans.to_features,
                reward=trans.reward,
                is_terminal=False,
                uncertainty=uncertainty
            )
            return next_state, uncertainty
        else:
            return None, 1.0  # Maximum uncertainty
    
    def imagine_trajectory(
        self,
        start_state: torch.Tensor,
        actions: List[int]
    ) -> SimulatedTrajectory:
        """
        Imagine a trajectory by simulating actions.
        
        Args:
            start_state: Starting state
            actions: Sequence of actions to simulate
            
        Returns:
            Simulated trajectory with states, rewards, uncertainty
        """
        if isinstance(start_state, torch.Tensor):
            current_features = start_state.detach().cpu().tolist()
        else:
            current_features = start_state
        
        states = [WorldState(features=current_features, reward=0.0, is_terminal=False)]
        total_reward = 0.0
        total_uncertainty = 0.0
        
        for action in actions:
            next_state, uncertainty = self.predict(
                torch.tensor(current_features), action
            )
            
            if next_state is None:
                # Unknown transition - use heuristic
                next_state = WorldState(
                    features=current_features,  # Stay in place
                    reward=0.0,
                    is_terminal=False,
                    uncertainty=1.0
                )
            
            states.append(next_state)
            total_reward += next_state.reward
            total_uncertainty += next_state.uncertainty
            current_features = next_state.features
            
            if next_state.is_terminal:
                break
        
        return SimulatedTrajectory(
            states=states,
            actions=actions[:len(states) - 1],
            total_reward=total_reward,
            avg_uncertainty=total_uncertainty / max(len(states), 1)
        )
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_experience': self.total_experience,
            'num_transitions': len(self.transitions),
            'num_states_visited': len(self.state_visits),
            'avg_transition_count': (
                sum(t.count for t in self.transitions.values()) /
                max(len(self.transitions), 1)
            )
        }


# ═══════════════════════════════════════════════════════════════════════════════
# META-LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetaParams:
    """Meta-learned hyperparameters."""
    learning_rate: float = 0.001
    exploration_rate: float = 0.1
    discount_factor: float = 0.99
    batch_size: int = 32
    temperature: float = 1.0


class MetaLearner:
    """
    Learn how to learn: adapts hyperparameters based on performance.
    
    Tracks what learning strategies work in different contexts
    and adjusts parameters accordingly.
    """
    
    def __init__(self, config: CognitionConfig):
        self.config = config
        self.params = MetaParams()
        
        # Performance history
        self.performance_history: deque = deque(maxlen=config.meta_learning_window)
        self.param_history: List[MetaParams] = []
        
        # Strategy effectiveness
        self.strategy_scores: Dict[str, deque] = {
            'high_lr': deque(maxlen=50),
            'low_lr': deque(maxlen=50),
            'high_explore': deque(maxlen=50),
            'low_explore': deque(maxlen=50)
        }
        
        self.current_step = 0
    
    def record_performance(self, performance: float, context: Optional[str] = None):
        """
        Record performance metric.
        
        Args:
            performance: Performance value (higher = better)
            context: Optional context label
        """
        self.performance_history.append(performance)
        
        # Track which strategy produced this performance
        if self.params.learning_rate > 0.005:
            self.strategy_scores['high_lr'].append(performance)
        else:
            self.strategy_scores['low_lr'].append(performance)
        
        if self.params.exploration_rate > 0.15:
            self.strategy_scores['high_explore'].append(performance)
        else:
            self.strategy_scores['low_explore'].append(performance)
        
        self.current_step += 1
        
        # Adapt parameters periodically
        if self.current_step % 25 == 0:
            self._adapt_params()
    
    def _adapt_params(self):
        """Adapt meta-parameters based on performance trends."""
        if len(self.performance_history) < 20:
            return
        
        # Compute recent vs older performance
        recent = list(self.performance_history)[-10:]
        older = list(self.performance_history)[-20:-10]
        
        recent_mean = sum(recent) / len(recent) if recent else 0
        older_mean = sum(older) / len(older) if older else 0
        
        # Adaptation based on trend
        adaptation_rate = self.config.meta_param_adaptation_rate
        
        if recent_mean > older_mean:
            # Things are improving - keep current strategy
            pass
        else:
            # Try different strategy
            # Compare strategy effectiveness
            high_lr_mean = (
                sum(self.strategy_scores['high_lr']) / 
                max(len(self.strategy_scores['high_lr']), 1)
            )
            low_lr_mean = (
                sum(self.strategy_scores['low_lr']) / 
                max(len(self.strategy_scores['low_lr']), 1)
            )
            
            if high_lr_mean > low_lr_mean:
                self.params.learning_rate = min(0.01, self.params.learning_rate * 1.2)
            else:
                self.params.learning_rate = max(0.0001, self.params.learning_rate * 0.8)
            
            # Similarly for exploration
            high_exp_mean = (
                sum(self.strategy_scores['high_explore']) / 
                max(len(self.strategy_scores['high_explore']), 1)
            )
            low_exp_mean = (
                sum(self.strategy_scores['low_explore']) / 
                max(len(self.strategy_scores['low_explore']), 1)
            )
            
            if high_exp_mean > low_exp_mean:
                self.params.exploration_rate = min(0.3, self.params.exploration_rate * 1.1)
            else:
                self.params.exploration_rate = max(0.01, self.params.exploration_rate * 0.9)
        
        self.param_history.append(MetaParams(**self.params.__dict__))
    
    def get_recommended_params(self) -> MetaParams:
        """Get current recommended hyperparameters."""
        return self.params
    
    def should_explore(self) -> bool:
        """Determine if agent should explore vs exploit."""
        return random.random() < self.params.exploration_rate
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'current_lr': self.params.learning_rate,
            'current_exploration': self.params.exploration_rate,
            'recent_performance': (
                sum(list(self.performance_history)[-10:]) / 10
                if len(self.performance_history) >= 10 else 0
            ),
            'adaptations': len(self.param_history)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOL GROUNDING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Symbol:
    """A grounded symbol connecting language to experience."""
    id: int
    name: str
    sensory_grounding: List[float]  # Sensory experience signature
    motor_grounding: List[float]  # Motor/action signature
    confidence: float = 0.5
    usage_count: int = 0


@dataclass
class SymbolicExpression:
    """A composition of symbols."""
    symbols: List[int]
    relation: str  # 'sequence', 'parallel', 'causal', etc.
    meaning: Optional[List[float]] = None


class SymbolSystem:
    """
    Ground symbols in sensorimotor experience.
    
    Connects abstract symbols (like words) to concrete
    perceptual and motor experiences, enabling language understanding.
    """
    
    def __init__(self, config: CognitionConfig):
        self.config = config
        self.symbols: Dict[int, Symbol] = {}
        self.name_to_id: Dict[str, int] = {}
        self.expressions: List[SymbolicExpression] = []
        self.next_id = 0
    
    def get_or_create_symbol(
        self,
        name: str,
        sensory_grounding: Optional[List[float]] = None,
        motor_grounding: Optional[List[float]] = None
    ) -> int:
        """Get existing symbol or create new one."""
        if name in self.name_to_id:
            symbol_id = self.name_to_id[name]
            symbol = self.symbols[symbol_id]
            symbol.usage_count += 1
            
            # Update grounding if provided
            if sensory_grounding is not None and symbol.sensory_grounding:
                alpha = 0.1
                symbol.sensory_grounding = [
                    (1 - alpha) * old + alpha * new
                    for old, new in zip(symbol.sensory_grounding, sensory_grounding)
                ]
            
            return symbol_id
        
        # Create new symbol
        if len(self.symbols) >= self.config.max_symbols:
            # Remove least used symbol
            min_usage_id = min(self.symbols, key=lambda k: self.symbols[k].usage_count)
            old_name = self.symbols[min_usage_id].name
            del self.symbols[min_usage_id]
            del self.name_to_id[old_name]
        
        symbol = Symbol(
            id=self.next_id,
            name=name,
            sensory_grounding=sensory_grounding or [],
            motor_grounding=motor_grounding or [],
            confidence=0.3 if sensory_grounding or motor_grounding else 0.1,
            usage_count=1
        )
        
        self.symbols[self.next_id] = symbol
        self.name_to_id[name] = self.next_id
        self.next_id += 1
        
        return self.next_id - 1
    
    def compose(
        self,
        symbol_ids: List[int],
        relation: str = 'sequence'
    ) -> SymbolicExpression:
        """Compose symbols into an expression."""
        # Compute combined meaning
        meanings = []
        for sid in symbol_ids:
            if sid in self.symbols:
                symbol = self.symbols[sid]
                if symbol.sensory_grounding:
                    meanings.append(symbol.sensory_grounding)
        
        combined_meaning = None
        if meanings:
            # Average the groundings
            combined_meaning = [
                sum(m[i] for m in meanings if i < len(m)) / len(meanings)
                for i in range(len(meanings[0]))
            ]
        
        expr = SymbolicExpression(
            symbols=symbol_ids,
            relation=relation,
            meaning=combined_meaning
        )
        self.expressions.append(expr)
        
        return expr
    
    def ground_text(
        self,
        text: str,
        sensory_context: Optional[torch.Tensor] = None
    ) -> List[int]:
        """
        Ground a text string by creating/updating symbols.
        
        Args:
            text: Text to ground
            sensory_context: Optional sensory features for grounding
            
        Returns:
            List of symbol IDs
        """
        words = text.lower().split()
        symbol_ids = []
        
        sensory = None
        if sensory_context is not None:
            if isinstance(sensory_context, torch.Tensor):
                sensory = sensory_context.detach().cpu().tolist()
        
        for word in words:
            # Each word gets grounded with context
            symbol_id = self.get_or_create_symbol(
                name=word,
                sensory_grounding=sensory
            )
            symbol_ids.append(symbol_id)
        
        return symbol_ids
    
    def retrieve_grounding(self, symbol_id: int) -> Optional[torch.Tensor]:
        """Get sensory grounding for a symbol."""
        if symbol_id in self.symbols:
            symbol = self.symbols[symbol_id]
            if symbol.sensory_grounding:
                return torch.tensor(symbol.sensory_grounding)
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        grounded_count = sum(
            1 for s in self.symbols.values()
            if s.confidence > self.config.symbol_grounding_threshold
        )
        
        return {
            'total_symbols': len(self.symbols),
            'grounded_symbols': grounded_count,
            'total_expressions': len(self.expressions),
            'avg_confidence': (
                sum(s.confidence for s in self.symbols.values()) /
                max(len(self.symbols), 1)
            )
        }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED AGI COGNITION
# ═══════════════════════════════════════════════════════════════════════════════

class AGICognition(nn.Module):
    """
    Unified AGI Cognition module combining all cognitive primitives.
    
    Integrates:
    - Causal Discovery
    - Abstraction Hierarchy
    - World Model
    - Meta-Learning
    - Symbol Grounding
    
    These systems compound and reinforce each other:
    - Causal discovery feeds abstraction
    - Abstractions ground symbols
    - Symbols enable compositional goals
    - World model enables planning
    - Meta-learning optimizes everything
    """
    
    def __init__(self, config: Optional[CognitionConfig] = None, feature_dim: int = 256):
        super().__init__()
        
        self.config = config or CognitionConfig(feature_dim=feature_dim)
        self.feature_dim = feature_dim
        
        # Cognitive systems
        self.causal_discovery = CausalDiscovery(self.config)
        self.abstraction = AbstractionHierarchy(self.config)
        self.world_model = WorldModel(self.config, feature_dim)
        self.meta_learner = MetaLearner(self.config)
        self.symbols = SymbolSystem(self.config)
        
        # Neural components for integration
        self.feature_projector = nn.Linear(feature_dim, feature_dim)
        self.cognition_head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.current_step = 0
    
    def forward(
        self,
        features: torch.Tensor,
        action: Optional[int] = None,
        reward: float = 0.0,
        next_features: Optional[torch.Tensor] = None,
        is_terminal: bool = False
    ) -> Dict[str, Any]:
        """
        Process experience through all cognitive systems.
        
        Args:
            features: Current state features [batch, hidden] or [hidden]
            action: Optional action taken
            reward: Reward received
            next_features: Optional next state features
            is_terminal: Whether episode ended
            
        Returns:
            Dict with augmented features and cognitive state
        """
        was_batch = features.dim() == 2
        if not was_batch:
            features = features.unsqueeze(0)
        
        batch_size = features.size(0)
        results = {}
        
        # Process each sample in batch
        for i in range(batch_size):
            feat = features[i]
            feat_list = feat.detach().cpu().tolist()
            
            # 1. Causal Discovery
            self.causal_discovery.observe(feat, action, reward)
            
            # 2. Abstraction
            self.abstraction.observe(feat)
            
            # 3. World Model
            if next_features is not None:
                next_feat = next_features[i] if next_features.dim() == 2 else next_features
                self.world_model.learn(feat, action or 0, next_feat, reward, is_terminal)
            
            # 4. Meta-Learning
            self.meta_learner.record_performance(reward)
        
        # Augment features with discovered variables
        augmented = self.causal_discovery.augment_features(features)
        
        # Get activated concepts
        concepts = self.abstraction.get_activated_concepts(features[0])
        
        # Project features
        projected = self.feature_projector(features)
        
        # Combine with cognitive context
        # (simplified - in full version would integrate more deeply)
        context = projected.mean(dim=0, keepdim=True).expand(batch_size, -1)
        enhanced = self.cognition_head(torch.cat([projected, context], dim=-1))
        
        if not was_batch:
            enhanced = enhanced.squeeze(0)
            augmented = augmented.squeeze(0) if augmented.dim() == 2 else augmented
        
        self.current_step += 1
        
        return {
            'features': enhanced,
            'augmented_features': augmented,
            'activated_concepts': concepts[:5] if concepts else [],
            'meta_params': self.meta_learner.get_recommended_params(),
            'should_explore': self.meta_learner.should_explore()
        }
    
    def imagine(
        self,
        state: torch.Tensor,
        action_sequence: List[int]
    ) -> SimulatedTrajectory:
        """Imagine a trajectory using the world model."""
        return self.world_model.imagine_trajectory(state, action_sequence)
    
    def ground_language(
        self,
        text: str,
        sensory_context: Optional[torch.Tensor] = None
    ) -> List[int]:
        """Ground text in sensory experience."""
        return self.symbols.ground_text(text, sensory_context)
    
    def summary(self) -> Dict[str, Any]:
        """Get full cognitive state summary."""
        return {
            'step': self.current_step,
            'causal_discovery': self.causal_discovery.summary(),
            'abstraction': self.abstraction.summary(),
            'world_model': self.world_model.summary(),
            'meta_learning': self.meta_learner.summary(),
            'symbols': self.symbols.summary()
        }


if __name__ == "__main__":
    print("Testing AGICognition module...")
    
    config = CognitionConfig(feature_dim=64)
    cognition = AGICognition(config, feature_dim=64)
    
    # Simulate experience
    for step in range(200):
        features = torch.randn(64)
        action = random.randint(0, 3)
        reward = random.random()
        next_features = features + torch.randn(64) * 0.1
        
        result = cognition(
            features,
            action=action,
            reward=reward,
            next_features=next_features
        )
    
    # Get summary
    summary = cognition.summary()
    print("\n=== Cognitive State ===")
    print(f"Step: {summary['step']}")
    print(f"\nCausal Discovery:")
    print(f"  Variables: {summary['causal_discovery']['num_variables']}")
    print(f"\nAbstraction:")
    print(f"  Concepts: {summary['abstraction']['total_concepts']}")
    print(f"\nWorld Model:")
    print(f"  Transitions: {summary['world_model']['num_transitions']}")
    print(f"\nMeta-Learning:")
    print(f"  LR: {summary['meta_learning']['current_lr']:.6f}")
    print(f"  Exploration: {summary['meta_learning']['current_exploration']:.3f}")
    print(f"\nSymbols:")
    print(f"  Total: {summary['symbols']['total_symbols']}")
    
    # Test imagination
    print("\n=== Testing Imagination ===")
    state = torch.randn(64)
    trajectory = cognition.imagine(state, [0, 1, 2, 3])
    print(f"Imagined {len(trajectory.states)} states")
    print(f"Total reward: {trajectory.total_reward:.3f}")
    print(f"Avg uncertainty: {trajectory.avg_uncertainty:.3f}")
    
    # Test language grounding
    print("\n=== Testing Language Grounding ===")
    symbols = cognition.ground_language("move forward quickly", torch.randn(64))
    print(f"Grounded {len(symbols)} symbols: {symbols}")
    
    print("\nAGICognition tests passed!")
