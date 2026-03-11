"""
OctoTetrahedral AGI - Main Model Integration
Combines all components into a unified architecture

Architecture:
    Input (tokens/embeddings)
         ↓
    Perception Limb (embedding + encoding)
         ↓
    RNA Editing Layer (dynamic adaptation)
         ↓
    Tetrahedral Core (geometry-aware transformer)
         ↓
    Geometric Physics Layer (Fuller/Lloyd/Morphogenesis/TPMS)
         ↓
    ┌─────────────────────────────────────────┐
    │           8-Limb Processing             │
    │  Memory ─── Planning ─── Language       │
    │     │          │           │            │
    │  Spatial ─── Reasoning ─── MetaCog      │
    └─────────────────────────────────────────┘
         ↓
    Quantum-Enhanced Hub Synchronization
         ↓
    AGICognition (causal discovery, world model, meta-learning)
         ↓
    Action Limb (output generation)
         ↓
    Output (logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

from config import Config, get_config
from core.tetrahedral_geometry import TetrahedralGeometry
from core.tetrahedral_attention import TetrahedralAttention
from core.tetrahedral_core import TetrahedralCore
from core.working_memory import WorkingMemory
from adaptation.lora import LoRALayer
from adaptation.rna_editing import RNAEditingLayer
from limbs.perception_limb import PerceptionLimb
from limbs.reasoning_limb import ReasoningLimb
from limbs.action_limb import ActionLimb
from limbs.memory_limb import MemoryLimb
from limbs.planning_limb import PlanningLimb
from limbs.language_limb import LanguageLimb
from limbs.spatial_limb import SpatialLimb
from limbs.metacognition_limb import MetaCognitionLimb
from limbs.visualization_limb import VisualizationLimb
from limbs.imagination_limb import ImaginationLimb
from limbs.empathy_limb import EmpathyLimb
from limbs.emotion_limb import EmotionLimb
from limbs.ethics_limb import EthicsLimb
from limbs.dream_mode import DreamMode
from sync.hub_sync import HubSync
from core.compound_braid import CompoundBraid
from core.compound_loop import CompoundLoopController, CompoundLoopConfig as _LoopCfg
from core.cognitive_geometry import CognitiveGeometryEngine, CognitiveGeometryConfig
from cognition import AGICognition, CognitionConfig

# Import new geometric physics modules
from physics.geometric_physics_layer import (
    GeometricPhysicsLayer,
    GeometricPhysicsConfig,
    QuantumEnhancedHubSync,
    create_geometric_physics_layer
)


class OctoTetrahedralModel(nn.Module):
    """
    OctoTetrahedral AGI - Main Model Class
    
    Combines:
    - Tetrahedral geometry (64-point structure for attention)
    - Octopus-inspired RNA editing (dynamic weight modulation)
    - Geometric Physics Layer (Fuller/Lloyd/Morphogenesis/TPMS)
    - Distributed 13-limb architecture (semi-autonomous processing units):
        * Perception: Input encoding
        * Memory: Episodic + semantic storage
        * Planning: Goal-directed action sequencing
        * Language: NLU/NLG with grounding
        * Spatial: Geometric/grid reasoning
        * Reasoning: Abstract pattern processing
        * MetaCognition: Self-monitoring and adaptation
        * Action: Output generation
        * Visualization: Reconstructive mental imagery
        * Imagination: Generative latent exploration
        * Empathy: Theory of Mind agent modeling
        * Emotion: Valence/arousal modulation
        * Ethics: Value alignment / safety contraction
    - Dream Mode (awake / daydream / dream)
    - Quantum-enhanced hub synchronization
    - Working memory (differentiable memory slots)
    - AGI Cognition (causal discovery, world model, meta-learning)
    """
    
    def __init__(self, config: Optional[Config] = None, use_geometric_physics: bool = True):
        super().__init__()
        
        self.config = config or get_config()
        self.use_geometric_physics = use_geometric_physics
        
        # Store key dimensions
        self.vocab_size = self.config.model.vocab_size
        self.hidden_dim = self.config.model.hidden_dim
        self.num_heads = self.config.model.num_heads
        self.num_layers = self.config.model.num_layers
        
        # === Tetrahedral Geometry ===
        self.geometry = TetrahedralGeometry()
        
        # === Perception Limb (Input Processing) ===
        self.perception = PerceptionLimb(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            max_seq_len=self.config.model.max_seq_len,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size
        )
        
        # === RNA Editing Layer (Dynamic Adaptation) ===
        self.rna_editing = RNAEditingLayer(
            hidden_dim=self.hidden_dim,
            num_heads=self.config.rna_editing.num_gated_heads,
            num_pathways=self.config.rna_editing.num_pathways,
            temperature_init=self.config.rna_editing.temperature_init
        )
        
        # === Tetrahedral Core (Main Transformer) ===
        moe_dict = None
        if self.config.moe.enabled:
            moe_dict = {
                "enabled": True,
                "num_experts": self.config.moe.num_experts,
                "top_k": self.config.moe.top_k,
                "expert_ffn_dim": self.config.moe.expert_ffn_dim,
                "jitter_noise": self.config.moe.jitter_noise,
                "compound_enabled": self.config.moe.compound_enabled,
                "compound_bias_scale": self.config.moe.compound_bias_scale,
                "enable_cross_transfer": self.config.moe.enable_cross_transfer,
            }
        self.core = TetrahedralCore(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ffn_dim=self.config.model.ffn_dim,
            dropout=self.config.model.dropout,
            moe_config=moe_dict,
        )
        
        # === Geometric Physics Layer (Fuller/Lloyd/Morphogenesis/TPMS/QbitNexus/ParallelUniverse) ===
        if self.use_geometric_physics:
            gp_config = GeometricPhysicsConfig(
                enable_fuller=self.config.geometric_physics.enable_fuller,
                enable_lloyd=self.config.geometric_physics.enable_lloyd,
                enable_morphogenesis=self.config.geometric_physics.enable_morphogenesis,
                enable_tpms=self.config.geometric_physics.enable_tpms,
                enable_qbit_nexus=self.config.geometric_physics.enable_qbit_nexus,
                enable_parallel_universe=self.config.geometric_physics.enable_parallel_universe,
                combination_mode=self.config.geometric_physics.combination_mode,
                fuller_ve_vertices=self.config.geometric_physics.fuller_ve_vertices,
                fuller_tensegrity_struts=self.config.geometric_physics.fuller_tensegrity_struts,
                fuller_geodesic_frequency=self.config.geometric_physics.fuller_geodesic_frequency,
                lloyd_energy_budget=self.config.geometric_physics.lloyd_energy_budget,
                lloyd_temperature=self.config.geometric_physics.lloyd_temperature,
                lloyd_reversible_fraction=self.config.geometric_physics.lloyd_reversible_fraction,
                morpho_diffusion_steps=self.config.geometric_physics.morpho_diffusion_steps,
                morpho_activator_diffusion=self.config.geometric_physics.morpho_activator_diffusion,
                morpho_inhibitor_diffusion=self.config.geometric_physics.morpho_inhibitor_diffusion,
                tpms_surface_type=self.config.geometric_physics.tpms_surface_type,
                tpms_num_heads=self.config.geometric_physics.tpms_num_heads,
                tpms_threshold=self.config.geometric_physics.tpms_threshold,
                qbit_num_vertices=self.config.geometric_physics.qbit_num_vertices,
                qbit_num_layers=self.config.geometric_physics.qbit_num_layers,
                qbit_dropout=self.config.geometric_physics.qbit_dropout,
                parallel_num_universes=self.config.geometric_physics.parallel_num_universes,
                parallel_num_dimensions=self.config.geometric_physics.parallel_num_dimensions,
                parallel_overlap_dims=self.config.geometric_physics.parallel_overlap_dims,
                parallel_collapse_mode=self.config.geometric_physics.parallel_collapse_mode,
                loss_weight_tensegrity=self.config.geometric_physics.loss_weight_tensegrity,
                loss_weight_lloyd=self.config.geometric_physics.loss_weight_lloyd,
                loss_weight_turing=self.config.geometric_physics.loss_weight_turing,
                loss_weight_equilibrium=self.config.geometric_physics.loss_weight_equilibrium,
                loss_weight_entanglement=self.config.geometric_physics.loss_weight_entanglement,
                loss_weight_universe_overlap=self.config.geometric_physics.loss_weight_universe_overlap,
                dropout=self.config.model.dropout
            )
            self.geometric_physics = GeometricPhysicsLayer(
                hidden_dim=self.hidden_dim,
                config=gp_config,
                num_heads=self.num_heads
            )
        else:
            self.geometric_physics = None
        
        # === Working Memory ===
        self.working_memory = WorkingMemory(
            hidden_dim=self.hidden_dim,
            num_slots=self.config.model.memory_slots,
            num_heads=self.num_heads // 2  # Use fewer heads for memory
        )
        
        # === Memory Limb (Episodic + Semantic Memory) ===
        self.memory_limb = MemoryLimb(
            hidden_dim=self.hidden_dim,
            num_memory_slots=128,
            episodic_capacity=1000,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size
        )
        
        # === Planning Limb (Goal-Directed Reasoning) ===
        self.planning = PlanningLimb(
            hidden_dim=self.hidden_dim,
            num_actions=10,
            planning_horizon=10,
            num_plans=5,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size
        )
        
        # === Language Limb (NLU/NLG) ===
        self.language = LanguageLimb(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            num_heads=self.num_heads // 2,
            num_layers=2,
            max_seq_len=self.config.model.max_seq_len,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size
        )
        
        # === Spatial Limb (Geometric/Grid Reasoning) ===
        self.spatial = SpatialLimb(
            hidden_dim=self.hidden_dim,
            max_grid_size=30,  # ARC grids up to 30x30
            num_heads=self.num_heads // 2,
            num_layers=2,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size
        )
        
        # === MetaCognition Limb (Self-Monitoring) ===
        self.metacognition = MetaCognitionLimb(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads // 2,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size
        )
        
        # === Visualization Limb (Reconstructive Mental Imagery) ===
        self.visualization = VisualizationLimb(
            hidden_dim=self.hidden_dim,
            num_scales=3,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size,
        )
        
        # === Imagination Limb (Generative Latent Exploration) ===
        self.imagination = ImaginationLimb(
            hidden_dim=self.hidden_dim,
            num_modalities=4,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size,
        )
        
        # === Empathy Limb (Theory of Mind) ===
        self.empathy = EmpathyLimb(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads // 2,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size,
        )
        
        # === Emotion Limb (Valence/Arousal Modulation) ===
        self.emotion = EmotionLimb(
            hidden_dim=self.hidden_dim,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size,
        )
        
        # === Ethics Limb (Safety Contraction) ===
        self.ethics = EthicsLimb(
            hidden_dim=self.hidden_dim,
            num_safety_axes=8,
            num_values=4,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size,
        )
        
        # === Dream Mode (Visualization + Imagination orchestrator) ===
        self.dream_mode = DreamMode(hidden_dim=self.hidden_dim)
        
        # === Compound Braid (Cross-Limb Information Exchange) ===
        moe_signal_dim = self.config.moe.num_experts if self.config.moe.enabled and self.config.moe.compound_enabled else 0
        self.compound_braid = CompoundBraid(
            hidden_dim=self.hidden_dim,
            num_limbs=11,  # 6 original + 5 new cognitive petals
            num_heads=self.num_heads // 4,
            dropout=self.config.model.dropout,
            braid_strength=0.3,
            moe_signal_dim=moe_signal_dim,
        )
        
        # === Compound Loop Controller (Adaptive-Depth Reasoning) ===
        self.use_compound_loop = self.config.compound_loop.enabled
        if self.use_compound_loop:
            loop_cfg = _LoopCfg(
                max_loops=self.config.compound_loop.max_loops,
                exit_threshold=self.config.compound_loop.exit_threshold,
                entropy_beta=self.config.compound_loop.entropy_beta,
                warmup_loops=self.config.compound_loop.warmup_loops,
                conciseness_reward=self.config.compound_loop.conciseness_reward,
                max_cheap_loops=self.config.compound_loop.max_cheap_loops,
            )
            self.compound_loop = CompoundLoopController(
                hidden_dim=self.hidden_dim, config=loop_cfg
            )
        else:
            self.compound_loop = None
        
        # === Reasoning Limb (Pattern Processing) ===
        self.reasoning = ReasoningLimb(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads // 2,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size
        )
        
        # === Action Limb (Output Generation) ===
        self.action = ActionLimb(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size,
            dropout=self.config.model.dropout,
            lora_rank=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            buffer_size=self.config.limb.buffer_size,
            tie_weights=True
        )
        
        # Tie embedding weights
        self.action.tie_embedding_weights(self.perception.get_embedding_weight())
        
        # === AGI Cognition (Causal Discovery, World Model, Meta-Learning) ===
        cognition_config = CognitionConfig(
            feature_dim=self.hidden_dim,
            max_causal_variables=64,
            max_abstraction_depth=5,
            world_model_horizon=20,
            meta_learning_window=100
        )
        self.cognition = AGICognition(
            config=cognition_config,
            feature_dim=self.hidden_dim
        )
        
        # === Hub Synchronization ===
        # Use quantum-enhanced sync if enabled
        if self.config.sync.use_quantum_sync:
            self.quantum_hub_sync = QuantumEnhancedHubSync(
                hidden_dim=self.hidden_dim,
                num_limbs=13,
                coupling_strength=self.config.sync.quantum_coupling_strength,
                use_tpms_routing=self.config.sync.use_tpms_routing,
                dropout=self.config.model.dropout
            )
        else:
            self.quantum_hub_sync = None
        
        # Standard hub sync (always available for FedAvg)
        self.hub_sync = HubSync(
            sync_frequency=self.config.sync.sync_frequency,
            max_drift=self.config.sync.max_drift,
            rollback_buffer_size=self.config.sync.rollback_buffer_size,
            use_performance_weighting=self.config.sync.use_performance_weighting
        )
        
        # Collect all 13 limbs for sync
        self._limbs = {
            'perception': self.perception,
            'memory': self.memory_limb,
            'planning': self.planning,
            'language': self.language,
            'spatial': self.spatial,
            'reasoning': self.reasoning,
            'metacognition': self.metacognition,
            'action': self.action,
            'visualization': self.visualization,
            'imagination': self.imagination,
            'empathy': self.empathy,
            'emotion': self.emotion,
            'ethics': self.ethics,
        }
        
        # === Cognitive Geometry Engine (ML Vocabulary Compound Integration) ===
        cg_cfg = self.config.cognitive_geometry
        cg_config = CognitiveGeometryConfig(
            enabled=cg_cfg.enabled,
            svd_enabled=cg_cfg.svd_enabled,
            svd_top_k=cg_cfg.svd_top_k,
            svd_loss_weight=cg_cfg.svd_loss_weight,
            alignment_enabled=cg_cfg.alignment_enabled,
            alignment_loss_weight=cg_cfg.alignment_loss_weight,
            entropy_monitor_enabled=cg_cfg.entropy_monitor_enabled,
            entropy_target=cg_cfg.entropy_target,
            entropy_loss_weight=cg_cfg.entropy_loss_weight,
            drift_enabled=cg_cfg.drift_enabled,
            drift_max_rotation=cg_cfg.drift_max_rotation,
            drift_loss_weight=cg_cfg.drift_loss_weight,
            anchor_enabled=cg_cfg.anchor_enabled,
            num_anchors=cg_cfg.num_anchors,
            anchor_decay=cg_cfg.anchor_decay,
            anchor_strength=cg_cfg.anchor_strength,
            repetition_dampen_enabled=cg_cfg.repetition_dampen_enabled,
            repetition_penalty=cg_cfg.repetition_penalty,
            repetition_window=cg_cfg.repetition_window,
            branch_scorer_enabled=cg_cfg.branch_scorer_enabled,
            branch_prune_threshold=cg_cfg.branch_prune_threshold,
            manifold_enabled=cg_cfg.manifold_enabled,
            manifold_loss_weight=cg_cfg.manifold_loss_weight,
            goal_vector_enabled=cg_cfg.goal_vector_enabled,
            goal_strength=cg_cfg.goal_strength,
            attention_plane_enabled=cg_cfg.attention_plane_enabled,
            plane_dim=cg_cfg.plane_dim,
            vector_field_enabled=cg_cfg.vector_field_enabled,
        )
        self.cognitive_geometry = CognitiveGeometryEngine(
            hidden_dim=self.hidden_dim,
            num_limbs=11,
            config=cg_config,
        )
        
        # === Initialization ===
        self._init_weights()
        
        # Statistics tracking
        self._forward_count = 0
        self._last_confidences: Dict[str, float] = {}
    
    def _init_weights(self):
        """Initialize model weights"""
        def _init_module(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Don't re-initialize limbs (they have their own init)
        self.core.apply(_init_module)
        self.working_memory.apply(_init_module)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_confidences: bool = False,
        use_memory: bool = True,
        pathway_hint: Optional[int] = None,
        dream_mode: str = 'awake',
    ) -> Dict[str, Any]:
        """
        Forward pass through the full model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            embeddings: Pre-computed embeddings (alternative to input_ids)
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for loss computation [batch, seq_len]
            return_dict: Whether to return dict (always True for now)
            return_confidences: Whether to compute limb confidences
            use_memory: Whether to use working memory
            pathway_hint: Optional hint for RNA editing pathway (0-2)
            
        Returns:
            Dict with logits, loss (if labels), and optional stats
        """
        self._forward_count += 1
        
        # === 1. Perception: Encode Input ===
        encoded, perception_conf = self.perception(
            token_ids=input_ids,
            embeddings=embeddings,
            return_confidence=return_confidences
        )
        # encoded: [batch, seq_len, hidden_dim]
        
        # === 2. RNA Editing: Dynamic Adaptation ===
        editing_result = self.rna_editing(encoded)
        edited = editing_result['output']
        editing_info = {
            'temperature': editing_result['temperature'],
            'head_gates': editing_result['head_gates'],
            'pathway_weights': editing_result['pathway_weights'],
            'confidence': editing_result['confidence'],
            'adaptation_magnitude': editing_result['temperature'].mean()
        }
        # edited: [batch, seq_len, hidden_dim]
        
        # === 3. Tetrahedral Core: Main Processing ===
        # Pass cached braid signal from previous step into MoE routing
        braid_signal = getattr(self, '_cached_braid_signal', None)
        core_result = self.core(
            edited,
            attention_mask=attention_mask,
            head_gates=editing_info['head_gates'],
            braid_signal=braid_signal,
        )
        core_output = core_result['hidden_states']
        reasoning_state = core_result['reasoning_state']
        moe_aux_loss = core_result.get('aux_loss', torch.tensor(0.0, device=core_output.device))
        # core_output: [batch, seq_len, hidden_dim]
        
        # === 3.5 Geometric Physics Layer (NEW) ===
        physics_loss = torch.tensor(0.0, device=core_output.device)
        geometric_physics_info = {}
        
        if self.use_geometric_physics and self.geometric_physics is not None:
            gp_result = self.geometric_physics(
                core_output,
                attention_mask=attention_mask,
                return_components=False
            )
            gp_output = gp_result['output']
            # Skip geometry entirely when NaN detected (prevents gradient contamination)
            if not torch.isnan(gp_output).any():
                core_output = gp_output
            physics_loss = gp_result['physics_loss']
            if torch.isnan(physics_loss):
                physics_loss = torch.tensor(0.0, device=core_output.device)
            geometric_physics_info = {
                'physics_loss': physics_loss,
                'physics_losses': gp_result.get('physics_losses', {}),
                'gate_weights': gp_result.get('gate_weights', None),
                'enabled_modules': gp_result.get('enabled_modules', [])
            }
        
        # === 4. Working Memory: Read/Write ===
        if use_memory:
            # Use reasoning state as query for memory
            memory_query = core_output.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
            
            # Read from working memory
            memory_read, _ = self.working_memory.read(
                core_output,  # Use full sequence as queries
                return_weights=False
            )
            
            # Write reasoning state to working memory
            self.working_memory.write(reasoning_state)  # [batch, hidden]
            
            # Blend memory with core output
            memory_enhanced = core_output + 0.1 * memory_read
        else:
            memory_enhanced = core_output
        
        # === 5-7. Limb Processing — optionally wrapped in compound loop ===
        _limb_outputs_for_cg = None  # Collected for cognitive geometry engine
        if self.use_compound_loop and self.compound_loop is not None:
            # Initialize loop memory state (differentiable within loop)
            self._loop_memory_state = self.working_memory.memory.clone()

            loop_result = self.compound_loop(
                memory_enhanced,
                process_fn=self._limb_loop_step,
                process_kwargs={
                    'attention_mask': attention_mask,
                    'return_confidences': return_confidences,
                    'encoded': encoded,
                }
            )
            multi_limb_output = loop_result['output']
            compound_loop_info = {
                'loop_count': loop_result['loop_count'],
                'exit_distribution': loop_result['exit_distribution'],
                'entropy_loss': loop_result['entropy_loss'],
            }

            # Commit final memory state (detach to prevent unbounded graph)
            self.working_memory.memory.data = self._loop_memory_state.detach()
            self._loop_memory_state = None

            # Use a dummy for reasoning out (already folded into loop output)
            reasoned = multi_limb_output
        else:
            # Original non-looped path (backward compatible)
            compound_loop_info = None
            # Memory Limb
            memory_out, memory_conf, _ = self.memory_limb(memory_enhanced)
            # Spatial Limb
            spatial_out, spatial_conf, _ = self.spatial(memory_enhanced)
            # Language Limb
            language_out, language_conf, _ = self.language(memory_enhanced)
            # MetaCognition
            meta_out, meta_conf, _ = self.metacognition(memory_enhanced)
            # Reasoning Limb
            reasoning_out, reasoning_conf, _ = self.reasoning(
                memory_enhanced,
                attention_mask=attention_mask,
                return_confidence=return_confidences
            )
            # Perception echo
            perception_echo = encoded
            
            # === New cognitive petals ===
            # Visualization (reconstructive, memory-based)
            vis_out, vis_conf, _ = self.visualization(memory_enhanced)
            # Imagination (generative, novelty-seeking)
            imag_out, imag_conf, _ = self.imagination(memory_enhanced)
            # Empathy (Theory of Mind)
            empathy_out, empathy_conf, _ = self.empathy(memory_enhanced)
            # Emotion (valence/arousal modulation)
            emotion_out, emotion_conf, _ = self.emotion(memory_enhanced)
            # Ethics (safety contraction)
            ethics_out, ethics_conf, _ = self.ethics(memory_enhanced)
            
            # Dream Mode: blend visualization + imagination
            dream_out, dream_info = self.dream_mode(
                vis_out, imag_out, ethics_out, dream_mode=dream_mode
            )
            
            # Emotion modulation: apply emotional signal to all limb outputs
            emo_signal = self.emotion.get_modulation_signal()
            if emo_signal is not None:
                emo_vec, emo_strength = emo_signal
                # Broadcast emotional modulation across sequence
                emo_bias = (emo_strength.unsqueeze(1) * emo_vec.unsqueeze(1)) * 0.05
                memory_out = memory_out + emo_bias
                reasoning_out = reasoning_out + emo_bias
                language_out = language_out + emo_bias
            
            # Store limb outputs for cognitive geometry engine (11 limbs)
            _limb_outputs_for_cg = [memory_out, spatial_out, language_out,
                                     meta_out, reasoning_out, perception_echo,
                                     dream_out, empathy_out, emotion_out,
                                     ethics_out, vis_out]
            
            # Compound Braid (11 limbs)
            moe_expert_loads = None
            if self.config.moe.enabled and self.config.moe.compound_enabled:
                moe_expert_loads = self._get_first_compound_moe_loads()
            combined_limbs, braid_info = self.compound_braid(
                [memory_out, spatial_out, language_out, meta_out,
                 reasoning_out, perception_echo, dream_out,
                 empathy_out, emotion_out, ethics_out, vis_out],
                attention_mask=attention_mask,
                moe_expert_loads=moe_expert_loads,
            )
            if braid_info.get('braid_signal') is not None:
                self._cached_braid_signal = braid_info['braid_signal'].detach()
            
            multi_limb_output = memory_enhanced + 0.3 * combined_limbs
            
            # Post-Braid Reasoning
            reasoned, _, _ = self.reasoning(
                multi_limb_output,
                attention_mask=attention_mask,
                return_confidence=return_confidences
            )
        
        # === 7. Planning Limb: Action Sequencing ===
        # Get current state from reasoned output
        current_state = reasoned.mean(dim=1)  # [batch, hidden]
        goal_state = current_state  # Use same as goal for now
        planning_output = self.planning(current_state, goal_state)
        
        # === 8. AGI Cognition: Higher-level reasoning ===
        # Feature vector for cognition module
        cognition_features = reasoned.mean(dim=1)  # [batch, hidden]
        cognition_output = self.cognition(cognition_features)
        
        # Use the projected features (fixed hidden_dim) instead of augmented
        # (augmented_features can have variable dimension due to discovered variables)
        enhanced_features = cognition_output['features']  # [batch, hidden] - fixed size
        cognition_enhanced = reasoned + 0.1 * enhanced_features.unsqueeze(1)
        
        # === 9. Action Limb: Generate Output ===
        logits, action_conf, gate_values = self.action(
            cognition_enhanced,
            return_confidence=return_confidences,
            embedding_weight=self.perception.get_embedding_weight()
        )
        # logits: [batch, seq_len, vocab_size]
        
        # === 9.5 Cognitive Geometry Engine: compound ML vocabulary integration ===
        cg_result = self.cognitive_geometry(
            hidden=cognition_enhanced,
            limb_outputs=_limb_outputs_for_cg,
            logits=logits,
            input_ids=input_ids,
        )
        # Apply modified hidden states and logits
        cognition_enhanced = cg_result['hidden']
        logits = cg_result['logits'] if cg_result['logits'] is not None else logits
        cg_aux_loss = cg_result['aux_loss']
        cognitive_geometry_info = cg_result['info']
        
        # === Compute Loss if Labels Provided ===
        loss = None
        if labels is not None:
            # Labels are PRE-SHIFTED by the dataset:
            #   input_ids = full_tokens[:-1], labels = full_tokens[1:]
            # So labels[i] is already the next token after input_ids[i].
            # No additional shift needed — CE(logits[i], labels[i]) is correct.
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Add information gain bonus (curiosity)
            if self.training:
                info_gain = self._compute_information_gain(
                    logits, editing_info
                )
                loss = loss - self.config.training.information_gain_weight * info_gain
                
                # Add physics-informed losses from GeometricPhysicsLayer
                if self.use_geometric_physics and physics_loss is not None:
                    loss = loss + physics_loss
                
                # Add MoE load-balancing auxiliary loss
                if self.config.moe.enabled:
                    loss = loss + self.config.moe.load_balance_weight * moe_aux_loss
                
                # Add compound loop entropy regularization
                if compound_loop_info is not None:
                    loss = loss + compound_loop_info['entropy_loss']
                
                # Add cognitive geometry auxiliary losses
                if self.config.cognitive_geometry.enabled:
                    loss = loss + cg_aux_loss
                
                # Add imagination KL divergence (encourages diverse generation)
                imag_kl = self.imagination.get_kl_loss()
                if isinstance(imag_kl, torch.Tensor) and imag_kl.item() > 0:
                    loss = loss + 0.01 * imag_kl
                
                # Add ethics value alignment loss
                ethics_align = self.ethics.get_alignment_loss()
                if isinstance(ethics_align, torch.Tensor):
                    loss = loss + 0.01 * ethics_align
        
        # === Build Output ===
        braid_info = braid_info if not self.use_compound_loop else {}
        dream_info_out = dream_info if not self.use_compound_loop else {}
        output = {
            'logits': logits,
            'loss': loss,
            'hidden_states': reasoned,
            'reasoning_state': reasoning_state,
            'braid_info': braid_info,
            'physics_loss': physics_loss if self.use_geometric_physics else None,
            'moe_aux_loss': moe_aux_loss if self.config.moe.enabled else None,
            'geometric_physics_info': geometric_physics_info if self.use_geometric_physics else None,
            'compound_loop_info': compound_loop_info,
            'cognitive_geometry_info': cognitive_geometry_info,
            'cg_aux_loss': cg_aux_loss,
            'dream_info': dream_info_out,
            'emotional_state': self.emotion.get_emotional_state(),
            'safety_state': self.ethics.get_safety_state(),
        }
        
        if return_confidences:
            # Compound loop path doesn't set individual limb confidences
            if self.use_compound_loop:
                memory_conf = memory_conf if 'memory_conf' in dir() else 0.5
                spatial_conf = spatial_conf if 'spatial_conf' in dir() else 0.5
                language_conf = language_conf if 'language_conf' in dir() else 0.5
                meta_conf = meta_conf if 'meta_conf' in dir() else 0.5
                reasoning_conf = reasoning_conf if 'reasoning_conf' in dir() else 0.5
                vis_conf = vis_conf if 'vis_conf' in dir() else 0.5
                imag_conf = imag_conf if 'imag_conf' in dir() else 0.5
                empathy_conf = empathy_conf if 'empathy_conf' in dir() else 0.5
                emotion_conf = emotion_conf if 'emotion_conf' in dir() else 0.5
                ethics_conf = ethics_conf if 'ethics_conf' in dir() else 0.5
            self._last_confidences = {
                'perception': perception_conf or 0.0,
                'memory': memory_conf or 0.0,
                'spatial': spatial_conf or 0.0,
                'language': language_conf or 0.0,
                'metacognition': meta_conf or 0.0,
                'reasoning': reasoning_conf or 0.0,
                'action': action_conf or 0.0,
                'visualization': vis_conf or 0.0,
                'imagination': imag_conf or 0.0,
                'empathy': empathy_conf or 0.0,
                'emotion': emotion_conf or 0.0,
                'ethics': ethics_conf or 0.0,
                'overall': (
                    (perception_conf or 0.5) * 
                    (reasoning_conf or 0.5) * 
                    (action_conf or 0.5)
                ) ** (1/3),  # Geometric mean
                'braid_gates': braid_info.get('gate_values', {}),
                'braid_weights': braid_info.get('combine_weights', {}),
            }
            output['confidences'] = self._last_confidences
        
        if self.training:
            output['editing_info'] = editing_info
        
        return output
    
    def _compute_information_gain(
        self,
        logits: torch.Tensor,
        editing_info: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute information gain (curiosity bonus).
        
        Higher uncertainty in predictions + RNA editing activity = more learning
        """
        # Entropy of predictions
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
        
        # RNA editing activity
        editing_activity = editing_info.get('adaptation_magnitude', 0.0)
        if isinstance(editing_activity, torch.Tensor):
            editing_activity = editing_activity.mean()
        
        # Combine: encourage exploration when uncertain and adapting
        info_gain = 0.1 * entropy + 0.05 * editing_activity
        
        return info_gain
    
    def _limb_loop_step(
        self,
        x: torch.Tensor,
        loop_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        return_confidences: bool = False,
        encoded: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One iteration of the compound-looped limb pipeline.
        Called by CompoundLoopController.forward() for each loop step.

        Consequential memory: reads working memory at loop entry, writes
        reasoning output back at loop exit. Fully differentiable within
        the loop — gradients flow through memory state across iterations.
        """
        # --- Consequential memory read (state from previous iteration) ---
        if self._loop_memory_state is not None:
            mem_read, _ = self.working_memory.read_from_state(
                x, self._loop_memory_state
            )
            x = x + 0.1 * mem_read

        # Parallel limb processing
        memory_out, _, _ = self.memory_limb(x)
        spatial_out, _, _ = self.spatial(x)
        language_out, _, _ = self.language(x)
        meta_out, _, _ = self.metacognition(x)
        reasoning_out, _, _ = self.reasoning(
            x, attention_mask=attention_mask,
            return_confidence=return_confidences
        )
        perception_echo = encoded if encoded is not None else x

        # New cognitive petals
        vis_out, _, _ = self.visualization(x)
        imag_out, _, _ = self.imagination(x)
        empathy_out, _, _ = self.empathy(x)
        emotion_out, _, _ = self.emotion(x)
        ethics_out, _, _ = self.ethics(x)
        dream_out, _ = self.dream_mode(vis_out, imag_out, ethics_out)

        # Compound Braid (11 limbs)
        moe_expert_loads = None
        if self.config.moe.enabled and self.config.moe.compound_enabled:
            moe_expert_loads = self._get_first_compound_moe_loads()
        combined_limbs, braid_info = self.compound_braid(
            [memory_out, spatial_out, language_out, meta_out,
             reasoning_out, perception_echo, dream_out,
             empathy_out, emotion_out, ethics_out, vis_out],
            attention_mask=attention_mask,
            moe_expert_loads=moe_expert_loads,
        )
        if braid_info.get('braid_signal') is not None:
            self._cached_braid_signal = braid_info['braid_signal'].detach()

        # Blend + post-braid reasoning
        blended = x + 0.3 * combined_limbs
        reasoned, _, _ = self.reasoning(
            blended, attention_mask=attention_mask,
            return_confidence=return_confidences
        )

        # --- Consequential memory write (feeds next iteration) ---
        if self._loop_memory_state is not None:
            write_content = reasoned.mean(dim=1)  # [batch, hidden]
            self._loop_memory_state = self.working_memory.write_to_state(
                write_content, self._loop_memory_state
            )

        return reasoned
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)
            
        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
        """
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if needed
                if generated.size(1) > self.config.model.max_seq_len:
                    context = generated[:, -self.config.model.max_seq_len:]
                else:
                    context = generated
                
                # Forward pass
                output = self.forward(input_ids=context)
                logits = output['logits']
                
                # Get next token logits
                next_logits = logits[:, -1, :]  # [batch, vocab]
                
                if do_sample:
                    # Sample with action limb's sampler
                    next_tokens, _ = self.action.sample(
                        next_logits.unsqueeze(1),
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    next_tokens = next_tokens.squeeze(1)
                else:
                    # Greedy
                    next_tokens = next_logits.argmax(dim=-1)
                
                # Append
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
                
                # Check for EOS (assuming token 0 or specific EOS)
                # For now, just generate max_new_tokens
        
        return generated

    def generate_diffusion(
        self,
        input_ids: torch.Tensor,
        num_tokens: int = 64,
        refine_steps: int = 8,
        temperature: float = 0.7,
        noise_schedule: str = 'cosine',
    ) -> torch.Tensor:
        """
        Diffusion-style generation: start from noise tokens, iteratively refine.
        
        Instead of generating one token at a time, generates all output tokens
        in parallel and refines them through multiple forward passes. Each step
        uses the model's RNA editing (per-input weight modulation) to adapt.
        
        Args:
            input_ids: Prompt token IDs [batch, prompt_len]
            num_tokens: Number of tokens to generate
            refine_steps: Number of refinement iterations
            temperature: Sampling temperature (decreases per step)
            noise_schedule: 'cosine' or 'linear' noise reduction
        """
        self.eval()
        B = input_ids.size(0)
        device = input_ids.device

        # Initialize output tokens from uniform random (structured noise)
        noisy_tokens = torch.randint(0, self.vocab_size, (B, num_tokens), device=device)
        
        with torch.no_grad():
            for step in range(refine_steps):
                # Decreasing temperature: more confident as we refine
                if noise_schedule == 'cosine':
                    progress = step / max(refine_steps - 1, 1)
                    t = temperature * (0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
                else:
                    t = temperature * (1.0 - step / refine_steps)
                t = max(t, 0.01)

                # Fraction of tokens to update this step (more early, fewer late)
                update_frac = 1.0 - 0.5 * (step / max(refine_steps - 1, 1))

                # Concatenate prompt + current noisy output
                full_seq = torch.cat([input_ids, noisy_tokens], dim=1)
                if full_seq.size(1) > self.config.model.max_seq_len:
                    full_seq = full_seq[:, -self.config.model.max_seq_len:]

                output = self.forward(input_ids=full_seq)
                logits = output['logits']

                # Get logits for the output positions only
                output_logits = logits[:, -num_tokens:, :]  # [B, num_tokens, vocab]

                # Compute confidence: how sure is the model about each position?
                probs = F.softmax(output_logits / t, dim=-1)
                confidence = probs.max(dim=-1).values  # [B, num_tokens]

                # Select which positions to update (lowest confidence first)
                num_update = max(1, int(num_tokens * update_frac))
                _, update_indices = confidence.topk(num_update, dim=-1, largest=False)

                # Sample new tokens for update positions
                new_tokens = torch.multinomial(
                    probs.view(-1, self.vocab_size), num_samples=1
                ).view(B, num_tokens)

                # Only update selected positions
                mask = torch.zeros_like(noisy_tokens, dtype=torch.bool)
                mask.scatter_(1, update_indices, True)
                noisy_tokens = torch.where(mask, new_tokens, noisy_tokens)

        return torch.cat([input_ids, noisy_tokens], dim=1)
    
    def sync_limbs(self, performance: float) -> Optional[Dict[str, Any]]:
        """
        Perform hub synchronization if needed.
        
        Args:
            performance: Current model performance (e.g., validation accuracy)
            
        Returns:
            Sync results if sync occurred, else None
        """
        self.hub_sync.step()
        
        if self.hub_sync.should_sync():
            return self.hub_sync.sync(self, self._limbs, performance)
        
        return None
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        """Get total parameter count"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_active_params(self) -> int:
        """Get active parameters per forward pass (accounts for MoE top-k routing)."""
        if not self.config.moe.enabled:
            return self.get_num_params()
        total = self.get_num_params()
        # Subtract inactive expert params: (E - K) experts × params_per_expert × L layers
        from core.moe import MoELayer
        from core.compound_moe import CompoundMoELayer
        for module in self.modules():
            if isinstance(module, (MoELayer, CompoundMoELayer)):
                inactive = module.total_params() - module.active_params()
                total -= inactive
        return total

    def _get_first_compound_moe_loads(self) -> Optional[torch.Tensor]:
        """Get expert loads from the first CompoundMoELayer for braid feedback."""
        from core.compound_moe import CompoundMoELayer
        for module in self.modules():
            if isinstance(module, CompoundMoELayer):
                return module.get_expert_loads()
        return None

    def prune_dead_experts(self) -> Dict[str, Any]:
        """
        LAEP: Prune dead experts across all CompoundMoE layers.
        Uses config thresholds. Returns pruning summary.
        """
        from core.compound_moe import CompoundMoELayer
        results = {}
        threshold = self.config.moe.expert_prune_threshold
        min_experts = self.config.moe.expert_prune_min
        for name, module in self.named_modules():
            if isinstance(module, CompoundMoELayer):
                candidates = module.get_pruning_candidates(threshold)
                if candidates:
                    pruned = module.prune_dead_experts(threshold, min_experts)
                    results[name] = {
                        'pruned': pruned,
                        'remaining': module.num_experts
                    }
        return results

    def get_compound_loop_stats(self) -> Dict[str, Any]:
        """Get stats from compound loop controller if active."""
        if hasattr(self, 'compound_loop') and self.compound_loop is not None:
            return self.compound_loop.get_stats()
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'total_params': self.get_num_params(),
            'trainable_params': self.get_num_params(trainable_only=True),
            'forward_count': self._forward_count,
            'last_confidences': self._last_confidences,
            'memory_utilization': self.working_memory.get_memory_state().norm().item(),
            'hub_sync_stats': self.hub_sync.get_stats(),
            'perception_stats': self.perception.get_stats(),
            'memory_limb_stats': self.memory_limb.get_stats(),
            'planning_stats': self.planning.get_stats(),
            'language_stats': self.language.get_stats(),
            'spatial_stats': self.spatial.get_stats(),
            'reasoning_stats': self.reasoning.get_stats(),
            'metacognition_stats': self.metacognition.get_stats(),
            'action_stats': self.action.get_stats()
        }
    
    def reset_memory(self):
        """Reset working memory slots"""
        self.working_memory.reset()
    
    def save_checkpoint(self, path: str, optimizer=None, step: int = 0):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'step': step,
            'stats': self.get_stats()
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu') -> Tuple['OctoTetrahedralModel', Dict]:
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        config = get_config()  # Default config
        # Could parse config from checkpoint['config'] here
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model, checkpoint


if __name__ == "__main__":
    print("Testing OctoTetrahedralModel...")
    
    # Create model with default config
    config = get_config()
    model = OctoTetrahedralModel(config)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {input_ids.shape}")
    
    output = model(
        input_ids=input_ids,
        labels=labels,
        return_confidences=True
    )
    
    print(f"Output keys: {output.keys()}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Confidences: {output['confidences']}")
    
    # Test generation
    print(f"\nTesting generation...")
    prompt = torch.randint(0, 1000, (1, 10))
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8
    )
    print(f"Prompt length: {prompt.size(1)}")
    print(f"Generated length: {generated.size(1)}")
    
    # Test hub sync
    print(f"\nTesting hub sync...")
    for _ in range(15):
        result = model.sync_limbs(performance=0.85)
        if result:
            print(f"Sync result: {result}")
    
    # Get stats
    stats = model.get_stats()
    print(f"\nModel stats:")
    print(f"  Forward count: {stats['forward_count']}")
    print(f"  Memory utilization: {stats['memory_utilization']:.4f}")
    
    print("\nAll OctoTetrahedralModel tests passed!")
