"""
OctoTetrahedral AGI - Limbs Module
Distributed semi-autonomous processing units

13 Limbs (Octopus + AGI Flower):
  Original 8 (Octopus-inspired):
    1. Perception - Sensory input processing
    2. Reasoning - Abstract pattern processing
    3. Action - Output generation
    4. Memory - Long-term and episodic memory
    5. Planning - Goal-directed planning
    6. Language - Natural language understanding
    7. Spatial - Spatial reasoning and geometry
    8. MetaCognition - Self-monitoring and meta-learning
  New 5 (AGI Flower cognitive petals):
    9. Visualization - Reconstructive mental imagery (memory → detail)
   10. Imagination - Generative exploration (novelty → experience)
   11. Empathy - Theory of Mind / agent modeling
   12. Emotion - Valence/arousal modulation of all processing
   13. Ethics - Value alignment and safety contraction

  Dream Mode orchestrator:
    - awake: full constraints + reality anchoring
    - daydream: visualization + imagination, loosened constraints
    - dream: pure imagination, no constraints
"""

from .base_limb import BaseLimb
from .perception_limb import PerceptionLimb
from .reasoning_limb import ReasoningLimb
from .action_limb import ActionLimb
from .memory_limb import MemoryLimb
from .planning_limb import PlanningLimb
from .language_limb import LanguageLimb
from .spatial_limb import SpatialLimb
from .metacognition_limb import MetaCognitionLimb
from .visualization_limb import VisualizationLimb
from .imagination_limb import ImaginationLimb
from .empathy_limb import EmpathyLimb
from .emotion_limb import EmotionLimb
from .ethics_limb import EthicsLimb
from .dream_mode import DreamMode

__all__ = [
    'BaseLimb',
    'PerceptionLimb',
    'ReasoningLimb',
    'ActionLimb',
    'MemoryLimb',
    'PlanningLimb',
    'LanguageLimb',
    'SpatialLimb',
    'MetaCognitionLimb',
    'VisualizationLimb',
    'ImaginationLimb',
    'EmpathyLimb',
    'EmotionLimb',
    'EthicsLimb',
    'DreamMode',
]
