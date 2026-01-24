"""
OctoTetrahedral AGI - Limbs Module
Distributed semi-autonomous processing units

8 Limbs (Octopus-inspired):
1. Perception - Sensory input processing
2. Reasoning - Abstract pattern processing
3. Action - Output generation
4. Memory - Long-term and episodic memory
5. Planning - Goal-directed planning
6. Language - Natural language understanding
7. Spatial - Spatial reasoning and geometry
8. MetaCognition - Self-monitoring and meta-learning
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

__all__ = [
    'BaseLimb',
    'PerceptionLimb',
    'ReasoningLimb', 
    'ActionLimb',
    'MemoryLimb',
    'PlanningLimb',
    'LanguageLimb',
    'SpatialLimb',
    'MetaCognitionLimb'
]
