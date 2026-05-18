"""
OctoTetrahedral AGI - Core Module
Tetrahedral geometry-based reasoning system
"""

from .tetrahedral_geometry import TetrahedralGeometry
from .tetrahedral_attention import TetrahedralAttention
from .tetrahedral_core import TetrahedralCore
from .working_memory import WorkingMemory
from .reservoir_dynamics import (
    ReservoirDynamics,
    EchoStateConstraint,
    NeuralPacemaker,
    TemporalBasisDiversity,
    ReservoirReadout,
    LIMB_LEAK_TARGETS,
)

__all__ = [
    'TetrahedralGeometry',
    'TetrahedralAttention',
    'TetrahedralCore',
    'WorkingMemory',
    'ReservoirDynamics',
    'EchoStateConstraint',
    'NeuralPacemaker',
    'TemporalBasisDiversity',
    'ReservoirReadout',
    'LIMB_LEAK_TARGETS',
]
