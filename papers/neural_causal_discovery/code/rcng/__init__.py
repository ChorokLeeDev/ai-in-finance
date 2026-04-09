"""
RCNG: Regime-Conditional Neural Granger Causality

End-to-end joint learning of regime discovery and causal graph structure.
Key novelty: Regimes are defined by causal structure differences, not just distributional differences.
"""

from .joint_model import JointRCNG
from .synthetic_data import RegimeSwitchingDGP
from .evaluation import evaluate_regime_causal_discovery

__all__ = [
    'JointRCNG',
    'RegimeSwitchingDGP',
    'evaluate_regime_causal_discovery',
]
