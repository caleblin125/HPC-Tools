"""Optimizers."""

from .base import Optimizer
from .random import RandomOptimizer
from .smac3 import SMAC3Optimizer
from .raytune import RayTuneOptimizer
from .hyperopt import HyperoptOptimizer
from .deap import DEAPOptimizer
from .cmaes import CMAESOptimizer
from .elite import EliteSearchOptimizer

OPTIMIZERS = {
    "random": RandomOptimizer,
    "smac3": SMAC3Optimizer,
    "raytune": RayTuneOptimizer,
    "hyperopt": HyperoptOptimizer,
    "deap": DEAPOptimizer,
    "cmaes": CMAESOptimizer,
    "elite": EliteSearchOptimizer,
}

__all__ = [
    "Optimizer",
    "RandomOptimizer",
    "SMAC3Optimizer",
    "RayTuneOptimizer",
    "HyperoptOptimizer",
    "DEAPOptimizer",
    "CMAESOptimizer",
    "EliteSearchOptimizer",
    "OPTIMIZERS",
]

