"""
Causal abstraction library

A framework for validating causal explanations of complex systems.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    # Read from the installed distribution metadata (distribution name
    # "causal_abstraction_eval"); pyproject.toml is the single source of truth.
    __version__ = _version("causal_abstraction_eval")
except PackageNotFoundError:  # running from an un-installed source checkout
    __version__ = "0.0.0+unknown"

# Analytical metrics
from .analytical_metrics import (
    AnalyticalMetric,
    StructuralDeviationMetric,
    CausalSensitivityIndexMetric,
    MallowsCpMetric,
    IBLagrangianMetric,
    CIBLagrangianMetric,
    MacroscopicInvarianceMetric,
    ComplexityShiftMetric,
    SobolSensitivityMetric,
    IIAMetric,
    BCCMetric,
    ProbingMetric,
    InfidelityMetric,
    SymbionMetric,
    DCCMetric,
    RelationalFidelityMetric,
)
from .config import EvaluationConfig
from .engine import EvaluationEngine
from .experiment import ExperimentResults, AnalyticalResults
# Metrics
from .metrics import (
    EvaluationMetric,
    SubspaceCheckMetric,
    L2Metric,
    MSEMetric,
    RMSEMetric,
    NMSEMetric,
    R2Metric,
    JSDivergenceMetric,
    KLDivergenceMetric,
    PrecisionMetric,
    MMDMetric,
    TrajectoryMSEMetric,
    DTWMetric,
    TemporalAutocorrelationMetric,
    SpectralMetric,
    ConditionalIndependenceMetric,
    VarianceDecompositionMetric,
    get_metric,
)
from .models.high_level import CausalGraph
from .models.low_level import LowLevelModel, NoisyLowLevelModel
# NeuralModel is imported lazily (see __getattr__ below) so that a bare
# `import causal_abstraction` does not eagerly load torch.
from .paths import DiagramBuilder, CausalPath
from .primitives import SystemState, AbstractVariable, ProbabilityDistribution, EmpiricalDistribution, UNMAPPED, \
    DiscreteDistribution
from .sampling import (
    TopDownSampler, BottomUpSampler, InterventionSampler,
    PairedSampler, NoisyMeasurementSampler, CombinedFaithfulnessSampler,
    PHI_DUMMY_NAME,
)
from .schema import MicroVariableSchema, CoarseGrainingMap, MicroSelector, Variable
from .spaces.base import (
    Subspace, RectSubspace, SphereSubspace, UnionSubspace,
    ComplementSubspace, FullSubspace, UniformSubspace,
    GaussianSubspace
)
from .tasks import StandardTasks, EvaluationTask, TaskResults
from .valuemap import ValueMap, ContinuousValueMap

__all__ = [
    # Version
    "__version__",
    # Core primitives
    "SystemState",
    "AbstractVariable",
    "Variable",
    "ProbabilityDistribution",
    "EmpiricalDistribution",
    "UNMAPPED",
    "DiscreteDistribution",
    # Schema & mapping
    "MicroVariableSchema",
    "CoarseGrainingMap",
    "MicroSelector",
    "ValueMap",
    "ContinuousValueMap",
    # Config & results
    "EvaluationConfig",
    "ExperimentResults",
    "AnalyticalResults",
    # Engine
    "EvaluationEngine",
    # Samplers
    "TopDownSampler",
    "BottomUpSampler",
    "InterventionSampler",
    "PairedSampler",
    "NoisyMeasurementSampler",
    "CombinedFaithfulnessSampler",
    "PHI_DUMMY_NAME",
    # Models
    "LowLevelModel",
    "NoisyLowLevelModel",
    "CausalGraph",
    "NeuralModel",
    # Spaces
    "Subspace",
    "RectSubspace",
    "SphereSubspace",
    "UnionSubspace",
    "ComplementSubspace",
    "FullSubspace",
    "UniformSubspace",
    "GaussianSubspace",
    # Paths
    "DiagramBuilder",
    "CausalPath",
    # EvaluationMetric classes
    "EvaluationMetric",
    "SubspaceCheckMetric",
    "L2Metric",
    "MSEMetric",
    "RMSEMetric",
    "NMSEMetric",
    "R2Metric",
    "JSDivergenceMetric",
    "KLDivergenceMetric",
    "PrecisionMetric",
    "MMDMetric",
    "TrajectoryMSEMetric",
    "DTWMetric",
    "TemporalAutocorrelationMetric",
    "SpectralMetric",
    "ConditionalIndependenceMetric",
    "VarianceDecompositionMetric",
    "get_metric",
    # AnalyticalMetric classes
    "AnalyticalMetric",
    "StructuralDeviationMetric",
    "CausalSensitivityIndexMetric",
    "MallowsCpMetric",
    "IBLagrangianMetric",
    "CIBLagrangianMetric",
    "MacroscopicInvarianceMetric",
    "ComplexityShiftMetric",
    "SobolSensitivityMetric",
    "IIAMetric",
    "BCCMetric",
    "DCCMetric",
    "ProbingMetric",
    "InfidelityMetric",
    "SymbionMetric",
    "RelationalFidelityMetric",
    # Tasks
    "StandardTasks",
    "EvaluationTask",
    "TaskResults",
]


def __getattr__(name):
    # PEP 562 lazy attribute access: defer the torch import behind
    # NeuralModel so a bare `import causal_abstraction` stays lightweight.
    if name == "NeuralModel":
        from .models.neural import NeuralModel
        return NeuralModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")