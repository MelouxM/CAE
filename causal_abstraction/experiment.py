"""Serialization and management of experimental results."""

import json
import logging
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

from .config import EvaluationConfig
from .metrics import SubspaceCheckMetric, EvaluationMetric

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Encodes Numpy types and custom classes into JSON primitives."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):
             return obj.to_dict()
        return str(obj)


class AnalyticalResults(dict):
    """Mapping of analytical-metric name → result from
    :meth:`~causal_abstraction.engine.EvaluationEngine.run_analytical_metrics`.

    A ``dict`` subclass mapping metric name to value: a successful metric maps to
    its value (typically a float), and a metric that raised during computation (in
    non-strict mode) maps to ``{'error': <message>}``.

    On top of the mapping it exposes typed failure introspection, so callers do not
    have to recover failures by sniffing for an ``'error'`` key (which is
    indistinguishable from a metric that legitimately returns such a dict):

    - :attr:`errors`: ``{name: message}`` for every metric that failed
    - :attr:`failed`: names of metrics that failed
    - :attr:`succeeded`: names of metrics that produced a value
    - :meth:`is_error`: whether a given metric failed
    """

    _ERROR_KEY = "error"

    def _set_error(self, name: str, exc: BaseException) -> None:
        """Record ``name`` as failed using the ``{'error': msg}`` shape."""
        self[name] = {self._ERROR_KEY: str(exc)}

    def is_error(self, name: str) -> bool:
        """True if ``name`` is an error sentinel (``{'error': <message>}``)."""
        value = self.get(name)
        return isinstance(value, dict) and set(value) == {self._ERROR_KEY}

    @property
    def errors(self) -> Dict[str, str]:
        """``{name: message}`` for every metric that errored."""
        return {name: self[name][self._ERROR_KEY] for name in self if self.is_error(name)}

    @property
    def failed(self) -> List[str]:
        """Names of metrics that errored."""
        return [name for name in self if self.is_error(name)]

    @property
    def succeeded(self) -> List[str]:
        """Names of metrics that produced a value."""
        return [name for name in self if not self.is_error(name)]


@dataclass
class ExperimentResults:
    """A record of an evaluation experiment, including config and results."""
    score: float
    score_se: float  # Standard error
    score_by_var: Dict[str, float]
    faithfulness: Optional[float] = None
    global_precision: Optional[float] = None
    precision_by_variable: Optional[Dict[str, float]] = None
    failures: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metric: EvaluationMetric = field(default_factory=SubspaceCheckMetric)

    name: Optional[str] = None
    config: Optional[EvaluationConfig] = None

    def save(self, filepath: str, allow_pickle: bool = False):
        """Save the manifest to a file (``.json`` or ``.pkl``).

        Prefer ``.json``. The ``.pkl`` path uses :mod:`pickle`, whose artifacts are
        unsafe to load from untrusted sources, so writing one requires
        ``allow_pickle=True`` (see :meth:`load`).

        Args:
            filepath: Destination path; the format is chosen from the ``.json`` /
                ``.pkl`` suffix.
            allow_pickle: Required to write a ``.pkl`` file.

        Raises:
            ValueError: If the extension is unsupported, or it is ``.pkl`` and
                ``allow_pickle`` is ``False``.
            TypeError: If this results object was built without a ``config``
                (``config`` is ``None``); the manifest cannot be serialized
                without one (``asdict(None)`` fails). The normal engine-produced
                path always sets ``config``.
        """
        path = Path(filepath)
        data = asdict(self)
        data['config'] = asdict(self.config)

        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str, cls=NumpyEncoder)
        elif path.suffix == '.pkl':
            if not allow_pickle:
                raise ValueError(
                    f"Refusing to write pickle file {filepath!r}: pickle artifacts "
                    "are unsafe to load from untrusted sources. Pass allow_pickle=True "
                    "to override, or save as .json instead."
                )
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Unsupported file extension. Use .json or .pkl")
        logger.info(f"Experiment saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, allow_pickle: bool = False) -> 'ExperimentResults':
        """Load results from a ``.json`` or ``.pkl`` file.

        Args:
            filepath: Path to a ``.json`` or ``.pkl`` results file.
            allow_pickle: ``.pkl`` files are deserialized with :mod:`pickle`, which
                can execute arbitrary code during loading. They are therefore
                rejected unless this is explicitly ``True``. Only enable it for
                files you created or otherwise trust; for untrusted or shared data
                use ``.json``, the safe, preferred format for this public API.

        Raises:
            FileNotFoundError: If ``filepath`` does not exist.
            ValueError: If the extension is unsupported, or it is ``.pkl`` and
                ``allow_pickle`` is ``False``.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"{filepath} not found.")

        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix == '.pkl':
            if not allow_pickle:
                raise ValueError(
                    f"Refusing to load pickle file {filepath!r}: pickle can execute "
                    "arbitrary code during deserialization. Pass allow_pickle=True only "
                    "if you trust this file's source, or use the .json format instead."
                )
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError("Unsupported file extension. Use .json or .pkl")

        config_data = data.pop('config')
        config = EvaluationConfig(**config_data)

        return cls(config=config, **data)
