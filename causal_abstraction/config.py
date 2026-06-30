"""Configuration data structures."""

import json
import logging
from dataclasses import dataclass, field, asdict, fields
from typing import List, Literal, Optional


@dataclass
class EvaluationConfig:
    """
    Configuration for an abstraction evaluation run.

    Attributes:
        metric (str): Dissimilarity ``D`` used to compare path outputs. One of
            ``'hard'`` (exact-match / discrete), ``'l2'``, ``'mse'``, ``'rmse'``,
            ``'nmse'``, ``'r2'`` (continuous), ``'jsd'`` (Jensen-Shannon),
            ``'kl'`` (Kullback-Leibler), ``'mmd'`` (maximum mean discrepancy),
            ``'traj_mse'``, ``'dtw'``, ``'autocorr'``, ``'spectral'`` (dynamical),
            ``'hsic'`` (HSIC-based conditional independence; resolves to
            :class:`~causal_abstraction.metrics.ConditionalIndependenceMetric`),
            and ``'var_decomp'`` (variance decomposition). See
            :func:`~causal_abstraction.metrics.get_metric` for the full mapping.
        check_precision (bool): If True, additionally compute the auxiliary
            (non-CAE) precision diagnostic over the ValueMap subspaces.
        return_detailed_failures (bool): If True, the result dictionary will
            include a list of specific per-sample failure cases (up to a cap).
        logging_level (int): Python logging level.
        n_jobs (int): The number of jobs to use for parallelization (joblib).
        error_threshold (float): Per-sample score above which a case is recorded
            as a detailed failure (only used when ``return_detailed_failures``).
        jsd_bins (int): Histogram bin count for the ``'jsd'`` metric.
        phi_noise_std (float): Standard deviation ``sigma_phi`` of the Gaussian
            noise overwritten onto continuous Phi variables during faithfulness
            testing.
        phi_noise_mode ({'gaussian', 'uniform'}): Noise law for continuous Phi
            variables: ``'gaussian'`` draws ``N(0, phi_noise_std**2)``,
            ``'uniform'`` draws ``Uniform(-phi_noise_std, phi_noise_std)``.
            (Integer Phi variables are overwritten with a uniform draw from
            ``{-1, 0, 1}`` and Booleans with a uniform resample, regardless of
            this setting.)
        phi_selection_prob (float or None): Probability with which the Phi
            sentinel is included among the intervention targets in
            :class:`~causal_abstraction.sampling.CombinedFaithfulnessSampler`.
            ``None`` (default) treats Phi as one extra candidate in a pool of
            ``len(variables) + 1`` (implicit probability).
        strict_mode (bool): If True, raise on mapping/abstraction failures
            instead of degrading to UNMAPPED.
        seed (int or None): Base seed for reproducible intervention sampling.
    """
    metric: Literal[
        'hard', 'l2', 'mse', 'nmse', 'rmse', 'r2', 'jsd', 'kl',
        'mmd', 'traj_mse', 'dtw', 'autocorr', 'spectral',
        'hsic', 'var_decomp',
    ] = 'hard'
    check_precision: bool = False
    return_detailed_failures: bool = False
    logging_level: int = logging.INFO
    n_jobs: int = 1

    # Metric parameters
    error_threshold: float = 0.01
    jsd_bins: int = 20

    # Faithfulness noise injection
    phi_noise_std: float = 0.1
    phi_noise_mode: Literal['gaussian', 'uniform'] = 'gaussian'
    phi_selection_prob: Optional[float] = None

    # Robustness
    strict_mode: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate numeric ranges; the ``metric`` string is validated lazily by
        :func:`~causal_abstraction.metrics.get_metric`.

        Raises:
            ValueError: If a numeric field is out of range (``phi_noise_std`` < 0,
                ``phi_selection_prob`` outside [0, 1], ``jsd_bins`` < 1,
                ``n_jobs`` == 0, or ``error_threshold`` < 0), or ``phi_noise_mode``
                is not ``'gaussian'``/``'uniform'``.
        """
        if self.phi_noise_std < 0:
            raise ValueError(
                f"phi_noise_std must be non-negative, got {self.phi_noise_std!r}."
            )
        if self.phi_noise_mode not in ('gaussian', 'uniform'):
            raise ValueError(
                "phi_noise_mode must be 'gaussian' or 'uniform', got "
                f"{self.phi_noise_mode!r}."
            )
        if self.phi_selection_prob is not None and not (
            0.0 <= self.phi_selection_prob <= 1.0
        ):
            raise ValueError(
                "phi_selection_prob must be in [0, 1] (or None), got "
                f"{self.phi_selection_prob!r}."
            )
        if self.jsd_bins < 1:
            raise ValueError(f"jsd_bins must be >= 1, got {self.jsd_bins!r}.")
        if self.n_jobs == 0:
            raise ValueError(
                f"n_jobs must be non-zero (use -1 for all cores), got "
                f"{self.n_jobs!r}."
            )
        if self.error_threshold < 0:
            raise ValueError(
                f"error_threshold must be non-negative, got {self.error_threshold!r}."
            )

    def to_json(self) -> str:
        """Serialize the config to a JSON string of all fields."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'EvaluationConfig':
        """Build a config from a JSON string produced by :meth:`to_json`.

        Raises:
            ValueError: If ``json_str`` contains field names not on EvaluationConfig,
                or if a decoded field value is out of range (validated by
                :meth:`__post_init__`).
        """
        data = json.loads(json_str)
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"Unknown EvaluationConfig field(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(valid)}."
            )
        return cls(**data)