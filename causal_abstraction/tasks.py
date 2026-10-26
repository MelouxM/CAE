"""
Self-contained evaluation tasks that can be composed into suites.

Each task specifies two paths to compare, a scorer, and optional overrides
for sampler, intervention domain, and sample count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .metrics import EvaluationMetric, get_metric, MSEMetric, R2Metric, NMSEMetric, KLDivergenceMetric, \
    JSDivergenceMetric
from .paths import CausalPath, DiagramBuilder
from .sampling import CombinedFaithfulnessSampler
from .sampling import InterventionSampler


@dataclass
class EvaluationTask:
    """
    A named, self-contained unit of evaluation.

    Compares path_a vs path_b using scorer.  Any field left as None falls
    back to the suite/engine default.

    Attributes:
        name:                Human-readable label shown in results.
        path_a:              The "ground truth" or reference path.
        path_b:              The "prediction" or approximation path.
        scorer:              Metric used to compare outputs.  Overrides engine default.
        sampler:             Sampler to use.  None -> engine default.
        intervention_domain: Variable names to intervene on.  None -> engine default.
        n_samples:           Sample count override.  None -> engine default.
        batch_size:          Batch size override.  None -> engine default.
        max_interventions:   Max simultaneous interventions.  None -> engine default.
        score_vars:          If set, only these output variables are scored.
                             Useful for observational tasks that should only
                             compare outputs, not all intermediate variables.
        aggregation:         The aggregation operator Agg of the metric family.
                             `'mean'` (default) is the CAE convention: the
                             dissimilarity is measured on all samples pooled per
                             variable, then averaged over variables.  `'sup'` and
                             `'inst_mean'` are instance-level: the target set of
                             each sampled cell is scored jointly, giving one
                             instance-level abstraction error per cell, and those
                             are aggregated by supremum (`'sup'`) or expectation
                             (`'inst_mean'`).  The instance-level forms are what
                             the ISIL/IC baselines require; see
                             `InstanceJSDMetric`.
    """
    name: str
    path_a: CausalPath
    path_b: CausalPath
    scorer: EvaluationMetric
    sampler: Optional[InterventionSampler] = None
    intervention_domain: Optional[List[str]] = None
    n_samples: Optional[int] = None
    batch_size: Optional[int] = None
    max_interventions: Optional[int] = None
    score_vars: Optional[List[str]] = None   # None = score everything path_a ∩ path_b returns
    aggregation: str = 'mean'

    AGGREGATIONS = ('mean', 'sup', 'inst_mean')

    def __post_init__(self):
        if self.aggregation not in self.AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {self.AGGREGATIONS}, got {self.aggregation!r}."
            )

    @property
    def is_instance_level(self) -> bool:
        """True when the task scores each sampled cell separately (ISIL/IC style)."""
        return self.aggregation in ('sup', 'inst_mean')


class TaskResults:
    """Results from a full task suite run."""

    def __init__(self, results: Dict[str, Any]):
        self._results = results

    def __getitem__(self, name: str):
        return self._results[name]

    def __repr__(self):
        lines = ["Results:"]
        for name, res in self._results.items():
            arrow = '↓' if res.metric.lower_is_better else '↑'
            lines.append(f"  {name:<20}: Score = {res.score:.4f} ± {1.96*res.score_se:.2e} / {res.metric.best_value} {arrow}")
        return "\n".join(lines)

    def summary(self) -> Dict[str, float]:
        """Return a mapping of task name to scalar score."""
        return {name: res.score for name, res in self._results.items()}

    def items(self):
        """Iterate over `(task_name, ExperimentResults)` pairs, mirroring `dict.items()`."""
        return self._results.items()


class StandardTasks:
    """
    Factory for the framework's evaluation tasks.

    Exposes exactly two causal-abstraction tasks, `CAE_up` (bottom-up) and
    `CAE_down` (top-down), both produced by `score`,
    plus a set of observational baselines (`observational` and its
    `observational_mse` / `observational_nmse` / `observational_r2` /
    `observational_kl` / `observational_jsd` shortcuts) provided only as
    reference points, not as validity metrics.

    Each causal-abstraction task supports an optional `include_faithfulness`
    parameter.  When True, a
    `CombinedFaithfulnessSampler` is
    automatically wrapped around the provided (or default) sampler so that
    phi-dummy interventions are mixed into every evaluation iteration.  This
    integrates the faithfulness check directly into the task score: poor
    faithfulness (low-level output shifts when phi is perturbed but the
    high-level model does not) raises the error, producing a unified score over
    both CAE↑/CAE↓ alignment and phi-variable leakage.
    """

    # Causal abstraction metrics
    @staticmethod
    def score(
        builder: DiagramBuilder,
        scorer: Optional[EvaluationMetric] = None,
        name: Optional[str] = None,
        include_faithfulness: bool = False,
        **overrides,
    ) -> EvaluationTask:
        """
        Standard CAE_up (bottom-up sampler) or CAE_down (top-down sampler).

        Args:
            builder: The diagram builder supplying the high- and low-level paths.
            scorer: Dissimilarity metric D. Defaults to `get_metric(builder.config)`.
            name: Task label used as the result-JSON key. Required and passed
                explicitly: one of `CAE_up` / `CAE_down` (faithful) or
                `CAE_up_nf` / `CAE_down_nf` (non-faithful), matching the sampler
                direction and the `include_faithfulness` flag. Naming is explicit
                rather than inferred from the sampler type, since a wrapped or
                subclassed `BottomUpSampler` would otherwise misname the key.
            include_faithfulness: If True, mix phi interventions into every iteration.
            **overrides: Extra `EvaluationTask` fields (e.g. `sampler`, `n_samples`).

        Returns:
            The configured `EvaluationTask`.

        Raises:
            ValueError: If `name` is not provided.
        """
        if name is None:
            raise ValueError(
                "StandardTasks.score() requires an explicit `name` (the result-JSON "
                "key), e.g. 'CAE_up', 'CAE_down', 'CAE_up_nf', or 'CAE_down_nf'."
            )

        from .sampling import BottomUpSampler

        def _check_direction(s):
            # Guard the CAE_up<->BottomUp / CAE_down<->TopDown alias so a mismatched
            # sampler can't be silently mislabeled
            is_bu = isinstance(s, BottomUpSampler)
            if name.startswith("CAE_up") and not is_bu:
                raise ValueError(
                    f"Task '{name}' is a bottom-up (CAE_up) task and requires a "
                    f"BottomUpSampler; got "
                    f"{type(s).__name__ if s is not None else 'the top-down faithful default'}."
                )
            if name.startswith("CAE_down") and is_bu:
                raise ValueError(
                    f"Task '{name}' is a top-down (CAE_down) task and must not use a "
                    f"BottomUpSampler."
                )

        if include_faithfulness:
            base_sampler = overrides.pop("sampler", None)
            if base_sampler is None:
                from .sampling import TopDownSampler
                base_sampler = TopDownSampler(builder.vm)
            _check_direction(base_sampler)
            combined_sampler = CombinedFaithfulnessSampler(
                base_sampler, phi_prob=getattr(builder.config, "phi_selection_prob", None)
            )
            return EvaluationTask(
                name=name,
                path_a=builder.build_path_standard_high_level_model(),
                path_b=builder.build_path_combined_low_level_model(),
                scorer=scorer or get_metric(builder.config),
                sampler=combined_sampler,
                **overrides,
            )

        _check_direction(overrides.get("sampler"))
        return EvaluationTask(
            name=name,
            path_a=builder.build_path_standard_high_level_model(),
            path_b=builder.build_path_standard_low_level_model(),
            scorer=scorer or get_metric(builder.config),
            **overrides,
        )

    # Observational metrics

    @staticmethod
    def observational(
        builder: DiagramBuilder,
        scorer: EvaluationMetric,
        sampler: Optional[InterventionSampler] = None,
        name: str = "observational",
        output_vars: Optional[List[str]] = None,
        **overrides,
    ) -> EvaluationTask:
        """
        Observational baseline: compare high-level model vs low-level model on inputs.
        """

        # Only intervene on root (input) variables
        if 'intervention_domain' not in overrides:
            root_names = [v.name for v in builder.high_level_model.get_roots()]
            overrides['intervention_domain'] = root_names

        path_high_level_model = CausalPath(
            f"{name}_high_level_model",
            [builder._step_high_level_model_predict],
            exclude_intervened=False,
        )
        path_low_level_model = CausalPath(
            f"{name}_low_level_model",
            [
                builder._step_ground_or_passthrough,
                builder._step_low_level_model_execute,
                builder._step_abstract,
            ],
            exclude_intervened=False,
        )
        return EvaluationTask(
            name=name,
            path_a=path_high_level_model,
            path_b=path_low_level_model,
            scorer=scorer,
            sampler=sampler,
            score_vars=output_vars,
            **overrides,
        )

    @staticmethod
    def observational_mse(
        builder: DiagramBuilder,
        sampler: Optional[InterventionSampler] = None,
        output_vars: Optional[List[str]] = None,
        name: str = "MSE",
        **overrides,
    ) -> EvaluationTask:
        """Observational baseline scored with MSE."""
        return StandardTasks.observational(
            builder, MSEMetric(), sampler=sampler, output_vars=output_vars, name=name, **overrides
        )

    @staticmethod
    def observational_nmse(
            builder: DiagramBuilder,
            sampler: Optional[InterventionSampler] = None,
            output_vars: Optional[List[str]] = None,
            name: str = "NMSE",
            **overrides,
    ) -> EvaluationTask:
        """Observational baseline scored with NMSE."""
        return StandardTasks.observational(
            builder, NMSEMetric(), sampler=sampler, output_vars=output_vars, name=name, **overrides
        )

    @staticmethod
    def observational_r2(
        builder: DiagramBuilder,
        sampler: Optional[InterventionSampler] = None,
        output_vars: Optional[List[str]] = None,
        name: str = "R^2",
        **overrides,
    ) -> EvaluationTask:
        """Observational baseline scored with R^2."""
        return StandardTasks.observational(
            builder, R2Metric(), sampler=sampler, output_vars=output_vars, name=name, **overrides
        )

    @staticmethod
    def observational_kl(
            builder: DiagramBuilder,
            sampler: Optional[InterventionSampler] = None,
            output_vars: Optional[List[str]] = None,
            name: str = "KL divergence",
            **overrides,
    ) -> EvaluationTask:
        """Observational baseline scored with KL divergence."""
        return StandardTasks.observational(
            builder, KLDivergenceMetric(), sampler=sampler, output_vars=output_vars, name=name, **overrides
        )

    @staticmethod
    def observational_jsd(
            builder: DiagramBuilder,
            sampler: Optional[InterventionSampler] = None,
            output_vars: Optional[List[str]] = None,
            name: str = "JSD divergence",
            **overrides,
    ) -> EvaluationTask:
        """Observational baseline scored with Jensen-Shannon divergence."""
        return StandardTasks.observational(
            builder, JSDivergenceMetric(), sampler=sampler, output_vars=output_vars, name=name, **overrides
        )


class ZennaroTasks:
    """ISIL and IC errors (Zennaro et al., 2023) as evaluation tasks.

    ISIL samples cells top-down and IC bottom-up, like CAE_down and CAE_up. Each
    cell is scored with `InstanceJSDMetric` and the cells are aggregated by sup
    (`*_sup`) or mean (`*_mean`). Phi is never intervened on. The sup is the max
    over the sampled cells, so it is a lower bound of the true sup.
    """

    SUP_NAMES = ("ISIL_sup", "IC_sup")
    MEAN_NAMES = ("ISIL_mean", "IC_mean")
    NAMES = SUP_NAMES + MEAN_NAMES

    @staticmethod
    def suite(
        builder: DiagramBuilder,
        n_instances: int,
        n_replicates: int = 1,
        bin_edges: Optional[Dict[str, Any]] = None,
        rtol: float = 1e-6,
        atol: float = 1e-12,
        top_down: Optional[InterventionSampler] = None,
        bottom_up: Optional[InterventionSampler] = None,
        extra_resolutions: Optional[Dict[str, Dict[str, Any]]] = None,
        **overrides,
    ) -> List[EvaluationTask]:
        """Build the ISIL_sup, ISIL_mean, IC_sup and IC_mean tasks.

        Args:
            builder: The diagram builder supplying the high- and low-level paths.
            n_instances: Number of cells to sample.
            n_replicates: Draws per cell. 1 is enough for deterministic models.
            bin_edges: Binning for continuous variables, see `InstanceJSDMetric`.
            rtol: Relative tolerance when comparing single draws.
            atol: Absolute tolerance when comparing single draws.
            top_down: Sampler for ISIL cells (default `TopDownSampler`).
            bottom_up: Sampler for IC cells (default `BottomUpSampler`).
            extra_resolutions: Optional `{suffix: bin_edges}`. Adds four tasks per
                entry (`ISIL_sup_<suffix>`, ...) scored on the same draws.
            **overrides: Extra `EvaluationTask` fields for all tasks.

        Returns:
            The tasks. Tasks of the same direction share a sampler, so the engine
            runs their interventions once.
        """
        from .metrics import InstanceJSDMetric
        from .sampling import BottomUpSampler, RepeatedInterventionSampler, TopDownSampler

        isil_sampler = RepeatedInterventionSampler(top_down or TopDownSampler(builder.vm), n_replicates)
        ic_sampler = RepeatedInterventionSampler(bottom_up or BottomUpSampler(builder.vm), n_replicates)

        resolutions = {"": bin_edges, **{f"_{k}": v for k, v in (extra_resolutions or {}).items()}}
        tasks: List[EvaluationTask] = []
        for res_suffix, edges in resolutions.items():
            scorer = InstanceJSDMetric(bin_edges=edges, rtol=rtol, atol=atol)
            for prefix, sampler in (("ISIL", isil_sampler), ("IC", ic_sampler)):
                for suffix, aggregation in (("sup", "sup"), ("mean", "inst_mean")):
                    tasks.append(EvaluationTask(
                        name=f"{prefix}_{suffix}{res_suffix}",
                        path_a=builder.build_path_standard_high_level_model(),
                        path_b=builder.build_path_standard_low_level_model(),
                        scorer=scorer,
                        sampler=sampler,
                        aggregation=aggregation,
                        n_samples=n_instances * n_replicates,
                        batch_size=n_replicates,
                        **overrides,
                    ))
        return tasks
