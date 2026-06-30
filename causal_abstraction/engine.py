"""
Orchestrates the execution of experiments by traversing the commuting diagram
and comparing the results using defined metrics.
"""

import logging
from collections import defaultdict
from typing import Callable, Dict, Any, List, Optional, Union, Tuple

import numpy as np
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm

from .analytical_metrics import AnalyticalMetric
from .tasks import TaskResults, EvaluationTask

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .primitives import AbstractVariable, UNMAPPED
from .schema import CoarseGrainingMap
from .valuemap import ValueMap
from .models.high_level import CausalGraph
from .models.low_level import LowLevelModel
from .sampling import InterventionSampler, TopDownSampler
from .metrics import get_metric, PrecisionMetric
from .config import EvaluationConfig
from .paths import DiagramBuilder, CausalPath
from .experiment import ExperimentResults, AnalyticalResults

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Orchestrates the verification of a causal abstraction."""

    def __init__(self,
                 high_level_model: CausalGraph,
                 low_level_model: LowLevelModel,
                 value_map: ValueMap,
                 cg_map: CoarseGrainingMap,
                 config: EvaluationConfig):
        """
        Args:
            high_level_model: The high-level model E.
            low_level_model: The low-level model M.
            value_map: The value map tau.
            cg_map: The coarse-graining map a.
            config: Evaluation configuration.
        """
        self.high_level_model = high_level_model
        self.low_level_model = low_level_model
        self.value_map = value_map
        self.cg_map = cg_map
        self.config = config
        self.scorer = get_metric(config)
        self.builder = DiagramBuilder(high_level_model, low_level_model, value_map, cg_map, config)
        # Set the level on the library's own logger namespace only. A library
        # must not call logging.basicConfig (which mutates the host's root logger
        # and is a no-op after the first call); the application owns handlers.
        logging.getLogger("causal_abstraction").setLevel(config.logging_level)

    def _compute_precision(self) -> Dict[str, float]:
        """
        Statically computes precision for each variable in the ValueMap.
        """
        scorer = PrecisionMetric()
        scores = {}

        for var_name, mapping in self.value_map.specs.items():
            try:
                subspaces = list(mapping.values())
                if not subspaces:
                    scores[var_name] = 0.0
                    continue

                total_vol = sum(s.volume() for s in subspaces)

                var_scores = []
                for label, s in mapping.items():
                    if total_vol == 0:
                        var_scores.append(1.0)
                    else:
                        var_scores.append(scorer.measure(s, total_vol))

                scores[var_name] = float(np.mean(var_scores))
            except Exception:
                logger.warning(
                    "Precision computation failed for variable %r; recording nan.",
                    var_name, exc_info=True,
                )
                scores[var_name] = float('nan')

        return scores

    def _sanitize_for_json(self, val: Any) -> Any:
        """Convert NumPy types to Python types for safe JSON serialization."""
        if hasattr(val, "item"):
            try:
                return val.item()
            except ValueError:
                pass
        if hasattr(val, "tolist"):
            return val.tolist()
        return val

    @staticmethod
    def _is_unmapped(v) -> bool:
        """Return True if v is or contains an UNMAPPED sentinel."""
        if v is UNMAPPED:
            return True
        if isinstance(v, list):
            return any(x is UNMAPPED for x in v)
        return False


    def _process_batch_raw(self,
                           sampler: InterventionSampler,
                           path_a: CausalPath,
                           path_b: CausalPath,
                           target_vars: List[AbstractVariable],
                           batch_size: int,
                           max_interventions: int,
                           seed: Optional[int] = None,
                           _all_roots: Optional[List[AbstractVariable]] = None,
                           ) -> Dict[str, Tuple[Any, Any]]:
        """
        Run a single batch of interventions and return raw (val_a, val_b) pairs
        without scoring.  UNMAPPED results are returned as (None, None) sentinels
        so callers can apply a penalty score.
        """
        rng = default_rng(seed)

        if _all_roots is None:
            _all_roots = self.high_level_model.get_roots()

        primary_spec = sampler.sample_intervention(target_vars, batch_size, max_interventions, rng=rng)
        intervention_spec = primary_spec.copy()

        missing_roots = [r for r in _all_roots if r.name not in intervention_spec]
        if missing_roots:
            bg_spec = sampler.sample_intervention(missing_roots, batch_size, force_all=True, rng=rng)
            if bg_spec:
                intervention_spec.update(bg_spec)

        results_a = path_a.execute(intervention_spec, rng=rng)
        results_b = path_b.execute(intervention_spec, rng=rng)

        common_vars = set(results_a.keys()) & set(results_b.keys())
        vars_to_compare = common_vars
        if path_a.exclude_intervened or path_b.exclude_intervened:
            vars_to_compare -= set(intervention_spec.keys())

        pairs: Dict[str, Tuple[Any, Any]] = {}
        n_unmapped = 0
        for var_name in vars_to_compare:
            val_a = results_a[var_name]
            val_b = results_b[var_name]
            if self._is_unmapped(val_a) or self._is_unmapped(val_b):
                pairs[var_name] = (None, None)
                n_unmapped += 1
            else:
                pairs[var_name] = (val_a, val_b)

        if n_unmapped > 0 and n_unmapped == len(vars_to_compare):
            logger.warning(
                "Batch produced no scoreable outputs. All %d variable(s) were "
                "UNMAPPED or had shape mismatches. Check low-level model output shapes and "
                "ValueMap coverage. Variables attempted: %s",
                len(vars_to_compare), sorted(vars_to_compare)
            )

        logger.debug(
            "Batch raw: %d vars scored, %d UNMAPPED, intervention keys: %s",
            len(vars_to_compare) - n_unmapped, n_unmapped,
            sorted(intervention_spec.keys())
        )

        return pairs

    def _dispatch_parallel_raw(
            self, sampler, path_a, path_b, targets, n_batches, batch_size, max_int, n_jobs,
            task_name: str = "",
        ) -> List[Dict[str, Tuple[Any, Any]]]:
        """Parallel dispatch using _process_batch_raw (returns raw value pairs)."""
        desc = f"Eval {task_name} {path_a.name} vs {path_b.name}" if task_name else f"Shared {path_a.name} vs {path_b.name}"
        ss = SeedSequence(self.config.seed)
        child_seeds = ss.spawn(n_batches)

        all_roots = [v for name, v in self.high_level_model.variables.items()
                     if not self.high_level_model._nodes.get(name, {}).get('parents')]

        if JOBLIB_AVAILABLE and n_jobs != 1:
            return Parallel(n_jobs=n_jobs)(
                delayed(self._process_batch_raw)(
                    sampler, path_a, path_b, targets, batch_size, max_int, seed=child_seeds[i], _all_roots=all_roots
                ) for i in tqdm(range(n_batches), desc=desc)
            )
        else:
            return [self._process_batch_raw(
                sampler, path_a, path_b, targets, batch_size, max_int,
                seed=child_seeds[i], _all_roots=all_roots
            ) for i in tqdm(range(n_batches), desc=desc)]

    def _score_collected_results(
        self,
        raw_batches: List[Dict[str, Tuple[Any, Any]]],
        scorer,
        score_vars: Optional[List[str]] = None,
        max_failures: int = 50,
        path_a_name: str = "",
        path_b_name: str = "",
    ) -> Tuple[Dict[str, float], float, List[Dict], Dict[str, float]]:
        """
        Score collected raw (val_a, val_b) pairs from _dispatch_parallel_raw.

        UNMAPPED sentinels (None, None) are counted and receive a penalty
        score of 1.0, blended with the distributional score from valid pairs.

        Returns:
            A 4-tuple ``(avg_scores, global_se, all_failures, coverage)``: per-variable
            mean scores, the global standard error, the recorded failure cases, and
            per-variable mapping coverage.
        """
        # Accumulate per-variable across all batches
        all_val_a: Dict[str, list] = defaultdict(list)
        all_val_b: Dict[str, list] = defaultdict(list)
        # Per-batch scores for SE estimation
        per_batch_scores: Dict[str, list] = defaultdict(list)
        # UNMAPPED tracking
        unmapped_counts: Dict[str, int] = defaultdict(int)
        # Track all variables seen (including fully-UNMAPPED ones)
        all_vars_seen: set = set()

        for raw in raw_batches:
            batch_a: Dict[str, list] = defaultdict(list)
            batch_b: Dict[str, list] = defaultdict(list)
            for var_name, (va, vb) in raw.items():
                if score_vars is not None and var_name not in score_vars:
                    continue
                all_vars_seen.add(var_name)
                if va is None or vb is None:
                    # UNMAPPED sentinel: penalty score for this batch
                    per_batch_scores[var_name].append(1.0)
                    unmapped_counts[var_name] += 1
                    continue
                va_arr = np.atleast_1d(np.asarray(va, dtype=float)).ravel()
                vb_arr = np.atleast_1d(np.asarray(vb, dtype=float)).ravel()
                all_val_a[var_name].append(va_arr)
                all_val_b[var_name].append(vb_arr)
                batch_a[var_name].append(va_arr)
                batch_b[var_name].append(vb_arr)

            # Per-batch score for valid (non-UNMAPPED) pairs in this batch
            for var_name in batch_a:
                ba = np.concatenate(batch_a[var_name])
                bb = np.concatenate(batch_b[var_name])
                per_batch_scores[var_name].append(float(scorer.measure(ba, bb)))

        # Main score: full distribution, blended with UNMAPPED penalty
        avg_scores: Dict[str, float] = {}
        coverage: Dict[str, float] = {}  # fraction of scoreable samples per variable
        for var_name in all_vars_seen:
            n_unmapped = unmapped_counts.get(var_name, 0)
            n_total = len(per_batch_scores.get(var_name, []))

            if var_name in all_val_a and all_val_a[var_name]:
                va_full = np.concatenate(all_val_a[var_name])
                vb_full = np.concatenate(all_val_b[var_name])
                measured_score = float(scorer.measure(va_full, vb_full))
                if n_total > 0 and n_unmapped > 0:
                    # weight valid measured score and explicit penalty by sample share.
                    n_valid = n_total - n_unmapped
                    frac_valid = n_valid / n_total
                    avg_scores[var_name] = measured_score * frac_valid + 1.0 * (1.0 - frac_valid)
                    coverage[var_name] = frac_valid
                else:
                    avg_scores[var_name] = measured_score
                    coverage[var_name] = 1.0
            else:
                avg_scores[var_name] = 1.0
                coverage[var_name] = 0.0

        # Log summary
        total_unmapped = sum(unmapped_counts.values())
        if total_unmapped > 0:
            logger.warning(
                "UNMAPPED results encountered during scoring: %s (penalty=1.0 applied)",
                dict(unmapped_counts),
            )

        # Global SE from per-batch score variance
        if per_batch_scores:
            per_var_se = []
            for var_name, batch_scores in per_batch_scores.items():
                # Drop NaN batch scores (e.g. HSIC on <20-sample batches, VarDecomp on
                # 1-sample batches) so full-sample estimators don't poison the SE.
                finite = [s for s in batch_scores if not np.isnan(s)]
                if len(finite) > 1:
                    per_var_se.append(
                        float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                    )
                else:
                    per_var_se.append(0.0)
            global_se = float(np.mean(per_var_se)) if per_var_se else 0.0
        else:
            global_se = 0.0

        logger.debug(
            "Scored %d variables: %s",
            len(avg_scores),
            {k: f"{v:.4f}" for k, v in avg_scores.items()},
        )

        # Failures
        all_failures: List[Dict] = []
        if self.config.return_detailed_failures:
            for var_name in all_val_a:
                if len(all_failures) >= max_failures:
                    break
                va_full = np.concatenate(all_val_a[var_name])
                vb_full = np.concatenate(all_val_b[var_name])
                for i in range(len(va_full)):
                    if len(all_failures) >= max_failures:
                        break
                    item_score = scorer.measure(
                        va_full[i:i + 1], vb_full[i:i + 1]
                    )
                    if item_score > self.config.error_threshold:
                        all_failures.append({
                            "variable": var_name,
                            "path_a": path_a_name,
                            "path_b": path_b_name,
                            "val_a": va_full[i],
                            "val_b": vb_full[i],
                            "score": item_score,
                            "sample_index": i,
                        })

        return avg_scores, global_se, all_failures, coverage

    def _task_group_key(self, task: EvaluationTask,
                        default_sampler: InterventionSampler,
                        default_targets: List[AbstractVariable],
                        default_n: int,
                        default_bs: int,
                        default_mi: int) -> tuple:
        """
        Return a hashable key that identifies tasks which can share intervention
        runs.  Two tasks with the same key have identical paths, sampler, and
        execution parameters - only their scorer may differ.
        """
        sampler = task.sampler if task.sampler is not None else default_sampler
        targets = (
            self._resolve_domain(task.intervention_domain) if task.intervention_domain is not None else default_targets
        )
        n = task.n_samples if task.n_samples is not None else default_n
        bs = task.batch_size if task.batch_size is not None else default_bs
        mi = task.max_interventions if task.max_interventions is not None else default_mi

        # Bound methods compare equal when __func__ and __self__ are the same,
        # so tasks built from the same DiagramBuilder will hash into the same group.
        return (
            tuple(task.path_a.steps),
            task.path_a.exclude_intervened,
            tuple(task.path_b.steps),
            task.path_b.exclude_intervened,
            id(sampler),
            n,
            bs,
            mi,
            frozenset(t.name for t in targets),
        )

    def run_tasks(
            self,
            tasks: List[EvaluationTask],
            n_samples: int,
            batch_size: int = 1,
            max_interventions: int = 1,
            intervention_domain: Optional[List[Union[str, AbstractVariable]]] = None,
            sampler: Optional[InterventionSampler] = None,
    ) -> TaskResults:
        """
        Run a list of EvaluationTasks, each with its own paths and scorer,
        sharing the remaining parameters as defaults.

        Per-task fields (sampler, intervention_domain, n_samples, batch_size,
        max_interventions, scorer) override shared defaults when set.

        Tasks that share identical paths, sampler, and execution parameters are
        grouped automatically: intervention runs are executed only once per group,
        and each task's scorer is applied to the cached (val_a, val_b) pairs.

        Args:
            tasks: The tasks to run.
            n_samples: Default sample count per task.
            batch_size: Default batch size.
            max_interventions: Default cap on simultaneous interventions.
            intervention_domain: Default variables to intervene on (None = all).
            sampler: Default sampler (None = TopDownSampler).

        Returns:
            A TaskResults keyed by task name.

        Raises:
            ValueError: If a domain name is not in the high-level model, or an
                AbstractVariable is not registered.
            TypeError: If a domain item is neither str nor AbstractVariable.
        """
        default_targets = self._resolve_domain(intervention_domain)
        default_sampler = sampler or TopDownSampler(self.value_map)

        # Group tasks that can share intervention runs
        # dict preserves insertion order, so result ordering matches the input task list.
        groups: Dict[tuple, List[EvaluationTask]] = {}
        for task in tasks:
            key = self._task_group_key(task, default_sampler, default_targets, n_samples, batch_size, max_interventions)
            groups.setdefault(key, []).append(task)

        all_results: Dict[str, Any] = {}

        for key, group_tasks in groups.items():
            first = group_tasks[0]

            # Resolve execution parameters from the first task in the group
            task_sampler = first.sampler if first.sampler is not None else default_sampler
            task_targets = (
                self._resolve_domain(first.intervention_domain)
                if first.intervention_domain is not None
                else default_targets
            )
            task_n = first.n_samples if first.n_samples is not None else n_samples
            task_bs = first.batch_size if first.batch_size is not None else batch_size
            task_mi = first.max_interventions if first.max_interventions is not None else max_interventions
            n_batches = max(1, task_n // task_bs)

            if len(group_tasks) == 1:
                task = group_tasks[0]
                logger.info(f"Running task {task.name}...")

                raw_batches = self._dispatch_parallel_raw(
                    task_sampler, task.path_a, task.path_b,
                    task_targets, n_batches, task_bs, task_mi,
                    self.config.n_jobs, task_name=task.name,
                )

                avg_scores, global_se, all_failures, coverage = self._score_collected_results(
                    raw_batches, task.scorer, score_vars=task.score_vars,
                    path_a_name=task.path_a.name, path_b_name=task.path_b.name,
                )

                global_score = float(np.mean(list(avg_scores.values()))) if avg_scores else float("nan")
                all_results[task.name] = ExperimentResults(
                    score=global_score, score_se=global_se, score_by_var=avg_scores,
                    failures=all_failures,
                    metadata={"coverage_by_var": coverage},
                    config=self.config, name=task.name, metric=task.scorer,
                )
            else:
                task_names = [t.name for t in group_tasks]
                logger.debug(f"Running shared interventions for tasks: {task_names}...")

                raw_batches = self._dispatch_parallel_raw(
                    task_sampler, first.path_a, first.path_b,
                    task_targets, n_batches, task_bs, task_mi,
                    self.config.n_jobs, task_name=f"shared({','.join(task_names)})",
                )

                for task in group_tasks:
                    avg_scores, global_se, all_failures, coverage = self._score_collected_results(
                        raw_batches, task.scorer, score_vars=task.score_vars,
                        path_a_name=first.path_a.name, path_b_name=first.path_b.name,
                    )

                    global_score = float(np.mean(list(avg_scores.values()))) if avg_scores else float("nan")
                    all_results[task.name] = ExperimentResults(
                        score=global_score, score_se=global_se, score_by_var=avg_scores,
                        failures=all_failures, metadata={"coverage_by_var": coverage},
                        config=self.config, name=task.name, metric=task.scorer,
                    )

        return TaskResults(all_results)

    def _resolve_domain(self, domain: Optional[List[Union[str, AbstractVariable]]]) -> List[AbstractVariable]:
        """Resolve a string/AbstractVariable list to registered AbstractVariables.

        Raises:
            ValueError: If a name is not in the high-level model, or an
                AbstractVariable is not registered.
            TypeError: If an item is neither str nor AbstractVariable.
        """
        # Resolve strings to objects
        target_vars = []
        if domain is None:
            target_vars = list(self.high_level_model.variables.values())
        else:
            for item in domain:
                if isinstance(item, str):
                    if item not in self.high_level_model.variables:
                        raise ValueError(f"Variable '{item}' not found in high-level model. "
                                         f"Available: {list(self.high_level_model.variables.keys())}")
                    target_vars.append(self.high_level_model.variables[item])
                elif isinstance(item, AbstractVariable):
                    # Validate it actually belongs to this high-level model
                    registered = self.high_level_model.variables.get(item.name)
                    if registered is None:
                        raise ValueError(f"AbstractVariable '{item.name}' not registered in high-level model.")
                    if registered is not item:
                        import warnings
                        warnings.warn(
                            f"AbstractVariable '{item.name}' is not the same object as the one "
                            f"registered in the high-level model. Using the registered version.",
                            UserWarning
                        )
                    target_vars.append(registered)
                else:
                    raise TypeError(f"domain items must be str or AbstractVariable, got {type(item)}")
        return target_vars

    def run_experiment(self,
                       sampler: InterventionSampler,
                       n_samples: int,
                       batch_size: int = 1,
                       max_interventions: int = 1,
                       intervention_domain: Optional[List[Union[str, AbstractVariable]]] = None,
                       save_metadata: bool = False,
                       additional_metadata: Optional[Dict] = None,
                       max_failures_to_report: int = 50,
                       custom_paths: Optional[Tuple[CausalPath, CausalPath]] = None,
                       ) -> ExperimentResults:
        """
        Execute the full evaluation pipeline for a single experiment.

        Args:
            sampler: The intervention sampler.
            n_samples: Total number of intervention samples.
            batch_size: Samples per batch.
            max_interventions: Cap on simultaneous interventions.
            intervention_domain: Variables to intervene on (None = all).
            save_metadata: If True, record run parameters in the result metadata.
            additional_metadata: Extra metadata merged in when ``save_metadata``.
            max_failures_to_report: Cap on recorded detailed failure cases.
            custom_paths: Optional (path_a, path_b) overriding the standard paths.

        Returns:
            The ExperimentResults for the run.

        Raises:
            ValueError: If a domain name is not in the high-level model, or an
                AbstractVariable is not registered.
            TypeError: If a domain item is neither str nor AbstractVariable.
        """
        is_surjective = self.value_map.validate_surjectivity()
        if not is_surjective:
            logger.warning("ValueMap failed surjectivity check. Some abstract states cannot be grounded.")

        target_vars = self._resolve_domain(intervention_domain)
        n_batches = max(1, n_samples // batch_size)

        if custom_paths:
            path_high_level_model, path_low_level_model = custom_paths
        else:
            path_high_level_model = self.builder.build_path_standard_high_level_model()
            path_low_level_model = self.builder.build_path_standard_low_level_model()

        raw_batches = self._dispatch_parallel_raw(
            sampler, path_high_level_model, path_low_level_model, target_vars, n_batches, batch_size,
            max_interventions, self.config.n_jobs, task_name="default",
        )

        avg_scores, global_se, all_failures, coverage = self._score_collected_results(
            raw_batches, self.scorer, max_failures=max_failures_to_report,
            path_a_name=path_high_level_model.name, path_b_name=path_low_level_model.name,
        )

        score_list = list(avg_scores.values())
        if not score_list:
            logger.warning(
                "Experiment produced no scores at all. This usually means every "
                "batch returned UNMAPPED for all variables. Check that:\n"
                "  1. low-level model output variable names match the CoarseGrainingMap selectors\n"
                "  2. ValueMap.abstract() can map the low-level model output values to labels\n"
                "  3. low-level model output arrays have shape (batch, *features), not (batch,)"
            )
        global_score = float(np.mean(score_list)) if score_list else float('nan')

        # Precision
        precision_scores = None
        global_precision = None
        if self.config.check_precision:
            precision_scores = self._compute_precision()
            valid_p_scores = [s for s in precision_scores.values() if not np.isnan(s)]
            global_precision = float(np.mean(valid_p_scores)) if valid_p_scores else 0.0

        if save_metadata:
            metadata = {
                "n_samples": n_samples,
                "batch_size": batch_size,
                "max_interventions": max_interventions,
                "intervention_domain": intervention_domain,
                "n_jobs": self.config.n_jobs,
                "coverage_by_var": coverage
            }
            if additional_metadata:
                metadata.update(additional_metadata)
        else:
            metadata = {}

        return ExperimentResults(
            score=global_score,
            score_se=global_se,
            faithfulness=None,
            score_by_var=avg_scores,
            global_precision=global_precision,
            precision_by_variable=precision_scores,
            failures=all_failures,
            metadata=metadata,
            config=self.config
        )


    def run_analytical_metrics(
        self,
        metrics: List[AnalyticalMetric],
        sampler: Optional[InterventionSampler] = None,
        n_samples: int = 100,
        config: Optional[EvaluationConfig] = None,
    ) -> AnalyticalResults:
        """
        Run a list of AnalyticalMetric objects and collect their results.

        Analytical metrics perform their own internal experiments (not path-based).

        Args:
            metrics: The metrics to evaluate.
            sampler: Sampler passed to each metric. Defaults to TopDownSampler.
            n_samples: Number of samples for each metric's internal experiment.
            config: Configuration forwarded to each metric. Falls back to the
                engine's own config.

        Returns:
            An AnalyticalResults (``dict`` subclass) mapping ``{metric_class_name:
            result}``. Successful metrics map to their value; a metric that raised
            (in non-strict mode) maps to ``{'error': <message>}``. The wrapper adds
            ``.errors`` / ``.failed`` / ``.succeeded`` / ``.is_error(name)`` so
            failures can be detected without sniffing for an ``'error'`` key.
        """
        if sampler is None:
            sampler = TopDownSampler(self.value_map)
        cfg = config if config is not None else self.config

        results = AnalyticalResults()
        errored: List[str] = []
        for i, metric in enumerate(metrics):
            name = type(metric).__name__
            logger.info(f"Running metric {i+1}/{len(metrics)}: {type(metric).__name__}")
            try:
                result = metric.compute(self.high_level_model, self.low_level_model, self.value_map, self.cg_map, sampler, n_samples, cfg)
                results[name] = result
                logger.debug(f"Score: {result}")
            except Exception as exc:
                if cfg.strict_mode:
                    raise
                logger.warning("AnalyticalMetric %s failed: %s", name, exc)
                results._set_error(name, exc)
                errored.append(name)
        if errored:
            # Errors are stored as {'error': ...} markers (queryable via
            # results.errors / results.failed); surface an aggregate count too, so a
            # silently half-failed battery is not mistaken for a clean run.
            logger.warning(
                "%d of %d analytical metric(s) errored and were stored as "
                "{'error': ...}, not as values: %s. Inspect before treating "
                "these results as reported numbers (set strict_mode=True to "
                "raise on the first failure instead, or check result.errors).",
                len(errored), len(metrics), ", ".join(errored),
            )
        return results
