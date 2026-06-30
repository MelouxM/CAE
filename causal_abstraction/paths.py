"""Graph traversal logic for the commuting diagram.

Composes operations (abstract -> intervene vs intervene -> abstract).
"""
import logging
from collections import defaultdict
from typing import Dict, Any, List, Callable, Optional, Union

import numpy as np

from .config import EvaluationConfig
from .models.high_level import CausalGraph
from .models.low_level import LowLevelModel
from .primitives import SystemState, UNMAPPED, EmpiricalDistribution, ensure_batch_2d, infer_batch_size, \
    GroundedInterventions, InterventionSpec
from .schema import CoarseGrainingMap, MicroSelector
from .valuemap import ValueMap

logger = logging.getLogger(__name__)

PipelineData = Union[Dict[str, Any], SystemState]


def _selector_width(index, var_dim: int) -> int:
    """Number of micro-dimensions a ``MicroSelector`` index spans within ``var_dim``."""
    if index is None:
        return var_dim
    if isinstance(index, slice):
        return len(range(*index.indices(var_dim)))
    if isinstance(index, tuple):
        return len(index)
    return 1  # single int index


class CausalPath:
    """Represents a sequence of operations in the commuting diagram."""

    def __init__(self, name: str, steps: List[Callable],
                 exclude_intervened: bool = True,
                 input_type: type = dict,
                 output_type: type = dict):
        """
        Args:
            name: Human-readable path identifier
            steps: List of callable steps that transform data
            exclude_intervened: Whether to exclude intervened variables from scoring (True for CAE consistency paths)
            input_type: Expected type of initial_data (for documentation)
            output_type: Expected type of return value (for documentation)
        """
        self.name = name
        self.steps = steps
        self.exclude_intervened = exclude_intervened
        self.input_type = input_type
        self.output_type = output_type

    def execute(self, initial_data: PipelineData,
                rng: Optional[np.random.Generator] = None) -> PipelineData:
        """Run each step in sequence, threading ``rng`` through every step.

        Args:
            initial_data: The starting intervention spec (dict) or SystemState.
            rng: Optional generator for reproducible sampling within steps.

        Returns:
            The output of the final step.
        """
        data = initial_data
        for step in self.steps:
            data = step(data, rng=rng)
        return data


class DiagramBuilder:
    """Helper to construct standard paths."""

    def __init__(self,
                 high_level_model: CausalGraph,
                 low_level_model: LowLevelModel,
                 vm: ValueMap,
                 cg: CoarseGrainingMap,
                 config: EvaluationConfig):
        """
        Args:
            high_level_model: The high-level model E.
            low_level_model: The low-level model M.
            vm: The value map tau.
            cg: The coarse-graining map a.
            config: Evaluation configuration.
        """
        self.high_level_model = high_level_model
        self.low_level_model = low_level_model
        self.vm = vm
        self.cg = cg
        self.config = config

        self._all_roots = self.high_level_model.get_roots()

    def _get_rng(self, rng: Optional[np.random.Generator]) -> np.random.Generator:
        return rng if rng is not None else np.random.default_rng()

    # Steps

    def _step_ground_or_passthrough(self, intervention_spec: 'InterventionSpec',
                                    rng: Optional[np.random.Generator] = None) -> 'GroundedInterventions':
        """
        Prepare low-level interventions by grounding abstract labels to micro-values.

        This grounding (the inverse value map tau^{-1}) is shared by both directions:
        bottom-up (CAE_up) paths originate interventions at the low level, while
        top-down (CAE_down) paths use it to lift high-level interventions down to the
        micro-level. In both cases it samples from the subspace of each abstract label
        via ValueMap.ground(), splitting grounded values across MicroSelectors.

        Raises:
            ValueError: In strict_mode, if an abstract variable has no micro-mapping
                in the CoarseGrainingMap.
        """
        low_level_model_interventions = {}
        batch_size = infer_batch_size(intervention_spec)

        pending_writes = {}

        for aname, data in intervention_spec.items():
            # Skip high-level model variables if the intervention spec contains both (custom samplers)
            if not self.cg.get_micro_vars(aname):
                if self.config.strict_mode:
                    raise ValueError(f"No micro-mapping for '{aname}' in CoarseGrainingMap.")
                logger.warning(
                    "Skipping '%s': no micro-mapping found in CoarseGrainingMap. Available abstract variables: %s",
                    aname, sorted(self.cg._fwd_map.keys())
                )
                continue

            # Get the full grounded tensor
            val = None
            if 'micro_values' in data and data['micro_values'] is not None:
                val = data['micro_values']
            elif 'labels' in data:
                labels = data['labels']
                if batch_size > 1:
                    grounded = [self.vm.ground(aname, lbl) for lbl in labels]
                    val = np.stack(grounded)
                else:
                    labels = labels[0]
                    val = self.vm.ground(aname, labels)
                    if val.ndim == 0:
                        val = val.reshape(1, 1)
                    elif val.ndim == 1:
                        val = val[np.newaxis, :]

            if val is None:
                continue

            # Split and assign to selectors
            selectors = self.cg.get_micro_vars(aname)

            if len(selectors) == 1:
                sel = selectors[0]
                self._add_pending_write(pending_writes, sel, val)
            else:
                # Split logic. Standard: elements correspond to selectors.
                if val.shape[-1] == len(selectors):
                    for i, sel in enumerate(selectors):
                        part_val = val[..., i:i + 1]
                        self._add_pending_write(pending_writes, sel, part_val)
                else:
                    # Dimensionality mismatch fallback: broadcast full val
                    for sel in selectors:
                        self._add_pending_write(pending_writes, sel, val)

        # Finalize interventions
        for base_var, updates in pending_writes.items():
            # updates is list of (index, val)
            if len(updates) == 1 and updates[0][0] is None:
                low_level_model_interventions[base_var] = updates[0][1]
            else:
                low_level_model_interventions[base_var] = updates

        return low_level_model_interventions

    def _add_pending_write(self, pending, selector, val):
        if selector.variable not in pending:
            pending[selector.variable] = []
        pending[selector.variable].append((selector.index, val))

    def _step_inject_phi_noise(self, low_level_model_interventions: 'GroundedInterventions',
                               rng: Optional[np.random.Generator] = None, batch_size=None) -> 'GroundedInterventions':
        """Injects noise into Phi (unmapped) variables.

        Each Phi element is either a whole-variable name (``str``) or a
        :class:`MicroSelector` pointing at the unmapped dimensions of a partially
        mapped variable. A whole-variable element overwrites the variable with
        noise; a selector element writes noise into only its indices, merged with
        (rather than clobbering) any grounded mapped dimensions of the same
        variable that are already present.
        """
        rng = self._get_rng(rng)

        if batch_size is None:
            batch_size = infer_batch_size(low_level_model_interventions)

        phi_vars = self.cg.phi_variables
        new_interventions = low_level_model_interventions.copy()

        for phi in phi_vars:
            if isinstance(phi, MicroSelector):
                var_name = phi.variable
                index = phi.index
                width = _selector_width(index, self.cg.schema.resolve_dim(var_name))
            else:
                var_name = phi
                index = None
                width = 1

            dtype = self.cg.schema.get_dtype(var_name)
            noise_std = self.config.phi_noise_std

            # Check explicit config mode or infer from dtype
            if dtype == float:
                if hasattr(self.config, 'phi_noise_mode') and self.config.phi_noise_mode == 'uniform':
                    var_noise = rng.uniform(-noise_std, noise_std, size=(batch_size, width))
                else:
                    var_noise = rng.normal(0, noise_std, size=(batch_size, width))
            elif dtype == int:
                var_noise = rng.integers(-1, 2, size=(batch_size, width))
            elif dtype == bool:
                var_noise = rng.choice([True, False], size=(batch_size, width))
            else:
                var_noise = rng.normal(0, noise_std, size=(batch_size, width))

            if index is None:
                # Whole-variable Phi: overwrite the variable with noise.
                new_interventions[var_name] = var_noise
            else:
                # Partial Phi: write noise into only the unmapped indices, layering
                # it on top of any grounded mapped dimensions already present as a
                # list of (index, value) writes.
                existing = new_interventions.get(var_name)
                if isinstance(existing, list):
                    merged = list(existing)
                elif existing is None:
                    merged = []
                else:
                    merged = [(None, existing)]
                merged.append((index, var_noise))
                new_interventions[var_name] = merged

        logger.debug("Injected noise into Phi variables: %s", new_interventions)
        return new_interventions

    def _step_low_level_model_execute(self, low_level_model_interventions: Dict[str, Any],
                          rng: Optional[np.random.Generator] = None) -> SystemState:
        """Execute the low-level model M under the grounded interventions."""
        base_state = SystemState()
        result = self.low_level_model.forward_with_interventions(base_state, low_level_model_interventions)
        # Normalize all array outputs to (batch, *features)
        normalized = {}
        for k, v in result.values.items():
            if isinstance(v, np.ndarray) and v.ndim == 1:
                normalized[k] = ensure_batch_2d(v)
            else:
                normalized[k] = v
        return SystemState(values=normalized)

    def _step_abstract(self, low_level_model_state: SystemState,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """
        Apply the value map tau (abstraction): map micro-state to abstract labels.

        Raises:
            ValueError: In strict_mode, if micro-variables cannot be retrieved, a
                value abstracts to UNMAPPED, or a value has an unexpected type.
        """
        results = {}
        for var_name in self.high_level_model.variables:
            selectors = self.cg.get_micro_vars(var_name)
            if not selectors: continue

            gathered_parts = []
            valid = True

            for sel in selectors:
                if sel.variable not in low_level_model_state.values:
                    valid = False
                    break

                full_val = low_level_model_state[sel.variable]

                if sel.index is not None:
                    try:
                        # Slice logic preserving dims
                        if isinstance(sel.index, int):
                            sl = (slice(None), slice(sel.index, sel.index + 1))
                        else:
                            sl = (slice(None), sel.index)
                        part = full_val[sl]
                        gathered_parts.append(part)
                    except (IndexError, TypeError):
                        valid = False
                        break
                else:
                    gathered_parts.append(full_val)

            if not valid:
                if self.config.strict_mode:
                    raise ValueError(
                        f"Strict mode: Failed to retrieve micro-variables for abstract variable '{var_name}'.")
                results[var_name] = UNMAPPED
                continue

            if len(gathered_parts) == 1:
                val = gathered_parts[0]
            else:
                val = np.concatenate(gathered_parts, axis=-1)

            # Abstract
            try:
                if isinstance(val, EmpiricalDistribution):
                    labels = [self.vm.abstract(var_name, s) for s in val.samples]
                    results[var_name] = EmpiricalDistribution(np.array(labels))
                elif isinstance(val, np.ndarray):
                    batch_size = val.shape[0]
                    labels = [self.vm.abstract(var_name, val[i]) for i in range(batch_size)]

                    if any(l is UNMAPPED for l in labels) and self.config.strict_mode:
                        raise ValueError(f"Strict mode: Variable '{var_name}' resulted in UNMAPPED during abstraction.")

                    results[var_name] = labels
                else:
                    if self.config.strict_mode:
                        raise ValueError(f"Strict Mode: Unknown value type for '{var_name}': {type(val)}")
                    results[var_name] = UNMAPPED
            except Exception as e:
                if self.config.strict_mode:
                    raise e
                logger.warning(
                    "Abstraction failed for '%s': %s. Value type=%s, shape=%s. Setting to UNMAPPED.",
                    var_name, e, type(val).__name__, getattr(val, 'shape', 'N/A'),
                )
                results[var_name] = UNMAPPED

        return results

    def _coerce_label(self, val):
        """Convert numpy scalars/arrays to Python primitives for high-level model equations."""
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            return val.tolist()  # lists are safer than arrays for user equations
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        return val

    def _step_high_level_model_predict(self, intervention_spec: 'InterventionSpec',
                          rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Execute the high-level model E under the macro-interventions."""
        # Skip low-level model variables if the intervention spec contains both (custom samplers),
        # and those whose labels are all None (e.g. from bottom-up samplers that only
        # provide micro_values).
        batch_size = infer_batch_size(intervention_spec)

        high_level_model_inputs = {}
        for k, v in intervention_spec.items():
            if k not in self.high_level_model.variables:
                continue
            labels = v.get('labels') if isinstance(v, dict) else v
            if isinstance(labels, (list, np.ndarray)) and all(l is None for l in labels):
                continue
            if isinstance(v, dict):
                high_level_model_inputs[k] = v['labels']
            else:
                high_level_model_inputs[k] = v

        results = {v: [] for v in self.high_level_model.variables}
        for i in range(batch_size):
            s_int = {k: self._coerce_label(v[i]) for k, v in high_level_model_inputs.items()}  # Coerce types for high-level model compat
            # A deterministic high-level model (the default) is evaluated with no
            # RNG. A stochastic model (some node carries an exogenous P_U) is given
            # the engine's seeded per-iteration rng so its noise draws stay on the
            # seeded streams and CAE numbers remain reproducible across processes.
            if self.high_level_model.is_stochastic:
                pred = self.high_level_model.predict(s_int, rng=rng)
            else:
                pred = self.high_level_model.predict(s_int)
            for k, val in pred.items():
                results[k].append(val)
        return results

    # Core path builders

    def _build_high_level_model_only(self, name: str) -> CausalPath:
        """Path p: evaluate the high-level model E under do(nu).

        Shared by CAE_up and CAE_down. The phi-dummy key ``'_phi'`` is not in
        ``high_level_model.variables``, so :meth:`_step_high_level_model_predict`
        silently ignores it (faithfulness interventions pass through harmlessly).
        """
        return CausalPath(
            name,
            [self._step_high_level_model_predict],
            exclude_intervened=True,
        )

    def _build_ground_low_level_model_abstract(self, name: str) -> CausalPath:
        """Path q: tau_Y o M o tau_X^{-1}.

        Grounds interventions to the micro-level (tau_X^{-1}), executes the
        low-level model M, then abstracts its outputs via tau_Y. Shared by CAE_up,
        CAE_down, and the faithfulness control.
        """
        return CausalPath(
            name,
            [
                self._step_ground_or_passthrough,
                self._step_low_level_model_execute,
                self._step_abstract,
            ],
            exclude_intervened=True,
        )

    # Public path factories

    # Standard causal-abstraction paths (shared by CAE_up and CAE_down).
    # p = E o tau_X  (predict E on abstracted inputs)
    # q = tau_Y o M  (abstract the low-level model's outputs)
    # Sampler choice (bottom-up vs top-down) selects CAE_up vs CAE_down.
    def build_path_standard_high_level_model(self) -> CausalPath:
        """Return path p: the high-level model E (shared by CAE_up and CAE_down)."""
        return self._build_high_level_model_only("CAE_high_level_model")

    def build_path_standard_low_level_model(self) -> CausalPath:
        """Return path q: ground -> low-level model M -> abstract."""
        return self._build_ground_low_level_model_abstract("CAE_low_level_model")
    def build_path_combined_low_level_model(self) -> CausalPath:
        """
        Combines the low-level model path with optional phi-noise injection.
        """
        from .sampling import PHI_DUMMY_NAME
        from .primitives import infer_batch_size
        builder = self

        def _step_combined_ground(spec, rng=None):
            clean_spec = {k: v for k, v in spec.items() if k != PHI_DUMMY_NAME}
            low_level_model_ints = builder._step_ground_or_passthrough(clean_spec, rng=rng)
            if PHI_DUMMY_NAME in spec:
                bs = infer_batch_size(spec)
                low_level_model_ints = builder._step_inject_phi_noise(low_level_model_ints, rng=rng, batch_size=bs)
            return low_level_model_ints

        return CausalPath(
            "Combined_CAE_low_level_model",
            [_step_combined_ground, self._step_low_level_model_execute, self._step_abstract],
            exclude_intervened=True,
        )
