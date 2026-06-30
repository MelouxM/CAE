"""
The value map (tau) and common variants, mapping between
micro-values (physical) and abstract labels (logical).
"""

import logging
from typing import Dict, Any, Optional, Callable, Union

import numpy as np

from .primitives import StateLabel, UNMAPPED
from .schema import CoarseGrainingMap
from .spaces.base import Subspace

logger = logging.getLogger(__name__)


class ValueMap:
    """The value map tau in constructive causal abstraction.

    Per macro-variable X, tau maps between micro-states and abstract labels:
    :meth:`abstract` (micro-values -> label) and :meth:`ground` (label ->
    micro-values), the latter being the inverse value map sampled from the
    label's subspace.
    """

    def __init__(self,
                 cg_map: CoarseGrainingMap,
                 variable_specs: Dict[str, Dict[StateLabel, Subspace]],
                 transforms: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None):
        """
        Args:
            cg_map: The structural coarse-graining definition (a).
            variable_specs: Maps each variable name to a ``{label: Subspace}`` dict.
            transforms: Optional functions to pre-process micro-values before abstraction.
        """
        self.cg_map = cg_map
        self.specs = variable_specs
        self.transforms = transforms or {}

    def _get_rng(self, rng: Optional[np.random.Generator]) -> np.random.Generator:
        return rng if rng is not None else np.random.default_rng()

    def abstract(self, abstract_var_name: str, micro_values: np.ndarray) -> Any:
        """Map micro-values to the abstract label of the subspace that contains them.

        Args:
            abstract_var_name: The macro-variable whose value map to apply.
            micro_values: The micro-state to abstract.

        Returns:
            The matching ``StateLabel``, or ``UNMAPPED`` if no subspace contains
            ``micro_values``.

        Raises:
            ValueError: If no value spec is defined for ``abstract_var_name``.
        """
        # Instance-level override hook
        if hasattr(self, '_custom_abstract'):
            return self._custom_abstract(micro_values)

        if abstract_var_name not in self.specs:
            raise ValueError(f"No value spec defined for '{abstract_var_name}'")

        if abstract_var_name in self.transforms:
            micro_values = self.transforms[abstract_var_name](micro_values)

        subspaces = self.specs[abstract_var_name]
        for label, subspace in subspaces.items():
            if np.all(subspace.contains(micro_values)):
                return label

        return UNMAPPED

    def ground(self, abstract_var_name: str, label: StateLabel,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample a micro-value realizing ``label`` (the inverse value map).

        Args:
            abstract_var_name: The macro-variable whose value map to invert.
            label: The abstract label to ground.
            rng: Optional generator for reproducible sampling from the subspace.

        Returns:
            A micro-value sampled from the subspace associated with ``label``.

        Raises:
            ValueError: If no value spec is defined for ``abstract_var_name``, or
                ``label`` is not one of its defined labels.
        """
        if hasattr(self, '_custom_ground'):
            return self._custom_ground(label)

        if abstract_var_name not in self.specs:
            raise ValueError(f"No value spec defined for '{abstract_var_name}'")

        subspaces = self.specs[abstract_var_name]
        if label not in subspaces:
            raise ValueError(f"Label '{label}' not defined for variable '{abstract_var_name}'")

        subspace = subspaces[label]
        sample = subspace.sample((1, subspace.dim), rng=rng)
        return sample.squeeze(0)

    def validate_surjectivity(self, n_samples: int = 100, strict_discrete: bool = True,
                              rng: Optional[np.random.Generator] = None) -> bool:
        """Check that micro-samples drawn from each label's subspace abstract back to that label.

        Args:
            n_samples: Number of samples drawn per label for continuous/large spaces.
            strict_discrete: If True, check every label of a small discrete map
                instead of sampling a subset.
            rng: Optional generator for reproducible sampling.

        Returns:
            True if every checked sample maps back to its source label.
        """
        rng = self._get_rng(rng)
        valid = True
        for var_name, subspace_map in self.specs.items():
            is_small_discrete = len(subspace_map) < 1000  # Heuristic: If the map has few keys, check them all
            labels_to_check = list(subspace_map.keys())

            if not (strict_discrete and is_small_discrete):  # Randomly sample n_samples labels
                keys = list(subspace_map.keys())
                indices = rng.choice(len(keys), size=min(len(keys), n_samples), replace=False)
                labels_to_check = [keys[i] for i in indices]

            for label in labels_to_check:
                subspace = subspace_map[label]
                try:
                    micro_val = subspace.sample((1, subspace.dim), rng=rng).squeeze(0)
                    mapped_label = self.abstract(var_name, micro_val)

                    if isinstance(label, (int, str, bool)):
                        match = (mapped_label == label)
                    else:  # Arrays/continuous
                        match = np.allclose(mapped_label, label, atol=1e-5)

                    if not match:
                        if mapped_label is UNMAPPED:
                            logger.error(f"Surjectivity fail: {var_name}[{label}] -> UNMAPPED (sample: {micro_val})")
                        else:
                            logger.error(f"Surjectivity fail: {var_name}[{label}] -> {mapped_label}")
                        valid = False

                except Exception as e:
                    logger.error(f"Surjectivity check error for '{var_name}', state '{label}': {e}")
                    valid = False

        return valid


class ContinuousValueMap(ValueMap):
    """
    Identity-like value map for continuous systems where the abstraction
    is a direct coarse-graining (e.g. mean kinetic energy, population count).

    Labels are the actual continuous values, not bin indices.
    The specs {0: subspace} are retained only so samplers know the valid range;
    the 0 key is never used as an actual label.
    """

    def abstract(self, name: str, val: np.ndarray) -> np.ndarray:
        """Return the value itself as the label (identity abstraction).

        Args:
            name: The macro-variable name (unused; identity map).
            val: The micro-value(s) to abstract.

        Returns:
            A float for scalar input, otherwise the array of labels (shape ``(B, d)``
            for batched input).
        """
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return arr.copy()
        # batch: (B, d) - return each row as its own label
        return arr.copy()

    def ground(self, name: str, label, rng=None) -> np.ndarray:
        """Return the label as the micro-value (identity grounding).

        Subclasses override this for systems with genuine many-to-one abstraction
        (e.g. a spatial ABM where the label is a total count but the micro-state is
        a spatial grid).

        Args:
            name: The macro-variable name (unused; identity map).
            label: The abstract label to ground.
            rng: Unused; accepted for signature compatibility with the base map.

        Returns:
            ``label`` as a float array.
        """
        return np.asarray(label, dtype=float)

    def validate_surjectivity(self, n_samples: int = 100, strict_discrete: bool = True, rng=None) -> bool:
        """For continuous identity maps, surjectivity holds if every variable's domain has positive volume."""
        for name, specs in self.specs.items():
            if not specs:
                return False
            if not any(s.volume() > 0 for s in specs.values()):
                return False
        return True