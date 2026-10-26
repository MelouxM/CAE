"""Structural schema of the low-level model and the coarse-graining topology."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Union, Tuple, Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class Variable:
    """
    Describes a single micro-level variable.

    Attributes:
        name:  Unique identifier used throughout the framework.
        shape: Shape of the variable's value tensor, excluding the batch
               dimension.  Defaults to `(1,)` for scalars.
        dtype: Python type hint used for noise injection and type-specific
               logic.  Defaults to `float`.

    Examples:
        Variable("x")                    # scalar float
        Variable("layer1", shape=(64,))  # 64-dim float vector
        Variable("gate_a", dtype=bool)   # scalar boolean
    """
    name: str
    shape: Tuple[int, ...] = (1,)
    dtype: type = float


@dataclass(frozen=True)
class MicroSelector:
    """
    Points to a specific part of a micro-variable.

    Examples:
        variable="layer1", index=2  -> layer1[2]
        variable="layer1", index=None -> layer1[:]
    """
    variable: str
    index: Optional[Union[int, slice, Tuple[int, ...]]] = None

    def __repr__(self):
        if self.index is None:
            return self.variable
        return f"{self.variable}[{self.index}]"


class MicroVariableSchema:
    """
    Defines the physical layout of the low-level model.

    Accepts a list of `Variable` objects.  Variables without an
    explicit shape default to `(1,)`.

    Example::

        schema = MicroVariableSchema([
            Variable("input", shape=(2,)),
            Variable("layer1", shape=(64,)),
            Variable("gate_a", dtype=bool),   # shape defaults to (1,)
        ])
    """

    _DEFAULT_SHAPE: Tuple[int, ...] = (1,)

    def __init__(self, variables: List[Variable]):
        self._variables: Dict[str, Variable] = {v.name: v for v in variables}
        self.variable_names: Set[str] = set(self._variables.keys())

    @staticmethod
    def from_names(names: Iterable[str]):
        """Build a schema of scalar variables (default shape `(1,)`, dtype `float`)."""
        return MicroVariableSchema([Variable(x) for x in names])

    def get_dtype(self, variable: str) -> type:
        """Return the variable's dtype, or `float` if it is not in the schema."""
        v = self._variables.get(variable)
        return v.dtype if v is not None else float

    def get_shape(self, variable: str) -> Tuple[int, ...]:
        """Return the variable's shape, or `(1,)` if it is not in the schema."""
        v = self._variables.get(variable)
        return v.shape if v is not None else (1,)

    def resolve_dim(self, variable: str) -> int:
        """Returns the size of the first dimension (used for slice resolution)."""
        return self.get_shape(variable)[0]

    def validate(self, selectors: List[MicroSelector]) -> bool:
        """Checks if the base variables referenced in selectors exist in the schema."""
        return all(sel.variable in self.variable_names for sel in selectors)


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Union of [start, stop) intervals, as sorted disjoint intervals"""
    merged: List[Tuple[int, int]] = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


class CoarseGrainingMap:
    """The coarse-graining map a: assigns micro-variables to macro-variables (cells) or to Phi.

    Maps abstract variable names to lists of `MicroSelector` (their cells). Cells may
    overlap, i.e. several macro-variables may read the same micro-variables; the
    overlapping pairs are listed in `overlaps`. Micro-resources used by no cell form
    the unmapped set Phi (`phi_variables`): a fully-unmapped variable is held by name,
    while the unmapped dimensions of a partially-mapped variable are held as
    per-dimension `MicroSelector` entries. `internal_variables` are micro-variables
    that are part of the mechanism but are not intervened on.
    """

    def __init__(self,
                 schema: MicroVariableSchema,
                 mapping: Dict[str, List[Union[str, Tuple[str, Any]]]],
                 internal_variables: Optional[List[str]] = None,
                 allow_overlap: bool = True):
        """
        Args:
            schema: The definition of the low-level model.
            mapping: Keys are abstract names. Values are lists of:
                     - Strings: "layer1" (implies full vector)
                     - Tuples: ("layer1", 2) or ("layer1", slice(0,2))
            internal_variables: Variable names that are part of the mechanism but
                should not be intervened on.
            allow_overlap: If False, cells must be disjoint.

        Raises:
            ValueError: If a selector is malformed, references an unknown variable,
                overlaps another cell while `allow_overlap` is False, or lists an
                internal variable absent from the schema.
            NotImplementedError: If a selector uses an index type other than int or slice.
        """
        self.schema = schema
        self._fwd_map: Dict[str, List[MicroSelector]] = {}
        self._inv_map: Dict[MicroSelector, List[str]] = {}
        self.phi_variables: Set[Union[str, MicroSelector]] = set()
        self.internal_variables = set(internal_variables) if internal_variables else set()
        # {frozenset({X, Y}): set of shared micro-variable names}
        self.overlaps: Dict[frozenset, Set[str]] = {}

        # Maps variable name -> sorted list of (start, stop, abstract name) intervals
        used_micro_resources: Dict[str, List[Tuple[int, int, str]]] = {}

        for abstract_name, raw_selectors in mapping.items():
            selectors = []

            # Parse input
            for item in raw_selectors:
                if isinstance(item, str):
                    sel = MicroSelector(item, None)
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    sel = MicroSelector(item[0], item[1])
                else:
                    raise ValueError(f"Invalid selector format: {item}")
                selectors.append(sel)

            # Validation
            if not self.schema.validate(selectors):
                raise ValueError(f"Abstract var '{abstract_name}' maps to unknown variables.")

            for sel in selectors:
                var = sel.variable
                var_dim = self.schema.resolve_dim(var)

                # Determine the range [start, stop) requested
                if sel.index is None:
                    req_start, req_stop = 0, var_dim
                elif isinstance(sel.index, int):
                    idx = sel.index if sel.index >= 0 else var_dim + sel.index
                    req_start, req_stop = idx, idx + 1
                elif isinstance(sel.index, slice):
                    req_start, req_stop, _ = sel.index.indices(var_dim)
                else:
                    raise NotImplementedError("Only int and slice indices supported for overlap check.")

                existing = used_micro_resources.setdefault(var, [])
                for occ_start, occ_stop, occ_name in existing:
                    if max(req_start, occ_start) < min(req_stop, occ_stop):
                        if not allow_overlap or occ_name == abstract_name:
                            raise ValueError(
                                f"Overlap detected in '{var}': Requested [{req_start}:{req_stop}] "
                                f"conflicts with existing [{occ_start}:{occ_stop}]"
                            )
                        self.overlaps.setdefault(frozenset((abstract_name, occ_name)), set()).add(var)
                existing.append((req_start, req_stop, abstract_name))

            # Build forward and inverse maps
            self._fwd_map[abstract_name] = selectors
            for sel in selectors:
                self._inv_map.setdefault(sel, []).append(abstract_name)

        if self.overlaps:
            logger.info("Overlapping cells: %s",
                        {"/".join(sorted(k)): sorted(v) for k, v in self.overlaps.items()})

        # Identify phi (unmapped) variables
        all_micro = self.schema.variable_names
        fully_mapped = set()
        partially_mapped = set()
        phi_partial_selectors: Set[MicroSelector] = set()

        for var in all_micro:
            if var not in used_micro_resources:
                continue

            var_dim = self.schema.resolve_dim(var)
            intervals = _merge_intervals([(a, b) for a, b, _ in used_micro_resources[var]])
            covered = sum(stop - start for start, stop in intervals)

            if covered >= var_dim:
                fully_mapped.add(var)
            elif covered > 0:
                partially_mapped.add(var)
                # Unmapped dimensions of a partially-mapped variable are treated
                # as Phi: one int selector per unmapped index, kept hashable and
                # matching the (index, value) intervention-write format
                mapped_idx = set()
                for start, stop in intervals:
                    mapped_idx.update(range(start, stop))
                for i in range(var_dim):
                    if i not in mapped_idx:
                        phi_partial_selectors.add(MicroSelector(var, i))
                logger.warning(
                    f"Variable '{var}' is only partially mapped ({covered}/{var_dim} dims). "
                    f"Unmapped dimensions will be treated as phi variables."
                )

        # Phi = fully-unmapped micro-variables (by name) plus the unmapped index
        # ranges of partially-mapped variables (as MicroSelectors), excluding any
        # variables declared internal.
        whole_var_phi = all_micro - fully_mapped - partially_mapped - self.internal_variables
        self.phi_variables = set(whole_var_phi) | phi_partial_selectors

        # Validate internal_variables exist in schema
        if not self.internal_variables.issubset(all_micro):
            invalid = self.internal_variables - all_micro
            raise ValueError(f"Internal variables not in schema: {invalid}")

    def get_micro_vars(self, abstract_name: str) -> List[MicroSelector]:
        """Return the selectors mapped to `abstract_name` (empty list if unknown)."""
        return self._fwd_map.get(abstract_name, [])

    def conflicts(self, abstract_names: Iterable[str]) -> List[Tuple[str, str]]:
        """Pairs among `abstract_names` whose cells overlap (sorted, empty if none)."""
        if not self.overlaps:
            return []
        names = sorted(set(abstract_names))
        return [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
                if frozenset((x, y)) in self.overlaps]