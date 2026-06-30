"""Defines the high-level model E (an SCM) as a graph of abstract variables."""

import inspect
import logging
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Callable, Union

import numpy as np

from ..primitives import AbstractVariable, ProbabilityDistribution, UNMAPPED

logger = logging.getLogger(__name__)


def _equation_accepts_u(equation: Callable) -> bool:
    """Whether a structural equation can receive the exogenous-noise keyword ``u``.

    True if the callable declares a ``u`` parameter or accepts arbitrary keywords
    (``**kwargs``). Callables whose signature cannot be introspected (e.g. some C
    builtins) are assumed compatible, leaving responsibility with the caller.
    """
    try:
        sig = inspect.signature(equation)
    except (TypeError, ValueError):
        return True
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == "u" and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


class CausalGraph:
    """
    Represents a high-level SCM defined by a graph of variables and their structural equations.
    """

    def __init__(self, variables: Optional[List[AbstractVariable]] = None):
        """
        Args:
            variables: Optional list of pre-defined AbstractVariables.
        """
        self.variables: Dict[str, AbstractVariable] = {}
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        self._sorted_nodes: Optional[List[str]] = None

        if variables:
            for v in variables:
                self.variables[v.name] = v

    def add_variable(self,
                     variable: Union[str, AbstractVariable],
                     equation: Callable,
                     parents: Optional[List[str]] = None,
                     distribution: Optional[Dict[Any, float]] = None,
                     domain: Any = None,
                     noise_dist: Optional[ProbabilityDistribution] = None):
        """
        Adds a variable node to the causal graph.

        Args:
            variable: A string name or an AbstractVariable object.
            equation: A function taking parent values as kwargs and returning the node value.
            parents: List of parent variable names.
            distribution: (Optional) If 'variable' is a string, defines its discrete probability.
            domain: (Optional) If 'variable' is a string, defines its subspace/domain.
            noise_dist: (Optional) Exogenous noise distribution ``P_{U_V}`` for this
                node, making the high-level model a full SCM. ``None`` (the default)
                keeps the node deterministic. When given, the node is stochastic: ``predict`` draws a
                noise value and passes it to ``equation`` via the ``u`` keyword, so
                ``equation`` must accept a ``u`` parameter (or ``**kwargs``). A
                point-mass ``noise_dist`` reproduces deterministic behavior.

        Raises:
            ValueError: If ``variable`` is neither a string nor an AbstractVariable,
                or if ``noise_dist`` is given but ``equation`` cannot receive ``u``.
        """
        if parents is None:
            parents = []

        # Resolve or create the AbstractVariable
        if isinstance(variable, AbstractVariable):
            name = variable.name
            self.variables[name] = variable
        elif isinstance(variable, str):
            name = variable
            # Create on the fly
            if name not in self.variables:
                self.variables[name] = AbstractVariable(name, distribution=distribution, domain=domain)
            else:
                # Update existing if specifically requested
                if distribution:
                    self.variables[name].distribution = distribution
                if domain:
                    self.variables[name].domain = domain
        else:
            raise ValueError("variable must be a string or AbstractVariable")

        # Opt-in exogenous noise: a stochastic node's equation must be able to
        # receive the noise draw via ``u`` (or **kwargs). Fail loud at construction
        # rather than silently corrupting results at predict() time.
        if noise_dist is not None and not _equation_accepts_u(equation):
            raise ValueError(
                f"Stochastic high-level node '{name}' was given a noise_dist, but its "
                f"equation does not accept a 'u' keyword argument. Add a 'u' parameter "
                f"(or **kwargs) so it can consume the exogenous noise."
            )

        # Register causal structure
        self._sorted_nodes = None  # Invalidate sort cache
        self._nodes[name] = {'parents': parents, 'equation': equation, 'noise_dist': noise_dist}

        # Update adjacency for topological sort (parent -> children)
        for p in parents:
            if p not in self._nodes and p not in self.variables:
                import warnings
                warnings.warn(f"Parent '{p}' of '{name}' is not yet registered. "
                              "Ensure it is added before calling predict().")
            if p not in self._adjacency:
                self._adjacency[p] = []
            self._adjacency[p].append(name)

        # Ensure self exists in adjacency keys
        if name not in self._adjacency:
            self._adjacency[name] = []

    def get_roots(self) -> List[AbstractVariable]:
        """Return the root variables (those with no registered parents)."""
        return [
        v for name, v in self.variables.items()
        if not self._nodes.get(name, {}).get('parents')
    ]

    @property
    def is_stochastic(self) -> bool:
        """True if any node carries an exogenous noise distribution ``P_U``.

        When False (the default), ``predict`` runs deterministically with no RNG use.
        """
        return any(n.get('noise_dist') is not None for n in self._nodes.values())

    def _topological_sort(self) -> List[str]:
        """Performs a topological sort to determine execution order."""
        if self._sorted_nodes:
            return self._sorted_nodes

        # Compute in-degree
        in_degree = {name: 0 for name in self._nodes}

        for name, props in self._nodes.items():
            for p in props['parents']:
                if p in self._nodes:
                    in_degree[name] += 1

        # Initialize queue with roots (in-degree 0)
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        sorted_order = []

        # Kahn's algorithm
        while queue:
            u = queue.popleft()
            sorted_order.append(u)

            if u in self._adjacency:
                for v in self._adjacency[u]:
                    if v in in_degree:
                        in_degree[v] -= 1
                        if in_degree[v] == 0:
                            queue.append(v)

        if len(sorted_order) != len(self._nodes):
            remaining = {n for n, d in in_degree.items() if d > 0}
            raise ValueError(f"Causal graph contains a cycle or unresolved dependencies: {remaining}")

        # Check for references to unregistered nodes
        all_parents = {p for props in self._nodes.values() for p in props['parents']}
        unregistered = all_parents - set(self._nodes.keys())
        if unregistered:
            raise ValueError(f"Parents referenced but never registered as nodes: {unregistered}")

        self._sorted_nodes = sorted_order
        return self._sorted_nodes

    def predict(self, interventions: Dict[str, Any], *,
                u: Optional[Dict[str, Any]] = None,
                rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """
        Execute the causal graph under a set of interventions (clamped values).

        Args:
            interventions: Mapping of variable name to clamped value.
            u: (Optional) Per-node exogenous-noise realizations, keyed by variable
                name, for stochastic nodes (those added with a ``noise_dist``). A
                stochastic node absent from this mapping draws from its own
                ``noise_dist``. Ignored for deterministic nodes. Keyword-only.
            rng: (Optional) Generator used for the exogenous-noise draws of stochastic
                nodes. Unused when the model is deterministic. Keyword-only.

        Returns:
            A mapping of every variable name to its computed value; variables whose
            parents are missing or failed are set to UNMAPPED.

        Raises:
            ValueError: If the graph contains a cycle/unresolved dependencies or
                references a parent that was never registered as a node.
        """
        execution_order = self._topological_sort()
        results = dict(interventions)

        for var_name in execution_order:
            if var_name in results:
                continue

            node_info = self._nodes.get(var_name)
            if not node_info or node_info['equation'] is None:
                continue

            parents = node_info['parents']

            # Check if parents are present
            if not all(p in results for p in parents):
                # Typically happens if a root was not provided
                results[var_name] = UNMAPPED
                continue

            parent_values = {p: results[p] for p in parents}

            # Propagate failure (None arises when a root's placeholder lambda: None is called)
            if any(v is UNMAPPED or v is None for v in parent_values.values()):
                results[var_name] = UNMAPPED
                continue

            noise_dist = node_info.get('noise_dist')
            try:
                if noise_dist is None:
                    # Deterministic node: equation called with parent kwargs only.
                    results[var_name] = node_info['equation'](**parent_values)
                else:
                    # Stochastic node: draw (or read) the exogenous noise U_V and pass
                    # it to the mechanism via the keyword ``u``. The draw uses the
                    # seeded, threaded rng so CAE numbers stay reproducible.
                    if u is not None and var_name in u:
                        u_v = u[var_name]
                    else:
                        u_v = noise_dist.sample(1, rng)[0]
                    results[var_name] = node_info['equation'](**parent_values, u=u_v)
            except Exception as e:
                logger.error(f"Error executing equation for '{var_name}': {e}")
                results[var_name] = UNMAPPED

        return results