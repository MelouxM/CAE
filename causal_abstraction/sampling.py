"""Sampling strategies for generating interventions."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .primitives import AbstractVariable, InterventionSpec
from .valuemap import ValueMap


class InterventionSampler(ABC):
    """Base interface for sampling interventions over abstract variables.

    Concrete samplers (:class:`TopDownSampler`, :class:`BottomUpSampler`, and the
    wrappers below) implement :meth:`sample_intervention`; the choice of sampler is
    what distinguishes CAE_down (top-down) from CAE_up (bottom-up).
    """

    def __init__(self, value_map: ValueMap):
        self.value_map = value_map

    @abstractmethod
    def sample_intervention(self,
                            variables: List[AbstractVariable],
                            batch_size: int = 1,
                            max_interventions: Optional[int] = None,
                            force_all: bool = False,
                            rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Sample an intervention spec over a random subset of ``variables``.

        Args:
            variables: Candidate abstract variables to intervene on.
            batch_size: Number of samples to draw per intervened variable.
            max_interventions: Cap on how many variables are intervened on at once.
            force_all: If True, intervene on every variable rather than a subset.
            rng: Optional generator for reproducible sampling.

        Returns:
            A spec mapping each intervened variable name to
            ``{'labels': [...], 'micro_values': ndarray | None}``.
        """
        pass

    def _select_targets(self,
                        variables: List[AbstractVariable],
                        max_interventions: Optional[int],
                        force_all: bool,
                        rng: np.random.Generator) -> List[AbstractVariable]:
        """Helper to randomly select a subset of variables to intervene on."""
        if not variables: return []
        if force_all: return variables

        limit = len(variables)
        if max_interventions is not None:
            limit = min(limit, max_interventions)

        if limit == 0: return []

        k = rng.integers(1, limit + 1)
        return list(rng.choice(variables, size=k, replace=False))

    def _get_rng(self, rng: Optional[np.random.Generator]) -> np.random.Generator:
        return rng if rng is not None else np.random.default_rng()


class TopDownSampler(InterventionSampler):
    """Samples macro-interventions from the high-level model's domain (CAE_down)."""

    def sample_intervention(self,
                            variables: List[AbstractVariable],
                            batch_size: int = 1,
                            max_interventions: Optional[int] = None,
                            force_all: bool = False,
                            rng: Optional[np.random.Generator] = None) -> 'InterventionSpec':
        """Draw labels from each variable's distribution/domain; defer grounding to the path."""
        rng = self._get_rng(rng)
        targets = self._select_targets(variables, max_interventions, force_all, rng)
        intervention_spec = {}

        for var in targets:
            labels = []

            # Discrete distribution
            if var.distribution:
                labels_pool = list(var.distribution.keys())
                probs = list(var.distribution.values())
                labels = rng.choice(labels_pool, size=batch_size, p=probs)

            # Continuous domain (abstract)
            elif var.domain is not None:
                samples = var.domain.sample((batch_size, var.domain.dim), rng=rng)
                for i in range(batch_size):
                    val = samples[i]
                    labels.append(val.item() if val.size == 1 else val)

            # Fallback: continuous physical geometry
            elif var.name in self.value_map.specs:
                first_subspace = next(iter(self.value_map.specs[var.name].values()))
                samples = first_subspace.sample((batch_size, first_subspace.dim), rng=rng)

                for i in range(batch_size):
                    labels.append(samples[i])

            else:
                continue

            intervention_spec[var.name] = {
                'labels': labels,
                'micro_values': None  # Defer to path step
            }

        return intervention_spec


class BottomUpSampler(InterventionSampler):
    """Samples micro-interventions from the low-level model's physical space (CAE_up)."""

    def sample_intervention(self,
                            variables: List[AbstractVariable],
                            batch_size: int = 1,
                            max_interventions: Optional[int] = None,
                            force_all: bool = False,
                            rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Sample a micro-value per variable from a random label subspace, then abstract it."""
        rng = self._get_rng(rng)
        targets = self._select_targets(variables, max_interventions, force_all, rng)
        intervention_spec = {}

        for var in targets:
            if var.name not in self.value_map.specs:
                continue

            specs = self.value_map.specs[var.name]
            all_subspaces = list(specs.values())
            # Randomly pick a subspace to sample from
            indices = rng.choice(len(all_subspaces), size=batch_size)

            micro_values_list = []
            labels = []

            for i in indices:
                subspace = all_subspaces[i]
                val = subspace.sample((1, subspace.dim), rng=rng).squeeze(0)
                micro_values_list.append(val)
                labels.append(self.value_map.abstract(var.name, val))

            micro_tensor = np.stack(micro_values_list, axis=0)
            intervention_spec[var.name] = {
                'labels': labels,
                'micro_values': micro_tensor
            }

        return intervention_spec


class NoisyMeasurementSampler(InterventionSampler):
    """
    Wraps any InterventionSampler and corrupts the sampled micro-values with
    additive noise before returning them.

    This models measurement noise at the sampling stage, as opposed to
    NoisyLowLevelModel which adds noise to the low-level model output.

    Args:
        base_sampler: The sampler to wrap.
        noise_std: Standard deviation of the noise.
        noise_vars: Abstract variable names to corrupt. None = all sampled variables.
        noise_type: ``'gaussian'`` adds N(0, noise_std²) noise; ``'relative'`` adds
            N(0, (noise_std · |x|)²) noise.
    """

    def __init__(
        self,
        base_sampler: InterventionSampler,
        noise_std: float,
        noise_vars: Optional[List[str]] = None,
        noise_type: str = 'gaussian',
    ):
        super().__init__(base_sampler.value_map)
        self.base_sampler = base_sampler
        self.noise_std = noise_std
        self.noise_vars = noise_vars
        self.noise_type = noise_type

    def sample_intervention(
        self,
        variables: List[AbstractVariable],
        batch_size: int = 1,
        max_interventions: Optional[int] = None,
        force_all: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        rng = self._get_rng(rng)
        spec = self.base_sampler.sample_intervention(variables, batch_size, max_interventions, force_all, rng)
        for var_name, entry in spec.items():
            if self.noise_vars is not None and var_name not in self.noise_vars:
                continue
            mv = entry.get('micro_values')
            if mv is None:
                continue
            mv = mv.astype(float)
            if self.noise_type == 'gaussian':
                mv = mv + rng.normal(0.0, self.noise_std, mv.shape)
            elif self.noise_type == 'relative':
                mv = mv + rng.normal(0.0, self.noise_std * np.abs(mv) + 1e-12)
            entry['micro_values'] = mv
        return spec


class PairedSampler(InterventionSampler):
    """
    Samples two independent intervention specs from the same abstract cells.

    Used by IIA (interchange intervention accuracy), BCC, and DCC metrics.
    Produces a pair of specs under the keys ``'_pair_a'`` and ``'_pair_b'``
    together with a ``'_is_paired': True`` flag.

    The IIA/BCC path steps in paths.py know how to unpack this format.

    Args:
        base_sampler: The underlying sampler for each half of the pair.
    """

    def __init__(self, base_sampler: InterventionSampler):
        super().__init__(base_sampler.value_map)
        self.base_sampler = base_sampler

    def sample_intervention(
        self,
        variables: List[AbstractVariable],
        batch_size: int = 1,
        max_interventions: Optional[int] = None,
        force_all: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        rng = self._get_rng(rng)
        # Use distinct sub-streams for the two halves so they are independent.
        ss = np.random.SeedSequence(int(rng.integers(2**32)))
        rng_a, rng_b = (np.random.default_rng(s) for s in ss.spawn(2))

        spec_a = self.base_sampler.sample_intervention(variables, batch_size, max_interventions, force_all, rng_a)
        spec_b = self.base_sampler.sample_intervention(variables, batch_size, max_interventions, force_all, rng_b)
        return {'_pair_a': spec_a, '_pair_b': spec_b, '_is_paired': True}


# Sentinel name for the phi dummy variable used by CombinedFaithfulnessSampler
PHI_DUMMY_NAME = '_phi'


class CombinedFaithfulnessSampler(InterventionSampler):
    """
    Extends any base sampler by adding a phi-dummy target to the intervention pool.

    When the phi dummy is selected as one of the intervention targets, a
    ``'_phi'`` key is added to the returned spec. The combined low-level model path
    (see :meth:`DiagramBuilder.build_path_combined_low_level_model`) detects this key
    and injects phi-variable noise, effectively merging the standard CAE_up/CAE_down
    check with an inline faithfulness test.

    The phi sentinel's inclusion probability is controlled by ``phi_prob``:

    * ``phi_prob=None`` (default) uses pool-based selection: for each
      iteration, :func:`_select_targets` randomly draws up to ``max_interventions``
      variables from the pool of ``len(variables) + 1`` candidates (the +1 being
      the phi dummy), so phi's inclusion probability is implicit (it depends on the
      pool size and ``max_interventions``).
    * ``phi_prob`` set to a float in ``[0, 1]`` selects the real intervention
      targets from ``variables`` as usual and then includes the phi sentinel
      independently with exactly that probability (a Bernoulli draw). This exposes
      the "configurable probability" described in the paper and is wired through
      :attr:`~causal_abstraction.config.EvaluationConfig.phi_selection_prob`.

    The caller sees a mix of pure CAE_up/CAE_down interventions and interventions
    that also inject phi noise.

    Args:
        base_sampler: The underlying sampler that handles real high-level model variables.
        phi_prob: Explicit inclusion probability for the phi sentinel (see above).
            ``None`` keeps the implicit pool-based behavior.

    Note:
        Background sampling (``force_all=True``) bypasses the phi dummy so that
        context variables are sampled normally.
    """

    def __init__(self, base_sampler: InterventionSampler, phi_prob: Optional[float] = None):
        super().__init__(base_sampler.value_map)
        self.base_sampler = base_sampler
        self.phi_prob = phi_prob

    def sample_intervention(
        self,
        variables: List[AbstractVariable],
        batch_size: int = 1,
        max_interventions: Optional[int] = None,
        force_all: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        rng = self._get_rng(rng)

        # Background / context sampling, do not inject phi
        if force_all:
            return self.base_sampler.sample_intervention(variables, batch_size, max_interventions, force_all, rng)

        if self.phi_prob is not None:
            # Explicit configurable inclusion probability for the phi sentinel:
            # draw the real targets normally, then add phi with prob phi_prob.
            real_targets = self._select_targets(list(variables), max_interventions, force_all, rng)
            phi_selected = bool(rng.random() < self.phi_prob)
        else:
            # Pool-based selection: phi is one extra candidate in a pool of len(variables)+1.
            phi_var = AbstractVariable(name=PHI_DUMMY_NAME, distribution={True: 1.0})
            extended = list(variables) + [phi_var]
            targets = self._select_targets(extended, max_interventions, force_all, rng)
            real_targets = [v for v in targets if v.name != PHI_DUMMY_NAME]
            phi_selected = len(real_targets) < len(targets)

        # Sample real interventions from the base sampler one variable at a time
        spec: Dict[str, Any] = {}
        for v in real_targets:
            single = self.base_sampler.sample_intervention(
                [v], batch_size, max_interventions=None, force_all=True, rng=rng
            )
            spec.update(single)

        if phi_selected:
            spec[PHI_DUMMY_NAME] = {'labels': [True] * batch_size, 'micro_values': None}

        return spec