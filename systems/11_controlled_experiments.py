"""
Controlled experiment systems for metric failure mode analysis.

Six experiments, each targeting a specific gap in existing metrics.

Exp 1 - Hidden Environmental Confounder (extends predator-prey)
Exp 2 - XOR-Masked Redundant Pathway (new binary system)
Exp 3 - Wrong Intermediate Representation (correct I/O, wrong mechanism)
Exp 4 - Spurious Mediator (fork-structured low-level model + chain high-level model)
Exp 5 - Unreachable Intermediate States (reachability-limited interventions)
Exp 6 - Wrong Causal Direction (chain-structured low-level model + fork high-level model)
"""

import numpy as np

from causal_abstraction import (
    LowLevelModel, SystemState, CausalGraph,
    MicroVariableSchema, CoarseGrainingMap, ValueMap, ContinuousValueMap,
    RectSubspace, FullSubspace, AbstractVariable,
)
from causal_abstraction.schema import Variable


# Precision for deterministic continuous systems.
# Eliminates IEEE-754 ULP differences between algebraically equivalent
# but arithmetically distinct high-level model equations (e.g. 1.5*(2x+1)+0.5 vs 3x+2).
_ROUND_DIGITS = 10


# Experiment 1: Hidden Environmental Confounder

class ConfounderPPModel(LowLevelModel):
    """
    Wraps any predator-prey low-level model and adds a hidden resource variable
    that multiplicatively modulates prey survival.

    At resource_level = 0 the prey output is unchanged (equilibrium).
    At resource_level != 0 the prey count is scaled by
    ``1 + resource_effect * resource_level``.

    Args:
        base_model: The underlying predator-prey ABM.
        resource_effect: Coupling strength. 0 = inert phi (faithful); >0 = leaky phi.
    """

    def __init__(self, base_model, resource_effect=0.0):
        self.base = base_model
        self.resource_effect = resource_effect

    def forward_with_interventions(self, input_state, interventions):
        clean = {k: v for k, v in interventions.items() if k != "resource_level"}

        result = self.base.forward_with_interventions(input_state, clean)

        # Resolve resource_level
        if "resource_level" in interventions:
            resource = np.atleast_2d(
                np.asarray(interventions["resource_level"], dtype=float)
            )
        else:
            resource = np.zeros((1, 1))  # equilibrium

        pops = result.values["final_populations"].copy()
        batch = pops.shape[0]
        r_flat = resource.ravel()
        for i in range(batch):
            r = r_flat[min(i, len(r_flat) - 1)]
            pops[i, 0] *= 1.0 + self.resource_effect * r

        result.values["final_populations"] = np.maximum(pops, 0.0)
        result.values["resource_level"] = resource.reshape(-1, 1)[:batch]

        return result

    def save_state(self):
        return self.base.save_state()

    def load_state(self, s):
        self.base.load_state(s)


# Experiment 2: XOR-Masked Redundant Pathway

class XORBackupLowLevelModel(LowLevelModel):
    """
    Binary system: x -> a = x, b1 = x, b2 = x, y = a OR (b1 XOR b2).

    Notation: code `a` is the paper's mediator `m` (mapped to macro M); code `b1, b2`
    are the paper's phi variables `a, b`. The causal structure is identical; only the
    symbol names differ.

    Since b1 always equals b2 under normal operation, the XOR is 0 and
    y = a = x.  But if b1 and b2 are independently perturbed (phi noise),
    the XOR may fire and force y = 1 regardless of a.

    Args:
        xor_active: If False, the XOR path is disconnected and y = a always (valid phi).
    """

    def __init__(self, xor_active=True):
        self.xor_active = xor_active

    def forward_with_interventions(self, input_state, interventions):
        x_raw = interventions.get("x", input_state.get("x", np.array([[0.0]])))
        x = np.atleast_2d(np.asarray(x_raw, dtype=float))

        a = x.copy()

        # b1, b2 default to x (same value -> XOR = 0)
        b1 = np.atleast_2d(np.asarray(
            interventions.get("b1", x.copy()), dtype=float))
        b2 = np.atleast_2d(np.asarray(
            interventions.get("b2", x.copy()), dtype=float))

        # Booleanize
        a_b = (a > 0.5).astype(float)
        b1_b = (b1 > 0.5).astype(float)
        b2_b = (b2 > 0.5).astype(float)

        if self.xor_active:
            xor_val = np.abs(b1_b - b2_b)            # XOR
            y = np.maximum(a_b, xor_val)              # OR
        else:
            y = a_b                                    # no backup path

        return SystemState(values={
            "x": x, "a": a, "b1": b1, "b2": b2, "y": y,
        })


def build_xor_schema():
    return MicroVariableSchema([
        Variable("x",  shape=(1,), dtype=bool),
        Variable("a",  shape=(1,), dtype=bool),
        Variable("b1", shape=(1,), dtype=bool),
        Variable("b2", shape=(1,), dtype=bool),
        Variable("y",  shape=(1,), dtype=bool),
    ])


def build_xor_cg(schema):
    """Maps x->X, a->M, y->Y.  b1, b2 are phi."""
    return CoarseGrainingMap(schema, {
        "X": ["x"],
        "M": ["a"],
        "Y": ["y"],
    })


def build_xor_vm(cg):
    bit = {0: RectSubspace((-0.5, 0.5)), 1: RectSubspace((0.5, 1.5))}
    return ValueMap(cg, {"X": bit, "M": bit, "Y": bit})


def build_xor_high_level_model():
    h = CausalGraph()
    d = {0: 0.5, 1: 0.5}
    h.add_variable("X", lambda: None, distribution=d)
    h.add_variable("M", lambda X: X, parents=["X"], distribution=d)
    h.add_variable("Y", lambda M: M, parents=["M"], distribution=d)
    return h


# Experiment 3: Wrong Intermediate Representation
#
# low-level model:  x -> m = 2x,  m -> z = m + 3   (so z = 2x + 3)
# high-level model valid: same equations
# high-level model wrong: x -> m = 3x,  m -> z = (2/3)m + 3   (so z = 2x + 3)
#
# Both high-level models produce identical z = 2x + 3 for any input x.
# But the intermediate m is wrong (3x vs 2x).
#
# Under IIA interchange on m:
#   high-level model counterfactual uses m_high_level_model = 3x_source, computes z = (2/3)*3x_s + 3 = 2x_s + 3
#   low-level model counterfactual uses m_low_level_model = 2x_source, computes z = 2x_s + 3
#   -> z matches! IIA passes. (The compensating equation masks the error.)
#
# Under CAE_down/CAE_up (checking m directly):
#   high-level model predicts m = 3x, low-level model gives m = 2x -> mismatch detected.
#
# All z computations are rounded to _ROUND_DIGITS to prevent IEEE-754
# ULP artifacts from creating spurious significance in the z dimension.

class IntermediateChainLowLevelModel(LowLevelModel):
    """
    Chain system with an accessible intermediate::

        x  ->  m = 2x  ->  z = m + 3   (= 2x + 3)

    Interventions on m are respected: z = m_intervened + 3.
    """

    def forward_with_interventions(self, input_state, interventions):
        x = self._resolve(interventions, input_state, "x", 5.0)
        x = np.atleast_2d(np.asarray(x, dtype=float))

        m_natural = 2.0 * x

        if "m" in interventions:
            m = np.atleast_2d(np.asarray(interventions["m"], dtype=float))
        else:
            m = m_natural

        z = np.round(m + 3.0, _ROUND_DIGITS)

        return SystemState(values={"x": x, "m": m, "z": z})

    @staticmethod
    def _resolve(interventions, state, name, default):
        if name in interventions:
            return interventions[name]
        v = state.get(name)
        return v if v is not None else np.array([[default]])


def build_intermediate_schema():
    return MicroVariableSchema([
        Variable("x", shape=(1,)),
        Variable("m", shape=(1,)),
        Variable("z", shape=(1,)),
    ])


def build_intermediate_cg(schema):
    return CoarseGrainingMap(schema, {"x": ["x"], "m": ["m"], "z": ["z"]})


def build_intermediate_vm(cg):
    x_sp = RectSubspace((0.0, 10.0))
    m_sp = RectSubspace((0.0, 30.0))   # wide enough for both 2x and 3x
    z_sp = RectSubspace((3.0, 23.0))   # z = 2x + 3, range [3, 23]
    return ContinuousValueMap(cg, {
        "x": {0: x_sp}, "m": {0: m_sp}, "z": {0: z_sp},
    })


def build_valid_intermediate_high_level_model():
    """Correct high-level model: x -> m = 2x -> z = m + 3."""
    h = CausalGraph()
    h.add_variable("x", lambda: None, domain=RectSubspace((0.0, 10.0)))
    h.add_variable("m", lambda x: 2.0 * x, parents=["x"],
                   domain=RectSubspace((0.0, 30.0)))
    h.add_variable("z", lambda m: round(m + 3.0, _ROUND_DIGITS), parents=["m"],
                   domain=RectSubspace((3.0, 23.0)))
    return h


def build_wrong_intermediate_high_level_model():
    """Wrong high-level model: x -> m = 3x (wrong!), m -> z = (2/3)m + 3 (compensates).
    Net effect: z = 2x + 3 (correct output, wrong mechanism).
    Rounding ensures z is bitwise identical to the valid high-level model's z output."""
    h = CausalGraph()
    h.add_variable("x", lambda: None, domain=RectSubspace((0.0, 10.0)))
    h.add_variable("m", lambda x: 3.0 * x, parents=["x"],
                   domain=RectSubspace((0.0, 30.0)))
    h.add_variable("z", lambda m: round((2.0 / 3.0) * m + 3.0, _ROUND_DIGITS),
                   parents=["m"], domain=RectSubspace((3.0, 23.0)))
    return h


# Experiment 4: Spurious Mediator  (fork low-level model + chain high-level model)
#
# All z equations use rounding to avoid IEEE-754 ULP artifacts
# between algebraically equivalent paths.

class ForkLowLevelModel(LowLevelModel):
    """
    Fork-structured system::

        x --> y = a*x + b
        x --> z = c*x + d

    Intervention on y does not affect z (z depends only on x).
    """

    def __init__(self, a=2.0, b=1.0, c=3.0, d=2.0):
        self.a, self.b = a, b
        self.c, self.d = c, d

    def forward_with_interventions(self, input_state, interventions):
        x = self._resolve(interventions, input_state, "x", 5.0)
        x = np.atleast_2d(np.asarray(x, dtype=float))

        y_natural = self.a * x + self.b
        z_from_x  = np.round(self.c * x + self.d, _ROUND_DIGITS)

        if "y" in interventions:
            y = np.atleast_2d(np.asarray(interventions["y"], dtype=float))
        else:
            y = y_natural

        if "z" in interventions:
            z = np.atleast_2d(np.asarray(interventions["z"], dtype=float))
        else:
            z = z_from_x  # z does not depend on y

        return SystemState(values={"x": x, "y": y, "z": z})

    @staticmethod
    def _resolve(interventions, state, name, default):
        if name in interventions:
            return interventions[name]
        v = state.get(name)
        return v if v is not None else np.array([[default]])


def build_fork_schema():
    return MicroVariableSchema([
        Variable("x", shape=(1,)),
        Variable("y", shape=(1,)),
        Variable("z", shape=(1,)),
    ])


def build_fork_cg(schema):
    return CoarseGrainingMap(schema, {"x": ["x"], "y": ["y"], "z": ["z"]})


def build_fork_vm(cg):
    x_sp = RectSubspace((0.0, 10.0))
    y_sp = RectSubspace((1.0, 21.0))
    z_sp = RectSubspace((2.0, 32.0))
    return ContinuousValueMap(cg, {
        "x": {0: x_sp}, "y": {0: y_sp}, "z": {0: z_sp},
    })


def build_fork_high_level_model():
    """Correct fork high-level model: x->y, x->z independently."""
    h = CausalGraph()
    h.add_variable("x", lambda: None, domain=RectSubspace((0.0, 10.0)))
    h.add_variable("y", lambda x: 2.0 * x + 1.0, parents=["x"],
                   domain=RectSubspace((1.0, 21.0)))
    h.add_variable("z", lambda x: round(3.0 * x + 2.0, _ROUND_DIGITS),
                   parents=["x"], domain=RectSubspace((2.0, 32.0)))
    return h


def build_chain_high_level_model():
    """Chain high-level model: x->y->z.  z = 1.5*y + 0.5 so that
    z = 1.5*(2x+1) + 0.5 = 3x + 2, matching the fork under observation.
    Uses rounding to match the ForkLowLevelModel's precision."""
    h = CausalGraph()
    h.add_variable("x", lambda: None, domain=RectSubspace((0.0, 10.0)))
    h.add_variable("y", lambda x: 2.0 * x + 1.0, parents=["x"],
                   domain=RectSubspace((1.0, 21.0)))
    h.add_variable("z", lambda y: round(1.5 * y + 0.5, _ROUND_DIGITS),
                   parents=["y"], domain=RectSubspace((2.0, 32.0)))
    return h


# Experiment 5: Unreachable Intermediate States
#
# Gap tested: Reachability-limited interventions (Gap 2).
#
# low-level model:  x in {0, 1} -> m = 2x in {0, 2} -> z = (m >= 2) ? 1 : 0
#
# M has domain {0, 1, 2, 3} but only {0, 2} are input-reachable.
#
# Valid high-level model:  M -> Z = (M >= 2) ? 1 : 0
#   M=0->Z=0, M=1->Z=0, M=2->Z=1, M=3->Z=1
#
# Wrong high-level model:  M -> Z = (M >= 1) ? 1 : 0
#   M=0->Z=0, M=1->Z=1(!!), M=2->Z=1, M=3->Z=1
#
# Both high-level models agree on M in {0, 2} (input-reachable).
# They disagree only at M=1 (unreachable from any input).
#
# IIA only swaps M values produced by inputs -> tests M in {0,2} -> passes.
# CAE_down samples M from its full domain {0,1,2,3} -> tests M=1 -> detects.
# Observational metrics: identical I/O for X in {0,1} -> score 0.

class ThresholdLowLevelModel(LowLevelModel):
    """
    Discrete threshold system::

        x in {0, 1} -> m = 2*x -> z = 1 if m >= 2 else 0

    Interventions on m are respected (z recomputed from intervened m).
    """

    def forward_with_interventions(self, input_state, interventions):
        x_raw = interventions.get("x", input_state.get("x", np.array([[0.0]])))
        x = np.atleast_2d(np.asarray(x_raw, dtype=float))

        m_natural = 2.0 * x

        if "m" in interventions:
            m = np.atleast_2d(np.asarray(interventions["m"], dtype=float))
        else:
            m = m_natural

        # Threshold at 1.5 in micro-space cleanly separates labels {0,1} from {2,3}
        z = (m >= 1.5).astype(float)

        return SystemState(values={"x": x, "m": m, "z": z})


def build_threshold_schema():
    return MicroVariableSchema([
        Variable("x", shape=(1,), dtype=int),
        Variable("m", shape=(1,), dtype=int),
        Variable("z", shape=(1,), dtype=int),
    ])


def build_threshold_cg(schema):
    return CoarseGrainingMap(schema, {"X": ["x"], "M": ["m"], "Z": ["z"]})


def build_threshold_vm(cg):
    bit = {0: RectSubspace((-0.5, 0.5)), 1: RectSubspace((0.5, 1.5))}
    m_labels = {
        0: RectSubspace((-0.5, 0.5)),
        1: RectSubspace((0.5, 1.5)),
        2: RectSubspace((1.5, 2.5)),
        3: RectSubspace((2.5, 3.5)),
    }
    return ValueMap(cg, {"X": bit, "M": m_labels, "Z": bit})


def build_valid_threshold_high_level_model():
    """Valid high-level model: Z = 1 if M >= 2, else 0."""
    h = CausalGraph()
    h.add_variable("X", lambda: None, distribution={0: 0.5, 1: 0.5})
    h.add_variable("M", lambda X: 2 * X, parents=["X"],
                   distribution={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    h.add_variable("Z", lambda M: int(M >= 2), parents=["M"],
                   distribution={0: 0.5, 1: 0.5})
    return h


def build_wrong_threshold_high_level_model():
    """Wrong high-level model: Z = 1 if M >= 1, else 0.
    Differs from valid only at M=1 (not input-reachable)."""
    h = CausalGraph()
    h.add_variable("X", lambda: None, distribution={0: 0.5, 1: 0.5})
    h.add_variable("M", lambda X: 2 * X, parents=["X"],
                   distribution={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    h.add_variable("Z", lambda M: int(M >= 1), parents=["M"],
                   distribution={0: 0.5, 1: 0.5})
    return h


# Experiment 6: Wrong Causal Direction  (chain low-level model + fork high-level model)

class ChainLowLevelModel(LowLevelModel):
    """
    Chain-structured system::

        x --> y = a*x + b
        y --> z = c*y + d

    Intervention on y does affect z (z depends on y, not directly on x).
    """

    def __init__(self, a=2.0, b=1.0, c=1.5, d=0.5):
        self.a, self.b = a, b
        self.c, self.d = c, d

    def forward_with_interventions(self, input_state, interventions):
        x = self._resolve(interventions, input_state, "x", 5.0)
        x = np.atleast_2d(np.asarray(x, dtype=float))

        y_natural = self.a * x + self.b

        if "y" in interventions:
            y = np.atleast_2d(np.asarray(interventions["y"], dtype=float))
        else:
            y = y_natural

        z_from_y = np.round(self.c * y + self.d, _ROUND_DIGITS)
        if "z" in interventions:
            z = np.atleast_2d(np.asarray(interventions["z"], dtype=float))
        else:
            z = z_from_y

        return SystemState(values={"x": x, "y": y, "z": z})

    @staticmethod
    def _resolve(interventions, state, name, default):
        if name in interventions:
            return interventions[name]
        v = state.get(name)
        return v if v is not None else np.array([[default]])


def build_wrong_fork_high_level_model():
    """Wrong fork high-level model applied to a chain low-level model: x->y, x->z (ignores y->z)."""
    h = CausalGraph()
    h.add_variable("x", lambda: None, domain=RectSubspace((0.0, 10.0)))
    h.add_variable("y", lambda x: 2.0 * x + 1.0, parents=["x"],
                   domain=RectSubspace((1.0, 21.0)))
    h.add_variable("z", lambda x: round(3.0 * x + 2.0, _ROUND_DIGITS),
                   parents=["x"], domain=RectSubspace((2.0, 32.0)))
    return h