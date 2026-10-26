"""The engine's seeded generator must reach stochastic low-level models and grounding."""
import numpy as np

from causal_abstraction import LowLevelModel, NoisyLowLevelModel, SystemState, run_low_level_model


class _Deterministic(LowLevelModel):
    def forward_with_interventions(self, input_state, interventions):
        return SystemState(values={"y": np.zeros((1, 1))})


class _Stochastic(LowLevelModel):
    def forward_with_interventions(self, input_state, interventions, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        return SystemState(values={"y": rng.normal(size=(1, 1))})


def test_deterministic_model_called_without_rng():
    out = run_low_level_model(_Deterministic(), SystemState(), {}, rng=np.random.default_rng(0))
    assert out["y"].shape == (1, 1)


def test_stochastic_model_receives_rng():
    a = run_low_level_model(_Stochastic(), SystemState(), {}, rng=np.random.default_rng(3))
    b = run_low_level_model(_Stochastic(), SystemState(), {}, rng=np.random.default_rng(3))
    assert np.array_equal(a["y"], b["y"])


def test_noisy_wrapper_uses_passed_rng():
    noisy = NoisyLowLevelModel(_Deterministic(), noise_std=1.0)
    a = noisy.forward_with_interventions(SystemState(), {}, rng=np.random.default_rng(7))
    b = noisy.forward_with_interventions(SystemState(), {}, rng=np.random.default_rng(7))
    assert np.array_equal(a["y"], b["y"])
    assert not np.allclose(a["y"], 0.0)
