"""Concrete LowLevelModel for PyTorch neural networks, bridging the NumPy-based
SystemState with PyTorch tensors.
"""

import warnings
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .low_level import LowLevelModel
from ..primitives import SystemState
from ..schema import MicroVariableSchema, Variable


class InterventionApplicator:
    """
    Applies interventions to PyTorch tensors (hard and soft, with broadcasting and slicing).
    """

    def __init__(self, device: str):
        self.device = device

    def _to_tensor(self, arr: Union[np.ndarray, 'torch.Tensor', float, int]) -> 'torch.Tensor':
        """Ensures the input is a tensor on the correct device."""
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device).float()
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        return torch.from_numpy(arr).float().to(self.device)

    def apply(self, target_tensor: 'torch.Tensor', raw_spec: Any):
        """
        Applies an intervention spec to a target tensor in-place.

        Args:
            target_tensor: The tensor to modify (B, ...).
            raw_spec: Can be:
                      - A value (numpy array, float)
                      - A tuple (indices, value)
                      - A tuple (indices, value, alpha)
                      - A list of the above
        """
        specs_list = raw_spec if isinstance(raw_spec, list) else [raw_spec]

        for spec in specs_list:
            self._apply_single_intervention(target_tensor, spec)

    def _apply_single_intervention(self, target_tensor: 'torch.Tensor', spec: Any):
        if isinstance(spec, (np.ndarray, torch.Tensor, float, int)):
            spec = (spec,)  # Normalize spec to tuple

        # Check for token-specific intervention (3d tensor: batch, seq, dim)
        token_idx = None
        if len(spec) > 0 and isinstance(spec[0], int) and not isinstance(spec[0], bool):
            if target_tensor.ndim == 3:
                token_idx = spec[0]
                spec = spec[1:]

        # Parse structure
        indices = None
        alpha = None

        if len(spec) == 3:
            indices, replacement, alpha = spec
        elif len(spec) == 2:
            indices, replacement = spec
        elif len(spec) == 1:
            replacement = spec[0]
        else:
            return  # Empty spec

        replacement_tensor = self._to_tensor(replacement)
        alpha_tensor = self._to_tensor(alpha) if alpha is not None else None

        # Broadcasting logic
        if replacement_tensor.ndim > 0 and replacement_tensor.shape[0] == 1 and target_tensor.shape[0] > 1:
            expand_shape = (target_tensor.shape[0], *replacement_tensor.shape[1:])
            replacement_tensor = replacement_tensor.expand(expand_shape)

        # Application logic
        if alpha_tensor is not None:
            self._apply_soft(target_tensor, indices, token_idx, replacement_tensor, alpha_tensor)
        else:
            self._apply_hard(target_tensor, indices, token_idx, replacement_tensor)

    def _apply_hard(self, target, indices, token_idx, val):
        if target.ndim == 3 and token_idx is not None:
            if indices is not None:
                # Val might be (batch, 1, dim)
                if isinstance(indices, int) and val.ndim == target.ndim:
                    val = val.squeeze(1)
                target[:, token_idx, indices] = val
            else:
                target[:, token_idx] = val
        elif target.ndim == 3:
            # Broadcast to (batch, seq, hidden)
            val_to_assign = val
            if val_to_assign.ndim == 2:
                val_to_assign = val_to_assign.unsqueeze(1)

            if indices is not None:
                if isinstance(indices, int) and val_to_assign.ndim == target.ndim:
                    # Usually indices for 3D is a slice or list (batch, 1, 1).
                    val_to_assign = val_to_assign.squeeze(-1)
                target[:, :, indices] = val_to_assign
            else:
                target.copy_(val_to_assign if val_to_assign.shape == target.shape else val)
        else:
            # Standard 2D (batch, dim)
            if indices is not None:
                if isinstance(indices, int) and val.ndim == target.ndim:
                    val = val.squeeze(-1)
                target[:, indices] = val
            else:
                target.copy_(val)

    def _apply_soft(self, target, indices, token_idx, val, alpha):
        # Soft: new = (1 - alpha) * old + alpha * new

        # Helper to get the relevant slice view
        if target.ndim == 3 and token_idx is not None:
            if indices is not None:
                view = target[:, token_idx, indices]
                if isinstance(indices, int) and val.ndim == target.ndim:
                    val = val.squeeze(1)
                    if alpha.ndim == target.ndim: alpha = alpha.squeeze(1)
            else:
                view = target[:, token_idx]
        elif target.ndim == 3:
            val = val.unsqueeze(1) if val.ndim == 2 else val
            alpha = alpha.unsqueeze(1) if alpha.ndim == 2 else alpha
            if indices is not None:
                view = target[:, :, indices]
                if isinstance(indices, int):
                    val = val.squeeze(-1)
                    alpha = alpha.squeeze(-1)
            else:
                view = target
        else:
            # 2D
            if indices is not None:
                view = target[:, indices]
                # If integer indexing, squeeze the trailing dim of inputs
                if isinstance(indices, int):
                    if val.ndim == view.ndim + 1:
                        val = val.squeeze(-1)
                    if alpha.ndim == view.ndim + 1:
                        alpha = alpha.squeeze(-1)
            else:
                view = target

        updated = (1 - alpha) * view + alpha * val
        view.copy_(updated)


class NeuralModel(LowLevelModel):
    """
    A wrapper for PyTorch nn.Module models (the low-level model M).
    """

    def __init__(self,
                 network: 'nn.Module',
                 layer_map: Dict[str, str],
                 schema: MicroVariableSchema,
                 device: str = "cpu",
                 input_shape: Optional[Tuple[int, ...]] = None):
        """
        Args:
            network: The PyTorch model.
            layer_map: Mapping from MicroVariable names -> PyTorch module names.
                       Use "__input__" to map a variable to the model's input tensor.
            schema: The variable schema.
            device: 'cpu' or 'cuda'.
            input_shape: Optional tuple to auto-generate inputs.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to use NeuralModel. "
                "Install it with `pip install causal_abstraction_eval[neural]`."
            )

        self.network = network
        self.layer_map = layer_map
        self.schema = schema
        self.device = device
        self.input_shape = input_shape
        self.network.to(self.device)
        self._module_dict = dict(self.network.named_modules())
        self.applicator = InterventionApplicator(self.device)

    @classmethod
    def infer_schema(cls,
                     network: 'nn.Module',
                     layer_map: Dict[str, str],
                     input_shape: Tuple[int, ...],
                     device: str = "cpu") -> MicroVariableSchema:
        """
        Infers a MicroVariableSchema by doing a single dry-run forward pass.

        Args:
            network: The PyTorch model.
            layer_map: Mapping from MicroVariable names -> PyTorch module names.
                       Use "__input__" for the model's input tensor.
            input_shape: Shape of a single input sample (without batch dimension).
            device: 'cpu' or 'cuda'.

        Returns:
            A MicroVariableSchema with shapes populated for every variable in layer_map.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If a ``layer_map`` module name is not found in the network.

        Example:
            schema = NeuralModel.infer_schema(network, layer_map, input_shape=(2,))
            model  = NeuralModel(network, layer_map, schema, input_shape=(2,))
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to use NeuralModel. "
                "Install it with `pip install causal_abstraction_eval[neural]`."
            )

        network = network.to(device)
        module_dict = dict(network.named_modules())
        captured: Dict[str, Tuple[int, ...]] = {}
        hooks = []

        # Hook to record output shape of each named layer
        def _make_shape_hook(micro_name: str):
            def hook(module, input, output):
                # output shape is (batch, *feature_dims); drop the batch dim
                captured[micro_name] = tuple(output.shape[1:])

            return hook

        try:
            for micro_name, module_name in layer_map.items():
                if module_name == "__input__":
                    # Input shape is known directly
                    captured[micro_name] = input_shape
                    continue

                module = module_dict.get(module_name)
                if module is None:
                    raise ValueError(
                        f"Module '{module_name}' (mapped from '{micro_name}') "
                        f"not found in network. Available: {list(module_dict.keys())}"
                    )
                hooks.append(module.register_forward_hook(_make_shape_hook(micro_name)))

            # Single dry-run with a dummy batch of size 1
            dummy = torch.zeros((1, *input_shape), device=device)
            with torch.no_grad():
                network(dummy)

        finally:
            for h in hooks:
                h.remove()

        variables = [Variable(name, shape=shape) for name, shape in captured.items()]
        return MicroVariableSchema(variables)

    @classmethod
    def from_network(cls,
                     network: 'nn.Module',
                     layer_map: Dict[str, str],
                     input_shape: Tuple[int, ...],
                     device: str = "cpu") -> Tuple['NeuralModel', MicroVariableSchema]:
        """
        Convenience factory: infers the schema and constructs the model in one call.

        Args:
            network: The PyTorch model.
            layer_map: Mapping from MicroVariable names -> PyTorch module names.
                       Use "__input__" for the model's input tensor.
            input_shape: Shape of a single input sample (without batch dimension).
            device: 'cpu' or 'cuda'.

        Returns:
            (NeuralModel, MicroVariableSchema), both fully initialized.

        Raises:
            ImportError: If PyTorch is not installed.
            ValueError: If a ``layer_map`` module name is not found in the network.

        Example:
            model, schema = NeuralModel.from_network(network, layer_map, input_shape=(2,))
            cg_map = CoarseGrainingMap(schema, cg_mapping)
        """
        schema = cls.infer_schema(network, layer_map, input_shape, device)
        model = cls(network, layer_map, schema, device=device, input_shape=input_shape)
        return model, schema

    def _numpy_to_torch(self, arr: np.ndarray) -> 'torch.Tensor':
        return torch.from_numpy(arr).float().to(self.device)

    def _torch_to_numpy(self, tensor: 'torch.Tensor') -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def forward_with_interventions(self,
                                   input_state: SystemState,
                                   interventions: Optional[Dict[str, Any]]) -> SystemState:
        """Run the network, applying interventions and capturing mapped activations.

        Args:
            input_state: Initial state; uses the ``'input'`` value if present,
                otherwise the first array value, otherwise zeros of ``input_shape``.
            interventions: Mapping of micro-variable name to an intervention spec.

        Returns:
            ``input_state`` merged with the captured activations and the ``'output'``.

        Raises:
            ValueError: If the input state is empty and no ``input_shape`` is set.
        """
        interventions = interventions or {}
        captured_activations = {}
        hooks = []

        # Hook factories
        def _make_activation_hook(micro_name: str):
            def hook(module, input, output):
                captured_activations[micro_name] = output.clone()

            return hook

        def _make_intervention_hook(micro_name: str, spec: Any):
            def hook_fn(module, input, output):
                # Modify output in-place via the applicator
                self.applicator.apply(output, spec)
                return output

            return hook_fn

        try:
            # Register hooks for internal layers
            for micro_name, module_name in self.layer_map.items():
                if module_name == "__input__":
                    continue  # Handled explicitly later

                module = self._module_dict.get(module_name)
                if module is None:
                    # A mapped micro-variable whose module name does not resolve
                    # would otherwise be silently skipped; warn instead.
                    warnings.warn(
                        f"NeuralModel: layer_map entry {micro_name!r} -> "
                        f"{module_name!r} matches no module in the network; "
                        f"this micro-variable is skipped (no activation "
                        f"captured, no intervention applied).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                if micro_name in interventions:
                    hooks.append(module.register_forward_hook(
                        _make_intervention_hook(micro_name, interventions[micro_name])
                    ))

                hooks.append(module.register_forward_hook(
                    _make_activation_hook(micro_name)
                ))

            # Prepare input tensor
            x = None
            if 'input' in input_state.values:
                x = self._numpy_to_torch(input_state['input'])
            elif input_state.values:
                # Fallback: use the first array-valued entry when no 'input' key exists
                first_val = next(iter(input_state.values.values()))
                if isinstance(first_val, np.ndarray):
                    x = self._numpy_to_torch(first_val)

            if x is None:
                if self.input_shape is None:
                    raise ValueError("Empty state and no input_shape.")

                batch_size = 1
                if interventions:
                    first_int = next(iter(interventions.values()))
                    # Extract batch size from first intervention if available
                    if isinstance(first_int, list):
                        first_int = first_int[0][1]
                    batch_size = first_int.shape[0]

                x = torch.zeros((batch_size, *self.input_shape)).to(self.device)

            # Apply input interventions
            input_vars = [mv for mv, mod in self.layer_map.items() if mod == "__input__"]
            for mv in input_vars:
                if mv in interventions:
                    self.applicator.apply(x, interventions[mv])
                captured_activations[mv] = x.clone()

            # Run forward
            output_tensor = self.network(x)

        finally:
            for h in hooks:
                h.remove()

        result_values = {'output': self._torch_to_numpy(output_tensor)}
        for k, v in captured_activations.items():
            result_values[k] = self._torch_to_numpy(v)

        final_state = input_state.merge(SystemState(values=result_values))
        return final_state