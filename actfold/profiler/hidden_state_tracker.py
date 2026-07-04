"""Hook-based hidden-state tracking for Diffusion LLMs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


class HiddenStateTracker:
    """Tracks and dumps hidden states at layer/token/step granularity.

    Registers forward hooks on model layers and stores the captured hidden
    states in memory. The captured tensors can be dumped to disk in
    ``safetensors`` or ``torch`` format.

    Args:
        storage_device: Device to move captured tensors to. Use "cpu" to save
            GPU memory during profiling.
    """

    def __init__(self, storage_device: str = "cpu") -> None:
        self.storage_device = storage_device
        self._states: dict[str, list[torch.Tensor]] = {}
        self._handles: list[Any] = []

    def register_hook(
        self,
        model: nn.Module,
        layer_indices: list[int] | None = None,
        target_module_type: type[nn.Module] | None = None,
    ) -> None:
        """Register forward hooks to capture hidden states.

        Args:
            model: The model to instrument.
            layer_indices: If provided, only instrument these layer indices.
            target_module_type: If provided, instrument modules of this type
                (e.g., ``nn.TransformerEncoderLayer``).
        """
        self.remove_hooks()
        self._states.clear()

        target_type = target_module_type or nn.TransformerEncoderLayer
        candidates: list[tuple[int, nn.Module]] = []

        for idx, module in enumerate(model.modules()):
            if isinstance(module, target_type):
                candidates.append((idx, module))

        if layer_indices is not None:
            candidates = [(idx, module) for idx, module in candidates if idx in layer_indices]

        for idx, module in candidates:
            handle = module.register_forward_hook(self._make_hook(str(idx)))
            self._handles.append(handle)

    def _make_hook(self, layer_key: str) -> Callable[..., None]:
        """Create a forward hook for a specific layer key."""

        def hook(
            module: nn.Module,
            input_args: tuple[Any, ...],
            output: Any,
        ) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            if layer_key not in self._states:
                self._states[layer_key] = []
            self._states[layer_key].append(tensor.detach().to(self.storage_device))

        return hook

    def capture(self, key: str, tensor: torch.Tensor) -> None:
        """Manually capture a tensor under a custom key.

        Args:
            key: Identifier for the captured tensor group.
            tensor: Tensor to store (will be detached and moved to storage device).
        """
        if key not in self._states:
            self._states[key] = []
        self._states[key].append(tensor.detach().to(self.storage_device))

    def get_states(self) -> dict[str, list[torch.Tensor]]:
        """Return the captured hidden states."""
        return self._states

    def dump(self, path: Path, format: str = "safetensors") -> None:
        """Persist tracked hidden states to disk.

        Args:
            path: Output file path.
            format: ``safetensors`` or ``torch``.

        Raises:
            ValueError: If format is unsupported.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "torch":
            torch.save(self._states, path)
        elif format == "safetensors":
            try:
                from safetensors.torch import save_file
            except ImportError as exc:
                raise ImportError(
                    "safetensors is required for this format. Install with: pip install safetensors"
                ) from exc

            flattened: dict[str, torch.Tensor] = {}
            for layer_key, tensors in self._states.items():
                for step_idx, tensor in enumerate(tensors):
                    flattened[f"{layer_key}_step{step_idx}"] = tensor
            save_file(flattened, str(path))
        else:
            raise ValueError(f"Unsupported dump format: {format}")

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        """Clear captured states and remove hooks."""
        self.remove_hooks()
        self._states.clear()
