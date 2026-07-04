"""Dynamic threshold and folding decision scheduler."""

from __future__ import annotations


class FoldingScheduler:
    """Adjusts the similarity threshold tau based on runtime context.

    The scheduler encodes the heuristic that:
    - Early layers are more stable -> higher tau (more reuse).
    - Early diffusion steps are less certain -> lower tau (more conservative).
    - Math tasks need higher accuracy -> lower tau.
    - Code tasks tolerate more reuse -> higher tau.

    Args:
        base_tau: Default threshold.
        num_layers: Total number of layers (used to normalize depth).
        num_steps: Total number of diffusion steps.
    """

    def __init__(
        self,
        base_tau: float = 0.95,
        num_layers: int = 12,
        num_steps: int = 10,
        disabled_layers: set[int] | None = None,
    ) -> None:
        if not 0.0 <= base_tau <= 1.0:
            raise ValueError(f"base_tau must be in [0, 1], got {base_tau}")
        if num_layers <= 0 or num_steps <= 0:
            raise ValueError("num_layers and num_steps must be positive.")

        self.base_tau = base_tau
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.disabled_layers = set(disabled_layers or [])

        # Tunable biases (small values to keep tau in a reasonable range).
        self.layer_bias_scale = 0.03  # early layers get +bias, late layers -bias
        self.step_bias_scale = 0.04  # early steps get -bias
        self.task_bias: dict[str, float] = {
            "math": -0.03,
            "code": 0.03,
            "general": 0.0,
        }

    def get_tau(
        self,
        layer_idx: int,
        step_idx: int,
        task_type: str = "general",
    ) -> float:
        """Return the threshold for a specific layer, step, and task.

        Args:
            layer_idx: Layer index.
            step_idx: Diffusion step index.
            task_type: One of "math", "code", "general".

        Returns:
            Clamped threshold in [0.80, 0.99].
        """
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        if not 0 <= step_idx < self.num_steps:
            raise ValueError(f"step_idx {step_idx} out of range [0, {self.num_steps})")

        # Depth: early layers are more reusable.
        depth_ratio = 1.0 - (layer_idx / max(self.num_layers - 1, 1))
        layer_bias = self.layer_bias_scale * (2.0 * depth_ratio - 1.0)

        # Step: early steps are more uncertain.
        step_ratio = 1.0 - (step_idx / max(self.num_steps - 1, 1))
        step_bias = -self.step_bias_scale * step_ratio

        task_bias = self.task_bias.get(task_type, 0.0)

        tau = self.base_tau + layer_bias + step_bias + task_bias
        return float(min(max(tau, 0.80), 0.99))

    def disable_layers(self, layer_indices: set[int]) -> None:
        """Mark ``layer_indices`` as disabled (no folding)."""
        self.disabled_layers.update(layer_indices)

    def enable_layers(self, layer_indices: set[int]) -> None:
        """Re-enable folding for ``layer_indices``."""
        self.disabled_layers.difference_update(layer_indices)

    def should_fold(
        self,
        layer_idx: int,
        step_idx: int,
    ) -> bool:
        """Return whether folding should be applied at a layer/step.

        Folding is enabled everywhere except:
        - layers in ``disabled_layers``,
        - the final layer,
        - the final diffusion step,
        where full recomputation is safer.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return False
        if step_idx < 0 or step_idx >= self.num_steps:
            return False
        if layer_idx in self.disabled_layers:
            return False
        # Disable folding for the final layer and final diffusion step.
        if layer_idx == self.num_layers - 1 or step_idx == self.num_steps - 1:
            return False
        return True

    def set_task_bias(self, task_type: str, bias: float) -> None:
        """Register or update a task-specific bias."""
        self.task_bias[task_type] = bias
