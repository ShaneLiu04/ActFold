"""Architecture detection and component extraction for Hugging Face models.

This module provides helpers that discover the embedding module, Transformer
layer stack, and language modeling head for a wide range of model families.
It is used by the demo and by generic folding wrappers so that Branch Folding
works without architecture-specific wiring.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import torch
import torch.nn as nn

_T = TypeVar("_T", bound=nn.Module)

# AR025 DD1: the single frozen message for the parent_branch_id /
# parent_branch_ids mutual exclusion (both singular and plural given).
_MULTI_PARENT_MSG = (
    "ManualFoldedForward accepts either parent_branch_id "
    "(shared) or parent_branch_ids (per-row), not both."
)


def _clear_cublas_workspaces() -> None:
    """Release torch's per-stream cuBLAS workspace cache (device memory).

    CUDA graph capture runs its warmup on a fresh side stream; the first
    matmul on that stream allocates a cuBLAS workspace that torch caches
    per stream forever. Evicting a runner releases the graph and its static
    buffers, but this process-wide cache is the only way to also return the
    workspace blocks the capture leaked (the resident-VRAM reclamation is
    pinned by UT-2106). Workspaces baked into a live graph's captured body
    live in that graph's private pool and are unaffected.
    """
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()


@dataclass
class ArchitectureProfile:
    """Discovered architecture components of a loaded model.

    Attributes:
        model_type: Canonical model type string (e.g. ``"gpt2"``, ``"llama"``).
        embed_module: Module that maps token ids to hidden states.
        layers: List of Transformer decoder/encoder layers.
        head_module: Optional language modeling head or output projection.
        layer_path: Dot-separated attribute path to ``layers``.
        embed_path: Dot-separated attribute path to ``embed_module``.
        head_path: Dot-separated attribute path to ``head_module``.
        supports_causal_mask: Whether the model expects a causal ``attention_mask``.
        final_norm: Optional final normalization module applied after the layer
            stack and before the head (e.g. LLaMA ``model.norm``, GPT-2
            ``transformer.ln_f``). ``None`` when the model has none.
    """

    model_type: str
    embed_module: nn.Module
    layers: nn.ModuleList
    head_module: nn.Module | None
    layer_path: str
    embed_path: str
    head_path: str | None
    supports_causal_mask: bool = True
    final_norm: nn.Module | None = None


# Ordered list of common layer-stack paths.  Earlier entries take precedence.
_DEFAULT_LAYER_PATHS: tuple[str, ...] = (
    # LLaMA / Qwen / Mistral / Gemma / Yi / InternLM style
    "model.layers",
    "transformer.h",
    "transformer.layers",
    # GPT-Neo / GPT-J / CodeGen
    "gpt_neox.layers",
    "transformer.blocks",
    # LLaDA (LLaDAModelLM: model.transformer.blocks)
    "model.transformer.blocks",
    # OPT
    "model.decoder.layers",
    "decoder.layers",
    # BERT / RoBERTa / DeBERTa
    "encoder.layer",
    "model.encoder.layer",
    "bert.encoder.layer",
    # T5 / UL2 / BART / mT5 (decoder path preferred for generation)
    "decoder.block",
    "model.decoder.block",
    "encoder.block",
    "model.encoder.block",
    # Falcon
    "transformer.h",
    # Phi / Phi-2
    "model.layers",
    # Generic
    "layers",
    "h",
    "blocks",
)

# Ordered list of common embedding paths.
_DEFAULT_EMBED_PATHS: tuple[str, ...] = (
    "model.embed_tokens",
    "model.transformer.wte",
    "transformer.wte",
    "transformer.word_embeddings",
    "transformer.embedding",
    "model.decoder.embed_tokens",
    "decoder.embed_tokens",
    "bert.embeddings.word_embeddings",
    "encoder.embed_tokens",
    "shared",
    "embeddings.word_embeddings",
    "embedding",
    "wte",
    "word_embeddings",
)

# Ordered list of common LM head paths.
_DEFAULT_HEAD_PATHS: tuple[str, ...] = (
    "lm_head",
    "model.lm_head",
    "transformer.lm_head",
    "model.decoder.lm_head",
    "encoder.lm_head",
    "cls.predictions",
    "cls",
    "head",
    "output_projection",
)


# Ordered list of common final-norm paths (applied after layers, before head).
_DEFAULT_FINAL_NORM_PATHS: tuple[str, ...] = (
    # LLaMA / Qwen / Mistral / Gemma
    "model.norm",
    # GPT-2 / GPT-Neo / GPT-J
    "transformer.ln_f",
    # OPT
    "model.decoder.norm",
    # BERT / RoBERTa / T5 / mT5
    "encoder.norm",
    "decoder.norm",
    # Generic
    "norm",
    "ln_f",
)


def _get_attr_path(model: nn.Module, path: str) -> Any | None:
    """Return the nested attribute at ``path`` or ``None`` if missing."""
    current: Any = model
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _find_path(
    model: nn.Module,
    candidates: tuple[str, ...],
    expected_type: type[_T] | tuple[type[_T], ...],
) -> tuple[str, _T] | None:
    """Find the first candidate path that resolves to an instance of ``expected_type``."""
    for path in candidates:
        obj = _get_attr_path(model, path)
        if isinstance(obj, expected_type):
            return path, obj
    return None


def find_layer_list(
    model: nn.Module,
    candidate_paths: tuple[str, ...] | None = None,
) -> tuple[str, nn.ModuleList] | None:
    """Discover the Transformer layer list in ``model``.

    Args:
        model: The model to inspect.
        candidate_paths: Optional ordered list of dot-separated paths to try.

    Returns:
        A tuple of ``(path, module_list)`` or ``None`` if no layer list is found.
    """
    paths = candidate_paths or _DEFAULT_LAYER_PATHS
    result = _find_path(model, paths, nn.ModuleList)
    return result


def find_embedding_module(
    model: nn.Module,
    candidate_paths: tuple[str, ...] | None = None,
) -> tuple[str, nn.Module] | None:
    """Discover the token embedding module in ``model``.

    Args:
        model: The model to inspect.
        candidate_paths: Optional ordered list of dot-separated paths to try.

    Returns:
        A tuple of ``(path, embedding_module)`` or ``None``.
    """
    paths = candidate_paths or _DEFAULT_EMBED_PATHS
    result = _find_path(model, paths, nn.Module)
    if result is not None:
        return result

    # Last resort: try get_input_embeddings for models that expose it.
    if hasattr(model, "get_input_embeddings"):
        getter = getattr(model, "get_input_embeddings")
        if callable(getter):
            emb = getter()
            if isinstance(emb, nn.Module):
                return "get_input_embeddings()", emb
    return None


def find_lm_head(
    model: nn.Module,
    candidate_paths: tuple[str, ...] | None = None,
) -> tuple[str, nn.Module] | None:
    """Discover the language modeling head in ``model``.

    Args:
        model: The model to inspect.
        candidate_paths: Optional ordered list of dot-separated paths to try.

    Returns:
        A tuple of ``(path, head_module)`` or ``None``.
    """
    paths = candidate_paths or _DEFAULT_HEAD_PATHS
    result = _find_path(model, paths, nn.Module)
    if result is not None:
        return result

    # Last resort: try get_output_embeddings for models that expose it.
    if hasattr(model, "get_output_embeddings"):
        getter = getattr(model, "get_output_embeddings")
        if callable(getter):
            head = getter()
            if isinstance(head, nn.Module):
                return "get_output_embeddings()", head
    return None


def _infer_model_type(model: nn.Module) -> str:
    """Infer a canonical model type from the model object or config."""
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str):
            return model_type.lower()
    cls_name = type(model).__name__.lower()
    for keyword in (
        "llada",
        "dream",
        "fastdllm",
        "gpt2",
        "gptneo",
        "gptj",
        "llama",
        "qwen",
        "mistral",
        "gemma",
        "opt",
        "falcon",
        "phi",
        "bert",
        "roberta",
        "t5",
        "bart",
    ):
        if keyword in cls_name:
            return keyword
    return "unknown"


def detect_architecture(
    model: nn.Module,
    layer_paths: tuple[str, ...] | None = None,
    embed_paths: tuple[str, ...] | None = None,
    head_paths: tuple[str, ...] | None = None,
) -> ArchitectureProfile:
    """Detect the embedding, layer stack, and head of ``model``.

    Args:
        model: A loaded Hugging Face model or ActFold adapter.
        layer_paths: Optional ordered list of layer-list paths.
        embed_paths: Optional ordered list of embedding paths.
        head_paths: Optional ordered list of LM-head paths.

    Returns:
        An :class:`ArchitectureProfile` describing the discovered components.

    Raises:
        RuntimeError: If the layer list cannot be discovered.
    """
    layer_result = find_layer_list(model, layer_paths)
    if layer_result is None:
        raise RuntimeError(
            "Could not discover the Transformer layer list. "
            f"Searched paths: {layer_paths or _DEFAULT_LAYER_PATHS}."
        )
    layer_path, layers = layer_result

    embed_result = find_embedding_module(model, embed_paths)
    if embed_result is None:
        raise RuntimeError(
            "Could not discover the token embedding module. "
            f"Searched paths: {embed_paths or _DEFAULT_EMBED_PATHS}."
        )
    embed_path, embed_module = embed_result

    head_result = find_lm_head(model, head_paths)
    head_module: nn.Module | None = None
    head_path: str | None = None
    if head_result is not None:
        head_path, head_module = head_result

    model_type = _infer_model_type(model)

    supports_causal_mask = model_type not in {"bert", "roberta", "deberta", "deberta-v2"}

    final_norm_module: nn.Module | None = None
    for norm_path in _DEFAULT_FINAL_NORM_PATHS:
        candidate = _get_attr_path(model, norm_path)
        if isinstance(candidate, nn.Module):
            final_norm_module = candidate
            break

    return ArchitectureProfile(
        model_type=model_type,
        embed_module=embed_module,
        layers=layers,
        head_module=head_module,
        layer_path=layer_path,
        embed_path=embed_path,
        head_path=head_path,
        supports_causal_mask=supports_causal_mask,
        final_norm=final_norm_module,
    )


def build_manual_folded_forward(
    model: nn.Module,
    cache: Any,
    gate: Any,
    scheduler: Any | None = None,
) -> "ManualFoldedForward":
    """Build a manual folded forward helper for ``model``.

    This is a convenience factory used when :class:`~actfold.core.model_wrapper.FoldedModel`
    cannot auto-discover the layer stack. It uses :func:`detect_architecture` to find
    the embedding, layers, and head, then wraps each layer with
    :class:`~actfold.core.folded_transformer.FoldedTransformerLayer`.

    Args:
        model: The model to wrap.
        cache: Activation cache shared across branches.
        gate: Similarity gate.
        scheduler: Optional folding scheduler.

    Returns:
        A :class:`ManualFoldedForward` instance.
    """
    return ManualFoldedForward(model, cache=cache, gate=gate, scheduler=scheduler)


class ManualFoldedForward(nn.Module):
    """Architecture-agnostic folded forward path for HF models.

    This is the recommended, **non-mutating** folded forward: it discovers the
    embedding, layer stack, and language modeling head via
    :func:`detect_architecture` and routes tokens through wrapped
    :class:`~actfold.core.folded_transformer.FoldedTransformerLayer` modules
    WITHOUT replacing anything on the base model — the base module tree,
    parameters, and ``state_dict()`` keys are untouched and the raw model stays
    directly callable at all times.  Branch context is threaded through
    explicit ``branch_id`` / ``parent_branch_id`` / ``step_idx`` arguments, so
    the thread-local ``FOLDING_CONTEXT`` fallback is never consulted.

    Use this instead of the legacy in-place
    :class:`~actfold.core.model_wrapper.FoldedModel`.

    Args:
        model: The base model to fold (never mutated).
        cache: Activation cache shared across branches.
        gate: Similarity gate.
        scheduler: Optional folding scheduler.
        split_layers: Wrap layers with
            :class:`~actfold.core.split_layer.SplitFoldedTransformerLayer` so
            stable tokens skip the FFN on recompute.  Layers without a
            detectable FFN chain silently fall back to full recompute.
        split_min_tokens: Minimum ``batch * seq`` for the split path to engage
            (below it the per-layer gather sync outweighs the FFN savings).
        use_cuda_graph: Opt in to CUDA-graph capture/replay of the folded
            verification forward (requires CUDA; wired up by the graph runner,
            ignored otherwise).
        graph_capacity_ratio: Fraction of tokens per layer reserved for
            divergent recompute under graph replay; must satisfy
            ``0 < ratio <= 1``.
        graph_adaptive_capacity: Opt in to adaptive divergent-capacity
            re-capture on the batched graph path (AR017): per-step
            divergence demand is observed and the runner is re-captured
            when the fixed capacity is persistently over- or
            under-provisioned. Inert without ``use_cuda_graph``.
        graph_adaptive_config: Optional
            :class:`~actfold.core.adaptive_capacity.AdaptiveCapacityConfig`
            policy hyper-parameters; a default config is constructed when
            ``None``. Must be ``None`` or a config instance when
            ``graph_adaptive_capacity`` is off too (type-checked always).
    """

    def __init__(
        self,
        model: nn.Module,
        cache: Any,
        gate: Any,
        scheduler: Any | None = None,
        split_layers: bool = False,
        split_min_tokens: int = 512,
        use_cuda_graph: bool = False,
        graph_capacity_ratio: float = 0.5,
        graph_adaptive_capacity: bool = False,
        graph_adaptive_config: Any | None = None,
        graph_runner_pool_size: int = 0,
        multi_parent_batch: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < graph_capacity_ratio <= 1.0:
            raise ValueError(
                "graph_capacity_ratio must satisfy 0 < ratio <= 1, got " f"{graph_capacity_ratio}"
            )
        if graph_runner_pool_size < 0:
            raise ValueError(
                "graph_runner_pool_size must be >= 0, got " f"{graph_runner_pool_size}"
            )
        self.profile = detect_architecture(model)
        # AR029: the cache setter validates graph compatibility, so the flag
        # must exist before the initial cache assignment.
        self.use_cuda_graph = bool(use_cuda_graph)
        self._validate_cache_graph_compat(cache)
        self.cache = cache
        self.gate = gate
        self.scheduler = scheduler
        self.split_layers = bool(split_layers)
        self.split_min_tokens = int(split_min_tokens)
        self.graph_capacity_ratio = float(graph_capacity_ratio)
        # Adaptive capacity (AR017): the policy exists only when the feature
        # is on -- the public probe ``graph_adaptive_policy`` reads ``None``
        # otherwise (UT-1606 zero-state contract: default-off must be
        # bit-identical to the AR016 wiring, no policy state ever built).
        if graph_adaptive_config is not None:
            from actfold.core.adaptive_capacity import AdaptiveCapacityConfig

            if not isinstance(graph_adaptive_config, AdaptiveCapacityConfig):
                raise ValueError(
                    "graph_adaptive_config must be an AdaptiveCapacityConfig "
                    f"instance, got {type(graph_adaptive_config).__name__}"
                )
        self.graph_adaptive_capacity = bool(graph_adaptive_capacity)
        self._graph_adaptive_policy: Any = None
        if self.graph_adaptive_capacity:
            from actfold.core.adaptive_capacity import (
                AdaptiveCapacityConfig,
                AdaptiveCapacityPolicy,
            )

            adaptive_config = (
                graph_adaptive_config
                if graph_adaptive_config is not None
                else AdaptiveCapacityConfig()
            )
            self._graph_adaptive_policy = AdaptiveCapacityPolicy(adaptive_config)
        self._graph_adaptive_retire_warned = False
        self._graph_adaptive_fail_warned = False
        # Lazily created by the CUDA graph path (AR002/T007); ``None`` on
        # the eager path.
        self.graph_runner: Any = None
        self._graph_mask: Any = None
        self._graph_pe: Any = None
        self._graph_position_ids: Any = None
        self._graph_degraded_warned = False
        self._graph_shape_warned = False
        self._graph_budget_warned = False
        self._graph_capture_failed = False
        # Batched-branch CUDA graph state (AR016): a separate runner for the
        # [N, T] capture shape (the serial runner holds [1, T] — the two
        # shapes can never share a runner). The shared degradation flag
        # above covers the scheduler/gate/CUDA/capture-fail family (one
        # warning per family, cross-reason suppression — serial precedent);
        # the shape/budget/capture-failed flags below are per-side.
        self.graph_runner_batched: Any = None
        self._graph_batched_mask: Any = None
        self._graph_batched_pe: Any = None
        self._graph_batched_position_ids: Any = None
        self._graph_batched_shape_warned = False
        # Multi-parent (AR025): the captured batched graph body serves one
        # shared parent; a per-row parent call takes the eager body with a
        # one-time warning (warn-once, never an error — the graph face is
        # Out-of-Scope for per-row parents).
        self._multi_parent_graph_warned = False
        self._graph_batched_budget_warned = False
        self._graph_batched_capture_failed = False
        # Shape-keyed graph runner pool (AR021 FR2). 0 = legacy: the
        # single-runner attributes above keep their exact warn-once /
        # never-recapture behavior. > 0 = LRU pool cap: one runner per
        # (tokens shape, attention-mask identity) fold window — the pinned
        # ctx mask makes the key semantically equal to the design's
        # (T, prompt_len) / (N, T, prompt_len) (same window -> same mask
        # object -> same key; same T with a different prompt_len -> a
        # different mask object -> a different key). The pools hold the
        # mask references so object ids stay valid; the legacy attributes
        # stay live aliases of the currently active runner (refreshed on
        # every pool capture/replay, so an LRU eviction drops the last
        # reference and reclaims the static-buffer VRAM).
        self.graph_runner_pool_size = int(graph_runner_pool_size)
        self._graph_runner_pool: "OrderedDict[Any, Any]" = OrderedDict()
        self._graph_runner_pool_stats: dict[Any, dict[str, int]] = {}
        self._graph_runner_batched_pool: "OrderedDict[Any, Any]" = OrderedDict()
        self._graph_runner_batched_pool_stats: dict[Any, dict[str, int]] = {}
        # Two-class eager counters (AR021 FR2 observation face):
        # degradation = every non-structural graph-path fallback (mask/pe
        # identity mismatch, forced identity mismatch, budget discard,
        # capture failure, scheduler/gate family, anomalously missing
        # parent entries); structural = the by-design eager forwards
        # (parent-None window populate, parent-at-old-shape window switch).
        self._degradation_fallback_count = 0
        self._structural_eager_count = 0
        # AR025 DD6: the single behavior switch. Off (default): the dispatch
        # predicate, the eager publish, and the generation loop are
        # byte-for-byte the pre-AR025 wiring. On: the predicate merges
        # per-row-parent candidate batches, the eager batched publish takes
        # one put_batch per layer (#42 probe, re-checked per call), and
        # ``folded_generate_batched`` merges chains. The API face
        # (``parent_branch_ids``) is NOT gated by this switch.
        self.multi_parent_batch = bool(multi_parent_batch)
        self._wrapped_layers = nn.ModuleList(
            [self._wrap_layer(layer, idx) for idx, layer in enumerate(self.profile.layers)]
        )

    def _validate_cache_graph_compat(self, value: Any) -> None:
        """Reject a graph-incompatible cache when CUDA-graph folding is on (AR029)."""
        if getattr(self, "use_cuda_graph", False) and getattr(value, "graph_incompatible", False):
            raise ValueError(
                "PyramidActivationCache is not CUDA-graph compatible; "
                "use use_cuda_graph=False or a dense cache."
            )

    @property
    def cache(self) -> Any:
        """Activation cache shared by the folded layers and graph runners."""
        return self._cache

    @cache.setter
    def cache(self, value: Any) -> None:
        """Set the cache and propagate the swap to every pinned reference.

        The wrapped layers and the (lazily captured) graph runners each hold
        their own cache reference, so a mid-life ``mff.cache = other`` (the
        AR020 A/B benchmark and the capability-probe tests) must reach all
        of them: eager layer stores, the publish-path capability probe and
        writes, parent-cache validation, and the runners' replay-time parent
        fetches. Without the propagation a swapped-in cache would never see
        the re-serialized parent and the batched graph path would silently
        fall back to eager. AR021 extends the reach to every runner held by
        the shape-keyed pools (including non-active members — an evicted
        runner is dead by definition and never replayed again).

        AR029 validates graph compatibility BEFORE mutating any reference, so
        a rejected swap leaves the instance unchanged.
        """
        self._validate_cache_graph_compat(value)
        self._cache = value
        layers = getattr(self, "_wrapped_layers", None)
        if layers is not None:
            for wrapped in layers:
                wrapped.cache = value
        for runner_attr in ("graph_runner", "graph_runner_batched"):
            runner = getattr(self, runner_attr, None)
            if runner is not None:
                runner._cache = value
        for pool_attr in ("_graph_runner_pool", "_graph_runner_batched_pool"):
            pool = getattr(self, pool_attr, None)
            if pool:
                for runner in pool.values():
                    runner._cache = value

    def _wrap_layer(self, layer: nn.Module, idx: int) -> nn.Module:
        """Wrap a single Transformer layer with a folded layer wrapper."""
        if self.split_layers:
            from actfold.core.split_layer import SplitFoldedTransformerLayer

            return SplitFoldedTransformerLayer(
                original_layer=layer,
                cache=self.cache,
                gate=self.gate,
                layer_idx=idx,
                scheduler=self.scheduler,
                min_split_tokens=self.split_min_tokens,
                multi_parent_batch=self.multi_parent_batch,
            )
        from actfold.core.folded_transformer import FoldedTransformerLayer

        return FoldedTransformerLayer(
            original_layer=layer,
            cache=self.cache,
            gate=self.gate,
            layer_idx=idx,
            scheduler=self.scheduler,
            multi_parent_batch=self.multi_parent_batch,
        )

    def forward(
        self,
        tokens: Any,
        branch_id: str | None,
        parent_branch_id: str | None = None,
        attention_mask: Any | None = None,
        step_idx: int = 0,
        branch_ids: Sequence[str] | None = None,
        position_embeddings: Any | None = None,
        forced_divergent: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        parent_branch_ids: Sequence[str] | None = None,
    ) -> Any:
        """Run a folded forward pass and return logits or hidden states.

        Args:
            tokens: Input token ids ``[batch, seq_len]`` (``[N, seq_len]``
                when ``branch_ids`` is set).
            branch_id: Identifier of the current branch (serial calls);
                must be ``None`` for batched calls.
            parent_branch_id: Optional parent branch identifier for reuse.
            attention_mask: Optional attention mask.
            step_idx: Current diffusion step index.
            branch_ids: Batched-branch identifiers (AR015): row ``i`` of
                ``tokens`` is verified as branch ``branch_ids[i]`` sharing
                ``parent_branch_id``. Batched calls use the batched CUDA
                graph path when ``use_cuda_graph`` is set and every batched
                precondition holds (AR016); any miss falls back to the
                batched eager body below.
            position_embeddings: Optional rotary ``(cos, sin)`` tuple
                (AR019) produced by the model's own ``rotary_emb`` under
                the constant-position scheme (``arange(T)``, no KV cache).
                Threaded through every construction face — serial/batched
                eager (forwarded as a layer kwarg), the CUDA-graph capture
                (baked by reference into the static body), and the adaptive
                re-capture. The graph path additionally requires the SAME
                tuple object on every replay after capture (``is`` identity,
                mirroring the ``attention_mask`` contract); a different
                object silently falls back to eager. ``None`` (the default)
                attaches no kwarg at all, so existing callers are
                byte-for-byte unchanged.
            forced_divergent: Optional boolean tensor (AR021 FR2) whose
                True positions are forced divergent in every layer's
                stability partition regardless of the gate (the layer ANDs
                the complement into the materialized stable mask before the
                profiler record and the count consumption). Serial calls
                accept ``[seq_len]`` or ``[1, seq_len]``; batched calls
                accept ``[N, seq_len]``. The graph paths bake the
                None/not-None identity plus the shape at capture and
                refresh the content through the runner's static buffer on
                every replay. ``None`` (the default) attaches no kwarg at
                all, so existing callers are byte-for-byte unchanged.
            position_ids: Optional ``position_ids`` tensor (AR023, the
                AR019 ``position_embeddings`` threading template applied to
                Dream's left-padded fold context — one
                ``compute_position_ids`` payload per ``(T, prompt_len)``
                window). Threaded through every construction face:
                serial/batched eager (forwarded as a layer kwarg, dropped
                by ``_recompute_all`` for layers without the parameter),
                the CUDA-graph capture (baked by reference into the static
                body), and the adaptive re-capture. The graph path requires
                the SAME tensor object on every replay after capture
                (``is`` identity, mirroring the ``position_embeddings``
                contract); a different object silently falls back to eager.
                ``None`` (the default) attaches no kwarg at all, so existing
                callers are byte-for-byte unchanged.
            parent_branch_ids: Per-row parent identifiers (AR025): row
                ``i`` of a batched call folds against
                ``parent_branch_ids[i]`` — the per-row generalization of
                the shared-``parent_branch_id`` expand contract (each
                parent must be cached at batch 1; a missing or
                length-mismatched parent degrades its row to a full
                recompute). Mutually exclusive with ``parent_branch_id``
                and requires ``branch_ids``. The batched CUDA-graph path
                does not perceive per-row parents (its captured body serves
                one shared parent): a multi-parent call with
                ``use_cuda_graph`` emits a one-time warning and takes the
                batched eager body. ``None`` (the default) attaches no
                kwarg at all, so existing callers are byte-for-byte
                unchanged.

        Returns:
            Output of the language modeling head, typically logits
            (``[N, seq_len, vocab]`` for batched calls).

        Raises:
            ValueError: If ``position_embeddings`` is not ``None`` and is
                not a 2-tuple of tensors whose seq dimension
                (``shape[-2]``) matches ``tokens``' sequence length and
                whose ``cos``/``sin`` head dimensions (``shape[-1]``)
                agree; if ``forced_divergent`` is not ``None`` and is not
                a boolean tensor whose last dimension matches ``tokens``
                sequence length with the serial/batched form above; or if
                ``position_ids`` is not ``None`` and is not an integer
                tensor of shape ``[seq_len]``, ``[1, seq_len]`` or
                ``[batch, seq_len]`` matching ``tokens`` (host-side
                metadata only, no device sync).
        """
        if position_embeddings is not None:
            if (
                not isinstance(position_embeddings, tuple)
                or len(position_embeddings) != 2
                or not isinstance(position_embeddings[0], torch.Tensor)
                or not isinstance(position_embeddings[1], torch.Tensor)
            ):
                raise ValueError(
                    "position_embeddings must be None or a (cos, sin) tuple "
                    f"of two torch.Tensors; got {position_embeddings!r}."
                )
            cos, sin = position_embeddings
            if (
                cos.dim() < 2
                or sin.dim() < 2
                or cos.shape[-2] != sin.shape[-2]
                or cos.shape[-2] != tokens.shape[-1]
                or cos.shape[-1] != sin.shape[-1]
            ):
                raise ValueError(
                    "position_embeddings seq dimension must match tokens "
                    f"({tuple(tokens.shape)}) and cos/sin head dimensions "
                    f"must agree; got cos {tuple(cos.shape)}, "
                    f"sin {tuple(sin.shape)}."
                )
        if forced_divergent is not None:
            if (
                not isinstance(forced_divergent, torch.Tensor)
                or forced_divergent.dtype != torch.bool
            ):
                raise ValueError(
                    "forced_divergent must be None or a boolean torch.Tensor; "
                    f"got dtype {getattr(forced_divergent, 'dtype', None)!r}."
                )
            seq_len = int(tokens.shape[-1])
            if branch_ids is not None:
                expected_forms = f"[{int(tokens.shape[0])}, {seq_len}] for batched calls"
                form_ok = (
                    forced_divergent.dim() == 2
                    and forced_divergent.shape[0] == tokens.shape[0]
                    and forced_divergent.shape[1] == seq_len
                )
            else:
                expected_forms = f"[{seq_len}] or [1, {seq_len}] for serial calls"
                form_ok = tuple(forced_divergent.shape) in (
                    (seq_len,),
                    (1, seq_len),
                )
            if not form_ok or forced_divergent.shape[-1] != seq_len:
                raise ValueError(
                    "forced_divergent must be None or a boolean tensor of "
                    f"shape {expected_forms}; got {tuple(forced_divergent.shape)} "
                    f"for tokens {tuple(tokens.shape)}."
                )
        if position_ids is not None:
            if not isinstance(position_ids, torch.Tensor) or position_ids.dtype not in (
                torch.int64,
                torch.int32,
            ):
                raise ValueError(
                    "position_ids must be None or an integer torch.Tensor; "
                    f"got dtype {getattr(position_ids, 'dtype', None)!r}."
                )
            seq_len = int(tokens.shape[-1])
            batch = int(tokens.shape[0])
            if position_ids.dim() == 1:
                form_ok = position_ids.shape[0] == seq_len
            elif position_ids.dim() == 2:
                form_ok = position_ids.shape[1] == seq_len and position_ids.shape[0] in (
                    1,
                    batch,
                )
            else:
                form_ok = False
            if not form_ok:
                raise ValueError(
                    "position_ids must be None or an integer tensor of shape "
                    f"[{seq_len}], [1, {seq_len}] or [{batch}, {seq_len}]; got "
                    f"{tuple(position_ids.shape)} for tokens {tuple(tokens.shape)}."
                )
        if parent_branch_ids is not None:
            if parent_branch_id is not None:
                raise ValueError(_MULTI_PARENT_MSG)
            if branch_ids is None:
                raise ValueError(
                    "parent_branch_ids (per-row parents) requires branch_ids "
                    "(batched); serial calls use parent_branch_id."
                )
            if len(parent_branch_ids) != len(branch_ids):
                raise ValueError(
                    f"parent_branch_ids length {len(parent_branch_ids)} does "
                    f"not match branch_ids length {len(branch_ids)}."
                )
        if branch_ids is not None:
            if branch_id is not None:
                raise ValueError(
                    "ManualFoldedForward accepts either branch_id (serial) "
                    "or branch_ids (batched), not both."
                )
            # Batched CUDA graph fast path (AR016): mirror the serial
            # dispatch below for the [N, T] capture shape; any miss falls
            # through to the batched eager body (a parent cached at
            # batch != 1 keeps its #41 RuntimeError — the graph layer never
            # swallows or alters the eager contract). AR025: the captured
            # batched body serves ONE shared parent, so a per-row
            # (multi-parent) call never enters the graph path — warn once
            # and take the eager body (the graph face is Out-of-Scope).
            if self.use_cuda_graph and parent_branch_ids is None:
                graph_out = self._graph_forward_batched_or_none(
                    tokens,
                    branch_ids,
                    parent_branch_id,
                    attention_mask,
                    step_idx,
                    position_embeddings,
                    forced_divergent,
                    position_ids,
                )
                if graph_out is not None:
                    return graph_out
        # Opt-in CUDA graph fast path (AR002/T008): replay the captured
        # static folded forward when every precondition holds; any miss falls
        # back to the eager body below.
        elif self.use_cuda_graph and branch_id is not None:
            graph_out = self._graph_forward_or_none(
                tokens,
                branch_id,
                parent_branch_id,
                attention_mask,
                step_idx,
                position_embeddings,
                forced_divergent,
                position_ids,
            )
            if graph_out is not None:
                return graph_out
        emb_fn: Callable[..., Any] = self.profile.embed_module
        x = emb_fn(tokens)

        # AR019: attach the rotary embeddings as a layer kwarg ONLY when
        # supplied — ``None`` adds no kwarg so the eager path stays
        # byte-for-byte identical for existing callers. The wrapped layers'
        # ``**kwargs`` channel carries it to ``_recompute_all``, which
        # filters it against each original layer's signature (real rotary
        # layers receive it; synthetic layers without the parameter drop it).
        pe_kwargs: dict[str, Any] = (
            {"position_embeddings": position_embeddings} if position_embeddings is not None else {}
        )
        # AR021 FR2: same attach-only-when-supplied rule for the forced
        # region; ``None`` adds no kwarg AND no layer op (zero disturbance).
        forced_kwargs: dict[str, Any] = (
            {"forced_divergent": forced_divergent} if forced_divergent is not None else {}
        )
        # AR023: same rule for the position payload (the Dream fold
        # context); ``None`` adds no kwarg so existing eager callers stay
        # byte-for-byte unchanged.
        pos_ids_kwargs: dict[str, Any] = (
            {"position_ids": position_ids} if position_ids is not None else {}
        )
        # AR025: same rule for the per-row parents; ``None`` adds no kwarg
        # so existing eager callers stay byte-for-byte unchanged.
        mp_kwargs: dict[str, Any] = (
            {"parent_branch_ids": list(parent_branch_ids)} if parent_branch_ids is not None else {}
        )
        if parent_branch_ids is not None and self.use_cuda_graph:
            # DD10 defense: the batched graph body perceives only one shared
            # parent; a per-row call degrades to the eager body below with a
            # one-time warning (never an error).
            if not self._multi_parent_graph_warned:
                warnings.warn(
                    "Multi-parent batched calls (parent_branch_ids) do not "
                    "support the CUDA graph path; running the batched eager "
                    "body instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._multi_parent_graph_warned = True
        for wrapped in self._wrapped_layers:
            x = wrapped(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                attention_mask=attention_mask,
                step_idx=step_idx,
                branch_ids=branch_ids,
                **pe_kwargs,
                **forced_kwargs,
                **pos_ids_kwargs,
                **mp_kwargs,
            )
            # Tuple-returning layers (e.g. the real LLaDA remote blocks): the
            # wrapped layer re-wraps its output as ``(hidden, None)`` for the
            # in-place FoldedModel control flow (AGENTS #13); MFF's own eager
            # loop consumes the first element -- the same convention
            # ``_recompute_all`` applies to the original layer outputs.
            if isinstance(x, tuple):
                x = x[0]

        # Apply the final normalization (e.g. LLaMA ``model.norm``) before the
        # head. Skipping it silently biased logits vs. the raw model forward.
        if self.profile.final_norm is not None:
            norm_fn: Callable[[Any], Any] = self.profile.final_norm
            x = norm_fn(x)

        if self.profile.head_module is not None:
            head_fn: Callable[..., Any] = self.profile.head_module
            return head_fn(x)

        warnings.warn(
            "ManualFoldedForward did not discover a language modeling head; "
            "returning hidden states instead of logits.",
            stacklevel=2,
        )
        return x

    def _warn_graph_degraded_once(self, reason: str) -> None:
        """Emit the one-time degradation warning for the graph path."""
        if self._graph_degraded_warned:
            return
        self._graph_degraded_warned = True
        warnings.warn(
            f"ManualFoldedForward CUDA graph path unavailable: {reason}; "
            "falling back to the eager path.",
            UserWarning,
            stacklevel=2,
        )

    def _parent_cache_complete(self, parent_branch_id: str, tokens: Any) -> bool:
        """Check the parent branch has every activation the capture needs.

        Every cached activation must also match ``tokens``' batch/seq shape:
        a parent cached at a different sequence length (e.g. the AR
        ``folded_generate`` growth) cannot be copied into the static buffers,
        so it counts as incomplete and the step stays eager.
        """
        try:
            embedding = self.cache.fetch(branch_id=parent_branch_id, layer_idx=0).get("embedding")
            if embedding is None:
                return False
            if tuple(embedding.shape[:2]) != tuple(tokens.shape):
                return False
            for layer_idx in range(len(self._wrapped_layers)):
                ffn_out = self.cache.fetch(branch_id=parent_branch_id, layer_idx=layer_idx).get(
                    "ffn_out"
                )
                if ffn_out is None:
                    return False
                if tuple(ffn_out.shape[:2]) != tuple(tokens.shape):
                    return False
        except (KeyError, RuntimeError):
            return False
        return True

    def _parent_incomplete_is_structural(self, parent_branch_id: str, tokens: Any) -> bool:
        """Classify an incomplete-parent graph fallback (AR021 FR2).

        Structural (by design, not degradation): the parent exists but was
        cached at a different batch/seq shape — the canonical canvas-growth
        window switch, where the child must run eager prefix folding against
        the shorter parent. Everything else (a missing parent entry, an
        evicted branch) is a degradation fallback.
        """
        try:
            embedding = self.cache.fetch(branch_id=parent_branch_id, layer_idx=0).get("embedding")
        except (KeyError, RuntimeError):
            return False
        if embedding is None:
            return False
        return tuple(embedding.shape[:2]) != tuple(tokens.shape)

    def _count_parent_incomplete_fallback(self, parent_branch_id: str, tokens: Any) -> None:
        """Bump the two-class eager counter for an incomplete-parent miss."""
        if self._parent_incomplete_is_structural(parent_branch_id, tokens):
            self._structural_eager_count += 1
        else:
            self._degradation_fallback_count += 1

    def _install_serial_runner(
        self,
        key: Any,
        runner: Any,
        attention_mask: Any,
        position_embeddings: Any | None,
        position_ids: torch.Tensor | None = None,
    ) -> None:
        """Insert a captured serial runner into the pool (LRU + stats).

        Eviction pops the least-recently-used entry (``OrderedDict`` order;
        hits ``move_to_end`` in the lookup path) and records the eviction on
        the entry's persistent stats. The legacy ``graph_runner`` /
        ``_graph_mask`` / ``_graph_pe`` attributes are re-pointed at the new
        runner on EVERY install and replay, so an evicted runner loses its
        last reference here and its static-buffer VRAM is reclaimed — the
        alias must never pin a dead pool member.
        """
        pool = self._graph_runner_pool
        pool[key] = runner
        pool.move_to_end(key)
        stats = self._graph_runner_pool_stats.setdefault(
            key, {"captures": 0, "replays": 0, "evictions": 0}
        )
        stats["captures"] += 1
        while len(pool) > self.graph_runner_pool_size:
            evicted_key, _evicted = pool.popitem(last=False)
            evicted_stats = self._graph_runner_pool_stats.get(evicted_key)
            if evicted_stats is not None:
                evicted_stats["evictions"] = evicted_stats.get("evictions", 0) + 1
            _clear_cublas_workspaces()
        self.graph_runner = runner
        self._graph_mask = attention_mask
        self._graph_pe = position_embeddings
        self._graph_position_ids = position_ids

    def _graph_forward_or_none(
        self,
        tokens: Any,
        branch_id: str,
        parent_branch_id: str | None,
        attention_mask: Any,
        step_idx: int,
        position_embeddings: Any | None = None,
        forced_divergent: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> Any:
        """Try the CUDA graph fast path; return logits or ``None`` for eager.

        Degradation matrix (design AR002/T008): once a runner exists, a token
        shape other than the captured one warns once and never re-captures;
        scheduler / non-cosine gate / non-CUDA warn once; a mask other than
        the one pinned at capture silently uses the eager path; an incomplete
        parent cache silently uses eager without disabling future captures;
        an exceeded divergent budget discards the replay output (one-time
        warning) and recomputes eagerly; a failed capture permanently
        disables the batched graph path (EX-002, one-time warning, no
        retry) without touching the serial one. With
        ``graph_adaptive_capacity`` on (AR017), two observation hooks
        bracket the outcome: the budget-exceeded branch runs the growth
        evaluation before the eager fallback, and a successful publish
        runs the shrink evaluation after the row-sliced publishes (the
        capacity swap never precedes the publish, which consumes the old
        runner's static buffers). AR019: ``position_embeddings`` mirrors
        the mask contract — baked by reference at capture (a non-CUDA
        tuple silently refuses the capture) and required to be the SAME
        object on every replay; any mismatch silently uses eager. AR021
        FR2: ``forced_divergent`` bakes the None/not-None identity plus
        the shape at capture and refreshes the CONTENT through the
        runner's static buffer on every replay (a pure device-to-device
        copy outside the captured body); a None/not-None mismatch on
        either side silently uses eager with no re-capture. AR021 FR2
        pool: with ``graph_runner_pool_size > 0`` the warn-once
        shape/mask pins are replaced by a shape+mask-identity keyed LRU
        pool — the pinned fold-context mask makes the key semantically
        ``(T, prompt_len)``: a new window captures a fresh pool runner
        (zero UserWarning), a same-window call replays, and the capacity
        evicts the least-recently-used runner with its VRAM reclaimed.
        Every graph-path eager fallback bumps exactly one of the two
        instance-cumulative counters: ``_structural_eager_count`` for the
        by-design misses (parent-None populate, parent-at-old-shape
        window switch), ``_degradation_fallback_count`` for everything
        else.
        """
        from actfold.core.similarity_gate import SimilarityGate

        if self._graph_capture_failed:
            # EX-002: a failed capture permanently disables the graph path
            # (the failure already warned once); never retry — capture is
            # expensive and its failure mode is not transient.
            self._degradation_fallback_count += 1
            return None
        pool_on = self.graph_runner_pool_size > 0
        key: Any = None
        runner: Any = None
        if pool_on:
            # The pool key is (tokens shape, attention-mask identity): the
            # fold context pins one mask object per (T, prompt_len) window,
            # so the key IS the design's shape window (same window -> same
            # mask object -> same key; same T with different padding -> a
            # different mask object -> a different key). The pool holds the
            # mask reference, so the id inside the key stays valid.
            key = (tuple(tokens.shape), attention_mask)
            runner = self._graph_runner_pool.get(key)
            if runner is not None:
                self._graph_runner_pool.move_to_end(key)
        else:
            runner = self.graph_runner
            if runner is not None and tuple(tokens.shape) != tuple(runner.tokens_static.shape):
                if not self._graph_shape_warned:
                    self._graph_shape_warned = True
                    warnings.warn(
                        "ManualFoldedForward graph path captured shape "
                        f"{tuple(runner.tokens_static.shape)}; tokens of shape "
                        f"{tuple(tokens.shape)} fall back to eager (no re-capture).",
                        UserWarning,
                        stacklevel=2,
                    )
                self._degradation_fallback_count += 1
                return None
        if parent_branch_id is None:
            # Parent passes always run eager: they populate the cache.
            self._structural_eager_count += 1
            return None
        if self.scheduler is not None:
            self._degradation_fallback_count += 1
            self._warn_graph_degraded_once("a folding scheduler is attached")
            return None
        if type(self.gate) is not SimilarityGate or self.gate.metric != "cosine":
            self._degradation_fallback_count += 1
            self._warn_graph_degraded_once("the gate is not an exact-type cosine SimilarityGate")
            return None
        if not tokens.is_cuda:
            self._degradation_fallback_count += 1
            self._warn_graph_degraded_once("CUDA is unavailable")
            return None

        fresh_capture = False
        if runner is None:
            # First graph-eligible call for this key: capture once. An
            # incomplete parent cache is a normal transient (e.g. after
            # eviction), so it falls back silently without disabling future
            # captures.
            if not self._parent_cache_complete(parent_branch_id, tokens):
                self._count_parent_incomplete_fallback(parent_branch_id, tokens)
                return None
            if attention_mask is not None and not attention_mask.is_cuda:
                self._degradation_fallback_count += 1
                return None
            if position_embeddings is not None and not position_embeddings[0].is_cuda:
                self._degradation_fallback_count += 1
                return None
            if position_ids is not None and not position_ids.is_cuda:
                self._degradation_fallback_count += 1
                return None
            if forced_divergent is not None and not forced_divergent.is_cuda:
                self._degradation_fallback_count += 1
                return None
            from actfold.core.cuda_graph import FoldedGraphRunner, safe_capture

            new_runner = FoldedGraphRunner(
                wrapped_layers=self._wrapped_layers,
                embed_fn=self.profile.embed_module,
                final_norm_fn=self.profile.final_norm,
                head_fn=self.profile.head_module,
                cache=self.cache,
                gate_tau=self.gate.tau,
                gate_eps=self.gate.eps,
                capacity_ratio=self.graph_capacity_ratio,
                attention_mask_static=attention_mask,
                position_embeddings_static=position_embeddings,
                forced_divergent_static=forced_divergent,
                position_ids_static=position_ids,
            )
            try:
                safe_capture(
                    new_runner,
                    "serial",
                    tokens,
                    branch_id,
                    parent_branch_id,
                    forced_divergent=forced_divergent,
                )
            except RuntimeError as exc:
                self._degradation_fallback_count += 1
                self._warn_graph_degraded_once(f"CUDA graph capture failed ({exc})")
                self._graph_capture_failed = True
                return None
            runner = new_runner
            fresh_capture = True
            if pool_on:
                self._install_serial_runner(
                    key, runner, attention_mask, position_embeddings, position_ids
                )
            else:
                self.graph_runner = new_runner
                self._graph_mask = attention_mask
                self._graph_pe = position_embeddings
                self._graph_position_ids = position_ids
        else:
            if pool_on:
                # Mask identity is guaranteed by the pool key; the rotary
                # tuple is checked per-runner against the captured object
                # (a same-key call with a different pe object is a
                # degradation, never a re-capture).
                if position_embeddings is not getattr(runner, "_position_embeddings_static", None):
                    self._degradation_fallback_count += 1
                    return None
                if position_ids is not getattr(runner, "_position_ids_static", None):
                    self._degradation_fallback_count += 1
                    return None
            elif (
                attention_mask is not self._graph_mask
                or position_embeddings is not self._graph_pe
                or position_ids is not self._graph_position_ids
            ):
                # The captured graph bakes the attention mask and the rotary
                # embeddings in; a different object must not silently reuse
                # either (AR019 mirrors the mask identity contract for pe).
                self._degradation_fallback_count += 1
                return None
            if (runner.forced_divergent_static is None) != (forced_divergent is None):
                # Forced-identity mismatch: the captured body either has a
                # forced AND that this call cannot feed or lacks one that
                # this call requests. Silent eager both directions — never a
                # re-capture (the runner stays valid for its own identity).
                self._degradation_fallback_count += 1
                return None

        logits = runner.replay(
            tokens, parent_branch_id, branch_id, forced_divergent=forced_divergent
        )
        if logits is None:
            # Parent cache incomplete for this step: plain eager fallback.
            self._count_parent_incomplete_fallback(parent_branch_id, tokens)
            return None
        if pool_on:
            stats = self._graph_runner_pool_stats.setdefault(
                key, {"captures": 0, "replays": 0, "evictions": 0}
            )
            if not fresh_capture:
                stats["replays"] += 1
            # Keep the legacy aliases on the ACTIVE pool member so an
            # evicted runner is never pinned past its eviction.
            self.graph_runner = runner
            self._graph_mask = attention_mask
            self._graph_pe = position_embeddings
            self._graph_position_ids = position_ids
        if runner.budget_exceeded:
            self._degradation_fallback_count += 1
            if not self._graph_budget_warned:
                self._graph_budget_warned = True
                warnings.warn(
                    "CUDA graph replay exceeded the divergent budget; "
                    "discarding the replay output and recomputing eagerly "
                    "(correctness first).",
                    UserWarning,
                    stacklevel=2,
                )
            return None
        self._publish_graph_child(runner, tokens, branch_id, parent_branch_id, step_idx)
        return logits.clone()

    def _publish_graph_child(
        self,
        runner: Any,
        tokens: Any,
        branch_id: str,
        parent_branch_id: str,
        step_idx: int,
    ) -> None:
        """Publish a validated replay into the cache and the profiler.

        Mirrors what the eager path stores: per-layer ``ffn_out`` (plus the
        layer-0 ``embedding`` = embed(tokens)) cloned from the runner's static
        buffers, and per-layer stability records derived from the static mask
        buffers (device-side sums, no extra host readback).
        """
        from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

        embedding = self.profile.embed_module(tokens)
        num_layers = len(self._wrapped_layers)
        for layer_idx in range(num_layers):
            activations: dict[str, Any] = {"ffn_out": runner.child_buf[layer_idx].clone()}
            if layer_idx == 0:
                activations["embedding"] = embedding
            self.cache.put(
                branch_id=branch_id,
                layer_idx=layer_idx,
                activations=activations,
            )
            GLOBAL_STABILITY_PROFILER.record(
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                layer_idx=layer_idx,
                step_idx=step_idx,
                stable_mask=runner.mask_buf[layer_idx],
                tau=self.gate.tau,
                metric="cosine",
            )

    def _install_batched_runner(
        self,
        key: Any,
        runner: Any,
        attention_mask: Any,
        position_embeddings: Any | None,
        position_ids: torch.Tensor | None = None,
    ) -> None:
        """Insert a captured batched runner into the pool (LRU + stats).

        Batched mirror of :meth:`_install_serial_runner`: LRU eviction with
        persistent per-key stats and legacy-alias refresh so evicted runners
        lose their last reference (VRAM conservation).
        """
        pool = self._graph_runner_batched_pool
        pool[key] = runner
        pool.move_to_end(key)
        stats = self._graph_runner_batched_pool_stats.setdefault(
            key, {"captures": 0, "replays": 0, "evictions": 0}
        )
        stats["captures"] += 1
        while len(pool) > self.graph_runner_pool_size:
            evicted_key, _evicted = pool.popitem(last=False)
            evicted_stats = self._graph_runner_batched_pool_stats.get(evicted_key)
            if evicted_stats is not None:
                evicted_stats["evictions"] = evicted_stats.get("evictions", 0) + 1
            _clear_cublas_workspaces()
        self.graph_runner_batched = runner
        self._graph_batched_mask = attention_mask
        self._graph_batched_pe = position_embeddings
        self._graph_batched_position_ids = position_ids

    def _graph_forward_batched_or_none(
        self,
        tokens: Any,
        branch_ids: Sequence[str],
        parent_branch_id: str | None,
        attention_mask: Any,
        step_idx: int,
        position_embeddings: Any | None = None,
        forced_divergent: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> Any:
        """Try the batched CUDA graph fast path; return logits or ``None``.

        Degradation matrix (AR016, mirrors the serial :meth:`_graph_forward_or_none`
        with batched differences): a token shape other than the captured one warns
        once (batched shape flag) and never re-captures; scheduler / non-cosine
        gate / non-CUDA warn once through the shared family flag; a mask other
        than the one pinned at capture silently uses the eager path; an
        incomplete parent cache, a sequence-length mismatch, or a parent cached
        at batch != 1 silently refuses **before every replay** (host-side shape
        metadata only — no device sync): the first two fall back to eager which
        succeeds, the last falls back to eager which raises the #41
        ``RuntimeError`` before any store (zero cache/profiler pollution — the
        graph layer never swallows the eager contract); an exceeded divergent
        budget discards the replay output (one-time warning) and recomputes
        eagerly; a failed capture permanently disables the batched graph path
        (EX-002, one-time warning, no retry) without touching the serial one.
        AR021 FR2: ``forced_divergent`` (``[N, seq]``) bakes the
        None/not-None identity at capture and refreshes the content on every
        replay; an identity mismatch silently uses eager. AR021 FR2 pool:
        with ``graph_runner_pool_size > 0`` the shape/mask pins are replaced
        by the (shape, mask-identity) keyed LRU pool — semantically
        ``(N, T, prompt_len)`` — with per-key stats and VRAM-conserving
        eviction; every eager fallback bumps the two-class counters exactly
        as on the serial side.
        """
        from actfold.core.similarity_gate import SimilarityGate

        if self._graph_batched_capture_failed:
            # EX-002: a failed capture permanently disables the batched graph
            # path; never retry — capture is expensive and its failure mode
            # is not transient.
            self._degradation_fallback_count += 1
            return None
        pool_on = self.graph_runner_pool_size > 0
        key: Any = None
        runner: Any = None
        if pool_on:
            key = (tuple(tokens.shape), attention_mask)
            runner = self._graph_runner_batched_pool.get(key)
            if runner is not None:
                self._graph_runner_batched_pool.move_to_end(key)
        else:
            runner = self.graph_runner_batched
            if runner is not None and tuple(tokens.shape) != tuple(runner.tokens_static.shape):
                if not self._graph_batched_shape_warned:
                    self._graph_batched_shape_warned = True
                    warnings.warn(
                        "ManualFoldedForward batched graph path captured shape "
                        f"{tuple(runner.tokens_static.shape)}; tokens of shape "
                        f"{tuple(tokens.shape)} fall back to eager (no re-capture).",
                        UserWarning,
                        stacklevel=2,
                    )
                self._degradation_fallback_count += 1
                return None
        if parent_branch_id is None:
            # Parent passes always run eager: they populate the cache.
            self._structural_eager_count += 1
            return None
        if self.scheduler is not None:
            self._degradation_fallback_count += 1
            self._warn_graph_degraded_once("a folding scheduler is attached")
            return None
        if type(self.gate) is not SimilarityGate or self.gate.metric != "cosine":
            self._degradation_fallback_count += 1
            self._warn_graph_degraded_once("the gate is not an exact-type cosine SimilarityGate")
            return None
        if not tokens.is_cuda:
            self._degradation_fallback_count += 1
            self._warn_graph_degraded_once("CUDA is unavailable")
            return None
        if len(branch_ids) != tokens.shape[0]:
            self._degradation_fallback_count += 1
            return None

        fresh_capture = False
        if runner is None:
            # First batched graph-eligible call for this key: capture once. A
            # parent cached at batch != 1 or a sequence-length mismatch is
            # refused silently here (the eager body decides the outcome — the
            # #41 contract for batch != 1).
            if not self._parent_cache_complete_batched(parent_branch_id, tokens):
                self._count_parent_incomplete_fallback(parent_branch_id, tokens)
                return None
            if attention_mask is not None and not attention_mask.is_cuda:
                self._degradation_fallback_count += 1
                return None
            if position_embeddings is not None and not position_embeddings[0].is_cuda:
                self._degradation_fallback_count += 1
                return None
            if position_ids is not None and not position_ids.is_cuda:
                self._degradation_fallback_count += 1
                return None
            if forced_divergent is not None and not forced_divergent.is_cuda:
                self._degradation_fallback_count += 1
                return None
            from actfold.core.cuda_graph import FoldedGraphRunner, safe_capture

            new_runner = FoldedGraphRunner(
                wrapped_layers=self._wrapped_layers,
                embed_fn=self.profile.embed_module,
                final_norm_fn=self.profile.final_norm,
                head_fn=self.profile.head_module,
                cache=self.cache,
                gate_tau=self.gate.tau,
                gate_eps=self.gate.eps,
                capacity_ratio=self.graph_capacity_ratio,
                attention_mask_static=attention_mask,
                position_embeddings_static=position_embeddings,
                forced_divergent_static=forced_divergent,
                position_ids_static=position_ids,
            )
            try:
                safe_capture(
                    new_runner,
                    "batched",
                    tokens,
                    branch_ids[0],
                    parent_branch_id,
                    forced_divergent=forced_divergent,
                )
            except RuntimeError as exc:
                self._degradation_fallback_count += 1
                self._warn_graph_degraded_once(f"CUDA graph capture failed ({exc})")
                self._graph_batched_capture_failed = True
                return None
            runner = new_runner
            fresh_capture = True
            if pool_on:
                self._install_batched_runner(
                    key, runner, attention_mask, position_embeddings, position_ids
                )
            else:
                self.graph_runner_batched = new_runner
                self._graph_batched_mask = attention_mask
                self._graph_batched_pe = position_embeddings
                self._graph_batched_position_ids = position_ids
        else:
            if pool_on:
                if position_embeddings is not getattr(runner, "_position_embeddings_static", None):
                    self._degradation_fallback_count += 1
                    return None
                if position_ids is not getattr(runner, "_position_ids_static", None):
                    self._degradation_fallback_count += 1
                    return None
            elif (
                attention_mask is not self._graph_batched_mask
                or position_embeddings is not self._graph_batched_pe
                or position_ids is not self._graph_batched_position_ids
            ):
                # The captured graph bakes the attention mask and the rotary
                # embeddings in; a different object must not silently reuse
                # either (AR019 mirrors the mask identity contract for pe).
                self._degradation_fallback_count += 1
                return None
            if (runner.forced_divergent_static is None) != (forced_divergent is None):
                # Forced-identity mismatch: silent eager both directions,
                # never a re-capture (AR021 FR2).
                self._degradation_fallback_count += 1
                return None

        # Replay-time parent re-validation (AR016 Major-1): re-check before
        # EVERY replay. A parent re-cached at [N, T, H] after capture would
        # copy cleanly into the [N, T, H] static parent buffers (each branch
        # folding against a different parent row) with the budget check none
        # the wiser — refuse here so the eager body raises the #41
        # RuntimeError before any store. Shape checks are host-side metadata
        # (no device-to-host sync).
        if not self._parent_cache_complete_batched(parent_branch_id, tokens):
            self._count_parent_incomplete_fallback(parent_branch_id, tokens)
            return None
        logits = runner.replay(
            tokens, parent_branch_id, branch_ids[0], forced_divergent=forced_divergent
        )
        if logits is None:
            # Parent cache incomplete for this step: plain eager fallback.
            self._count_parent_incomplete_fallback(parent_branch_id, tokens)
            return None
        if pool_on:
            stats = self._graph_runner_batched_pool_stats.setdefault(
                key, {"captures": 0, "replays": 0, "evictions": 0}
            )
            if not fresh_capture:
                stats["replays"] += 1
            self.graph_runner_batched = runner
            self._graph_batched_mask = attention_mask
            self._graph_batched_pe = position_embeddings
            self._graph_batched_position_ids = position_ids
        if runner.budget_exceeded:
            self._degradation_fallback_count += 1
            if not self._graph_batched_budget_warned:
                self._graph_batched_budget_warned = True
                warnings.warn(
                    "Batched CUDA graph replay exceeded the divergent budget; "
                    "discarding the replay output and recomputing eagerly "
                    "(correctness first).",
                    UserWarning,
                    stacklevel=2,
                )
            # Growth hook (AR017): the violation is the growth signal, so the
            # observation must happen inside the exceeded branch -- after the
            # one-time warning, before the eager fallback.
            self._observe_adaptive_capacity(
                runner,
                tokens,
                branch_ids,
                parent_branch_id,
                attention_mask,
                exceeded=True,
                position_embeddings=position_embeddings,
                forced_divergent=forced_divergent,
                position_ids=position_ids,
            )
            return None
        self._publish_graph_children(runner, tokens, branch_ids, parent_branch_id, step_idx)
        # Shrink hook (AR017): placed after the publish -- the swap must
        # never precede it (the publish reads the old runner's static
        # buffers).
        self._observe_adaptive_capacity(
            runner,
            tokens,
            branch_ids,
            parent_branch_id,
            attention_mask,
            exceeded=False,
            position_embeddings=position_embeddings,
            forced_divergent=forced_divergent,
            position_ids=position_ids,
        )
        return logits.clone()

    def _parent_cache_complete_batched(self, parent_branch_id: str, tokens: Any) -> bool:
        """Check the parent branch has every batched capture/replay input.

        Mirrors :meth:`_parent_cache_complete` with the expand contract
        (#41): the expected parent shape is ``[1, seq]`` — a parent cached at
        any other batch (including ``N``) or sequence length is refused, so
        the distinction between "incomplete", "seq mismatch", and "batch
        != 1" is delegated to the eager body (the first two succeed there,
        the last raises the #41 RuntimeError before any store).
        """
        expected = (1, tokens.shape[1])
        try:
            embedding = self.cache.fetch(branch_id=parent_branch_id, layer_idx=0).get("embedding")
            if embedding is None:
                return False
            if tuple(embedding.shape[:2]) != expected:
                return False
            for layer_idx in range(len(self._wrapped_layers)):
                ffn_out = self.cache.fetch(branch_id=parent_branch_id, layer_idx=layer_idx).get(
                    "ffn_out"
                )
                if ffn_out is None:
                    return False
                if tuple(ffn_out.shape[:2]) != expected:
                    return False
        except (KeyError, RuntimeError):
            return False
        return True

    def _publish_graph_children(
        self,
        runner: Any,
        tokens: Any,
        branch_ids: Sequence[str],
        parent_branch_id: str,
        step_idx: int,
    ) -> None:
        """Publish a validated batched replay into the cache and the profiler.

        Row-sliced mirror of :meth:`_publish_graph_child`: each branch ``i``
        receives the ``[1, T, H]`` slice of the static child buffer plus the
        layer-0 embedding row, and a per-layer stability record. AR018 host-cost
        compression: (1) the whole ``[L, N, T, H]`` child buffer is cloned ONCE
        into a fresh snapshot -- ``cache.put`` copies into its own storage, so
        the snapshot's row views are safe against static-buffer reuse and
        against by-reference caches alike (the snapshot is private and never
        written by this side after the publish); (2) all (layer, row) stability
        sums come from ONE ``mask_buf.sum(dim=2)`` reduction recorded through
        ``StabilityProfiler.record_precomputed`` (0-dim views, zero extra
        launches, zero host readback). The debug path
        (``profiler.debug_enabled``) falls back to the per-row ``record``
        entry, which is the only one that can produce ``divergence_positions``.
        AR020: a ``BatchedPutCapable`` cache receives all L*N entries through
        one ``put_batch`` call whose payloads mirror the per-entry ``put``
        payloads field-for-field (the entry's ``step_idx`` keeps the loop
        variant's default — the folded stack stores every activation at step
        0); incapable caches keep the original double loop byte-for-byte.
        """
        from actfold.core.cache_protocol import BatchedPutCapable, CachePutEntry
        from actfold.profiler.stability_profiler import GLOBAL_STABILITY_PROFILER

        embedding = self.profile.embed_module(tokens)
        num_layers = len(self._wrapped_layers)
        # FR2: one fresh snapshot clone of the whole child buffer replaces the
        # per-row clones (cache.put copies, so the intermediate per-row clone
        # was pure overhead).
        snapshot = runner.child_buf.clone()
        # FR1: one batched reduction serves every (layer, row) stability count.
        profiler = GLOBAL_STABILITY_PROFILER
        debug_path = profiler.enabled and profiler.debug_enabled
        sums: Any = None
        if profiler.enabled and not debug_path:
            sums = runner.mask_buf.sum(dim=2)  # [L, N], int64
        seq_len = int(tokens.shape[1])
        # AR020 batched dispatch: probe the capability on EVERY publish — the
        # result is never cached on the instance (the cache attribute is
        # swappable and the A/B benchmark relies on re-probing after a swap).
        # The probe is an O(1) attribute-existence check, zero device sync.
        batched = isinstance(self.cache, BatchedPutCapable)
        if batched:
            entries: list[CachePutEntry] = []
            for layer_idx in range(num_layers):
                for row, branch_id in enumerate(branch_ids):
                    activations: dict[str, Any] = {"ffn_out": snapshot[layer_idx][row : row + 1]}
                    if layer_idx == 0:
                        activations["embedding"] = embedding[row : row + 1]
                    entries.append(
                        CachePutEntry(
                            branch_id=branch_id,
                            layer_idx=layer_idx,
                            activations=activations,
                        )
                    )
            # Exceptions propagate unchanged (no swallowing, no re-grading).
            self.cache.put_batch(entries)
        for layer_idx in range(num_layers):
            for row, branch_id in enumerate(branch_ids):
                if not batched:
                    activations = {"ffn_out": snapshot[layer_idx][row : row + 1]}
                    if layer_idx == 0:
                        activations["embedding"] = embedding[row : row + 1]
                    self.cache.put(
                        branch_id=branch_id,
                        layer_idx=layer_idx,
                        activations=activations,
                    )
                if debug_path:
                    # Debug positions need the mask itself: per-row fallback to
                    # the original entry (correctness over launch count).
                    profiler.record(
                        branch_id=branch_id,
                        parent_branch_id=parent_branch_id,
                        layer_idx=layer_idx,
                        step_idx=step_idx,
                        stable_mask=runner.mask_buf[layer_idx][row : row + 1],
                        tau=self.gate.tau,
                        metric="cosine",
                    )
                elif sums is not None:
                    profiler.record_precomputed(
                        branch_id=branch_id,
                        parent_branch_id=parent_branch_id,
                        layer_idx=layer_idx,
                        step_idx=step_idx,
                        stable_sum=sums[layer_idx][row],
                        num_tokens=seq_len,
                        tau=self.gate.tau,
                        metric="cosine",
                    )

    def _observe_adaptive_capacity(
        self,
        runner: Any,
        tokens: Any,
        branch_ids: Sequence[str],
        parent_branch_id: str,
        attention_mask: Any,
        exceeded: bool,
        position_embeddings: Any | None = None,
        forced_divergent: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> None:
        """Run the adaptive-capacity policy hook (AR017; no-op when off).

        Called from exactly two points in
        :meth:`_graph_forward_batched_or_none`: inside the budget-exceeded
        branch (``exceeded=True`` -- the growth evaluation must see the
        violation) and after a successful publish (``exceeded=False`` -- the
        shrink evaluation). Steps whose replay returned ``None`` (incomplete
        parent cache) never reach this hook, so the staged divergences are
        always fresh here. A ``"recapture"`` verdict re-captures through
        :meth:`_recapture_batched_runner`; a ``"retired"`` verdict warns
        exactly once (the capacity stays fixed from here on). AR019: the
        caller's ``position_embeddings`` is threaded into the re-capture;
        AR021 FR2: so is the caller's ``forced_divergent``; both hooks fire
        only on replay paths where the pe/forced identity checks have
        already passed, so the re-capture always receives the pinned
        objects.
        """
        policy = self._graph_adaptive_policy
        if policy is None:
            return
        divergences = runner.last_layer_divergences
        if divergences is None:
            # Defensive: unreachable through the two call sites (replay ran
            # validate_budgets before either hook); never consume a stale
            # staging.
            return
        num_tokens = int(tokens.shape[0]) * int(tokens.shape[1])
        action = policy.observe(max(divergences), exceeded, int(runner.capacity), num_tokens)
        if action.kind == "recapture" and action.target_capacity is not None:
            self._recapture_batched_runner(
                action.target_capacity,
                int(runner.capacity),
                tokens,
                branch_ids,
                parent_branch_id,
                attention_mask,
                position_embeddings=position_embeddings,
                forced_divergent=forced_divergent,
                position_ids=position_ids,
            )
        elif action.kind == "retired":
            if not self._graph_adaptive_retire_warned:
                self._graph_adaptive_retire_warned = True
                warnings.warn(
                    "Adaptive graph capacity reached the max_recaptures cap; "
                    "the capacity is now fixed at the current value.",
                    UserWarning,
                    stacklevel=2,
                )

    def _recapture_batched_runner(
        self,
        target_capacity: int,
        current_capacity: int,
        tokens: Any,
        branch_ids: Sequence[str],
        parent_branch_id: str,
        attention_mask: Any,
        position_embeddings: Any | None = None,
        forced_divergent: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> None:
        """Re-capture the batched runner at a new capacity (AR017).

        Capture-new-first, swap-on-success, drop-old-last: a failed
        re-capture keeps the old runner in service and retires only the
        adaptive path. ``_graph_batched_capture_failed`` is deliberately
        untouched -- its semantics disable the whole batched graph path and
        would discard the healthy old runner (req Major-3); the adaptive
        retirement flag is an independent, weaker degradation.

        The decision uses the capacity the new ratio will actually produce
        (``ceil(ratio * N * T)``), not the policy's target: the IEEE product
        ``target / n * n`` only rounds up (the ceil lands in
        ``{target, target + 1}``), so a shrink at ``target == C - 1`` may
        collapse back to the current capacity -- skip instead of paying for
        a pointless capture. Exact for power-of-two token totals (design
        SS4.4; all in-scope graph geometries). AR019: the re-captured
        runner bakes the caller's pinned ``position_embeddings`` by
        reference and re-pins it on success (face 4 of the pe threading).
        AR021 FR2: the caller's ``forced_divergent`` is baked with the same
        contract, and a pool-mode re-capture replaces the pool entry for
        the active key (the evicted old runner loses its pool reference —
        the legacy alias never pins a dead member).
        """
        num_tokens = int(tokens.shape[0]) * int(tokens.shape[1])
        new_ratio = min(1.0, target_capacity / num_tokens)
        prospective = max(1, min(math.ceil(new_ratio * num_tokens), num_tokens))
        if prospective == current_capacity:
            return
        from actfold.core.cuda_graph import FoldedGraphRunner, safe_capture

        new_runner = FoldedGraphRunner(
            wrapped_layers=self._wrapped_layers,
            embed_fn=self.profile.embed_module,
            final_norm_fn=self.profile.final_norm,
            head_fn=self.profile.head_module,
            cache=self.cache,
            gate_tau=self.gate.tau,
            gate_eps=self.gate.eps,
            capacity_ratio=new_ratio,
            attention_mask_static=attention_mask,
            position_embeddings_static=position_embeddings,
            forced_divergent_static=forced_divergent,
            position_ids_static=position_ids,
        )
        try:
            safe_capture(
                new_runner,
                "adaptive",
                tokens,
                branch_ids[0],
                parent_branch_id,
                forced_divergent=forced_divergent,
            )
        except RuntimeError as exc:
            if not self._graph_adaptive_fail_warned:
                self._graph_adaptive_fail_warned = True
                warnings.warn(
                    f"Adaptive capacity re-capture failed ({exc}); adaptive "
                    "capacity retired, keeping the current graph.",
                    UserWarning,
                    stacklevel=2,
                )
            policy = self._graph_adaptive_policy
            if policy is not None:
                policy.retire()
            return
        if self.graph_runner_pool_size > 0:
            key = (tuple(tokens.shape), attention_mask)
            self._install_batched_runner(
                key, new_runner, attention_mask, position_embeddings, position_ids
            )
        else:
            self.graph_runner_batched = new_runner
            self._graph_batched_mask = attention_mask
            self._graph_batched_pe = position_embeddings
            self._graph_batched_position_ids = position_ids
        policy = self._graph_adaptive_policy
        if policy is not None:
            policy.note_recaptured()

    @property
    def folding_applied(self) -> bool:
        """Return ``True`` because layers are explicitly wrapped."""
        return True

    @property
    def graph_pool_stats(self) -> dict[str, Any]:
        """Shape-keyed graph runner pool statistics (AR021 FR2 probe).

        Schema: ``{"serial": {"pool_size": int, <pool key>: {"captures",
        "replays", "evictions"}}, "batched": {same layout},
        "degradation_fallback_count": int, "structural_eager_count": int}``.
        The per-key stats are instance-cumulative and persist after an LRU
        eviction (the evicted entry's ``evictions`` count stays readable);
        the two eager counters count every graph-path eager fallback in the
        two-class split (degradation vs structural). Pool sections carry
        only ``pool_size`` while the pool is off (no runner is ever pooled).
        AR026 adds the process-global capture-safety probe keys:
        ``capture_diagnostics`` (the first recorded failure records from
        ``safe_capture`` — copies, capped) and
        ``capture_diagnostics_overflow`` (failures past the cap).
        """
        from actfold.core.cuda_graph import (
            capture_diagnostics_overflow,
            capture_diagnostics_snapshot,
        )

        serial: dict[str, Any] = {"pool_size": len(self._graph_runner_pool)}
        for pool_key, entry in self._graph_runner_pool_stats.items():
            serial[pool_key] = dict(entry)
        batched: dict[str, Any] = {"pool_size": len(self._graph_runner_batched_pool)}
        for pool_key, entry in self._graph_runner_batched_pool_stats.items():
            batched[pool_key] = dict(entry)
        return {
            "serial": serial,
            "batched": batched,
            "degradation_fallback_count": self._degradation_fallback_count,
            "structural_eager_count": self._structural_eager_count,
            "capture_diagnostics": capture_diagnostics_snapshot(),
            "capture_diagnostics_overflow": capture_diagnostics_overflow(),
        }

    @property
    def graph_adaptive_policy(self) -> Any:
        """The adaptive-capacity policy instance (AR017 public probe).

        ``None`` when ``graph_adaptive_capacity`` is off (no policy state is
        ever built); otherwise the live
        :class:`~actfold.core.adaptive_capacity.AdaptiveCapacityPolicy`
        whose ``events`` list records successful re-captures and the
        retirement.
        """
        return self._graph_adaptive_policy
