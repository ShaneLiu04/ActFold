"""Architecture detection and component extraction for Hugging Face models.

This module provides helpers that discover the embedding module, Transformer
layer stack, and language modeling head for a wide range of model families.
It is used by the demo and by generic folding wrappers so that Branch Folding
works without architecture-specific wiring.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import torch.nn as nn

_T = TypeVar("_T", bound=nn.Module)


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
    ) -> None:
        super().__init__()
        if not 0.0 < graph_capacity_ratio <= 1.0:
            raise ValueError(
                "graph_capacity_ratio must satisfy 0 < ratio <= 1, got "
                f"{graph_capacity_ratio}"
            )
        self.profile = detect_architecture(model)
        self.cache = cache
        self.gate = gate
        self.scheduler = scheduler
        self.split_layers = bool(split_layers)
        self.split_min_tokens = int(split_min_tokens)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph_capacity_ratio = float(graph_capacity_ratio)
        # Lazily created by the CUDA graph path (AR002/T007); ``None`` on
        # the eager path.
        self.graph_runner: Any = None
        self._graph_mask: Any = None
        self._graph_degraded_warned = False
        self._graph_shape_warned = False
        self._graph_budget_warned = False
        self._graph_capture_failed = False
        self._wrapped_layers = nn.ModuleList(
            [self._wrap_layer(layer, idx) for idx, layer in enumerate(self.profile.layers)]
        )

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
            )
        from actfold.core.folded_transformer import FoldedTransformerLayer

        return FoldedTransformerLayer(
            original_layer=layer,
            cache=self.cache,
            gate=self.gate,
            layer_idx=idx,
            scheduler=self.scheduler,
        )

    def forward(
        self,
        tokens: Any,
        branch_id: str,
        parent_branch_id: str | None = None,
        attention_mask: Any | None = None,
        step_idx: int = 0,
    ) -> Any:
        """Run a folded forward pass and return logits or hidden states.

        Args:
            tokens: Input token ids ``[batch, seq_len]``.
            branch_id: Identifier of the current branch.
            parent_branch_id: Optional parent branch identifier for reuse.
            attention_mask: Optional attention mask.
            step_idx: Current diffusion step index.

        Returns:
            Output of the language modeling head, typically logits.
        """
        # Opt-in CUDA graph fast path (AR002/T008): replay the captured
        # static folded forward when every precondition holds; any miss falls
        # back to the eager body below.
        if self.use_cuda_graph:
            graph_out = self._graph_forward_or_none(
                tokens, branch_id, parent_branch_id, attention_mask, step_idx
            )
            if graph_out is not None:
                return graph_out
        emb_fn: Callable[..., Any] = self.profile.embed_module
        x = emb_fn(tokens)

        for wrapped in self._wrapped_layers:
            x = wrapped(
                x,
                branch_id=branch_id,
                parent_branch_id=parent_branch_id,
                attention_mask=attention_mask,
                step_idx=step_idx,
            )

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
            embedding = self.cache.fetch(branch_id=parent_branch_id, layer_idx=0).get(
                "embedding"
            )
            if embedding is None:
                return False
            if tuple(embedding.shape[:2]) != tuple(tokens.shape):
                return False
            for layer_idx in range(len(self._wrapped_layers)):
                ffn_out = self.cache.fetch(
                    branch_id=parent_branch_id, layer_idx=layer_idx
                ).get("ffn_out")
                if ffn_out is None:
                    return False
                if tuple(ffn_out.shape[:2]) != tuple(tokens.shape):
                    return False
        except (KeyError, RuntimeError):
            return False
        return True

    def _graph_forward_or_none(
        self,
        tokens: Any,
        branch_id: str,
        parent_branch_id: str | None,
        attention_mask: Any,
        step_idx: int,
    ) -> Any:
        """Try the CUDA graph fast path; return logits or ``None`` for eager.

        Degradation matrix (design AR002/T008): once a runner exists, a token
        shape other than the captured one warns once and never re-captures;
        scheduler / non-cosine gate / non-CUDA warn once; a mask other than
        the one pinned at capture silently uses the eager path; an incomplete
        parent cache silently uses eager without disabling future captures;
        an exceeded divergent budget discards the replay output (one-time
        warning) and recomputes eagerly; a failed capture permanently disables
        the graph path (EX-002, one-time warning, no retry) — correctness
        first.
        """
        from actfold.core.similarity_gate import SimilarityGate

        runner = self.graph_runner
        if self._graph_capture_failed:
            # EX-002: a failed capture permanently disables the graph path
            # (the failure already warned once); never retry — capture is
            # expensive and its failure mode is not transient.
            return None
        if runner is not None and tuple(tokens.shape) != tuple(
            runner.tokens_static.shape
        ):
            if not self._graph_shape_warned:
                self._graph_shape_warned = True
                warnings.warn(
                    "ManualFoldedForward graph path captured shape "
                    f"{tuple(runner.tokens_static.shape)}; tokens of shape "
                    f"{tuple(tokens.shape)} fall back to eager (no re-capture).",
                    UserWarning,
                    stacklevel=2,
                )
            return None
        if parent_branch_id is None:
            # Parent passes always run eager: they populate the cache.
            return None
        if self.scheduler is not None:
            self._warn_graph_degraded_once("a folding scheduler is attached")
            return None
        if type(self.gate) is not SimilarityGate or self.gate.metric != "cosine":
            self._warn_graph_degraded_once(
                "the gate is not an exact-type cosine SimilarityGate"
            )
            return None
        if not tokens.is_cuda:
            self._warn_graph_degraded_once("CUDA is unavailable")
            return None

        if runner is None:
            # First graph-eligible call: capture once. An incomplete parent
            # cache is a normal transient (e.g. after eviction), so it falls
            # back silently without disabling future captures.
            if not self._parent_cache_complete(parent_branch_id, tokens):
                return None
            if attention_mask is not None and not attention_mask.is_cuda:
                return None
            from actfold.core.cuda_graph import FoldedGraphRunner

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
            )
            try:
                new_runner.capture(tokens, branch_id, parent_branch_id)
            except RuntimeError as exc:
                self._warn_graph_degraded_once(f"CUDA graph capture failed ({exc})")
                self._graph_capture_failed = True
                return None
            self.graph_runner = new_runner
            self._graph_mask = attention_mask
            runner = new_runner
        elif attention_mask is not self._graph_mask:
            # The captured graph bakes the attention mask in; a different
            # mask object must not silently reuse it.
            return None

        logits = runner.replay(tokens, parent_branch_id, branch_id)
        if logits is None:
            # Parent cache incomplete for this step: plain eager fallback.
            return None
        if runner.budget_exceeded:
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
            activations: dict[str, Any] = {
                "ffn_out": runner.child_buf[layer_idx].clone()
            }
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

    @property
    def folding_applied(self) -> bool:
        """Return ``True`` because layers are explicitly wrapped."""
        return True
