"""Fused CUDA kernels for Branch Folding activation merging.

This module provides optional Triton-accelerated implementations of the
stable/divergent token merge and the activation-cache gather.  When Triton is
not installed or the input tensors live on CPU, transparent PyTorch fallbacks
are used so that behavior and numerical results are identical.
"""

from __future__ import annotations

import warnings

import torch

# ---------------------------------------------------------------------------
# Triton availability probe
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is optional
    triton = None
    tl = None
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ActivationCacheDict = dict[tuple[str, int, int, int], dict[str, torch.Tensor]]


# ---------------------------------------------------------------------------
# PyTorch fallbacks
# ---------------------------------------------------------------------------
def _merge_stable_divergent_torch(
    parent_ffn: torch.Tensor,
    child_out: torch.Tensor,
    stable_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation of the stable/divergent merge.

    For every token position ``(b, t)`` the hidden vector is taken from
    ``parent_ffn`` when ``stable_mask[b, t]`` is True, otherwise from
    ``child_out``.
    """
    # stable_mask is [B, T]; broadcast across hidden dim [B, T, H].
    expanded = stable_mask.unsqueeze(-1)
    return torch.where(expanded, parent_ffn, child_out)


def _gather_cached_activations_torch(
    cache: ActivationCacheDict,
    branch_id: str,
    layer_idx: int,
    token_mask: torch.Tensor,
    step_idx: int,
) -> dict[str, torch.Tensor]:
    """Vectorized cache reconstruction used as fallback and CPU path."""
    batch_size, seq_len = token_mask.shape

    sample_key = None
    for token_idx in range(seq_len):
        key = (branch_id, layer_idx, token_idx, step_idx)
        if key in cache:
            sample_key = key
            break
    if sample_key is None:
        raise KeyError(f"No cache entry for branch={branch_id}, layer={layer_idx}, step={step_idx}")

    sample_entry = cache[sample_key]
    device = next(iter(sample_entry.values())).device
    dtype = next(iter(sample_entry.values())).dtype

    output: dict[str, torch.Tensor] = {}
    for name, sample_tensor in sample_entry.items():
        leading = [batch_size, seq_len] + list(sample_tensor.shape[1:])
        output[name] = torch.zeros(leading, dtype=dtype, device=device)

    for token_idx in range(seq_len):
        key = (branch_id, layer_idx, token_idx, step_idx)
        if key not in cache:
            continue
        if not token_mask[:, token_idx].any():
            continue
        entry = cache[key]
        for name, tensor in entry.items():
            output[name][:, token_idx, ...] = tensor

    return output


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit  # type: ignore[untyped-decorator]
    def _merge_kernel(  # type: ignore[no-untyped-def]
        parent_ptr,
        child_ptr,
        mask_ptr,
        out_ptr,
        parent_batch_stride,
        parent_seq_stride,
        parent_hidden_stride,
        child_batch_stride,
        child_seq_stride,
        child_hidden_stride,
        mask_batch_stride,
        mask_seq_stride,
        out_batch_stride,
        out_seq_stride,
        out_hidden_stride,
        seq_len,
        hidden_size,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        """Element-wise select between parent and child activations.

        Each program instance processes one hidden vector ``out[b, t, :]``.
        The 1-D pid is mapped to ``(b, t)`` by integer division.  Only the
        selected source row is read (single-sided load), and all strides are
        explicit so non-contiguous inputs are supported without copies.
        """
        pid = tl.program_id(0)
        b = pid // seq_len
        t = pid % seq_len

        # Load the boolean mask for this token.
        mask_off = b * mask_batch_stride + t * mask_seq_stride
        stable = tl.load(mask_ptr + mask_off).to(tl.int1)

        if stable:
            src = parent_ptr + b * parent_batch_stride + t * parent_seq_stride
            src_hidden_stride = parent_hidden_stride
        else:
            src = child_ptr + b * child_batch_stride + t * child_seq_stride
            src_hidden_stride = child_hidden_stride
        dst = out_ptr + b * out_batch_stride + t * out_seq_stride

        # Vectorized load/store over the hidden dimension (tail-masked, so
        # any hidden size is supported).
        for h_off in range(0, hidden_size, BLOCK_SIZE):
            h = h_off + tl.arange(0, BLOCK_SIZE)
            mask = h < hidden_size

            vec = tl.load(src + h * src_hidden_stride, mask=mask)
            tl.store(dst + h * out_hidden_stride, vec, mask=mask)


_TRITON_MERGE_DISABLED = False
_TRITON_GATHER_SELECT_DISABLED = False


def _merge_stable_divergent_triton(
    parent_ffn: torch.Tensor,
    child_out: torch.Tensor,
    stable_mask: torch.Tensor,
) -> torch.Tensor:
    """Triton implementation of the merge.

    Falls back to PyTorch if inputs do not live on CUDA or use unsupported
    dtypes.  Arbitrary hidden sizes are handled via tail masking and arbitrary
    strides are passed to the kernel, so no caller-side ``.contiguous()``
    copies are made.  If the kernel fails to compile or launch on the
    installed Triton version, the PyTorch fallback is used for all subsequent
    calls.
    """
    global _TRITON_MERGE_DISABLED

    if not _HAS_TRITON or _TRITON_MERGE_DISABLED:
        return _merge_stable_divergent_torch(parent_ffn, child_out, stable_mask)

    # Only CUDA tensors are supported by the Triton path; the mask must also
    # live on CUDA because the kernel reads it directly.
    if (
        parent_ffn.device.type != "cuda"
        or child_out.device.type != "cuda"
        or stable_mask.device.type != "cuda"
    ):
        return _merge_stable_divergent_torch(parent_ffn, child_out, stable_mask)

    # fp32, fp16 and bf16 are the dtypes typically used by LLMs.
    if parent_ffn.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return _merge_stable_divergent_torch(parent_ffn, child_out, stable_mask)

    if parent_ffn.dtype != child_out.dtype:
        return _merge_stable_divergent_torch(parent_ffn, child_out, stable_mask)

    batch_size, seq_len, hidden_dim = parent_ffn.shape

    block_size = 128

    # The output is always contiguous; strides of the (possibly
    # non-contiguous) inputs are passed to the kernel explicitly.
    out = torch.empty(parent_ffn.shape, dtype=parent_ffn.dtype, device=parent_ffn.device)

    total_tokens = batch_size * seq_len
    grid = (total_tokens,)

    try:
        _merge_kernel[grid](
            parent_ffn,
            child_out,
            stable_mask,
            out,
            parent_ffn.stride(0),
            parent_ffn.stride(1),
            parent_ffn.stride(2),
            child_out.stride(0),
            child_out.stride(1),
            child_out.stride(2),
            stable_mask.stride(0),
            stable_mask.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            seq_len,
            hidden_dim,
            BLOCK_SIZE=block_size,
        )
    except Exception as exc:
        _TRITON_MERGE_DISABLED = True
        warnings.warn(
            f"Triton merge kernel unavailable on this Triton installation ({exc}); "
            "falling back to the PyTorch merge.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _merge_stable_divergent_torch(parent_ffn, child_out, stable_mask)
    return out


# ---------------------------------------------------------------------------
# Fused gather + select (optimization #4)
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit  # type: ignore[untyped-decorator]
    def _gather_select_kernel(  # type: ignore[no-untyped-def]
        parent_ptr,
        index_ptr,
        child_ptr,
        mask_ptr,
        out_ptr,
        seq_len,
        hidden_dim,
        parent_stride,
        child_stride,
        BLOCK_H: tl.constexpr,
    ) -> None:
        """Copy a parent row (by index) or the child row into ``out``.

        One program handles one token row: stable tokens gather row
        ``index[t]`` from the contiguous parent buffer, divergent tokens copy the
        child row. This fuses ``index_select`` and ``torch.where`` into a single
        memory pass with no intermediate gather tensor.
        """
        t = tl.program_id(0)
        stable = tl.load(mask_ptr + t) != 0
        if stable:
            row = tl.load(index_ptr + t)
            src = parent_ptr + row * parent_stride
        else:
            src = child_ptr + t * child_stride
        dst = out_ptr + t * hidden_dim
        for h in range(0, hidden_dim, BLOCK_H):
            offs = h + tl.arange(0, BLOCK_H)
            m = offs < hidden_dim
            v = tl.load(src + offs, mask=m)
            tl.store(dst + offs, v, mask=m)


def gather_select(
    parent_buffer: torch.Tensor,
    rows: torch.Tensor,
    child_out: torch.Tensor,
    stable_mask: torch.Tensor,
) -> torch.Tensor:
    """Fused stable/divergent token selection from a contiguous parent buffer.

    For every token position ``(b, t)`` the output equals
    ``parent_buffer[rows[b, t]]`` when ``stable_mask[b, t]`` is True, otherwise
    ``child_out[b, t]``.

    Args:
        parent_buffer: Contiguous parent activations ``[capacity, hidden]``.
        rows: Parent row index per token ``[batch, seq_len]`` (int).
        child_out: Recomputed child activations ``[batch, seq_len, hidden]``.
        stable_mask: Boolean stability mask ``[batch, seq_len]``.

    Returns:
        Merged tensor ``[batch, seq_len, hidden]``.
    """
    global _TRITON_GATHER_SELECT_DISABLED
    if child_out.ndim != 3 or stable_mask.shape != child_out.shape[:2]:
        raise ValueError(
            f"child_out must be [B, T, H] and stable_mask [B, T], got "
            f"{child_out.shape} and {stable_mask.shape}"
        )
    batch, seq_len, hidden_dim = child_out.shape
    if parent_buffer.ndim != 2:
        raise ValueError(f"parent_buffer must be [cap, H], got {parent_buffer.shape}")
    if parent_buffer.shape[1] != hidden_dim:
        raise ValueError(
            f"parent_buffer hidden {parent_buffer.shape[1]} != child hidden {hidden_dim}"
        )

    mask = stable_mask.to(device=child_out.device, dtype=torch.bool)
    flat_rows = rows.reshape(-1).to(device=child_out.device, dtype=torch.int64)
    flat_mask = mask.reshape(-1)
    flat_child = child_out.reshape(batch * seq_len, hidden_dim)

    use_triton = (
        _HAS_TRITON
        and not _TRITON_GATHER_SELECT_DISABLED
        and child_out.is_cuda
        and parent_buffer.is_cuda
        and child_out.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and parent_buffer.dtype == child_out.dtype
    )
    if not use_triton:
        gathered = parent_buffer.index_select(0, flat_rows).reshape(batch, seq_len, hidden_dim)
        return torch.where(mask.unsqueeze(-1), gathered, child_out)

    parent_buffer = parent_buffer.contiguous()
    flat_child = flat_child.contiguous()
    flat_mask_u8 = flat_mask.view(torch.uint8) if flat_mask.dtype == torch.bool else flat_mask
    out = torch.empty_like(flat_child)
    grid = (batch * seq_len,)
    try:
        _gather_select_kernel[grid](
            parent_buffer,
            flat_rows,
            flat_child,
            flat_mask_u8,
            out,
            seq_len,
            hidden_dim,
            parent_buffer.stride(0),
            flat_child.stride(0),
            BLOCK_H=128,
        )
    except Exception as exc:
        _TRITON_GATHER_SELECT_DISABLED = True
        warnings.warn(
            f"Triton gather/select kernel unavailable ({exc}); using PyTorch fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        gathered = parent_buffer.index_select(0, flat_rows).reshape(batch, seq_len, hidden_dim)
        return torch.where(mask.unsqueeze(-1), gathered, child_out)
    return out.reshape(batch, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def merge_stable_divergent(
    parent_ffn: torch.Tensor,
    child_out: torch.Tensor,
    stable_mask: torch.Tensor,
) -> torch.Tensor:
    """Fuse cached parent FFN output with recomputed child output.

    For every token position ``(b, t)`` the output hidden vector equals
    ``parent_ffn[b, t, :]`` when ``stable_mask[b, t]`` is True, otherwise
    ``child_out[b, t, :]``.

    This function automatically selects a Triton CUDA kernel when available
    and appropriate, otherwise a vectorized PyTorch fallback.

    Args:
        parent_ffn: Cached parent FFN output ``[B, T, H]``.
        child_out: Full child layer output ``[B, T, H]``.
        stable_mask: Boolean stability mask ``[B, T]``.

    Returns:
        Merged output ``[B, T, H]`` with the same dtype/device as inputs.

    Raises:
        ValueError: If input shapes are incompatible.
    """
    if parent_ffn.ndim != 3 or child_out.ndim != 3:
        raise ValueError(
            f"parent_ffn and child_out must be 3-D [B, T, H], got "
            f"{parent_ffn.shape} and {child_out.shape}"
        )
    if parent_ffn.shape != child_out.shape:
        raise ValueError(
            f"Shape mismatch: parent_ffn {parent_ffn.shape} vs child_out {child_out.shape}"
        )
    if stable_mask.ndim != 2 or stable_mask.shape != parent_ffn.shape[:2]:
        raise ValueError(
            f"stable_mask must be 2-D [B, T] matching parent_ffn {parent_ffn.shape[:2]}, "
            f"got {stable_mask.shape}"
        )

    # Ensure the mask lives on the same device as the activations and is a
    # boolean tensor so the PyTorch fallback can broadcast without device errors.
    stable_mask = stable_mask.to(parent_ffn.device, dtype=torch.bool)

    # Prefer Triton on CUDA; fall back otherwise.
    if _HAS_TRITON and parent_ffn.is_cuda:
        return _merge_stable_divergent_triton(parent_ffn, child_out, stable_mask)
    return _merge_stable_divergent_torch(parent_ffn, child_out, stable_mask)


def gather_cached_activations(
    cache: ActivationCacheDict,
    branch_id: str,
    layer_idx: int,
    token_mask: torch.Tensor,
    step_idx: int = 0,
) -> dict[str, torch.Tensor]:
    """Reconstruct activation tensors from per-token cache entries.

    This is a vectorized replacement for the Python-loop cache retrieval used
    by ``ActivationCache.get``.  When all token entries are present and the
    cache is dense, it stacks them in a single ``torch.stack`` call and then
    applies ``token_mask`` to zero out masked positions.

    Args:
        cache: Underlying OrderedDict-like cache mapping keys to per-token
            activation dictionaries.
        branch_id: Branch to retrieve from.
        layer_idx: Layer index.
        token_mask: Boolean mask ``[B, T]``.
        step_idx: Diffusion step index.

    Returns:
        Dictionary of reconstructed activations with the same leading shape
        ``[B, T, ...]``; masked positions are zero-filled.

    Raises:
        KeyError: If the first token entry is missing (same behavior as
            ``ActivationCache.get``).
    """
    batch_size, seq_len = token_mask.shape

    sample_key = None
    for token_idx in range(seq_len):
        key = (branch_id, layer_idx, token_idx, step_idx)
        if key in cache:
            sample_key = key
            break
    if sample_key is None:
        raise KeyError(f"No cache entry for branch={branch_id}, layer={layer_idx}, step={step_idx}")

    sample_entry = cache[sample_key]

    # Fast vectorized path: all token entries are present.
    dense = all(
        (branch_id, layer_idx, token_idx, step_idx) in cache for token_idx in range(seq_len)
    )

    if not dense:
        # Sparse cache: fall back to the loop-based implementation which
        # already handles missing entries correctly.
        return _gather_cached_activations_torch(cache, branch_id, layer_idx, token_mask, step_idx)
    output: dict[str, torch.Tensor] = {}
    for name, sample_tensor in sample_entry.items():
        per_token_tensors = [
            cache[(branch_id, layer_idx, token_idx, step_idx)][name] for token_idx in range(seq_len)
        ]
        # Stack along the sequence dimension.
        stacked = torch.stack(per_token_tensors, dim=1)

        if stacked.shape[0] != batch_size:
            # The stored batch size may differ from the requested mask; expand
            # or narrow to match.  This mirrors the original implementation
            # which copies ``[:, token_idx, ...]``.
            if stacked.shape[0] == 1 and batch_size > 1:
                stacked = stacked.expand(batch_size, *stacked.shape[1:])
            else:
                stacked = stacked[:batch_size]

        # Apply the mask: True positions keep the cached value, False -> zero.
        expanded_mask = token_mask.view(batch_size, seq_len, *([1] * (stacked.ndim - 2)))
        output[name] = torch.where(expanded_mask, stacked, torch.zeros_like(stacked))

    return output
