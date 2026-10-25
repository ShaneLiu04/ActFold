"""AR036 T001 probe: is commit-draft fusion numerically sound?

P1  fused [1,2B] logits for block b  == standalone commit [1,B] logits
P2  fused next-block decision rows   ~= standalone draft [1,B] (maxdiff + argmax rate)
P3  fused KV cropped to hist+B       == standalone commit KV (per-layer maxdiff)
P4  cache crop API availability

Real checkpoint, no synthetic tensors beyond real token ids.
"""

from __future__ import annotations

import argparse
from typing import Any

from scripts._hf_env import add_hf_env_arguments, apply_hf_env

apply_hf_env()

import torch  # noqa: E402

from scripts.ar021_generation_loop_bench import (  # noqa: E402
    BLOCK,
    _load_model,
    _prepare_prompts,
    _resolve_model_path,
)


def _clone_kv(kv: Any) -> Any:
    """Container rebuild + detached clone (deepcopy rejects non-leaf KV)."""
    out = type(kv)()
    layers = getattr(kv, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            out.update(layer.keys.detach().clone(), layer.values.detach().clone(), i)
        return out
    for i, (k, v) in enumerate(zip(kv.key_cache, kv.value_cache)):
        out.update(k.detach().clone(), v.detach().clone(), i)
    return out


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def _crop(kv: Any, keep: int) -> Any | None:
    """Crop a DynamicCache-like object in place to ``keep`` positions."""
    if not hasattr(kv, "crop"):
        return None
    try:
        res = kv.crop(keep)
        out = kv if res is None else res
        seq = getattr(out, "get_seq_length", None)
        if callable(seq) and int(seq()) != keep:
            return None
        return out
    except Exception:
        return None


def _kv_layers(kv: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layers = getattr(kv, "layers", None)
    if layers is not None:
        return [(layer.keys, layer.values) for layer in layers]
    return list(zip(kv.key_cache, kv.value_cache))


def main() -> int:
    ap = argparse.ArgumentParser()
    add_hf_env_arguments(ap)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    model, tokenizer, meta = _load_model(_resolve_model_path(args.model))
    prompts = _prepare_prompts(tokenizer, quick=True)
    ids = prompts[0]["ids"][:, : prompts[0]["raw_len"]]
    B = BLOCK
    need = 3 * B
    if ids.shape[1] < need:
        reps = (need + ids.shape[1] - 1) // ids.shape[1]
        ids = ids.repeat(1, reps)
    ids = ids[:, :need]
    mask_id = int(getattr(tokenizer, "mask_token_id", None) or 0) or int(
        tokenizer.convert_tokens_to_ids("|<MASK>|")
    )
    print(f"model={meta.get('model_path')} B={B} mask_id={mask_id} ids={tuple(ids.shape)}")

    hist = ids[:, : 2 * B]
    block_b = ids[:, 2 * B : 3 * B]
    prefill = model(
        input_ids=hist,
        use_cache=True,
        update_past_key_values=True,
        block_size=B,
    )
    kv = prefill.past_key_values

    commit = model(
        input_ids=block_b,
        use_cache=True,
        past_key_values=_clone_kv(kv),
        update_past_key_values=True,
        block_size=B,
    )
    logits_c = commit.logits
    kv_c = commit.past_key_values
    advance = logits_c[:, -1:, :].argmax(dim=-1)
    next_canvas = torch.cat(
        [advance, torch.full((1, B - 1), mask_id, dtype=ids.dtype, device=ids.device)], dim=1
    )

    fused = model(
        input_ids=torch.cat([block_b, next_canvas], dim=1),
        use_cache=True,
        past_key_values=_clone_kv(kv),
        update_past_key_values=True,
        block_size=B,
    )
    logits_f = fused.logits
    kv_f = fused.past_key_values

    p1 = _maxdiff(logits_f[:, :B], logits_c)

    draft = model(
        input_ids=next_canvas,
        use_cache=True,
        past_key_values=_clone_kv(kv_c),
        update_past_key_values=False,
        block_size=B,
    )
    draft_rows = torch.cat([logits_c[:, -1:, :], draft.logits[:, :-1, :]], dim=1)
    fused_rows = logits_f[:, B - 1 : 2 * B - 1]
    p2_diff = _maxdiff(fused_rows, draft_rows)
    p2_argmax = float((fused_rows.argmax(-1) == draft_rows.argmax(-1)).float().mean().item())

    keep = hist.shape[1] + B
    cropped = _crop(kv_f, keep)
    p4 = cropped is not None
    p3 = None
    if cropped is not None:
        la = _kv_layers(cropped)
        lb = _kv_layers(kv_c)
        p3 = max(max(_maxdiff(ka, kb), _maxdiff(va, vb)) for (ka, va), (kb, vb) in zip(la, lb))

    print(f"P1_blockb_logits_maxdiff={p1:.6f}")
    print(f"P2_nextblock_rows_maxdiff={p2_diff:.6f} argmax_match={p2_argmax:.6f}")
    print(f"P3_cropped_kv_maxdiff={p3}")
    print(f"P4_crop_available={p4} cache_type={type(kv_f).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
