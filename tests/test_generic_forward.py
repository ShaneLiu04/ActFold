"""Red-phase tests for AR008: on-demand ``output_hidden_states`` in generic forward.

UT-701~705 (AR008-T001). These tests are written BEFORE the implementation
change (TDD Red): the current ``GenericDiffusionLLM.forward`` unconditionally
passes ``output_hidden_states=True`` (generic.py:129-135), so the ``is False``
and call-count assertions below must fail until the on-demand behavior lands.

Stub construction follows design.md §4.1: real ``SimpleNamespace`` output
objects (not MagicMock return values) so ``getattr(outputs, field, None)``
probing semantics are real, and ``nn.Module`` stubs that record every call's
kwargs in ``call_log``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from actfold.models.generic import GenericDiffusionLLM

_VOCAB = 100
_HIDDEN = 16
_BATCH = 2
_SEQ = 4

# Verbatim RuntimeError text from generic.py:148-150 (AR008 D3: preserved word
# for word by the on-demand implementation).
_RUNTIME_ERROR_MESSAGE = (
    "Model output exposes neither 'logits', 'last_hidden_state', nor "
    "'hidden_states'; cannot extract a tensor for ActFold."
)


class _RecordingStub(nn.Module):
    """Base nn.Module stub recording every forward call's kwargs in ``call_log``.

    Subclasses override ``forward`` to produce the output shape under test.
    The recorded entry contains ``input_ids``, ``attention_mask``, and all
    forwarded kwargs (``output_hidden_states`` / ``return_dict`` included).
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_hidden_layers=2,
            num_attention_heads=2,
        )
        self.call_log: list[dict[str, Any]] = []

    def _record(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> None:
        entry: dict[str, Any] = dict(kwargs)
        entry["input_ids"] = input_ids
        entry["attention_mask"] = attention_mask
        self.call_log.append(entry)


class _LogitsOnlyStub(_RecordingStub):
    """Stub whose output exposes only ``logits`` (srs 3.1-1)."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.randn(_BATCH, _SEQ, _VOCAB)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self._record(input_ids, attention_mask, kwargs)
        return SimpleNamespace(logits=self.logits)


class _LastHiddenOnlyStub(_RecordingStub):
    """Stub whose output exposes only ``last_hidden_state`` (srs 3.1-2)."""

    def __init__(self) -> None:
        super().__init__()
        self.last_hidden = torch.randn(_BATCH, _SEQ, _HIDDEN)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self._record(input_ids, attention_mask, kwargs)
        return SimpleNamespace(last_hidden_state=self.last_hidden)


class _HiddenStatesFlagStub(_RecordingStub):
    """Flag-conditional stub honoring ``output_hidden_states`` like a real HF model.

    ``False`` -> ``SimpleNamespace(hidden_states=None)`` (no collection);
    ``True`` -> ``SimpleNamespace(hidden_states=(h0, h1))``.

    Design §4.1 WARN-1: a stub that always returns the tuple would make the
    "first forward = 2 calls" assertion a false Red.
    """

    def __init__(self) -> None:
        super().__init__()
        self.h0 = torch.randn(_BATCH, _SEQ, _HIDDEN)
        self.h1 = torch.randn(_BATCH, _SEQ, _HIDDEN)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self._record(input_ids, attention_mask, kwargs)
        if kwargs.get("output_hidden_states"):
            return SimpleNamespace(hidden_states=(self.h0, self.h1))
        return SimpleNamespace(hidden_states=None)


class _NoOutputStub(_RecordingStub):
    """Stub whose output exposes none of the three fields (srs 3.1-4)."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self._record(input_ids, attention_mask, kwargs)
        return SimpleNamespace()


class _EmptyHiddenStub(_RecordingStub):
    """Stub returning an empty ``hidden_states`` tuple (design §4.1 WARN-4)."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self._record(input_ids, attention_mask, kwargs)
        return SimpleNamespace(hidden_states=())


def _make_generic(stub: nn.Module) -> GenericDiffusionLLM:
    """Build a ``GenericDiffusionLLM`` whose ``self.model`` is ``stub``.

    Follows the test_models.py MagicMock + patch ``from_pretrained`` pattern
    (:62-99 precedent). The stub carries ``.config`` (hidden_size /
    vocab_size / layer / head counts) for ``__init__`` (generic.py:70-74) and
    exposes no ``lm_head`` / output embeddings, so the wrapper constructs the
    LM head itself (``allow_random_head=True``; deterministic under the
    autouse ``seed`` fixture).

    Args:
        stub: The recording stub module to install as ``self.model``.

    Returns:
        A ``GenericDiffusionLLM`` instance wrapping ``stub``.
    """
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "pad"
    mock_tokenizer.__len__ = lambda _: _VOCAB
    model_patch = patch("actfold.models.generic.AutoModel.from_pretrained", return_value=stub)
    tokenizer_patch = patch(
        "actfold.models.generic.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )
    with model_patch, tokenizer_patch:
        return GenericDiffusionLLM("dummy/ar008-stub", allow_random_head=True)


def _tokens() -> torch.Tensor:
    """Return a small deterministic-shape token batch ``[2, 4]``."""
    return torch.randint(0, _VOCAB, (_BATCH, _SEQ))


def test_ut_701_logits_only_passes_output_hidden_states_false() -> None:
    """UT-701 (AR008, srs 3.1-1): logits-only models must not request hidden states.

    Two forward calls; every underlying call must receive
    ``output_hidden_states is False`` and the returned logits must equal the
    stub's ``logits`` tensor bit-for-bit.
    """
    stub = _LogitsOnlyStub()
    model = _make_generic(stub)
    tokens = _tokens()
    for _ in range(2):
        result = model(tokens)
        torch.testing.assert_close(result, stub.logits)
    assert len(stub.call_log) == 2
    for call in stub.call_log:
        assert call["output_hidden_states"] is False


def test_ut_702_last_hidden_only_passes_false_and_projects() -> None:
    """UT-702 (AR008, srs 3.1-2): last_hidden_state-only models get False and a projection.

    The underlying call must receive ``output_hidden_states is False`` and the
    wrapper must return ``lm_head(last_hidden_state)`` bit-for-bit.
    """
    stub = _LastHiddenOnlyStub()
    model = _make_generic(stub)
    result = model(_tokens())
    assert len(stub.call_log) == 1
    assert stub.call_log[0]["output_hidden_states"] is False
    torch.testing.assert_close(result, model.lm_head(stub.last_hidden))


def test_ut_703_hidden_states_only_reruns_once_then_caches_flag() -> None:
    """UT-703 (AR008, srs 3.1-3): hidden_states-only models rerun once, then cache the flag.

    First forward: exactly two underlying calls (``output_hidden_states``
    False then True) and the result equals ``lm_head(h1)``. Second forward:
    exactly one additional call with ``True`` (the instance flag is cached).
    """
    stub = _HiddenStatesFlagStub()
    model = _make_generic(stub)
    tokens = _tokens()
    first = model(tokens)
    assert len(stub.call_log) == 2
    assert stub.call_log[0]["output_hidden_states"] is False
    assert stub.call_log[1]["output_hidden_states"] is True
    torch.testing.assert_close(first, model.lm_head(stub.h1))
    second = model(tokens)
    assert len(stub.call_log) == 3
    assert stub.call_log[2]["output_hidden_states"] is True
    torch.testing.assert_close(second, model.lm_head(stub.h1))


@pytest.mark.parametrize(
    "stub_factory",
    [_NoOutputStub, _EmptyHiddenStub],
    ids=["no-output-fields", "empty-hidden-tuple"],
)
def test_ut_704_missing_outputs_raise_verbatim_after_single_rerun(
    stub_factory: Callable[[], _RecordingStub],
) -> None:
    """UT-704 (AR008, srs 3.1-4): unusable outputs raise the verbatim RuntimeError.

    Covers both the no-fields stub and the ``hidden_states=()`` empty-tuple
    variant (design §4.1 WARN-4). The message must match the current text
    word for word, and the underlying model must be called exactly twice
    (False then True — no infinite rerun).
    """
    stub = stub_factory()
    model = _make_generic(stub)
    with pytest.raises(RuntimeError) as excinfo:
        model(_tokens())
    assert str(excinfo.value) == _RUNTIME_ERROR_MESSAGE
    assert len(stub.call_log) == 2
    assert stub.call_log[0]["output_hidden_states"] is False
    assert stub.call_log[1]["output_hidden_states"] is True


def test_ut_705_branch_kwargs_stripped_from_both_calls() -> None:
    """UT-705 (AR008, srs 3.1-5): branch kwargs never reach the underlying model.

    Forwarding ``branch_id`` / ``parent_branch_id`` / ``step_idx`` against a
    flag-conditional hidden_states-only stub: both underlying calls (the
    initial False pass and the True rerun) must be free of branch keys, and
    the result must equal ``lm_head(h1)``.
    """
    stub = _HiddenStatesFlagStub()
    model = _make_generic(stub)
    result = model(
        _tokens(),
        branch_id=1,
        parent_branch_id=0,
        step_idx=3,
    )
    assert len(stub.call_log) == 2
    for call in stub.call_log:
        assert "branch_id" not in call
        assert "parent_branch_id" not in call
        assert "step_idx" not in call
    assert stub.call_log[0]["output_hidden_states"] is False
    assert stub.call_log[1]["output_hidden_states"] is True
    torch.testing.assert_close(result, model.lm_head(stub.h1))


# ---------------------------------------------------------------------------
# EX-805 (AR008-T002): CUDA peak-memory evidence for the on-demand flag
# ---------------------------------------------------------------------------

_MEM_LAYERS = 24
_MEM_BATCH, _MEM_SEQ, _MEM_HIDDEN = 1, 512, 4096
_MEM_MIN_DELTA_BYTES = 150 * 1024 * 1024


class _PeakMemoryStub(nn.Module):
    """Parameter-free last_hidden_state stub with optional hidden-state tuple.

    Holds one ``[batch, seq, hidden]`` fp32 buffer (8 MiB at the EX-805
    shape). When the wrapper requests ``output_hidden_states`` the stub
    clones the buffer ``_MEM_LAYERS`` times into a tuple (~192 MiB at the
    EX-805 shape) — the all-layer collection that the on-demand flag
    avoids. No layer weights: the tuple is a runtime allocation, so the
    measured peak delta isolates the collection itself.
    """

    def __init__(self, device: str) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=_VOCAB,
            hidden_size=_MEM_HIDDEN,
            num_hidden_layers=_MEM_LAYERS,
            num_attention_heads=2,
        )
        self.register_buffer("buf", torch.randn(_MEM_BATCH, _MEM_SEQ, _MEM_HIDDEN, device=device))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        result: dict[str, Any] = {"last_hidden_state": self.buf}
        if kwargs.get("output_hidden_states"):
            result["hidden_states"] = tuple(self.buf.clone() for _ in range(_MEM_LAYERS))
        return SimpleNamespace(**result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ex_805_on_demand_flag_cuts_peak_memory() -> None:
    """EX-805 (AR008, srs 3.3): the all-layer tuple costs ~192 MiB of peak.

    Two wrappers over the same stub: the default (on-demand, flag False)
    instance never collects hidden states, while a flag-True instance
    (manually set: the last_hidden_state-only stub never triggers the
    rerun naturally) collects the 24-layer tuple on every forward. The
    CUDA peak-memory delta between the two must exceed 150 MiB
    (theoretical 24 x 512 x 4096 x 4B = 192 MiB; threshold leaves ~22%
    headroom for allocator noise).
    """
    device = "cuda"
    stub = _PeakMemoryStub(device).to(device)
    model_off = _make_generic(stub)
    model_on = _make_generic(stub)
    # The wrapper's randomly initialized head is created on CPU; move it
    # to the stub's device (production code would move the whole wrapper).
    model_off.lm_head = model_off.lm_head.to(device)
    model_on.lm_head = model_on.lm_head.to(device)
    model_on._needs_hidden_states = True
    tokens = torch.randint(0, _VOCAB, (_MEM_BATCH, 8), device=device)

    with torch.no_grad():
        # Warm-up pass: first-touch allocations (lm_head etc.) must not
        # pollute the comparison.
        model_off(tokens)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        model_off(tokens)
        peak_off = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        model_on(tokens)
        peak_on = torch.cuda.max_memory_allocated()

    assert peak_on - peak_off >= _MEM_MIN_DELTA_BYTES
