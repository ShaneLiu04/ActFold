"""Unit tests for AR010: model family inference unification.

Covers the dedup of ``ModelRegistry._resolve_family`` onto
``actfold.models.utils.infer_model_family`` (single implementation source,
srs 3.1) and the UserWarning raised when auto-detection falls back to
``causal_lm`` (fallback visibility, srs 3.2). UT-901..904 per
``specs/changes/AR010-model-family-inference-unification/design.md`` §4.1.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from actfold.models import ModelRegistry

_RECORDED_CALLS: list[tuple[str, str]] = []


class _RecordingDummy:
    """Dummy wrapper class recording ``(family, model_name)`` on construction."""

    family: str

    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
        _RECORDED_CALLS.append((type(self).family, model_name_or_path))


def _make_dummy(family: str) -> type[_RecordingDummy]:
    """Build a per-family ``_RecordingDummy`` subclass.

    Args:
        family: Registry key the dummy is installed under.

    Returns:
        A fresh dummy class tagged with ``family``.
    """
    return type(f"_Dummy_{family}", (_RecordingDummy,), {"family": family})


def _install_dummy_registry(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Replace ``ModelRegistry._REGISTRY`` with recording dummy constructors.

    Avoids real ``from_pretrained`` network loads while still verifying
    which family class ``load`` constructs.

    Args:
        monkeypatch: pytest monkeypatch fixture (auto-restores the dict).

    Returns:
        The shared call log, cleared before installation.
    """
    _RECORDED_CALLS.clear()
    families = (
        "llada",
        "dream",
        "fast_dllm",
        "fast-dllm",
        "causal_lm",
        "causal",
        "generic",
    )
    monkeypatch.setattr(
        ModelRegistry,
        "_REGISTRY",
        {family: _make_dummy(family) for family in families},
    )
    return _RECORDED_CALLS


@pytest.mark.parametrize(
    ("model_name", "model_family", "expected"),
    [
        ("GPT-LLaDA-8B", "auto", "llada"),
        ("FastDLLM-v2", "auto", "fast_dllm"),
        ("some/dream-7b", "auto", "dream"),
        ("gpt2", "auto", "causal_lm"),
        ("gpt2", "causal_lm", "causal_lm"),
        ("anything", "dream", "dream"),
    ],
)
def test_resolve_family_equivalence_matrix(
    model_name: str,
    model_family: str,
    expected: str,
) -> None:
    """UT-901 (srs 3.1): auto-inference equivalence matrix survives dedup.

    Mixed-case, concatenated and explicit-passthrough identifiers must
    resolve bit-identically to the pre-change substring rules. The
    gpt2-auto case triggers the new fallback UserWarning (owned by
    UT-902), so it runs under ``simplefilter("ignore")`` — this matrix
    asserts return values only.
    """
    if model_family == "auto" and expected == "causal_lm":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ModelRegistry._resolve_family(model_name, model_family)
        assert result == expected
    else:
        assert ModelRegistry._resolve_family(model_name, model_family) == expected


def test_auto_fallback_warns_and_uses_causal_lm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-902 (srs 3.2-①): auto fallback to causal_lm emits one UserWarning.

    ``"gpt2"`` contains no known family keyword, so auto-detection falls
    back to ``causal_lm``. The warning must carry the
    "auto-detect fell back to causal_lm" phrase and the model name, and
    the fallback must still construct the causal_lm family class
    (behavior unchanged, visibility added).
    """
    calls = _install_dummy_registry(monkeypatch)
    with pytest.warns(
        UserWarning,
        match="auto-detect fell back to causal_lm",
    ) as record:
        model = ModelRegistry.load("gpt2", model_family="auto")
    assert len(record) == 1
    assert "gpt2" in str(record[0].message)
    assert calls == [("causal_lm", "gpt2")]
    assert isinstance(model, _RecordingDummy)


def test_explicit_family_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-903 (srs 3.2-②): explicit model_family emits no UserWarning.

    An explicit ``model_family="causal_lm"`` is a user choice: the load
    must succeed with zero UserWarnings even under
    ``simplefilter("error")``.
    """
    calls = _install_dummy_registry(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ModelRegistry.load("gpt2", model_family="causal_lm")
    assert calls == [("causal_lm", "gpt2")]
    assert isinstance(model, _RecordingDummy)


def test_auto_known_substring_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UT-904 (srs 3.2-③): auto with a known substring emits no warning.

    ``"some/llada-8b"`` hits the ``llada`` substring, so auto-detection
    resolves a known family: the load must succeed under
    ``simplefilter("error")`` and construct the llada dummy.
    """
    calls = _install_dummy_registry(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ModelRegistry.load("some/llada-8b", model_family="auto")
    assert calls == [("llada", "some/llada-8b")]
    assert isinstance(model, _RecordingDummy)
    assert type(model).family == "llada"
