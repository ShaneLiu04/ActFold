"""Tests for T019 script portability: no sys.path hacks, no hardcoded HF mirror
paths, and a shared `scripts/_hf_env.py` helper for HF endpoint/home configuration.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import List

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

HF_SCRIPT_NAMES = (
    "algo_experiments.py",
    "opt1_cache_bench.py",
    "opt2_long_seq.py",
    "opt2_shape_bench.py",
    "opt2_split_bench.py",
    "opt3_adaptive_bench.py",
    "overhead_bench.py",
)

ACTFOLD_IMPORT_RE = re.compile(r"^\s*(?:from actfold|import actfold)", re.MULTILINE)


def _all_script_files() -> List[Path]:
    return sorted(SCRIPTS_DIR.glob("*.py"))


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _load_hf_env_module() -> ModuleType:
    module_path = SCRIPTS_DIR / "_hf_env.py"
    spec = importlib.util.spec_from_file_location("scripts_hf_env_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_sys_path_hacks() -> None:
    offenders = [p.name for p in _all_script_files() if "sys.path" in p.read_text(encoding="utf-8")]
    assert offenders == [], f"scripts still contain sys.path hacks: {offenders}"


def test_no_hardcoded_hf_mirror_or_autodl() -> None:
    offenders = [
        p.name
        for p in _all_script_files()
        if "hf-mirror.com" in p.read_text(encoding="utf-8")
        or "/root/autodl" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"scripts still hardcode platform-specific HF paths: {offenders}"


def test_hf_scripts_apply_hf_env_before_actfold_import() -> None:
    failures: List[str] = []
    for name in HF_SCRIPT_NAMES:
        source = _read_script(name)
        apply_pos = source.find("apply_hf_env()")
        if apply_pos == -1:
            failures.append(f"{name}: missing apply_hf_env() call")
            continue
        match = ACTFOLD_IMPORT_RE.search(source)
        if match is not None and apply_pos > match.start():
            failures.append(f"{name}: apply_hf_env() call is after the first actfold import")
    assert failures == [], f"apply_hf_env ordering violations: {failures}"


def test_hf_scripts_declare_hf_cli_arguments() -> None:
    # Single source of truth: the flags are declared once in the shared
    # helper; each HF script wires them into its parser via
    # ``add_hf_env_arguments(parser)``.
    helper_source = _read_script("_hf_env.py")
    assert "--hf-endpoint" in helper_source
    assert "--hf-home" in helper_source
    failures: List[str] = []
    for name in HF_SCRIPT_NAMES:
        source = _read_script(name)
        if "add_hf_env_arguments(" not in source:
            failures.append(f"{name}: missing add_hf_env_arguments(parser) call")
    assert failures == [], f"HF CLI argument declaration violations: {failures}"


def test_apply_hf_env_cli_space_form(monkeypatch) -> None:
    module = _load_hf_env_module()
    monkeypatch.setattr(sys, "argv", ["prog", "--hf-endpoint", "https://example.custom/v1", "--hf-home", "D:/cache/hf"])
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    module.apply_hf_env()
    assert os.environ["HF_ENDPOINT"] == "https://example.custom/v1"
    assert os.environ["HF_HOME"] == "D:/cache/hf"


def test_apply_hf_env_cli_equals_form(monkeypatch) -> None:
    module = _load_hf_env_module()
    monkeypatch.setattr(sys, "argv", ["prog", "--hf-endpoint=https://a.example", "--hf-home=/tmp/x"])
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    module.apply_hf_env()
    assert os.environ["HF_ENDPOINT"] == "https://a.example"
    assert os.environ["HF_HOME"] == "/tmp/x"


def test_apply_hf_env_env_fallback(monkeypatch) -> None:
    module = _load_hf_env_module()
    monkeypatch.setenv("HF_ENDPOINT", "https://env.example")
    monkeypatch.setattr(sys, "argv", ["prog"])
    module.apply_hf_env()
    assert os.environ["HF_ENDPOINT"] == "https://env.example"


def test_apply_hf_env_cli_priority_over_env(monkeypatch) -> None:
    module = _load_hf_env_module()
    monkeypatch.setenv("HF_ENDPOINT", "https://env.example")
    monkeypatch.setattr(sys, "argv", ["prog", "--hf-endpoint", "https://cli.example"])
    module.apply_hf_env()
    assert os.environ["HF_ENDPOINT"] == "https://cli.example"


def test_apply_hf_env_no_flags_no_env(monkeypatch) -> None:
    module = _load_hf_env_module()
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog"])
    module.apply_hf_env()
    assert "HF_ENDPOINT" not in os.environ
    assert "HF_HOME" not in os.environ


def test_apply_hf_env_only_hf_home_flag(monkeypatch) -> None:
    module = _load_hf_env_module()
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--hf-home", "/tmp/only-home"])
    module.apply_hf_env()
    assert os.environ["HF_HOME"] == "/tmp/only-home"
    assert "HF_ENDPOINT" not in os.environ


def test_add_hf_env_arguments_parses_values() -> None:
    module = _load_hf_env_module()
    parser = argparse.ArgumentParser()
    module.add_hf_env_arguments(parser)
    args = parser.parse_args(["--hf-endpoint", "X", "--hf-home", "Y"])
    assert args.hf_endpoint == "X"
    assert args.hf_home == "Y"


def test_add_hf_env_arguments_defaults_none() -> None:
    module = _load_hf_env_module()
    parser = argparse.ArgumentParser()
    module.add_hf_env_arguments(parser)
    args = parser.parse_args([])
    assert args.hf_endpoint is None
    assert args.hf_home is None


def test_rerun_checklist_has_no_hardcoded_platform_paths() -> None:
    """docs/RERUN_CHECKLIST.md must stay free of platform-specific paths/mirrors."""
    repo_root = SCRIPTS_DIR.parent
    text = (repo_root / "docs" / "RERUN_CHECKLIST.md").read_text(encoding="utf-8")
    assert "/root/autodl-tmp" not in text
    assert "hf-mirror.com" not in text


def test_invalidated_markers_cover_all_results() -> None:
    """Every artifact under results/ is covered by an INVALIDATED.md marker."""
    results_root = SCRIPTS_DIR.parent / "results"
    assert (results_root / "INVALIDATED.md").exists()
    for artifact in results_root.rglob("*"):
        if not artifact.is_file() or artifact.name == "INVALIDATED.md":
            continue
        rel = artifact.relative_to(results_root)
        ancestors = [results_root] + [
            results_root.joinpath(*rel.parts[:i]) for i in range(1, len(rel.parts))
        ]
        assert any((a / "INVALIDATED.md").exists() for a in ancestors), (
            f"no INVALIDATED.md covers {artifact}"
        )
