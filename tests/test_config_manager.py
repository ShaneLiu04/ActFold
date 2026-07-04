"""Tests for actfold.utils.config_manager."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from actfold.utils.config_manager import ActFoldConfig, default_config, load_config


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Write a temporary YAML config file."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tau": 0.93,
                "metric": "l2",
                "max_entries_per_layer": 512,
                "enable_dynamic_tau": True,
                "device": "cpu",
                "seed": 123,
                "num_layers": 6,
                "hidden_dim": 256,
                "num_heads": 8,
                "seq_len": 32,
                "vocab_size": 2000,
                "num_steps": 20,
                "model_name_or_path": "gpt2",
                "model_family": "causal_lm",
                "torch_dtype": "float16",
                "device_map": "auto",
                "use_real_eval": True,
                "eval_backend": "lm-eval",
                "eval_batch_size": 4,
                "eval_num_fewshot": 5,
                "eval_limit": 10,
                "eval_base_only": True,
                "unknown_key": "ignored",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_default_config() -> None:
    config = default_config()
    assert config.tau == 0.95
    assert config.metric == "cosine"
    assert config.num_layers == 4
    assert config.use_real_eval is True
    assert config.eval_backend == "auto"


def test_load_config(temp_config: Path) -> None:
    with pytest.warns(UserWarning, match="unknown_key"):
        config = load_config(temp_config)
    assert isinstance(config, ActFoldConfig)
    assert config.tau == 0.93
    assert config.metric == "l2"
    assert config.max_entries_per_layer == 512
    assert config.enable_dynamic_tau is True
    assert config.device == "cpu"
    assert config.seed == 123
    assert config.num_layers == 6
    assert config.hidden_dim == 256
    assert config.num_heads == 8
    assert config.seq_len == 32
    assert config.vocab_size == 2000
    assert config.num_steps == 20
    assert config.model_name_or_path == "gpt2"
    assert config.model_family == "causal_lm"
    assert config.torch_dtype == "float16"
    assert config.device_map == "auto"
    assert config.use_real_eval is True
    assert config.eval_backend == "lm-eval"
    assert config.eval_batch_size == 4
    assert config.eval_num_fewshot == 5
    assert config.eval_limit == 10
    assert config.eval_base_only is True


def test_load_config_warns_on_unknown_keys(temp_config: Path) -> None:
    with pytest.warns(UserWarning, match="unknown_key"):
        config = load_config(temp_config)
    assert not hasattr(config, "unknown_key")


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ActFoldConfig(tau=1.5)
    with pytest.raises(ValueError):
        ActFoldConfig(metric="invalid")
    with pytest.raises(ValueError):
        ActFoldConfig(max_entries_per_layer=0)
    with pytest.raises(ValueError):
        ActFoldConfig(load_in_8bit=True, load_in_4bit=True)
    with pytest.raises(ValueError):
        ActFoldConfig(torch_dtype="int8")
    with pytest.raises(ValueError):
        ActFoldConfig(eval_backend="mock")
    with pytest.raises(ValueError):
        ActFoldConfig(use_real_eval=False)
