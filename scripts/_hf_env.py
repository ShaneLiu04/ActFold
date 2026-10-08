"""Portable Hugging Face environment handling for benchmark scripts.

Historically the experiment scripts hardcoded a third-party mirror endpoint
and a platform-specific (cloud-vendor) cache directory, which broke
portability across machines.

This module centralizes the portable replacement:

* ``apply_hf_env()`` must be called at module top, **before** importing
  ``actfold.models`` / ``transformers`` / ``huggingface_hub`` (these libraries
  read ``HF_ENDPOINT`` / ``HF_HOME`` at import time). Explicit CLI flags take
  priority; without them existing environment variables are respected (env
  fallback). No hardcoded defaults.
* ``add_hf_env_arguments(parser)`` declares the flags on each script's own
  ``argparse`` parser so they are documented in ``--help`` and do not trip
  argument parsing (the environment is applied earlier by ``apply_hf_env()``).
"""

from __future__ import annotations

import argparse
import os
import sys

__all__ = ["apply_hf_env", "add_hf_env_arguments"]

_HF_FLAGS = (("--hf-endpoint", "HF_ENDPOINT"), ("--hf-home", "HF_HOME"))


def apply_hf_env() -> None:
    """Apply ``--hf-endpoint`` / ``--hf-home`` CLI overrides to ``os.environ``.

    Scans :data:`sys.argv` for the two flags (both ``--flag VALUE`` and
    ``--flag=VALUE`` forms). CLI values take priority; when a flag is absent
    the corresponding environment variable (``HF_ENDPOINT`` / ``HF_HOME``)
    is left untouched.

    Must be called before ``transformers`` / ``huggingface_hub`` are imported
    so the libraries observe the overrides at import time.
    """
    for flag, env_var in _HF_FLAGS:
        for i, arg in enumerate(sys.argv):
            if arg == flag and i + 1 < len(sys.argv):
                os.environ[env_var] = sys.argv[i + 1]
                break
            if arg.startswith(flag + "="):
                os.environ[env_var] = arg.split("=", 1)[1]
                break


def add_hf_env_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Declare ``--hf-endpoint`` / ``--hf-home`` on ``parser`` and return it.

    The flags default to ``None`` (i.e. "respect the environment"). The actual
    environment override is applied earlier by :func:`apply_hf_env`; declaring
    them here only keeps the script's own ``argparse`` from rejecting the
    flags and documents them in ``--help``.
    """
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Hugging Face endpoint override (env fallback: HF_ENDPOINT).",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Hugging Face cache directory override (env fallback: HF_HOME).",
    )
    return parser
