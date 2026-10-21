"""Host-side frozen execution plan for the AR030 paged activation cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from actfold.core.pyramid_plan import PyramidPlan


class PlanStaleError(ValueError):
    """Raised when a plan is entered after the cache epoch changed."""

    def __init__(self, plan_epoch: int, cache_epoch: int) -> None:
        self.plan_epoch = plan_epoch
        self.cache_epoch = cache_epoch
        super().__init__(f"plan epoch {plan_epoch} is stale (cache epoch {cache_epoch})")


def freeze_mapping(value: Mapping[Any, Any]) -> Mapping[Any, Any]:
    frozen: dict[Any, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            frozen[key] = freeze_mapping(item)
        elif isinstance(item, tuple):
            frozen[key] = tuple(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


def availability_words(lens: Mapping[int, int], seq_len: int, page_tokens: int) -> tuple[int, ...]:
    """Encode a page-lens mapping as a 32-bit token bitmap."""

    words = [0] * ((seq_len + 31) // 32)
    for logical, length in lens.items():
        start = logical * page_tokens
        for token in range(start, min(start + length, seq_len)):
            words[token // 32] |= 1 << (token % 32)
    return tuple(words)


def decode_availability(words: tuple[int, ...], seq_len: int) -> tuple[bool, ...]:
    out: list[bool] = []
    for token in range(seq_len):
        word = words[token // 32] if token // 32 < len(words) else 0
        out.append(bool(word & (1 << (token % 32))))
    return tuple(out)


@dataclass(frozen=True)
class FoldPagePlan:
    """Immutable host plan: mappings, availability bitmap and byte budgets."""

    plan_id: str
    epoch: int
    branches: tuple[str, ...]
    layers: tuple[int, ...]
    seq_len: Mapping[str, Mapping[int, int]]
    tables: Mapping[str, Mapping[int, Mapping[int, int]]]
    availability: Mapping[str, Mapping[int, tuple[tuple[int, ...], int]]]
    expected_bytes: int
    budget_pages: Mapping[int, int]
    config: Mapping[str, Any]

    def availability_for(self, branch: str, layer: int) -> tuple[bool, ...] | None:
        item = self.availability.get(branch, {}).get(layer)
        if item is None:
            return None
        words, seq_len = item
        return decode_availability(words, seq_len)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "epoch": self.epoch,
            "branches": list(self.branches),
            "layers": list(self.layers),
            "seq_len": {
                b: {int(k): int(v) for k, v in layers.items()} for b, layers in self.seq_len.items()
            },
            "tables": {
                b: {
                    int(l): {int(p): int(pid) for p, pid in pages.items()}
                    for l, pages in layers.items()
                }
                for b, layers in self.tables.items()
            },
            "availability": {
                b: {int(l): [list(words), int(seq_len)] for l, (words, seq_len) in layers.items()}
                for b, layers in self.availability.items()
            },
            "expected_bytes": int(self.expected_bytes),
            "budget_pages": {int(k): int(v) for k, v in self.budget_pages.items()},
            "config": _plain(self.config),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FoldPagePlan":
        if not isinstance(payload, Mapping):
            raise ValueError("plan payload must be a mapping")
        try:
            seq_len = {
                str(b): {int(l): int(v) for l, v in layers.items()}
                for b, layers in payload["seq_len"].items()
            }
            tables = {
                str(b): {
                    int(l): {int(p): int(pid) for p, pid in pages.items()}
                    for l, pages in layers.items()
                }
                for b, layers in payload["tables"].items()
            }
            availability = {
                str(b): {
                    int(l): (tuple(int(w) for w in item[0]), int(item[1]))
                    for l, item in layers.items()
                }
                for b, layers in payload["availability"].items()
            }
            plan = cls(
                plan_id=str(payload["plan_id"]),
                epoch=int(payload["epoch"]),
                branches=tuple(str(b) for b in payload["branches"]),
                layers=tuple(int(l) for l in payload["layers"]),
                seq_len=freeze_mapping(seq_len),
                tables=freeze_mapping(tables),
                availability=freeze_mapping(availability),
                expected_bytes=int(payload["expected_bytes"]),
                budget_pages=freeze_mapping(
                    {int(k): int(v) for k, v in payload["budget_pages"].items()}
                ),
                config=freeze_mapping(dict(payload["config"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid plan payload: {exc}") from exc
        if plan.plan_id != compute_plan_id(plan):
            raise ValueError("plan payload id does not match its contents")
        return plan


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def compute_plan_id(plan: FoldPagePlan) -> str:
    payload = {
        "epoch": plan.epoch,
        "branches": list(plan.branches),
        "layers": list(plan.layers),
        "seq_len": _plain(plan.seq_len),
        "tables": _plain(plan.tables),
        "availability": _plain(plan.availability),
        "expected_bytes": plan.expected_bytes,
        "budget_pages": _plain(plan.budget_pages),
        "config": _plain(plan.config),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def page_bytes_for(
    page_tokens: int,
    batch: int,
    feat_shape: tuple[int, ...],
    names: tuple[str, ...],
    dtype: Any,
) -> int:
    element = max(1, int(getattr(dtype, "itemsize", 4)))
    per_token = max(1, batch) * element
    for dim in feat_shape:
        per_token *= int(dim)
    return page_tokens * per_token * max(1, len(names))


def budget_pages(plan: PyramidPlan, page_tokens: int) -> dict[int, int]:
    return {
        layer: max(1, int(plan.budget_tokens(layer)) // page_tokens) for layer in range(plan.depth)
    }
