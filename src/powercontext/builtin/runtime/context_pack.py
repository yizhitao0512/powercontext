# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Incremental context-pack wire protocol for reusing prior prepared context."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from powercontext.errors import PowerContextError

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES = 512
# 增量路径故意使用更细的分块：条目插入只影响相邻块，提高跨轮前缀复用率。
INCREMENTAL_CHUNK_BYTES = 96
_DIGEST_PREFIX = "sha256:"
_DIGEST_HEX_LENGTH = 64
_PACK_KEYS = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_KEYS = frozenset({"schema", "base_digest", "target_digest", "prefix_items", "appended"})


class ContextPackError(PowerContextError, ValueError):
    """Raised when a context pack or delta violates the wire protocol."""


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete, self-describing chunk of prepared context."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: Iterable[bytes], /) -> ContextPack:
        """Build a full pack from a bytes tuple without re-chunking."""

        normalized = tuple(_normalize_item(item) for item in items)
        digest = _pack_digest(normalized)
        return cls(normalized, digest, _pack_byte_count(normalized, digest))

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Deterministically chunk UTF-8 content while preserving raw bytes."""

        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be a positive integer")
        if not isinstance(content, str):
            raise ContextPackError("context content must be text")
        return cls.build(_chunk_utf8(content.encode("utf-8"), chunk_bytes))

    def to_bytes(self) -> bytes:
        """Serialize the pack to canonical wire JSON."""

        return _pack_wire(self.items, self.digest, self.byte_count)

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPack:
        """Decode and fully validate a serialized pack."""

        payload = _load_json_object(data, _PACK_KEYS)
        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError("unsupported context pack schema")
        items = _decode_items(payload["items"])
        digest = payload["digest"]
        if not isinstance(digest, str) or not _is_digest(digest):
            raise ContextPackError("context pack digest is invalid")
        if _pack_digest(items) != digest:
            raise ContextPackError("context pack digest does not match its items")
        byte_count = payload["byte_count"]
        if type(byte_count) is not int or byte_count < 0 or byte_count != len(bytes(data)):
            raise ContextPackError("context pack byte_count does not match the payload")
        return cls(items, digest, byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A prefix-reusing suffix delta between two full packs."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        """Keep the common prefix of ``base`` and append the target suffix."""

        prefix = 0
        for left, right in zip(base.items, target.items):
            if left != right:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        """Serialized delta size in bytes, including protocol fields."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize the delta to canonical wire JSON."""

        return _canonical(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": self.base_digest,
                "target_digest": self.target_digest,
                "prefix_items": self.prefix_items,
                "appended": [_b64(item) for item in self.appended],
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ContextPackDelta:
        """Decode and validate a serialized delta."""

        payload = _load_json_object(data, _DELTA_KEYS)
        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError("unsupported context pack delta schema")
        base_digest = payload["base_digest"]
        target_digest = payload["target_digest"]
        if not isinstance(base_digest, str) or not _is_digest(base_digest):
            raise ContextPackError("delta base_digest is invalid")
        if not isinstance(target_digest, str) or not _is_digest(target_digest):
            raise ContextPackError("delta target_digest is invalid")
        prefix_items = payload["prefix_items"]
        if type(prefix_items) is not int or prefix_items < 0:
            raise ContextPackError("delta prefix_items must be a non-negative integer")
        return cls(base_digest, target_digest, prefix_items, _decode_items(payload["appended"]))


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Validate ``delta`` against ``base`` and rebuild the target pack."""

    if delta.base_digest != base.digest:
        raise ContextPackError("delta does not apply to this base pack")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base pack item count")
    target = ContextPack.build((*base.items[: delta.prefix_items], *delta.appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("delta target digest does not match the rebuilt pack")
    return target


def _normalize_item(item: Any) -> bytes:
    if not isinstance(item, (bytes, bytearray, memoryview)):
        raise ContextPackError("context pack items must be bytes")
    return bytes(item)


def _b64(item: bytes) -> str:
    return base64.b64encode(item).decode("ascii")


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pack_digest(items: tuple[bytes, ...]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": [_b64(item) for item in items]}
    return _DIGEST_PREFIX + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _pack_wire(items: tuple[bytes, ...], digest: str, byte_count: int) -> bytes:
    return _canonical(
        {
            "schema": PACK_SCHEMA,
            "items": [_b64(item) for item in items],
            "digest": digest,
            "byte_count": byte_count,
        }
    )


def _pack_byte_count(items: tuple[bytes, ...], digest: str) -> int:
    count = 0
    for _ in range(16):
        length = len(_pack_wire(items, digest, count))
        if length == count:
            return count
        count = length
    raise ContextPackError("context pack byte_count did not converge")


def _is_digest(value: str) -> bool:
    if not value.startswith(_DIGEST_PREFIX):
        return False
    hex_part = value[len(_DIGEST_PREFIX) :]
    return len(hex_part) == _DIGEST_HEX_LENGTH and all(char in "0123456789abcdef" for char in hex_part)


def _load_json_object(data: Any, expected_keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ContextPackError("wire payload must be bytes")
    try:
        payload = json.loads(bytes(data))
    except (ValueError, UnicodeDecodeError) as error:
        raise ContextPackError("wire payload must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ContextPackError("wire payload must be a JSON object")
    if set(payload) != expected_keys:
        raise ContextPackError("wire payload fields do not match its schema")
    return payload


def _decode_items(values: Any) -> tuple[bytes, ...]:
    if not isinstance(values, list):
        raise ContextPackError("wire items must be a list")
    items: list[bytes] = []
    for value in values:
        if not isinstance(value, str):
            raise ContextPackError("wire items must be base64 strings")
        try:
            items.append(base64.b64decode(value, validate=True))
        except (ValueError, TypeError) as error:
            raise ContextPackError("wire item is not valid base64") from error
    return tuple(items)


def _chunk_utf8(data: bytes, limit: int) -> tuple[bytes, ...]:
    chunks: list[bytes] = []
    start = 0
    total = len(data)
    while start < total:
        end = min(start + limit, total)
        if end < total:
            while end > start and (data[end] & 0xC0) == 0x80:
                end -= 1
        if end <= start:
            raise ContextPackError("chunk_bytes cannot form a valid UTF-8 boundary")
        chunk = data[start:end]
        try:
            chunk.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContextPackError("chunk is not valid UTF-8") from error
        chunks.append(chunk)
        start = end
    return tuple(chunks)


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "INCREMENTAL_CHUNK_BYTES",
    "PACK_SCHEMA",
    "apply_context_pack_delta",
]
