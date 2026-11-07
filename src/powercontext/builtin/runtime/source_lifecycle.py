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

"""Propagate Source availability through the evidence a Scope depends on.

Callers register a small DAG: Sources are the roots, Memory entries and Topic
Memory artifacts are the nodes derived from them. Withdrawing a Source removes
the support of every descendant according to the ``all`` / ``any`` policy each
node declared, and restoring it puts the supported nodes back. Deactivation only
changes eligibility: entry bodies, revisions, and artifact revisions stay
exactly as they were, so historical references keep resolving for audit.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.artifacts.memory import Memory, MemoryEntryVersion, MemoryService
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_EVENTS_TABLE,
    SOURCE_LIFECYCLE_NODES_TABLE,
)
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.errors import ArtifactNotFoundError
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection

    from powercontext.builtin.runtime.application import BuiltinRuntime
    from powercontext.builtin.runtime.relational import BuiltinContext

NodeFamily: TypeAlias = Literal["source", "memory", "topic"]
DependencyPolicy: TypeAlias = Literal["all", "any"]
NodeReference: TypeAlias = SourceRef | MemoryCitation | ArtifactRef
ResolvePurpose: TypeAlias = Literal["generation", "audit"]

_POLICIES: tuple[DependencyPolicy, ...] = ("all", "any")
_PURPOSES: tuple[ResolvePurpose, ...] = ("generation", "audit")
_LIFECYCLE_REASON = "source-lifecycle"


class SourceLifecycleError(ValueError):
    """Reject a lifecycle request that cannot change the tracked graph."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SourceLifecycleEvent(BaseModel):
    """One Source lifecycle transition, validated before it reaches the graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: Annotated[int, Field(strict=True, gt=0)]
    sequence: Annotated[int, Field(strict=True, gt=0)]
    available: Annotated[bool, Field(strict=True)]
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank")
        return value


@dataclass(frozen=True, slots=True)
class _Node:
    node_id: str
    ordinal: int
    family: NodeFamily
    policy: DependencyPolicy
    parents: tuple[str, ...]
    reference: NodeReference
    available: bool
    deactivated: bool
    source_revision: int | None


@dataclass(frozen=True, slots=True)
class _EventRecord:
    sequence: int
    revision: int
    available: bool
    reason: str

    def matches(self, event: SourceLifecycleEvent) -> bool:
        return (self.revision, self.available, self.reason) == (
            event.revision,
            event.available,
            event.reason,
        )


def _dump_reference(reference: NodeReference) -> str:
    return json.dumps(reference.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _load_reference(family: str, payload: Mapping[str, object]) -> NodeReference:
    if family == "source":
        return SourceRef.model_validate(payload)
    if family == "memory":
        return MemoryCitation.model_validate(payload)
    return ArtifactRef.model_validate(payload)


def _node_from_row(row: Mapping[str, object]) -> _Node:
    family = str(row["family"])
    return _Node(
        node_id=str(row["node_id"]),
        ordinal=int(str(row["ordinal"])),
        family=family,  # type: ignore[arg-type]
        policy=str(row["policy"]),  # type: ignore[arg-type]
        parents=tuple(str(parent) for parent in json.loads(str(row["parents_json"]))),
        reference=_load_reference(family, json.loads(str(row["reference_json"]))),
        available=bool(row["available"]),
        deactivated=bool(row["deactivated"]),
        source_revision=None if row["source_revision"] is None else int(str(row["source_revision"])),
    )


def _support(states: Sequence[bool], policy: DependencyPolicy) -> bool:
    return all(states) if policy == "all" else any(states)


def _require_node_id(node_id: object) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("node-id")
    return node_id


def _require_policy(policy: object) -> DependencyPolicy:
    if policy not in _POLICIES:
        raise SourceLifecycleError("policy")
    return policy


def _require_parents(parents: object) -> tuple[str, ...]:
    if isinstance(parents, str) or not isinstance(parents, Sequence):
        raise SourceLifecycleError("parents")
    resolved = tuple(_require_node_id(parent) for parent in parents)
    if len(set(resolved)) != len(resolved):
        raise SourceLifecycleError("parents")
    return resolved


def _require_revision(revision: object) -> int:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise SourceLifecycleError("revision")
    return revision


class SourceLifecycleRepository:
    """Read and write the Scope's lifecycle graph without owning transactions."""

    def __init__(self, database: AsyncDatabase) -> None:
        self._database = database

    def transaction(self):
        return self._database.transaction()

    async def nodes(self, connection: AsyncConnection, scope_id: str, /) -> tuple[_Node, ...]:
        query = (
            select(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
            .order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)
        )
        rows = (await connection.execute(query)).mappings().all()
        return tuple(_node_from_row(row) for row in rows)

    async def node(self, connection: AsyncConnection, scope_id: str, node_id: str, /) -> _Node | None:
        query = select(SOURCE_LIFECYCLE_NODES_TABLE).where(
            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
            SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
        )
        row = (await connection.execute(query)).mappings().first()
        return None if row is None else _node_from_row(row)

    async def next_ordinal(self, connection: AsyncConnection, scope_id: str, /) -> int:
        query = select(func.max(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)).where(
            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id
        )
        highest = (await connection.execute(query)).scalar()
        return 1 if highest is None else int(highest) + 1

    async def insert_node(self, connection: AsyncConnection, scope_id: str, node: _Node, /) -> None:
        await connection.execute(
            insert(SOURCE_LIFECYCLE_NODES_TABLE).values(
                scope_id=scope_id,
                node_id=node.node_id,
                ordinal=node.ordinal,
                family=node.family,
                policy=node.policy,
                parents_json=json.dumps(node.parents),
                reference_json=_dump_reference(node.reference),
                available=node.available,
                deactivated=node.deactivated,
                source_revision=node.source_revision,
                created_at=datetime.now(UTC),
            )
        )

    async def update_node(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
        *,
        available: bool | None = None,
        deactivated: bool | None = None,
    ) -> None:
        changes: dict[str, object] = {}
        if available is not None:
            changes["available"] = available
        if deactivated is not None:
            changes["deactivated"] = deactivated
        if not changes:
            return
        await connection.execute(
            update(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
            )
            .values(**changes)
        )

    async def last_event(self, connection: AsyncConnection, scope_id: str, node_id: str, /) -> _EventRecord | None:
        query = (
            select(SOURCE_LIFECYCLE_EVENTS_TABLE)
            .where(
                SOURCE_LIFECYCLE_EVENTS_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_EVENTS_TABLE.c.node_id == node_id,
            )
            .order_by(SOURCE_LIFECYCLE_EVENTS_TABLE.c.sequence.desc())
            .limit(1)
        )
        row = (await connection.execute(query)).mappings().first()
        if row is None:
            return None
        return _EventRecord(
            sequence=int(str(row["sequence"])),
            revision=int(str(row["revision"])),
            available=bool(row["available"]),
            reason=str(row["reason"]),
        )

    async def insert_event(
        self,
        connection: AsyncConnection,
        scope_id: str,
        event: SourceLifecycleEvent,
        /,
    ) -> None:
        await connection.execute(
            insert(SOURCE_LIFECYCLE_EVENTS_TABLE).values(
                scope_id=scope_id,
                node_id=event.node_id,
                sequence=event.sequence,
                revision=event.revision,
                available=event.available,
                reason=event.reason,
                applied_at=datetime.now(UTC),
            )
        )

    async def hidden_topic_ids(self, connection: AsyncConnection, scope_id: str, /) -> tuple[str, ...]:
        query = select(SOURCE_LIFECYCLE_NODES_TABLE.c.reference_json).where(
            SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
            SOURCE_LIFECYCLE_NODES_TABLE.c.family == "topic",
            SOURCE_LIFECYCLE_NODES_TABLE.c.available.is_(False),
        )
        rows = (await connection.execute(query)).scalars().all()
        return tuple(str(json.loads(payload)["artifact_id"]) for payload in rows)


class SourceLifecycleApplication:
    """Select the lifecycle service for one Scope and cache hidden Topic ids."""

    def __init__(self, runtime: BuiltinRuntime) -> None:
        self._runtime = runtime
        self._hidden_topics: dict[str, frozenset[str]] = {}

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, self, validate_scope_id(scope_id))

    def invalidate(self, scope_id: str, /) -> None:
        self._hidden_topics.pop(scope_id, None)

    async def hidden_topic_ids(self, scope_id: str, /) -> frozenset[str]:
        """Return Topic artifact ids the lifecycle keeps out of recall."""

        resolved = validate_scope_id(scope_id)
        cached = self._hidden_topics.get(resolved)
        if cached is not None:
            return cached
        repository = self._repository()
        if repository is None:
            self._hidden_topics[resolved] = frozenset()
            return frozenset()
        async with repository.transaction() as connection:
            hidden = frozenset(await repository.hidden_topic_ids(connection, resolved))
        self._hidden_topics[resolved] = hidden
        return hidden

    def _repository(self) -> SourceLifecycleRepository | None:
        database = getattr(self._runtime._provider, "database", None)
        return None if database is None else SourceLifecycleRepository(database)


class ScopedSourceLifecycle:
    """Register evidence nodes and apply Source lifecycle events in one Scope."""

    def __init__(
        self,
        runtime: BuiltinRuntime,
        application: SourceLifecycleApplication,
        scope_id: str,
    ) -> None:
        self._runtime = runtime
        self._application = application
        self.scope_id = scope_id

    def _repository(self) -> SourceLifecycleRepository:
        repository = self._application._repository()
        if repository is None:
            raise SourceLifecycleError("store")
        return repository

    async def register_source(self, node_id: str, source_ref: SourceRef, /, *, revision: int = 1) -> None:
        """Track one already captured Source at the given lifecycle revision."""

        await self._register(
            node_id,
            family="source",
            reference=source_ref,
            parents=(),
            policy="all",
            source_revision=_require_revision(revision),
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Sequence[str],
        policy: DependencyPolicy = "all",
    ) -> None:
        """Track one Memory entry and the nodes that support its body as a whole."""

        await self._register(node_id, family="memory", reference=citation, parents=parents, policy=policy)

    async def register_topic(
        self,
        node_id: str,
        artifact_ref: ArtifactRef,
        /,
        *,
        parents: Sequence[str],
        policy: DependencyPolicy = "all",
    ) -> None:
        """Track one Topic Memory artifact and the nodes that support its body."""

        await self._register(node_id, family="topic", reference=artifact_ref, parents=parents, policy=policy)

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one Source event and return every currently available node id."""

        if not isinstance(event, SourceLifecycleEvent):
            raise SourceLifecycleError("event")
        repository = self._repository()
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            async with repository.transaction() as connection:
                nodes = await repository.nodes(connection, self.scope_id)
                if not nodes:
                    raise SourceLifecycleError("unknown-source")
                by_id = {node.node_id: node for node in nodes}
                target = by_id.get(event.node_id)
                if target is None or target.family != "source":
                    raise SourceLifecycleError("unknown-source")
                if target.source_revision != event.revision:
                    raise SourceLifecycleError("event-revision")
                last = await repository.last_event(connection, self.scope_id, event.node_id)
                replay = self._require_acceptable_sequence(last, event)
                states = self._propagate(nodes, event)
                if not replay:
                    await repository.insert_event(connection, self.scope_id, event)
                await self._store_states(connection, nodes, states)
            await self._sync_memory(context)
            self._application.invalidate(self.scope_id)
            return sorted(node_id for node_id, available in states.items() if available)

    async def resolve(self, node_id: str, /, *, purpose: ResolvePurpose) -> NodeReference:
        """Return the evidence a node stands on, honoring generation eligibility."""

        resolved = _require_node_id(node_id)
        if purpose not in _PURPOSES:
            raise SourceLifecycleError("purpose")
        repository = self._repository()
        async with repository.transaction() as connection:
            node = await repository.node(connection, self.scope_id, resolved)
        if node is None:
            raise SourceLifecycleError("unknown-node")
        if purpose == "generation" and not node.available:
            raise SourceLifecycleError("unavailable-node")
        return node.reference

    async def _register(
        self,
        node_id: object,
        *,
        family: NodeFamily,
        reference: NodeReference,
        parents: object,
        policy: object,
        source_revision: int | None = None,
    ) -> None:
        resolved_id = _require_node_id(node_id)
        resolved_policy = _require_policy(policy)
        resolved_parents = () if family == "source" else _require_parents(parents)
        if family != "source" and not resolved_parents:
            raise SourceLifecycleError("parents")
        await self._verify_reference(family, reference)
        repository = self._repository()
        async with self._runtime._context(self.scope_id) as context, self._runtime._locked(self.scope_id):
            async with repository.transaction() as connection:
                existing = await repository.node(connection, self.scope_id, resolved_id)
                if existing is not None:
                    same = _same_registration(
                        existing,
                        family,
                        reference,
                        resolved_parents,
                        resolved_policy,
                        source_revision,
                    )
                    if not same:
                        raise SourceLifecycleError("conflict")
                else:
                    await self._insert_node(
                        connection,
                        _Node(
                            node_id=resolved_id,
                            ordinal=await repository.next_ordinal(connection, self.scope_id),
                            family=family,
                            policy=resolved_policy,
                            parents=resolved_parents,
                            reference=reference,
                            available=True,
                            deactivated=False,
                            source_revision=source_revision,
                        ),
                    )
            if family == "memory":
                await self._sync_memory(context)
            self._application.invalidate(self.scope_id)

    async def _insert_node(self, connection: AsyncConnection, node: _Node) -> None:
        """Store a node whose availability is decided now, not by a later event."""

        parent_states = await self._parent_states(connection, node.parents)
        available = True if node.family == "source" else _support(parent_states, node.policy)
        stored = _Node(
            node_id=node.node_id,
            ordinal=node.ordinal,
            family=node.family,
            policy=node.policy,
            parents=node.parents,
            reference=node.reference,
            available=available,
            deactivated=node.family == "memory" and not available,
            source_revision=node.source_revision,
        )
        await self._repository().insert_node(connection, self.scope_id, stored)

    async def _parent_states(self, connection: AsyncConnection, parents: tuple[str, ...]) -> tuple[bool, ...]:
        states: list[bool] = []
        for parent in parents:
            node = await self._repository().node(connection, self.scope_id, parent)
            if node is None:
                raise SourceLifecycleError("unknown-parent")
            states.append(node.available)
        return tuple(states)

    async def _verify_reference(self, family: NodeFamily, reference: NodeReference) -> None:
        expected: tuple[type[object], ...] = {
            "source": (SourceRef,),
            "memory": (MemoryCitation,),
            "topic": (ArtifactRef,),
        }[family]
        if not isinstance(reference, expected):
            raise SourceLifecycleError("reference")
        try:
            await self._read_reference(family, reference)
        except Exception as error:
            raise SourceLifecycleError("unknown-reference") from error

    async def _read_reference(self, family: NodeFamily, reference: NodeReference) -> None:
        if family == "source" and isinstance(reference, SourceRef):
            await self._runtime.records.for_scope(self.scope_id).get_source(
                reference.source_type,
                reference.source_id,
            )
        elif family == "memory" and isinstance(reference, MemoryCitation):
            await self._runtime.memory.for_scope(self.scope_id).get(GetMemoryEntryRequest(citation=reference))
        elif isinstance(reference, ArtifactRef):
            await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                reference.family,
                reference.artifact_id,
                reference.revision,
            )

    def _require_acceptable_sequence(
        self,
        last: _EventRecord | None,
        event: SourceLifecycleEvent,
    ) -> bool:
        """Return True when the event only replays the sequence already stored."""

        if last is None:
            return False
        if event.sequence < last.sequence:
            raise SourceLifecycleError("event-sequence")
        if event.sequence == last.sequence:
            if not last.matches(event):
                raise SourceLifecycleError("event-sequence")
            return True
        return False

    def _propagate(
        self,
        nodes: Sequence[_Node],
        event: SourceLifecycleEvent,
    ) -> dict[str, bool]:
        """Recompute every derived node in registration order, which is topological."""

        states = {node.node_id: node.available for node in nodes}
        states[event.node_id] = event.available
        for node in nodes:
            if node.family == "source":
                continue
            states[node.node_id] = _support(
                tuple(states[parent] for parent in node.parents),
                node.policy,
            )
        return states

    async def _store_states(
        self,
        connection: AsyncConnection,
        nodes: Sequence[_Node],
        states: Mapping[str, bool],
    ) -> None:
        repository = self._repository()
        for node in nodes:
            available = states[node.node_id]
            if available == node.available:
                continue
            await repository.update_node(
                connection,
                self.scope_id,
                node.node_id,
                available=available,
                deactivated=True if node.family == "memory" and not available else None,
            )

    async def _sync_memory(self, context: BuiltinContext) -> None:
        """Make the native Memory manifest agree with the propagated verdicts."""

        nodes = await self._load_nodes()
        restored: tuple[MemoryEntryVersion, ...] = ()
        service = context.artifacts.memory
        head = await _head_or_none(service, context.artifacts.memory_artifact_id)
        if head is None:
            return
        entries = await service.entries(head)
        states = {item.entry_id: item.state for item in head.content.manifest.entries}
        to_inactive, restored = _memory_transitions(nodes, entries, states)
        if to_inactive:
            head = await service.forget(head, entries=to_inactive, reason=_LIFECYCLE_REASON)
        if restored:
            await service.reactivate(head, entries=restored, reason=_LIFECYCLE_REASON)
        settled = _settled_nodes(nodes, states, restored)
        if settled:
            await self._clear_deactivated(settled)

    async def _clear_deactivated(self, node_ids: Sequence[str]) -> None:
        repository = self._repository()
        async with repository.transaction() as connection:
            for node_id in node_ids:
                await repository.update_node(connection, self.scope_id, node_id, deactivated=False)

    async def _load_nodes(self) -> tuple[_Node, ...]:
        repository = self._repository()
        async with repository.transaction() as connection:
            return await repository.nodes(connection, self.scope_id)


def _same_registration(
    node: _Node,
    family: NodeFamily,
    reference: NodeReference,
    parents: tuple[str, ...],
    policy: DependencyPolicy,
    source_revision: int | None,
) -> bool:
    return (
        node.family == family
        and node.policy == policy
        and node.parents == parents
        and node.source_revision == source_revision
        and _dump_reference(node.reference) == _dump_reference(reference)
    )


def _settled_nodes(
    nodes: Sequence[_Node],
    states: Mapping[str, str],
    restored: Sequence[MemoryEntryVersion],
) -> tuple[str, ...]:
    """Return entries the lifecycle deactivated that are active again."""

    restored_ids = {entry.entry_id for entry in restored}
    settled: list[str] = []
    for node in nodes:
        if node.family != "memory" or not node.available or not node.deactivated:
            continue
        entry_id = _citation_of(node).entry_id
        if entry_id in restored_ids or states.get(entry_id) == "active":
            settled.append(node.node_id)
    return tuple(settled)


def _citation_of(node: _Node) -> MemoryCitation:
    if not isinstance(node.reference, MemoryCitation):
        raise SourceLifecycleError("reference")
    return node.reference


def _memory_transitions(
    nodes: Sequence[_Node],
    entries: Sequence[MemoryEntryVersion],
    states: Mapping[str, str],
) -> tuple[tuple[MemoryEntryVersion, ...], tuple[MemoryEntryVersion, ...]]:
    """Split the Memory entries that must leave or rejoin the active set."""

    versions = {entry.entry_id: entry for entry in entries}
    to_inactive: list[MemoryEntryVersion] = []
    to_active: list[MemoryEntryVersion] = []
    for node in nodes:
        if node.family != "memory":
            continue
        entry_id = _citation_of(node).entry_id
        state = states.get(entry_id)
        entry = versions.get(entry_id)
        if state is None or entry is None:
            continue
        if not node.available and state == "active":
            to_inactive.append(entry)
        elif node.available and node.deactivated and state == "inactive":
            to_active.append(entry)
    return tuple(to_inactive), tuple(to_active)


async def _head_or_none(service: MemoryService, artifact_id: str) -> Memory | None:
    try:
        return await service.head(artifact_id)
    except ArtifactNotFoundError:
        return None


__all__ = ["SourceLifecycleError", "SourceLifecycleEvent"]
