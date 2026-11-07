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

"""Pi session branch history captured as durable, incrementally imported evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
_CHECKPOINT_EVENTS_KEY = "events"
_VISIBLE_MESSAGE_ROLES = frozenset({"user", "assistant", "toolResult", "bashExecution"})


class PiSessionInput(BaseModel):
    """One offline Pi session archive plus the leaf whose ancestry is captured."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible Pi event with role, rendered text, and the original record."""

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured branch event that replays through the Pi event Definition."""

    event: PiEventInput


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the stable identity for one Pi event source."""

    identity = f"pi:{quote(session_id, safe='')}:{quote(event_id, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")
    return identity


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Parse one Pi session archive into the visible ancestry of the chosen leaf."""

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be valid UTF-8") from error

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Pi session archive must be valid JSONL") from error
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be JSON objects")
        records.append(record)

    if not records:
        raise ValueError("Pi session archive must start with a session header")

    header = records[0]
    if header.get("type") != "session" or header.get("version") != 3:
        raise ValueError("Pi session header must declare type=session and version=3")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must declare a non-empty string id")

    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records[1:]:
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Pi session records must declare a non-empty string id")
        if event_id in by_id:
            raise ValueError("Pi session record ids must be unique")
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str):
                raise ValueError("Pi session record parentId must be a string or null")
            if parent_id == event_id:
                raise ValueError("Pi session record cannot be its own parent")
            if parent_id not in by_id:
                raise ValueError("Pi session record parentId must reference an earlier node")
        by_id[event_id] = record
        order.append(event_id)

    if not order:
        raise ValueError("Pi session archive must contain at least one record")

    leaf_id = value.leaf_id if value.leaf_id is not None else order[-1]
    if leaf_id not in by_id:
        raise ValueError("leaf_id must reference an existing Pi session node")

    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            raise ValueError("Pi session graph must not contain cycles")
        seen.add(current)
        record = by_id[current]
        chain.append(record)
        parent = record.get("parentId")
        current = parent if isinstance(parent, str) else None
    chain.reverse()

    events: list[PiEventInput] = []
    for record in chain:
        event = _event_from_record(session_id, record)
        if event is not None:
            events.append(event)
    return tuple(events)


def _event_from_record(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    record_type = record.get("type")
    event_id = record["id"]
    parent_id = record.get("parentId")

    role: str
    body: str | None
    if record_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        message_role = message.get("role")
        role = message_role if isinstance(message_role, str) else "message"
        if role == "toolResult":
            body = _tool_result_body(message)
        elif role == "bashExecution":
            body = _bash_body(message)
        else:
            body = _content_body(message.get("content"))
    elif record_type == "compaction" or record_type == "branch_summary":
        summary = record.get("summary")
        if not isinstance(summary, str):
            return None
        role = record_type
        body = summary
    elif record_type == "custom_message":
        role = "custom"
        body = _content_body(record.get("content"))
    else:
        return None

    if body is None:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=event_id,
        parent_id=parent_id if isinstance(parent_id, str) else None,
        event_type=record_type,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _tool_result_body(message: dict[str, Any]) -> str | None:
    tool_name = message.get("toolName")
    call_id = message.get("toolCallId")
    text = _content_body(message.get("content")) or ""
    return f"[tool_result {tool_name} {call_id}] {text}"


def _bash_body(message: dict[str, Any]) -> str | None:
    command = message.get("command")
    output = message.get("output")
    command_text = command if isinstance(command, str) else ""
    output_text = output if isinstance(output, str) else ""
    return f"$ {command_text}\n{output_text}"


def _content_body(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content if content else None
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "toolCall":
            name = block.get("name")
            call_id = block.get("id")
            arguments = block.get("arguments")
            parts.append(f"[tool_call {name} {call_id}] {_compact_json(arguments)}")
    if not parts:
        return None
    return "\n".join(parts)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class PiEventSourceAdapter:
    """Resolve and read Pi event Sources from parsed events."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=pi_event_source_id(value.session_id, value.event_id),
            materialization=SourceMaterialization.CAPTURED,
            event=value,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return source.event


class PiEventTextEvidenceProjection:
    """Expose one branch event as canonical textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        event = source.event
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.name,
            content=event.content,
            metadata={
                "session_id": event.session_id,
                "event_id": event.event_id,
                "parent_id": event.parent_id,
                "event_type": event.event_type,
                "role": event.role,
            },
        )


class PiSessionConnector:
    """Incrementally capture the visible ancestry of a Pi session leaf."""

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: Path | str, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self._path.read_bytes()
        events = parse_pi_session(PiSessionInput(content=content, leaf_id=self._leaf_id))

        imported = _checkpoint_events(session.checkpoint)
        for event in events:
            checksum = _record_checksum(event.record)
            if event.event_id in imported:
                if imported[event.event_id] != checksum:
                    raise ValueError("imported Pi event was rewritten")
                continue
            await session.submit(event.event_id, PI_EVENT_SOURCE_NAME, event)
            imported[event.event_id] = checksum

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint={_CHECKPOINT_EVENTS_KEY: imported},
        )


def _checkpoint_events(checkpoint: JsonValue | None) -> dict[str, str]:
    if not isinstance(checkpoint, dict):
        return {}
    events = checkpoint.get(_CHECKPOINT_EVENTS_KEY)
    if not isinstance(events, dict):
        return {}
    return {str(key): value for key, value in events.items() if isinstance(value, str)}


def _record_checksum(record: dict[str, JsonValue]) -> str:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    projections=(PiEventTextEvidenceProjection(),),
)

__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
