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

"""Cross-call embedding coalescing with a bounded success-only vector cache.

One wrapper instance owns the shared state for its event loop: a coalescing
window that batches the texts of concurrent callers, an in-flight table that
lets a late caller join work already started, and an LRU cache of successful
vectors. Each caller keeps its own tickets, so cancellation, failures, and
usage accounting stay isolated per call.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from powercontext.builtin.inference.models import EmbeddingVector
    from powercontext.builtin.inference.protocols import EmbeddingModel

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_COALESCE_MS = 5.0
DEFAULT_CACHE_SIZE = 1024


@dataclass(slots=True)
class _Batch:
    """One provider request and the usage it reported."""

    index: int
    texts: tuple[str, ...]
    usage: InferenceUsage = field(default_factory=lambda: InferenceUsage(requests=1))


@dataclass(slots=True)
class _Ticket:
    """One deduplicated text that a caller waits on."""

    text: str
    future: asyncio.Future[EmbeddingVector]
    batch: _Batch | None = None


def _consume_exception(future: asyncio.Future[object]) -> None:
    """Mark a settled exception as retrieved so an abandoned caller stays silent."""

    if not future.cancelled():
        future.exception()


def _validated(result: EmbeddingResult, expected: int) -> tuple[EmbeddingVector, ...]:
    """Return the batch vectors, rejecting a count that cannot match the batch."""

    vectors = tuple(result.vectors)
    if len(vectors) != expected:
        raise ValueError("embedding batch returned a mismatched vector count")  # noqa: TRY003
    return vectors


def _add_optional(total: int | None, value: int | None) -> int | None:
    if value is None:
        return total
    return value if total is None else total + value


def _aggregate(batches: Iterable[_Batch]) -> InferenceUsage:
    requests = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    for batch in batches:
        requests += batch.usage.requests
        input_tokens = _add_optional(input_tokens, batch.usage.input_tokens)
        output_tokens = _add_optional(output_tokens, batch.usage.output_tokens)
    return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


class CoalescingEmbeddingModel:
    """Share identical text work across callers and cache successful vectors."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        coalesce_ms: float = DEFAULT_COALESCE_MS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least one")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003
        self._model = model
        self.profile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._pending: dict[str, _Ticket] = {}
        self._window: dict[str, _Ticket] = {}
        self._timer: asyncio.TimerHandle | None = None
        self._draining: set[asyncio.Task[None]] = set()
        self._batches = 0

    async def embed(self, texts: Sequence[str], /) -> EmbeddingResult:
        """Return one vector per input text, in order, with the usage this call waited on."""

        tickets: list[_Ticket] = []
        fresh: list[_Ticket] = []
        for text in texts:
            ticket = self._cached(text) or self._pending.get(text)
            if ticket is None:
                ticket = _Ticket(text=text, future=self._open_future())
                self._pending[text] = ticket
                fresh.append(ticket)
            tickets.append(ticket)
        if not tickets:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))
        if fresh:
            self._admit(fresh)
        return await self._collect(tickets)

    async def aclose(self) -> None:
        """Cancel queued window work; already started provider calls are left alone."""

        handle, self._timer = self._timer, None
        if handle is not None:
            handle.cancel()
        self._window.clear()

    def _open_future(self) -> asyncio.Future[EmbeddingVector]:
        future: asyncio.Future[EmbeddingVector] = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_exception)
        return future

    def _cached(self, text: str) -> _Ticket | None:
        if self._cache_size == 0:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        future = self._open_future()
        future.set_result(vector)
        return _Ticket(text=text, future=future)

    def _admit(self, fresh: Sequence[_Ticket]) -> None:
        for ticket in fresh:
            self._window[ticket.text] = ticket
        if self._coalesce_seconds <= 0:
            self._flush()
        elif self._timer is None:
            self._timer = asyncio.get_running_loop().call_later(self._coalesce_seconds, self._flush)

    def _flush(self) -> None:
        handle, self._timer = self._timer, None
        if handle is not None:
            handle.cancel()
        if not self._window:
            return
        queued = list(self._window.values())
        self._window.clear()
        loop = asyncio.get_running_loop()
        for start in range(0, len(queued), self._max_batch_size):
            chunk = queued[start : start + self._max_batch_size]
            self._batches += 1
            batch = _Batch(index=self._batches, texts=tuple(ticket.text for ticket in chunk))
            for ticket in chunk:
                ticket.batch = batch
            task = loop.create_task(self._run(batch, chunk))
            self._draining.add(task)
            task.add_done_callback(self._draining.discard)

    async def _run(self, batch: _Batch, tickets: Sequence[_Ticket]) -> None:
        try:
            result = await self._model.embed(batch.texts)
        except asyncio.CancelledError:
            self._settle(tickets, asyncio.CancelledError())
            raise
        except Exception as error:
            self._settle(tickets, error)
            return
        try:
            vectors = _validated(result, len(batch.texts))
        except ValueError as error:
            self._settle(tickets, error)
            return
        batch.usage = result.usage
        for ticket, vector in zip(tickets, vectors, strict=True):
            settled = tuple(vector)
            self._remember(ticket.text, settled)
            self._release(ticket)
            if not ticket.future.done():
                ticket.future.set_result(settled)

    def _settle(self, tickets: Sequence[_Ticket], error: BaseException) -> None:
        for ticket in tickets:
            self._release(ticket)
            if not ticket.future.done():
                ticket.future.set_exception(error)

    def _release(self, ticket: _Ticket) -> None:
        if self._pending.get(ticket.text) is ticket:
            del self._pending[ticket.text]

    def _remember(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size == 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    async def _collect(self, tickets: Sequence[_Ticket]) -> EmbeddingResult:
        batches: dict[int, _Batch] = {}
        vectors: list[EmbeddingVector] = []
        for ticket in tickets:
            # Shield the shared future: a cancelled caller must not cancel the work
            # that other callers are still waiting on.
            vectors.append(await asyncio.shield(ticket.future))
            if ticket.batch is not None:
                batches[ticket.batch.index] = ticket.batch
        return EmbeddingResult(vectors=tuple(vectors), usage=_aggregate(batches.values()))


__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_COALESCE_MS",
    "DEFAULT_MAX_BATCH_SIZE",
    "CoalescingEmbeddingModel",
]
