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

"""HTTP cross-encoder reranker that ranks a coarse Memory candidate pool."""

from __future__ import annotations

import math
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage

DEFAULT_HTTP_RERANK_BATCH_SIZE = 32
HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score candidates in bounded HTTP batches and merge them globally."""

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = DEFAULT_HTTP_RERANK_BATCH_SIZE,
        policy_id: str = HTTP_RERANK_POLICY_ID,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate, then keep the best ``limit`` original ranks."""

        if len(candidates) < 1:
            raise ValueError("rerank requires at least one candidate")
        if limit < 1 or limit > len(candidates):
            raise ValueError("rerank limit must be between one and the candidate count")

        ranked = tuple(enumerate(candidates, start=1))
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(ranked), self._batch_size):
            batch = ranked[start : start + self._batch_size]
            expected_ranks = {rank for rank, _ in batch}
            payload = {
                "query": query,
                "candidates": [{"rank": rank, "text": hit.text} for rank, hit in batch],
            }
            response = await self._client.post(self._endpoint, json=payload)
            requests += 1
            scores.update(_parse_scores(response, expected_ranks))

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in ordered),
            usage=InferenceUsage(requests=requests),
        )


def _parse_scores(response: httpx.Response, expected_ranks: set[int]) -> dict[int, float]:
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("reranker response must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise ValueError("reranker response must contain exactly 'scores'")
    items = payload["scores"]
    if not isinstance(items, list):
        raise ValueError("reranker scores must be a list")

    parsed: dict[int, float] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"rank", "score"}:
            raise ValueError("reranker score entries must contain exactly 'rank' and 'score'")
        rank = item["rank"]
        score = item["score"]
        if type(rank) is not int or rank not in expected_ranks:
            raise ValueError("reranker rank must be an expected integer rank")
        if rank in parsed:
            raise ValueError("reranker returned a duplicate rank")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("reranker score must be a finite number")
        parsed[rank] = float(score)

    if set(parsed) != expected_ranks:
        raise ValueError("reranker scores must cover every candidate rank exactly once")
    return parsed


__all__: tuple[Any, ...] = (
    "DEFAULT_HTTP_RERANK_BATCH_SIZE",
    "HTTP_RERANK_POLICY_ID",
    "HTTPMemoryReranker",
)
