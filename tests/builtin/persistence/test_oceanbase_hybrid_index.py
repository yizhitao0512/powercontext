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

"""Statement shape and channel recovery for the native hybrid index.

These checks need no server: the statement is compiled against the MySQL dialect, and the
union result is fed to the splitter as plain mappings. What only a live seekdb can prove —
that the plan really uses both indexes — is asserted by the published acceptance test.
"""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import mysql

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.persistence.oceanbase.memory_index import (
    OceanBaseMemoryHybridIndex,
    _hybrid_channels,
)

PROFILE = EmbeddingProfile(
    profile_id="hybrid-index-test",
    model="deterministic",
    dimension=3,
    distance="l2",
    normalization="unit",
)
MEMORY = ArtifactRef(family="memory", artifact_id="mem-a", revision=7)


def _compile(index: OceanBaseMemoryHybridIndex, *, tagged: bool) -> str:
    statement = index._tagged_search_sql if tagged else index._search_sql
    return statement.compile(dialect=mysql.dialect(paramstyle="pyformat")).string


def _row(
    channel: str,
    *,
    artifact: str = "mem-a",
    entry: str = "e1",
    version: str = "v1",
    score: float | None = None,
    distance: float | None = None,
) -> dict[str, object]:
    return {
        "channel": channel,
        "memory_artifact_id": artifact,
        "head_revision": 3,
        "entry_id": entry,
        "entry_version_id": version,
        "text": "body",
        "match_score": score,
        "distance": distance,
    }


def test_hybrid_statement_carries_both_channels_in_one_query() -> None:
    sql = _compile(OceanBaseMemoryHybridIndex(PROFILE), tagged=False)
    upper = sql.upper()
    assert "pc_memory_entry_heads" in sql
    assert "pc_memory_vector_entries" in sql
    assert "AGAINST" in upper
    assert "l2_distance" in sql.lower()
    assert "UNION ALL" in upper
    assert "APPROXIMATE" in upper
    # One round trip means one statement, so both candidate limits live inside it.
    assert upper.count("LIMIT ") == 2


def test_tagged_hybrid_statement_prefilters_inside_the_database() -> None:
    sql = _compile(OceanBaseMemoryHybridIndex(PROFILE), tagged=True)
    assert sql.count("pc_artifact_tags") == 2
    assert "APPROXIMATE" not in sql.upper()


def test_union_result_is_split_back_into_two_rankings() -> None:
    # The union does not promise a row order, so the rankings must be rebuilt from the
    # projected score and distance rather than from the order the rows arrive in.
    rows = [
        _row("vector", entry="e2", version="v2", distance=0.9),
        _row("fts", entry="e2", version="v2", score=1.2),
        _row("vector", entry="e1", version="v1", distance=0.1),
        _row("fts", entry="e1", version="v1", score=4.5),
        _row("fts", entry="e3", version="v3", score=4.5),
    ]
    channels = _hybrid_channels(rows, (MEMORY,))
    assert [hit.entry_id for hit in channels.fts] == ["e1", "e3", "e2"]
    assert [hit.entry_id for hit in channels.vector] == ["e1", "e2"]
    assert all(hit.distance is None for hit in channels.fts)
    assert [hit.distance for hit in channels.vector] == [0.1, 0.9]
    assert all(hit.memory_ref == MEMORY for hit in channels.fts + channels.vector)


def test_rows_outside_the_requested_heads_are_dropped() -> None:
    channels = _hybrid_channels([_row("fts", artifact="mem-z", entry="ez", score=9.9)], (MEMORY,))
    assert channels.fts == ()
    assert channels.vector == ()


def test_native_hybrid_index_advertises_both_channels() -> None:
    index = OceanBaseMemoryHybridIndex(PROFILE)
    assert index.capabilities.fts
    assert index.capabilities.vector
    assert index.capabilities.hybrid
    assert index.capabilities.tag_filter
    assert index.capabilities.embedding_profile == PROFILE
    assert [table.name for table in index.tables] == ["pc_memory_vector_entries"]


def test_hybrid_index_rejects_a_foreign_embedding_profile() -> None:
    from powercontext.builtin.artifacts.memory import CapabilityNotSupportedError, MemorySearchRequest

    index = OceanBaseMemoryHybridIndex(PROFILE)
    request = MemorySearchRequest(
        query="alpha",
        analyzed_query="alpha",
        memories=(MEMORY,),
        candidate_limit=8,
        mode="hybrid",
        query_vector=(1.0, 0.0, 0.0),
        embedding_profile=PROFILE.model_copy(update={"profile_id": "other"}),
    )
    with pytest.raises(CapabilityNotSupportedError):
        # The profile guard runs before the connection is used, so no server is required.
        import asyncio

        asyncio.run(index.search(None, "scope", request))
