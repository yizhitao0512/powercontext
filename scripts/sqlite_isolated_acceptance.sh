#!/usr/bin/env bash
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
set -euo pipefail
[[ ${GITHUB_RUN_ID:?} =~ ^[0-9]+$ && ${GITHUB_RUN_ATTEMPT:?} =~ ^[0-9]+$ ]]
project="pc-sqlite-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"

helper() {
    docker run --rm --network none \
        --mount "type=bind,source=$PC_EVIDENCE_DIR,target=/owned" \
        --env "PC_MARKER=$PC_MARKER" --env "PC_UID=$(id -u)" --env "PC_GID=$(id -g)" \
        --entrypoint python python:3.12-slim-bookworm /owned/helper.py "$1"
}

case "${1:?}" in
    prepare)
        dir=$(mktemp -d "${RUNNER_TEMP:?}/sqlite-evidence.XXXXXXXX")
        marker=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
        printf 'PC_EVIDENCE_DIR=%s\nPC_MARKER=%s\n' "$dir" "$marker" >> "$GITHUB_ENV"
        printf '%s' "$marker" > "$dir/.marker"
        cp scripts/sqlite_probe_files.py "$dir/helper.py"
        mkdir "$dir/evidence"
        export PC_EVIDENCE_DIR="$dir" PC_MARKER="$marker"
        for kind in container volume network; do
            case "$kind" in
                container) existing=$(docker ps -aq --filter "label=com.docker.compose.project=$project") ;;
                volume) existing=$(docker volume ls -q --filter "label=com.docker.compose.project=$project") ;;
                network) existing=$(docker network ls -q --filter "label=com.docker.compose.project=$project") ;;
            esac
            [[ -z "$existing" ]] || { echo "Project collision: $kind" >&2; exit 1; }
        done
        for tag in "$project-harness:latest" "$project-powercontext:latest"; do
            if docker image inspect "$tag" >/dev/null 2>&1; then echo "Image tag collision" >&2; exit 1; fi
        done
        helper probe
        printf 'PC_RESOURCES_OWNED=yes\n' >> "$GITHUB_ENV"
        ;;
    run)
        : "${PC_EVIDENCE_DIR:?}" "${PC_MARKER:?}" "${PC_RESOURCES_OWNED:?}"
        test "$PC_RESOURCES_OWNED" = yes
        git diff --exit-code HEAD^ HEAD -- e2e/bub/Dockerfile docker/Dockerfile src
        echo "Validated candidate parent: $(git rev-parse HEAD^)"
        export PC_SQLITE_PROJECT="$project"
        python3 - <<'PY'
import os
from pathlib import Path
p = Path("e2e/bub/run.sh")
s = p.read_text()
line = next(line for line in s.splitlines() if line.startswith("export COMPOSE_PROJECT_NAME="))
p.write_text(s.replace(line, 'export COMPOSE_PROJECT_NAME="$PC_SQLITE_PROJECT"'))
for name, cache in (("e2e/bub/Dockerfile", "powercontext-bub-e2e-uv"), ("docker/Dockerfile", "powercontext-uv")):
    p = Path(name)
    s = p.read_text()
    assert s.count("id=" + cache) == 1
    p.write_text(s.replace("id=" + cache, "id=" + os.environ["PC_SQLITE_PROJECT"] + "-" + cache))
PY
        export TMPDIR="$PC_EVIDENCE_DIR/tmp" XDG_CACHE_HOME="$PC_EVIDENCE_DIR/cache"
        export UV_CACHE_DIR="$PC_EVIDENCE_DIR/cache/uv" UV_PYTHON_INSTALL_DIR="$PC_EVIDENCE_DIR/python"
        export CODEX_HOME="$PC_EVIDENCE_DIR/codex" POWERCONTEXT_E2E_DATABASE=sqlite
        export POWERCONTEXT_E2E_OUTPUT="$PC_EVIDENCE_DIR/evidence"
        mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$CODEX_HOME"
        make harness-compose-acceptance
        ;;
    scan)
        : "${PC_EVIDENCE_DIR:?}" "${PC_MARKER:?}" "${SCENARIO_OUTCOME:?}"
        helper own
        export DATABASE=sqlite DIAGNOSTICS_PATH="$PC_EVIDENCE_DIR/diagnostics" EVIDENCE_PATH="$PC_EVIDENCE_DIR/evidence"
        bash scripts/scan_e2e_evidence.sh
        cat "$DIAGNOSTICS_PATH/summary.txt"
        ;;
    cleanup)
        if [[ -z "${PC_EVIDENCE_DIR:-}" || -z "${PC_MARKER:-}" ]]; then echo "No temporary directory created"; exit 0; fi
        case "$PC_EVIDENCE_DIR" in "${RUNNER_TEMP:?}"/sqlite-evidence.*) ;; *) echo "Unexpected path" >&2; exit 1 ;; esac
        [[ -d "$PC_EVIDENCE_DIR" && ! -L "$PC_EVIDENCE_DIR" ]]
        [[ $(cat "$PC_EVIDENCE_DIR/.marker") == "$PC_MARKER" ]]
        if [[ ${PC_RESOURCES_OWNED:-} == yes ]]; then
            export POWERCONTEXT_E2E_OUTPUT="$PC_EVIDENCE_DIR/evidence" POWERCONTEXT_E2E_CODEX_AUTH_MOUNT=/dev/null
            docker compose --project-name "$project" -f e2e/bub/compose.yaml down --volumes --remove-orphans
            for tag in "$project-harness:latest" "$project-powercontext:latest"; do
                docker image rm "$tag" 2>/dev/null || true
            done
            helper clean
        fi
        rm -rf --one-file-system "$PC_EVIDENCE_DIR"
        test ! -e "$PC_EVIDENCE_DIR"
        echo "Removed this run's exclusive directory"
        ;;
    *) exit 2 ;;
esac
