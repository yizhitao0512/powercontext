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

import os
import shutil
import stat
import sys
from pathlib import Path

base = Path("/owned")
marker_fd = os.open(base / ".marker", os.O_RDONLY | os.O_NOFOLLOW)
with os.fdopen(marker_fd) as marker:
    if marker.read() != os.environ["PC_MARKER"]:
        raise SystemExit("Runner and Docker daemon do not share this directory")

mode = sys.argv[1]
if mode == "probe":
    print("Exclusive bind mount identity verified")
    raise SystemExit(0)

root = base / "evidence"
if root.is_symlink() or not root.is_dir():
    raise SystemExit("Evidence root is missing or a symlink")
device = root.stat().st_dev
entries = []
for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
    for name in [*dirs, *files]:
        path = Path(directory) / name
        info = path.lstat()
        if info.st_dev != device:
            raise SystemExit("Refusing to cross a filesystem boundary")
        if mode == "own" and (stat.S_ISLNK(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)):
            raise SystemExit("Refusing to upload links from evidence")
        entries.append(path)
if mode == "own":
    uid, gid = int(os.environ["PC_UID"]), int(os.environ["PC_GID"])
    for path in [*entries, root]:
        os.chown(path, uid, gid, follow_symlinks=False)
    print(f"Prepared {len(entries)} evidence entries for scanning")
elif mode == "clean":
    shutil.rmtree(root)
    print("Removed only the verified evidence tree")
else:
    raise SystemExit("Unsupported operation")
