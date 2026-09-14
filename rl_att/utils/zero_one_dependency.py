"""Hash-pinned optional wheel; never install into the frozen victim runtime."""

import hashlib
import os
from pathlib import Path
import sys

WHEEL = "zoopt-0.4.2-py3-none-any.whl"
SHA256 = "d015ab3633b8f1951c5caa59b05adff08bcd6655e6cc3a5f397b7abd2171faa5"


def dependency_record(path):
    path = Path(path).resolve()
    if path.name != WHEEL or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError("ZOOpt wheel differs from pinned 0.4.2 artifact")
    return dict(name="zoopt", version="0.4.2", path=str(path), sha256=SHA256,
                license="MIT", loading="isolated wheel zipimport; no environment installation")


def load_zoopt():
    path = os.environ.get("RL_ATT_ZOOPT_WHEEL")
    if not path:
        raise RuntimeError("Set RL_ATT_ZOOPT_WHEEL to the pinned optional dependency")
    record = dependency_record(path)
    if record["path"] not in sys.path:
        sys.path.insert(0, record["path"])
    import zoopt
    if not str(zoopt.__file__).startswith(record["path"] + os.sep):
        raise RuntimeError("ZOOpt was imported from an unpinned location")
    return zoopt
