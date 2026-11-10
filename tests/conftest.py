"""Offline suite guard installed before collection, including Python children.

No test collection exclusions. Loopback fake HTTP servers remain available.
"""
import os
from pathlib import Path
import tempfile
from tests import network_guard

_child_guard = None
_old_pythonpath = None
_old_proxy_env = {}


def pytest_sessionstart(session):
    global _child_guard, _old_pythonpath
    network_guard.install()
    _child_guard = tempfile.TemporaryDirectory(prefix="graphintel-offline-")
    source = Path(network_guard.__file__).read_text(encoding="utf-8")
    (Path(_child_guard.name) / "sitecustomize.py").write_text(source, encoding="utf-8")
    _old_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = _child_guard.name + (os.pathsep + _old_pythonpath if _old_pythonpath else "")
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            _old_proxy_env[name] = os.environ.pop(name)


def pytest_sessionfinish(session, exitstatus):
    network_guard.restore()
    if _old_pythonpath is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = _old_pythonpath
    os.environ.update(_old_proxy_env)
    if _child_guard is not None:
        _child_guard.cleanup()
