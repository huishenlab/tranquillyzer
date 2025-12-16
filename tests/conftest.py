import importlib.util
import site
import sys
import types
import pytest
from pathlib import Path

# Prefer an installed tranquillyzer; fall back to local source when not available.
if importlib.util.find_spec("tranquillyzer") is None:
    site.addsitedir(str(Path(__file__).resolve().parent.parent))

# Provide lightweight tensorflow stub if not installed to keep unit tests fast.
if importlib.util.find_spec("tensorflow") is None:
    tf_stub = types.ModuleType("tensorflow")
    tf_stub.keras = types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda: None))
    sys.modules.setdefault("tensorflow", tf_stub)


@pytest.fixture
def fake_tf(monkeypatch):
    """Alias for monkeypatch to avoid TF-heavy dependency name leakage in tests."""
    return monkeypatch
