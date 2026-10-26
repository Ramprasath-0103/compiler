import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nlpcc import models

CORPUS = Path(__file__).parent / "corpus"


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):

    for probe in ("_openai_compatible", "_anthropic", "_ollama"):
        monkeypatch.setattr(models, probe, lambda: None)
    monkeypatch.setenv("NLPCC_PROVIDER", "")
    monkeypatch.setenv("NLPCC_LOCAL_MODEL", "")
    monkeypatch.setattr(models, "LOCAL_MODEL", "")
