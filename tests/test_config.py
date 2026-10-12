import importlib.util

import pytest

from rag import config


def load_config():
    """Execute a fresh copy of rag/config.py so the real module is left untouched."""
    spec = importlib.util.spec_from_file_location("config_copy", config.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "env, message",
    [
        ({"CHUNK_SIZE": "big"}, "CHUNK_SIZE must be an integer"),
        ({"RETRIEVAL_K": "0"}, "RETRIEVAL_K must be >= 1"),
        ({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "100"}, "must be smaller than CHUNK_SIZE"),
    ],
)
def test_invalid_settings(monkeypatch, env, message):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(RuntimeError, match=message):
        load_config()


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    cfg = load_config()
    assert (cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP) == (500, 50)


def test_require(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "  ")
    with pytest.raises(config.ConfigError, match="SOME_KEY is not set"):
        config.require("SOME_KEY")
    monkeypatch.setenv("SOME_KEY", "abc")
    assert config.require("SOME_KEY") == "abc"
