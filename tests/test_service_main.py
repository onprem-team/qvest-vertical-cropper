"""The console entry point the container actually runs.

Never exercised by the API tests, which build the app directly. A typo here surfaces only
as a container that will not boot on the Brev instance.
"""
from __future__ import annotations

import pytest

from v_cropper.service import main as service_main


@pytest.fixture
def run_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        service_main.uvicorn, "run",
        lambda app, **kwargs: captured.update(app=app, **kwargs),
    )
    return captured


def test_serves_the_app_on_all_interfaces_by_default(monkeypatch, run_kwargs):
    for name in ("CROPPER_HOST", "CROPPER_PORT", "CROPPER_LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)

    service_main.main()

    # Binding to 0.0.0.0 is correct only because compose publishes to loopback; see compose.yaml.
    assert run_kwargs == {
        "app": "v_cropper.service.app:app",
        "host": "0.0.0.0",
        "port": 8080,
        "log_level": "info",
    }


def test_host_port_and_log_level_are_configurable(monkeypatch, run_kwargs):
    monkeypatch.setenv("CROPPER_HOST", "127.0.0.1")
    monkeypatch.setenv("CROPPER_PORT", "9000")
    monkeypatch.setenv("CROPPER_LOG_LEVEL", "debug")

    service_main.main()

    assert run_kwargs["host"] == "127.0.0.1"
    assert run_kwargs["port"] == 9000
    assert run_kwargs["log_level"] == "debug"


def test_a_non_numeric_port_fails_loudly(monkeypatch, run_kwargs):
    """Better to crash on boot than to silently listen on a default port."""
    monkeypatch.setenv("CROPPER_PORT", "not-a-port")
    with pytest.raises(ValueError):
        service_main.main()
