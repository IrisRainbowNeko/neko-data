import os
import sys

import pytest

from neko_data.runtime.cache import DiskShardCache


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 process-query API")
def test_windows_pid_probe_never_sends_console_signals(monkeypatch):
    def forbidden(*_args):
        pytest.fail("Windows lease liveness must never call os.kill")

    monkeypatch.setattr(os, "kill", forbidden)
    assert DiskShardCache._pid_is_alive(os.getpid())
    assert not DiskShardCache._pid_is_alive(2**31 - 1)
