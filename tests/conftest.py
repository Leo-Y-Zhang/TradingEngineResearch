"""Shared pytest configuration for the TradingEngineResearch suite.

**Isolate tests from a local ``.env``.** An operator running a live/paper session keeps a ``.env``
in the repo root (mode, broker, secrets). The test suite must be deterministic regardless of any
``.env`` present — otherwise e.g. a ``MODE=LIVE`` / ``CONFIRM_LIVE=true`` ``.env`` would make the
fail-closed safety tests (which expect those UNSET) pass spuriously or fail. So we disable env-file
loading for the whole session here, at conftest import (before any settings are constructed), and
reset the settings singleton. Tests that need specific settings construct them explicitly.
"""

from __future__ import annotations

from core.config import EngineSettings, reset_settings

# Done at import (not in a fixture) so it precedes any module-level settings construction.
EngineSettings.model_config["env_file"] = None
reset_settings()
