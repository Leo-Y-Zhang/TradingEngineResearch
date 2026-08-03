"""Guards for the deploy artifacts (ROADMAP Phase 6 item 6).

These cannot build an image (no Docker daemon in CI here), so they check the
things that silently break a deploy: the files exist, the entrypoint dispatches
the documented modes, the Dockerfile wires the right entrypoint/extras, and the
deploy entry points the container invokes are importable.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_deploy_files_exist():
    for rel in ("Dockerfile", ".dockerignore", "docker-compose.yml",
                "scripts/entrypoint.sh", ".env.example", "docs/DEPLOY.md"):
        assert (ROOT / rel).exists(), f"missing deploy artifact: {rel}"


def test_entrypoint_dispatches_documented_modes():
    text = _read("scripts/entrypoint.sh")
    for marker in ("combined)", "loop)", "api)",
                   "python -m ops.run_loop --serve",
                   "python -m ops.run_loop",
                   "ops.api:create_app_from_settings"):
        assert marker in text, f"entrypoint.sh missing {marker!r}"


def test_dockerfile_wires_entrypoint_and_extras():
    text = _read("Dockerfile")
    assert 'ENTRYPOINT ["entrypoint.sh"]' in text
    assert 'CMD ["combined"]' in text
    assert "app,persistence,ingestion,vault" in text  # runtime extras
    assert "constraints.txt" in text                   # pinned, reproducible
    assert "USER tradingengineresearch" in text                    # non-root runtime


def test_compose_is_loopback_only_and_has_state_volume():
    text = _read("docker-compose.yml")
    assert "127.0.0.1:8000:8000" in text   # unauthenticated API → loopback only
    assert "engine-state:/app/state" in text   # durable state volume


def test_deploy_entry_points_importable():
    # The exact symbols the container entrypoint invokes must exist.
    from ops.api import create_app_from_settings  # noqa: F401
    from ops.run_loop import serve_combined  # noqa: F401
