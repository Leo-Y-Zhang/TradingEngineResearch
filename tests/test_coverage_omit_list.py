"""The coverage omit list must never name a file that something imports.

``[tool.coverage.run].omit`` in pyproject.toml carries an authored list of research
modules excluded from the coverage floor. Its stated rule is:

    a file under research/ stays measured if it is reachable, transitively, from the
    test suite or from a production package.

so putting a file on that list is a factual claim -- "nothing imports this" -- that
silently rots. It has rotted once already: ``institutional_flow.py`` was added while a
test importing it was landing in the same window, which would have excluded a module the
suite actually exercises and quietly shrunk the measured surface. It was caught by hand.
Twice before that, a regex tried to derive the list and got it wrong; one attempt would
have dropped ``riskparity.py``, which had just been given sixteen tests.

This test replaces the hand-check. It parses the real import graph rather than matching
names, and it deliberately imports nothing from research/ -- an import here would drag
modules into the measured surface and change the number this suite is meant to police.

Two failure modes are covered: an omitted file that IS reachable (the list is stale and
under-measures), and an omitted path that no longer exists (the list is stale and lies).
"""

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The packages that ship. A research module reachable from one of these is production
# code by consequence, whatever directory it lives in, and must stay measured.
PROD_PACKAGES = (
    "core", "data", "strategies", "execution", "learning",
    "ops", "nlp", "backtesting", "broker",
)

# Vendored/extracted trees and build output. `codebase/` holds a second, older copy of
# this project; walking it would map duplicate module names onto the wrong files.
SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "build", "dist", ".hypothesis",
    "tradingengineresearch.egg-info", "codebase", "backend", "frontend", "secrets",
})


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _source_files() -> dict[str, Path]:
    """Every importable module in the repo, keyed by dotted name."""
    found: dict[str, Path] = {}
    for py in REPO_ROOT.rglob("*.py"):
        if SKIP_DIRS.isdisjoint(py.relative_to(REPO_ROOT).parts):
            found[_module_name(py)] = py
    return found


def _imported_names(path: Path) -> set[str]:
    """Dotted names imported by a file, including imports nested inside functions.

    Function-local imports are not incidental here -- several sleeve tests defer
    `from research.sleeves import ...` into the test body, so a module-level-only
    scan would report those modules as unreachable and wave through a stale entry.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover - a syntax error is another test's problem
        return set()

    names: set[str] = set()
    own = _module_name(path)
    package = own.rsplit(".", 1)[0] if "." in own else ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                if node.level > 1:
                    base = base[: len(base) - (node.level - 1)]
                prefix = ".".join(base)
                module = f"{prefix}.{node.module}" if node.module else prefix
            else:
                module = node.module or ""
            if module:
                names.add(module)
                # `from research.sleeves import pead` imports a module, not an attribute.
                names.update(f"{module}.{alias.name}" for alias in node.names)
    return names


def _reachable_files() -> set[Path]:
    """Transitive closure of imports from the test suite and the shipping packages."""
    modules = _source_files()

    roots = list((REPO_ROOT / "tests").rglob("*.py"))
    for package in PROD_PACKAGES:
        roots.extend((REPO_ROOT / package).rglob("*.py"))

    seen = set(roots)
    pending = list(roots)
    while pending:
        for name in _imported_names(pending.pop()):
            target = modules.get(name)
            if target is not None and target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def _omitted_files() -> list[str]:
    """Omit entries that name a specific file (skipping the `*` and `tests/` globs)."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    entries = config["tool"]["coverage"]["run"]["omit"]
    return [e for e in entries if not e.startswith(("*", "tests"))]


def test_no_omitted_file_is_reachable_from_tests_or_production() -> None:
    """An omitted file that something imports is measured code being hidden."""
    reachable = _reachable_files()
    stale = sorted(e for e in _omitted_files() if (REPO_ROOT / e) in reachable)

    assert not stale, (
        "These files are on the coverage omit list but ARE reachable from the test "
        "suite or a production package, so they must be REMOVED from the list "
        f"(the rule is stated in pyproject.toml): {stale}"
    )


def test_every_omitted_path_exists() -> None:
    """A path that no longer exists silently stops omitting anything."""
    missing = sorted(e for e in _omitted_files() if not (REPO_ROOT / e).is_file())

    assert not missing, (
        "These coverage omit entries do not point at a file that exists; they are "
        f"dead and should be deleted or corrected: {missing}"
    )


def test_the_graph_walk_actually_finds_things() -> None:
    """Guard the guard: a broken walk would return almost nothing and pass vacuously.

    Both tests above assert an emptiness. If `_reachable_files` silently degraded --
    a renamed source root, an exclusion that swallowed the tree -- they would still
    pass while checking nothing. This pins the walk to observable facts instead.
    """
    reachable = _reachable_files()
    assert len(reachable) > 100, f"import graph collapsed to {len(reachable)} files"

    # `research/validation.py` is imported at module level by several tests, so it is
    # reachable by inspection; if the walk cannot see it, the walk is broken.
    assert (REPO_ROOT / "research" / "validation.py") in reachable

    assert _omitted_files(), "omit list parsed as empty -- pyproject structure changed"
