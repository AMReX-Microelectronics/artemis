"""Discover and resolve diags_* diagnostic series under a run prefix."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

# Fallback when --no-auto-series is used (and --series is not set).
DEFAULT_SERIES: tuple[tuple[str, str], ...] = (
    ("FDTD CFL=0.8", "diags_fdtd_08"),
    ("ADI CFL=1.6", "diags_adi_16"),
    ("ADI CFL=3.2", "diags_adi_32"),
    ("ADI CFL=6.4", "diags_adi_64"),
    ("ADI CFL=12.8", "diags_adi_128"),
    ("ADI CFL=25.6", "diags_adi_256"),
    ("ADI CFL=51.2", "diags_adi_512"),
)

_DIAGS_RE = re.compile(r"^diags_(fdtd|adi)_(\d+)$", re.IGNORECASE)


def format_series_label(dirname: str) -> str:
    """diags_fdtd_08 -> 'FDTD CFL=0.8'; diags_adi_64 -> 'ADI CFL=6.4'."""
    match = _DIAGS_RE.fullmatch(dirname)
    if not match:
        return dirname
    method = match.group(1).upper()
    cfl = int(match.group(2)) / 10.0
    return f"{method} CFL={cfl:g}"


def _sort_key(path: Path) -> tuple[int, int, str]:
    match = _DIAGS_RE.fullmatch(path.name)
    if not match:
        return (2, 0, path.name)
    # FDTD first, then ADI ordered by CFL digit suffix.
    order = 0 if match.group(1).lower() == "fdtd" else 1
    return (order, int(match.group(2)), path.name)


def discover_series(
    prefix: Path,
    *,
    require_plotfiles: bool = False,
    require_reducedfiles: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Return (label, dirname) pairs for diags_* folders under prefix."""
    if not prefix.is_dir():
        raise FileNotFoundError(f"prefix not found: {prefix.resolve()}")

    found: list[tuple[str, str]] = []
    for path in sorted(prefix.iterdir(), key=_sort_key):
        if not path.is_dir() or not path.name.startswith("diags_"):
            continue
        if require_plotfiles and not any(path.glob("plt*")):
            continue
        if require_reducedfiles and not (path / "reducedfiles").is_dir():
            continue
        found.append((format_series_label(path.name), path.name))

    if not found:
        raise FileNotFoundError(f"No diags_* series found under {prefix.resolve()}")
    return tuple(found)


def series_from_names(
    prefix: Path,
    names: Sequence[str],
    *,
    require_plotfiles: bool = False,
    require_reducedfiles: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Build (label, dirname) pairs from explicit directory names, in given order."""
    if not names:
        raise ValueError("--series requires at least one directory name")
    if not prefix.is_dir():
        raise FileNotFoundError(f"prefix not found: {prefix.resolve()}")

    found: list[tuple[str, str]] = []
    for name in names:
        path = prefix / name
        if not path.is_dir():
            raise FileNotFoundError(f"series directory not found: {path.resolve()}")
        if require_plotfiles and not any(path.glob("plt*")):
            raise FileNotFoundError(f"no plotfiles in series directory: {path.resolve()}")
        if require_reducedfiles and not (path / "reducedfiles").is_dir():
            raise FileNotFoundError(
                f"no reducedfiles/ in series directory: {path.resolve()}"
            )
        found.append((format_series_label(name), name))
    return tuple(found)


def resolve_series(
    prefix: Path,
    auto: bool,
    fallback: tuple[tuple[str, str], ...] = DEFAULT_SERIES,
    *,
    selected: Sequence[str] | None = None,
    require_plotfiles: bool = False,
    require_reducedfiles: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Resolve series: --series overrides, else auto-discover, else fallback."""
    if selected:
        return series_from_names(
            prefix,
            selected,
            require_plotfiles=require_plotfiles,
            require_reducedfiles=require_reducedfiles,
        )
    if auto:
        return discover_series(
            prefix,
            require_plotfiles=require_plotfiles,
            require_reducedfiles=require_reducedfiles,
        )
    return fallback


def series_directories(
    prefix: Path,
    series: tuple[tuple[str, str], ...],
) -> list[tuple[str, Path]]:
    """Map (label, dirname) pairs to (label, absolute directory) paths."""
    return [(label, prefix / name) for label, name in series]


def add_prefix_series_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prefix",
        type=Path,
        default=Path("run_archive"),
        help="directory containing the diags_* folders",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        metavar="DIR",
        help="explicit diags_* directory names under --prefix to plot "
        "(order preserved; overrides --auto-series / --no-auto-series)",
    )
    parser.add_argument(
        "--auto-series",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="discover diags_* folders under --prefix (default). "
        "Use --no-auto-series for the built-in SERIES list. "
        "Ignored when --series is set.",
    )
