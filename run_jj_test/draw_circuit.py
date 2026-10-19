#!/usr/bin/env python3
"""Draw the crosstalk circuit from its NumPy mask and Artemis input decks."""

from __future__ import annotations

import argparse
import ast
import operator
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def numeric_value(expression: str, names: dict[str, float]) -> float:
    """Evaluate the small arithmetic language used by my_constants."""

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError(f"unsupported numeric expression: {expression!r}")

    return visit(ast.parse(expression, mode="eval"))


def read_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        assignments[key.strip()] = value.strip()
    return assignments


def read_constants(assignments: dict[str, str]) -> dict[str, float]:
    pending = {
        key.removeprefix("my_constants."): value
        for key, value in assignments.items()
        if key.startswith("my_constants.")
    }
    values: dict[str, float] = {}
    while pending:
        progressed = False
        for key, expression in list(pending.items()):
            try:
                values[key] = numeric_value(expression, values)
            except (KeyError, ValueError, SyntaxError):
                continue
            del pending[key]
            progressed = True
        if not progressed:
            break
    return values


def quoted_path(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def resolve_mask_path(input_path: Path, mask_name: str) -> Path:
    candidate = input_path.parent / mask_name
    if candidate.exists():
        return candidate.resolve()
    candidate = input_path.parent.parent / mask_name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"cannot find {mask_name!r} relative to {input_path}")


def xy_mask(array: np.ndarray, nx: int, ny: int, k_index: int) -> np.ndarray:
    """Return the mask with array axes ordered as (x, y)."""
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[:2] == (nx, ny):
        # Some exported masks contain one extruded plane even when npy_k_index
        # refers to the corresponding z index in the simulation grid.
        return array[:, :, min(k_index, array.shape[2] - 1)]
    if array.ndim == 3 and array.shape[1:] == (nx, ny):
        return array[min(k_index, array.shape[0] - 1), :, :]
    array = np.squeeze(array)
    if array.shape == (nx, ny):
        return array
    if array.shape == (ny, nx):
        return array.T
    raise ValueError(
        f"mask shape {array.shape} is incompatible with input grid ({nx}, {ny})"
    )


def runs(values: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, values.astype(bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def pad_centers(
    mask: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, x_jj: float, y_jj: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    j = int(np.clip(np.searchsorted(y_edges, y_jj) - 1, 0, mask.shape[1] - 1))
    candidates = []
    for start, stop in runs(mask[:, j] > 0):
        center = 0.5 * (x_edges[start] + x_edges[stop + 1])
        # Ignore the ground-plane runs touching either domain boundary.
        if start > 0 and stop < mask.shape[0] - 1:
            candidates.append(center)
    left = max((x for x in candidates if x < x_jj), default=0.35 * x_jj)
    right = min((x for x in candidates if x > x_jj), default=1.65 * x_jj)
    label_y = y_edges[0] + 0.34 * (y_edges[-1] - y_edges[0])
    return (left, label_y), (right, label_y)


def coordinate_from_function(expression: str, axis: str) -> float:
    match = re.search(rf"\({axis}\s*>?=?\s*({NUMBER})", expression)
    if not match:
        raise ValueError(f"could not read {axis}-coordinate from {expression!r}")
    return float(match.group(1))


def drive_geometry(assignments: dict[str, str]) -> tuple[float, float, float]:
    expression = assignments["warpx.Ex_excitation_flag_function(x,y,z)"]
    xlo = re.search(rf"\(x\s*>=\s*({NUMBER})", expression)
    xhi = re.search(rf"\(x\s*<=\s*({NUMBER})", expression)
    y = re.search(rf"\(y\s*>\s*({NUMBER})", expression)
    if not (xlo and xhi and y):
        raise ValueError("could not parse the Ex excitation footprint")
    return float(xlo.group(1)), float(xhi.group(1)), float(y.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[Path("inputs_artemis_crosstalk_rcsj"), Path("inputs_artemis_crosstalk_L")],
        help="Artemis input deck(s) (default: the local RCSJ and linear decks)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("circuit_layout_from_data.png"),
        help="output image path",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [path.resolve() for path in args.inputs]
    decks = [read_assignments(path) for path in input_paths]
    constants = [read_constants(deck) for deck in decks]
    deck, values = decks[0], constants[0]

    nx, ny = int(values["n_cellx"]), int(values["n_celly"])
    xlo, xhi = values["prob_lox"], values["prob_hix"]
    ylo, yhi = values["prob_loy"], values["prob_hiy"]
    mask_name = quoted_path(deck["macroscopic.sigma_npy_file"])
    mask_path = resolve_mask_path(input_paths[0], mask_name)
    k_index = int(float(deck.get("macroscopic.npy_k_index", "0")))
    mask = xy_mask(np.load(mask_path), nx, ny, k_index)

    # Require all supplied decks to describe the same physical layout.
    shared = ("n_cellx", "n_celly", "prob_hix", "prob_hiy", "x_gap_lo", "x_gap_hi", "y_jj", "L_jj", "R_jj")
    for path, other_deck, other_values in zip(input_paths[1:], decks[1:], constants[1:]):
        for key in shared:
            if not np.isclose(other_values[key], values[key]):
                raise ValueError(f"{path}: {key} does not match the first input deck")
        other_mask = quoted_path(other_deck["macroscopic.sigma_npy_file"])
        if resolve_mask_path(path, other_mask) != mask_path:
            raise ValueError(f"{path}: mask file does not match the first input deck")

    x_edges = np.linspace(xlo, xhi, nx + 1)
    y_edges = np.linspace(ylo, yhi, ny + 1)
    gap_x0, gap_x1, y_jj = values["x_gap_lo"], values["x_gap_hi"], values["y_jj"]
    x_jj = 0.5 * (gap_x0 + gap_x1)
    n_edges = int(round((gap_x1 - gap_x0) / values["dx"]))
    (pad0_x, pad_y), (pad1_x, _) = pad_centers(mask, x_edges, y_edges, x_jj, y_jj)
    drive_x0, drive_x1, drive_y = drive_geometry(deck)
    probe0 = (values["x0"], values["y0"])
    probe1 = (values["x1"], values["y1"])
    frequency_ghz = values["freq"] / 1e9

    scale = 1e6  # meters -> micrometers
    navy, red, green, purple = "#17394a", "#e83d32", "#148a48", "#71348f"
    fig, ax = plt.subplots(figsize=(7.4, 9.5), constrained_layout=True)
    cmap = ListedColormap(["#f7f5f1", navy])
    ax.imshow(
        (mask.T > 0).astype(int),
        origin="lower",
        extent=np.array([xlo, xhi, ylo, yhi]) * scale,
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    # Overlays come directly from coordinates in the input deck.
    ax.plot(np.array([drive_x0, drive_x1]) * scale, [drive_y * scale] * 2,
            color=red, lw=4, solid_capstyle="butt", zorder=4)
    ax.annotate(
        f"{frequency_ghz:g} GHz drive",
        xy=(drive_x1 * scale, drive_y * scale),
        xytext=(455, 745), color=red, fontsize=11,
        arrowprops={"arrowstyle": "-", "color": red, "lw": 1},
    )

    enabled_rcsj = any(d.get("algo.use_josephson_junction") == "1" for d in decks)
    enabled_l = any(d.get("algo.use_lumped_inductor") == "1" for d in decks)
    device_label = " / ".join(name for name, on in (("RCSJ", enabled_rcsj), (r"$L_J$", enabled_l)) if on)
    shunted = all(d.get("warpx.use_lumped_resistor") == "1" for d in decks)
    shunt_note = f"\nshunt R = {values['R_jj']:g} $\\Omega$/edge" if shunted else ""
    # Both decks place the RCSJ/L_J on every x-directed edge across the metal
    # gap, x_gap_lo < x < x_gap_hi, so the element connects the two pads.
    ax.plot(
        np.array([gap_x0, gap_x1]) * scale,
        [y_jj * scale, y_jj * scale],
        color=green,
        lw=5,
        solid_capstyle="butt",
        zorder=5,
    )
    ax.annotate(
        f"{device_label}\n{n_edges} edges in series\n"
        f"{values['L_jj'] * 1e12:.4g} pH/edge{shunt_note}",
        xy=(gap_x1 * scale, y_jj * scale), xytext=(395, 495),
        color=green, fontsize=10,
        bbox={"fc": "#f7f5f1", "ec": "none", "alpha": 0.9},
        arrowprops={"arrowstyle": "-", "color": green, "lw": 1},
    )

    for name, (x, y), text_offset in (
        ("E0", probe0, (-46, -22)),
        ("E1", probe1, (-46, 26)),
    ):
        # E1 sits on one device edge, so draw it as a ring to keep the
        # green RCSJ/L_J segment visible through the probe marker.
        if name == "E1":
            ax.scatter([x * scale], [y * scale], s=70, facecolors="none",
                       edgecolors=purple, linewidths=2, zorder=6)
        else:
            ax.scatter([x * scale], [y * scale], s=48, color=purple, zorder=6)
        ax.annotate(name, (x * scale, y * scale), xytext=text_offset,
                    textcoords="offset points", color=purple, fontsize=11)

    # Egap integrates Ex over every device edge, i.e. the full gap voltage.
    box_x = np.array([gap_x0, gap_x1, gap_x1, gap_x0, gap_x0]) * scale
    box_y = (y_jj + np.array([-1, -1, 1, 1, -1]) * 3 * values["dy"]) * scale
    ax.plot(box_x, box_y, color=purple, lw=1.2, ls="--", zorder=6)
    ax.annotate("Egap", (gap_x0 * scale, (y_jj - 3 * values["dy"]) * scale),
                xytext=(-34, -16), textcoords="offset points",
                color=purple, fontsize=11,
                bbox={"fc": "#f7f5f1", "ec": "none", "alpha": 0.9})

    for x in (pad0_x, pad1_x):
        ax.text(x * scale, pad_y * scale, "pad", color="white", ha="center",
                va="center", fontsize=12)
    ax.text((xlo + 0.035 * (xhi - xlo)) * scale, 0.5 * (ylo + yhi) * scale,
            "ground", color="white", rotation=90, ha="center", va="center", fontsize=10)

    legend = [
        Line2D([0], [0], color=navy, lw=8, label="metal mask = 1"),
        Line2D([0], [0], color=red, lw=3, label=f"{frequency_ghz:g} GHz drive"),
        Line2D([0], [0], color=green, lw=4, solid_capstyle="round",
               label=device_label),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=purple,
               markersize=7, label="E probes"),
        Line2D([0], [0], color=purple, lw=1.2, ls="--",
               label="Egap probe (whole gap)"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9)
    ax.set(xlabel=r"$x$ ($\mu$m)", ylabel=r"$y$ ($\mu$m)",
           xlim=(xlo * scale, xhi * scale), ylim=(ylo * scale, yhi * scale))
    ax.set_title("Crosstalk circuit from simulation mask and input decks", loc="left")
    ax.tick_params(direction="out", length=3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output.resolve()}")
    print(f"mask: {mask_path} (shape={mask.shape})")
    print("inputs: " + ", ".join(str(path) for path in input_paths))


if __name__ == "__main__":
    main()
