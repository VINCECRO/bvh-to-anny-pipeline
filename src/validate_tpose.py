"""
Validation de la T-pose : BVH frame 0 ↔ Anny calibrée Blender.

Produit output/validate_tpose.png avec 4 panneaux :
  [A] BVH frame 0  — squelette en espace Anny (X, Z) centré sur Hips
  [B] Anny rest     — pose de repos native (identité)
  [C] Anny T-pose   — matrices Blender (absolute)
  [D] Overlay       — BVH (pointillé) + Anny Blender (plein)

Usage :
    uv run python src/validate_tpose.py
    uv run python src/validate_tpose.py --bvh data/cmu/data/013/13_01.bvh
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

from bvh_reader import BVHFile
from retarget import (
    _C, CMU_TO_ANNY,
    load_blender_matrices, load_anny_model,
    tpose_params, get_rest_bone_poses,
    bvh_joint_positions_in_anny_space,
)

# ─────────────────────────────────────────────────────────────────────────────
# Connexions squelettes pour l'affichage
# ─────────────────────────────────────────────────────────────────────────────

_CMU_BONES = [
    ("Hips", "LowerBack"), ("LowerBack", "Spine"), ("Spine", "Spine1"),
    ("Spine1", "Neck"), ("Neck", "Neck1"), ("Neck1", "Head"),
    ("Spine1", "LeftShoulder"), ("Spine1", "RightShoulder"),
    ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LHipJoint"), ("Hips", "RHipJoint"),
    ("LHipJoint", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
    ("RHipJoint", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"),
]

_ANNY_BONES = [
    ("root", "spine05"), ("spine05", "spine04"), ("spine04", "spine03"),
    ("spine03", "spine02"), ("spine02", "spine01"),
    ("spine01", "neck01"), ("neck01", "neck02"), ("neck02", "neck03"), ("neck03", "head"),
    ("spine01", "clavicle.L"), ("clavicle.L", "shoulder01.L"),
    ("shoulder01.L", "upperarm01.L"), ("upperarm01.L", "upperarm02.L"),
    ("upperarm02.L", "lowerarm01.L"), ("lowerarm01.L", "lowerarm02.L"),
    ("lowerarm02.L", "wrist.L"),
    ("spine01", "clavicle.R"), ("clavicle.R", "shoulder01.R"),
    ("shoulder01.R", "upperarm01.R"), ("upperarm01.R", "upperarm02.R"),
    ("upperarm02.R", "lowerarm01.R"), ("lowerarm01.R", "lowerarm02.R"),
    ("lowerarm02.R", "wrist.R"),
    ("root", "pelvis.L"), ("pelvis.L", "upperleg01.L"), ("upperleg01.L", "upperleg02.L"),
    ("upperleg02.L", "lowerleg01.L"), ("lowerleg01.L", "lowerleg02.L"),
    ("lowerleg02.L", "foot.L"), ("foot.L", "toe1-1.L"),
    ("root", "pelvis.R"), ("pelvis.R", "upperleg01.R"), ("upperleg01.R", "upperleg02.R"),
    ("upperleg02.R", "lowerleg01.R"), ("lowerleg01.R", "lowerleg02.R"),
    ("lowerleg02.R", "foot.R"), ("foot.R", "toe1-1.R"),
]

# Couleurs par côté
_CL = "#00e5ff"  # cyan  — gauche
_CR = "#ff9500"  # orange — droite
_CC = "#e0e0e0"  # gris  — centre / colonne

BG = "#0d0d1a"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers affichage
# ─────────────────────────────────────────────────────────────────────────────

def _color_cmu(a: str, b: str) -> str:
    if "Left" in a or "Left" in b or "LHip" in a or "LHip" in b:
        return _CL
    if "Right" in a or "Right" in b or "RHip" in a or "RHip" in b:
        return _CR
    return _CC


def _color_anny(a: str, b: str) -> str:
    if ".L" in b:
        return _CL
    if ".R" in b:
        return _CR
    return _CC


def _proj_xz(p: np.ndarray) -> np.ndarray:
    """Projection frontale : X horizontal, Z vertical."""
    return np.array([p[0], p[2]])


def _proj_yz(p: np.ndarray) -> np.ndarray:
    """Projection latérale : -Y horizontal (face vers droite), Z vertical."""
    return np.array([-p[1], p[2]])


def _center(pos: dict[str, np.ndarray], ref_key: str) -> dict[str, np.ndarray]:
    """Centre les positions sur le joint de référence."""
    origin = pos.get(ref_key, np.zeros(3))
    return {k: v - origin for k, v in pos.items()}


def _normalize_height(pos: dict[str, np.ndarray],
                      top_key: str, bot_key: str) -> dict[str, np.ndarray]:
    """
    Normalise les positions par la hauteur totale (top_key → bot_key).
    Utile pour comparer deux squelettes à des échelles différentes.
    """
    if top_key not in pos or bot_key not in pos:
        return pos
    h = np.linalg.norm(pos[top_key] - pos[bot_key])
    if h < 1e-6:
        return pos
    return {k: v / h for k, v in pos.items()}


def _setup_ax(ax, title: str):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=8, pad=4)
    ax.tick_params(colors="#555", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.grid(True, alpha=0.07, color="white")


def _autoscale(ax, pts: list[np.ndarray], pad_frac: float = 0.20):
    if not pts:
        return
    arr = np.array(pts)
    span = arr.max(0) - arr.min(0)
    pad  = np.maximum(span * pad_frac, 0.05)
    ax.set_xlim(arr[:, 0].min() - pad[0], arr[:, 0].max() + pad[0])
    ax.set_ylim(arr[:, 1].min() - pad[1], arr[:, 1].max() + pad[1])


def draw_cmu(ax, pos3d: dict[str, np.ndarray], proj,
             alpha: float = 1.0, lw: float = 2.0, ls: str = "solid"):
    pos2d = {k: proj(v) for k, v in pos3d.items()}
    segs   = []
    colors = []
    for a, b in _CMU_BONES:
        if a in pos2d and b in pos2d:
            segs.append([pos2d[a], pos2d[b]])
            colors.append(_color_cmu(a, b))
    if segs:
        for seg, col in zip(segs, colors):
            ax.add_collection(LineCollection([seg], colors=col, linewidths=lw,
                                              linestyles=ls, alpha=alpha, zorder=3))
    pts = [p for p in pos2d.values()]
    for p in pts:
        ax.scatter(p[0], p[1], c="white", s=12, zorder=5, alpha=alpha)
    return list(pos2d.values())


def draw_anny(ax, pos3d: dict[str, np.ndarray], proj,
              alpha: float = 1.0, lw: float = 2.0, ls: str = "solid"):
    pos2d = {k: proj(v) for k, v in pos3d.items()}
    for a, b in _ANNY_BONES:
        if a in pos2d and b in pos2d:
            ax.add_collection(LineCollection([[pos2d[a], pos2d[b]]],
                                              colors=_color_anny(a, b),
                                              linewidths=lw, linestyles=ls,
                                              alpha=alpha, zorder=3))
    pts = [p for p in pos2d.values()]
    for p in pts:
        ax.scatter(p[0], p[1], c="#ff4444", s=10, zorder=5, alpha=alpha)
    return list(pos2d.values())


# ─────────────────────────────────────────────────────────────────────────────
# Estimation de l'échelle BVH → Anny
# ─────────────────────────────────────────────────────────────────────────────

def estimate_scale(
    bvh_pos_raw: dict[str, np.ndarray],
    anny_pos: dict[str, np.ndarray],
) -> float:
    """
    Estime le facteur d'échelle BVH (unités brutes) → Anny (mètres)
    depuis la hauteur du squelette (pieds → tête).
    """
    pairs = [
        (("LeftFoot", "RightFoot"), ("foot.L", "foot.R")),
        (("Head",),                 ("head",)),
    ]
    # Position moyenne des pieds BVH
    feet_bvh = np.mean(
        [bvh_pos_raw[j] for j in ("LeftFoot", "RightFoot") if j in bvh_pos_raw], axis=0
    )
    head_bvh = bvh_pos_raw.get("Head", None)
    if feet_bvh is None or head_bvh is None:
        return 0.06405  # fallback

    # Hauteur BVH en Y (brut)
    h_bvh = abs(head_bvh[1] - feet_bvh[1])  # Y = up dans BVH

    # Hauteur Anny en Z
    feet_anny = np.mean(
        [anny_pos[j] for j in ("foot.L", "foot.R") if j in anny_pos], axis=0
    )
    head_anny = anny_pos.get("head", None)
    if feet_anny is None or head_anny is None:
        return 0.06405

    h_anny = abs(head_anny[2] - feet_anny[2])  # Z = up dans Anny

    if h_bvh < 1e-6:
        return 0.06405
    return float(h_anny / h_bvh)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", default="data/cmu/data/005/05_03.bvh")
    ap.add_argument("--out", default="output/validate_tpose.png")
    args = ap.parse_args()

    bvh_path = Path(args.bvh)
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    print(f"BVH     : {bvh_path}")
    print(f"Sortie  : {out_path}")

    # ── 1. BVH frame 0 ────────────────────────────────────────────────────────
    print("\n[1] Lecture BVH frame 0…")
    bvh = BVHFile(bvh_path)
    bvh_pos_raw = bvh.joint_world_positions(0)   # Y-up, unités BVH
    print(f"    {bvh.n_frames} frames  |  {len(bvh_pos_raw)} joints")

    # ── 2. Modèle Anny ─────────────────────────────────────────────────────────
    print("[2] Chargement Anny…")
    model = load_anny_model()
    blender_mats = load_blender_matrices()
    print(f"    {model.bone_count} os  |  {len(blender_mats)} matrices Blender")

    # ── 3. T-pose Anny via Blender (absolute) ─────────────────────────────────
    print("[3] T-pose Anny (Blender absolute)…")
    tpose_p = tpose_params(model, blender_mats)
    with torch.no_grad():
        tpose_out = model(pose_parameters=tpose_p, pose_parameterization="absolute")
    anny_tpose_pos = {
        model.bone_labels[i]: tpose_out["bone_poses"].squeeze(0)[i, :3, 3].numpy()
        for i in range(model.bone_count)
    }

    # ── 4. Pose de repos native Anny (identité) ────────────────────────────────
    print("[4] Pose de repos native Anny…")
    identity = {label: torch.eye(4).unsqueeze(0) for label in model.bone_labels}
    with torch.no_grad():
        rest_out = model(pose_parameters=identity, pose_parameterization="rest_relative")
    anny_rest_pos = {
        model.bone_labels[i]: rest_out["bone_poses"].squeeze(0)[i, :3, 3].numpy()
        for i in range(model.bone_count)
    }

    # ── 5. Échelle BVH → Anny ─────────────────────────────────────────────────
    scale = estimate_scale(bvh_pos_raw, anny_tpose_pos)
    print(f"[5] Échelle BVH → Anny : {scale:.5f} m/unité")

    bvh_pos_anny = bvh_joint_positions_in_anny_space(bvh, 0, scale=scale)

    # ── 6. Centrage sur root / Hips ────────────────────────────────────────────
    bvh_c      = _center(bvh_pos_anny,   "Hips")
    anny_rest_c = _center(anny_rest_pos,  "root")
    anny_tp_c   = _center(anny_tpose_pos, "root")

    # ── 7. Figure ──────────────────────────────────────────────────────────────
    print("[6] Rendu figure…")
    fig, axes = plt.subplots(2, 4, figsize=(24, 14), facecolor=BG)
    fig.suptitle(
        f"Validation T-pose  |  {bvh_path.stem}  frame 0",
        color="white", fontsize=12,
    )
    fig.subplots_adjust(hspace=0.25, wspace=0.15,
                        left=0.02, right=0.99, top=0.93, bottom=0.03)

    projs = [
        ("Face (X,Z)",  _proj_xz),
        ("Côté (-Y,Z)", _proj_yz),
    ]

    for row, (proj_name, proj) in enumerate(projs):
        # Col 0 : BVH
        pts = draw_cmu(axes[row, 0], bvh_c, proj)
        _setup_ax(axes[row, 0], f"BVH  — {proj_name}")
        _autoscale(axes[row, 0], pts)

        # Col 1 : Anny native rest
        pts = draw_anny(axes[row, 1], anny_rest_c, proj)
        _setup_ax(axes[row, 1], f"Anny repos natif  — {proj_name}")
        _autoscale(axes[row, 1], pts)

        # Col 2 : Anny T-pose Blender
        pts = draw_anny(axes[row, 2], anny_tp_c, proj)
        _setup_ax(axes[row, 2], f"Anny T-pose Blender  — {proj_name}")
        _autoscale(axes[row, 2], pts)

        # Col 3 : Overlay BVH + Anny Blender
        pts_b = draw_cmu(axes[row, 3], bvh_c, proj, alpha=0.5, ls="dashed", lw=1.5)
        pts_a = draw_anny(axes[row, 3], anny_tp_c, proj, alpha=1.0, lw=2.0)
        _setup_ax(axes[row, 3], f"Overlay  — {proj_name}")
        _autoscale(axes[row, 3], pts_b + pts_a)

    # Légende globale
    legend_handles = [
        mpatches.Patch(color=_CL,      label="Gauche"),
        mpatches.Patch(color=_CR,      label="Droite"),
        mpatches.Patch(color="white",  label="BVH joint"),
        mpatches.Patch(color="#ff4444", label="Anny joint"),
    ]
    fig.legend(handles=legend_handles, loc="lower right",
               facecolor="#1a1a2e", labelcolor="white", fontsize=8, framealpha=0.8)

    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nImage sauvegardée : {out_path}")

    # ── Rapport numérique ──────────────────────────────────────────────────────
    print("\n── Positions clés centrées sur root/Hips (mètres) ──")
    landmarks = [
        ("BVH", bvh_c,       [("Head","Head"), ("LeftFoot","LeftFoot"), ("RightFoot","RightFoot"),
                               ("LeftHand","LeftHand"), ("RightHand","RightHand")]),
        ("Anny Blender", anny_tp_c, [("head","head"), ("foot.L","foot.L"), ("foot.R","foot.R"),
                                     ("wrist.L","wrist.L"), ("wrist.R","wrist.R")]),
    ]
    for name, pos, keys in landmarks:
        print(f"\n  [{name}]")
        for _, k in keys:
            if k in pos:
                p = pos[k]
                print(f"    {k:20s}  ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m")


if __name__ == "__main__":
    main()
