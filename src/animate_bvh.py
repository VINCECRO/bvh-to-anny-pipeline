"""
Animation BVH brut vs Anny — comparaison côte à côte.
Génère un GIF animé (4 panneaux) pour valider le mapping CMU→Anny.

Layout :
  ┌──────────────┬──────────────┐
  │  BVH  front  │  Anny front  │
  ├──────────────┼──────────────┤
  │  BVH  côté   │  Anny côté   │
  └──────────────┴──────────────┘

BVH  : Y = vertical, X = gauche/droite, Z = profondeur (face → +Z)
Anny : Z = vertical, X = gauche/droite, Y = profondeur (face → -Y)

Usage :
  uv run python src/animate_bvh.py --bvh data/cmu/data/005/05_01.bvh
  uv run python src/animate_bvh.py --bvh data/cmu/data/005/05_01.bvh --start 0 --end 120 --step 3
  uv run python src/animate_bvh.py --bvh data/cmu/data/013/13_29.bvh --step 4 --fps 12 --with-mesh
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from bvh_reader import BVHFile
from visualize import (
    ANNY_DISPLAY_BONES,
    _CMU_LEFT_BONES, _CMU_RIGHT_BONES, _CMU_CENTER_BONES,
    _C_LEFT, _C_RIGHT, _C_CENTER,
    get_anny_joint_positions,
)
from retarget import load_anny_model, load_blender_matrices, retarget_frame

BG = "#0d0d1a"


# ── Dessin ────────────────────────────────────────────────────────────────────

def _draw_bvh_skeleton(ax, pos2d: dict):
    """Squelette BVH avec coloration gauche/droite/centre."""
    for bones, color in [
        (_CMU_LEFT_BONES, _C_LEFT),
        (_CMU_RIGHT_BONES, _C_RIGHT),
        (_CMU_CENTER_BONES, _C_CENTER),
    ]:
        segs = [[pos2d[a], pos2d[b]] for a, b in bones if a in pos2d and b in pos2d]
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=2.0, zorder=3))

    major = {"Hips", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot",
             "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"}
    for name, p in pos2d.items():
        size = 60 if name in major else 15
        color = "yellow" if name == "Hips" else "white"
        ax.scatter(p[0], p[1], c=color, s=size, zorder=6)


_ANNY_LEFT_BONES  = {(a, b) for a, b in ANNY_DISPLAY_BONES if ".L" in b}
_ANNY_RIGHT_BONES = {(a, b) for a, b in ANNY_DISPLAY_BONES if ".R" in b}
_ANNY_CENTER_BONES = [(a, b) for a, b in ANNY_DISPLAY_BONES
                      if ".L" not in b and ".R" not in b]


def _draw_anny_skeleton(ax, pos2d: dict, verts2d=None, faces=None):
    """Squelette Anny coloré gauche/droite (même convention que BVH)."""
    if verts2d is not None and faces is not None:
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        patches = []
        for face in faces:
            pts = verts2d[face]
            v0, v1, v2 = pts
            cross = (v1[0]-v0[0])*(v2[1]-v0[1]) - (v1[1]-v0[1])*(v2[0]-v0[0])
            if cross < 0:
                patches.append(Polygon(pts, closed=True))
        if patches:
            pc = PatchCollection(patches, alpha=0.18, facecolor="lightblue", edgecolor="none")
            ax.add_collection(pc)

    _display_joints = {j for pair in ANNY_DISPLAY_BONES for j in pair}
    for bones, color in [
        (_ANNY_LEFT_BONES,   _C_LEFT),
        (_ANNY_RIGHT_BONES,  _C_RIGHT),
        (_ANNY_CENTER_BONES, _C_CENTER),
    ]:
        segs = [[pos2d[a], pos2d[b]] for a, b in bones if a in pos2d and b in pos2d]
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=2.0, zorder=3))
    # Uniquement les joints anatomiques (pas les os faciaux)
    for name, p in pos2d.items():
        if name in _display_joints:
            ax.scatter(p[0], p[1], c="white", s=12, zorder=5)


def _setup_ax(ax, title: str):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=9, pad=5)
    ax.tick_params(colors="#555")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.grid(True, alpha=0.1, color="white")


# ── Pré-calcul ────────────────────────────────────────────────────────────────

def precompute(bvh_path: Path, frame_indices: list[int], model,
               blender_mats: dict,
               with_mesh: bool) -> list[dict]:
    """Calcule squelette BVH + pose Anny pour chaque frame."""
    bvh = BVHFile(bvh_path)
    results = []

    for i, frame_idx in enumerate(frame_indices):
        sys.stdout.write(f"\r  Frame {i+1}/{len(frame_indices)}  (BVH {frame_idx:4d})…")
        sys.stdout.flush()

        bvh_positions = bvh.joint_world_positions(frame_idx)
        pose_params   = retarget_frame(bvh, frame_idx, model, blender_mats)

        with torch.no_grad():
            output = model(pose_parameters=pose_params,
                           pose_parameterization="absolute")

        anny_positions = get_anny_joint_positions(output, model)
        verts = output["vertices"].squeeze(0).cpu().numpy() if with_mesh else None

        results.append({
            "frame_idx": frame_idx,
            "bvh": bvh_positions,
            "anny": anny_positions,
            "verts": verts,
        })

    print(f"\r  {len(frame_indices)} frames calculées.{' '*30}")
    return results


# ── Calcul des bornes globales (axes fixes = pas de zoom qui saute) ────────────

def _global_bounds(frames_data: list[dict], pad_frac=0.18) -> dict:
    bvh_front_x, bvh_front_y = [], []
    bvh_side_x, bvh_side_y = [], []
    af_x, af_y = [], []
    as_x, as_y = [], []

    for fd in frames_data:
        cx_bvh = np.mean([p[0] for p in fd["bvh"].values()])
        for p in fd["bvh"].values():
            bvh_front_x.append(p[0] - cx_bvh)
            bvh_front_y.append(p[1])            # Y = vertical CMU
            bvh_side_x.append(p[2])             # Z = profondeur CMU
            bvh_side_y.append(p[1])

        cx_anny = np.mean([p[0] for p in fd["anny"].values()])
        for p in fd["anny"].values():
            af_x.append(p[0] - cx_anny)
            af_y.append(p[2])                   # Z = vertical Anny
            as_x.append(-p[1])                  # -Y = profondeur Anny (→ même sens que BVH)
            as_y.append(p[2])

    def _b(xs, ys):
        xs, ys = np.array(xs), np.array(ys)
        px = max((xs.max()-xs.min())*pad_frac, 0.2)
        py = max((ys.max()-ys.min())*pad_frac, 0.2)
        return xs.min()-px, xs.max()+px, ys.min()-py, ys.max()+py

    return {
        "bvh_front": _b(bvh_front_x, bvh_front_y),
        "bvh_side":  _b(bvh_side_x,  bvh_side_y),
        "anny_front": _b(af_x, af_y),
        "anny_side":  _b(as_x, as_y),
    }


# ── Animation ─────────────────────────────────────────────────────────────────

def make_animation(frames_data: list[dict], bvh_stem: str, fps: int,
                   faces_np: np.ndarray) -> tuple:
    bounds = _global_bounds(frames_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    fig.subplots_adjust(hspace=0.35, wspace=0.25, left=0.04, right=0.97, top=0.92, bottom=0.04)
    (ax_bf, ax_af), (ax_bs, ax_as) = axes

    fig.suptitle(f"BVH CMU  vs  Anny — {bvh_stem}", color="white", fontsize=13, y=0.97)

    # Légende côté (communes à tous les panels BVH)
    legend_patch = [
        mpatches.Patch(color=_C_LEFT,   label="Gauche"),
        mpatches.Patch(color=_C_RIGHT,  label="Droite"),
        mpatches.Patch(color=_C_CENTER, label="Centre"),
    ]

    def update(i):
        fd = frames_data[i]

        for ax in axes.flat:
            ax.cla()

        # ── Projections BVH ───────────────────────────────────────────────────
        cx_bvh = np.mean([p[0] for p in fd["bvh"].values()])
        bvh_front = {k: np.array([p[0]-cx_bvh, p[1]]) for k, p in fd["bvh"].items()}
        bvh_side  = {k: np.array([p[2],         p[1]]) for k, p in fd["bvh"].items()}

        # ── Projections Anny ──────────────────────────────────────────────────
        cx_anny = np.mean([p[0] for p in fd["anny"].values()])
        anny_front = {k: np.array([p[0]-cx_anny,  p[2]]) for k, p in fd["anny"].items()}
        anny_side  = {k: np.array([-p[1],          p[2]]) for k, p in fd["anny"].items()}

        # ── Mesh silhouettes optionnelles ─────────────────────────────────────
        verts_front_2d = verts_side_2d = None
        if fd["verts"] is not None:
            cx_v = fd["verts"][:, 0].mean()
            verts_front_2d = np.column_stack([fd["verts"][:, 0]-cx_v, fd["verts"][:, 2]])
            verts_side_2d  = np.column_stack([-fd["verts"][:, 1],     fd["verts"][:, 2]])

        # ── Panel BVH front ───────────────────────────────────────────────────
        _draw_bvh_skeleton(ax_bf, bvh_front)
        _setup_ax(ax_bf, f"BVH CMU — face  (X, Y)   frame {fd['frame_idx']}")
        ax_bf.set_xlim(bounds["bvh_front"][0], bounds["bvh_front"][1])
        ax_bf.set_ylim(bounds["bvh_front"][2], bounds["bvh_front"][3])
        ax_bf.legend(handles=legend_patch, loc="lower right",
                     facecolor=BG, labelcolor="white", fontsize=7)

        # ── Panel BVH côté ────────────────────────────────────────────────────
        _draw_bvh_skeleton(ax_bs, bvh_side)
        _setup_ax(ax_bs, "BVH CMU — côté  (Z=prof., Y)")
        ax_bs.set_xlim(bounds["bvh_side"][0], bounds["bvh_side"][1])
        ax_bs.set_ylim(bounds["bvh_side"][2], bounds["bvh_side"][3])

        # ── Panel Anny front ──────────────────────────────────────────────────
        _draw_anny_skeleton(ax_af, anny_front, verts_front_2d, faces_np if verts_front_2d is not None else None)
        _setup_ax(ax_af, "Anny — face  (X, Z)")
        ax_af.set_xlim(bounds["anny_front"][0], bounds["anny_front"][1])
        ax_af.set_ylim(bounds["anny_front"][2], bounds["anny_front"][3])

        # ── Panel Anny côté ───────────────────────────────────────────────────
        _draw_anny_skeleton(ax_as, anny_side, verts_side_2d, faces_np if verts_side_2d is not None else None)
        _setup_ax(ax_as, "Anny — côté  (−Y=prof., Z)")
        ax_as.set_xlim(bounds["anny_side"][0], bounds["anny_side"][1])
        ax_as.set_ylim(bounds["anny_side"][2], bounds["anny_side"][3])

    anim = FuncAnimation(fig, update, frames=len(frames_data), interval=1000//fps, blit=False)
    return fig, anim


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Animation BVH brut vs Anny (GIF)")
    parser.add_argument("--bvh",       default="data/cmu/data/005/05_01.bvh",
                        help="Fichier BVH source")
    parser.add_argument("--start",     type=int, default=0,
                        help="Première frame (défaut 0)")
    parser.add_argument("--end",       type=int, default=None,
                        help="Dernière frame excluante (défaut : toutes)")
    parser.add_argument("--step",      type=int, default=5,
                        help="Pas entre frames (défaut 5 → ~24fps source → ~5fps effectif)")
    parser.add_argument("--fps",       type=int, default=10,
                        help="FPS du GIF de sortie (défaut 10)")
    parser.add_argument("--out",       default=None,
                        help="Chemin GIF de sortie (défaut output/anim_<stem>.gif)")
    parser.add_argument("--with-mesh", action="store_true",
                        help="Ajouter la silhouette mesh Anny (plus lent)")
    parser.add_argument("--dpi",       type=int, default=90,
                        help="Résolution du GIF (défaut 90)")
    args = parser.parse_args()

    bvh_path = Path(args.bvh)
    out_path = Path(args.out) if args.out else Path("output") / f"anim_{bvh_path.stem}.gif"
    out_path.parent.mkdir(exist_ok=True)

    print(f"[1] Chargement BVH : {bvh_path.name}")
    _bvh_tmp = BVHFile(bvh_path)
    n_frames = _bvh_tmp.n_frames
    end = min(args.end, n_frames) if args.end else n_frames
    frame_indices = list(range(args.start, end, args.step))
    print(f"    {n_frames} frames dispo → animation : {len(frame_indices)} frames "
          f"[{args.start}:{end}:{args.step}]")

    print("[2] Chargement modèle Anny…")
    model        = load_anny_model()
    blender_mats = load_blender_matrices()
    faces_np     = model.faces if isinstance(model.faces, np.ndarray) else model.faces.cpu().numpy()

    print(f"[3] Pré-calcul de {len(frame_indices)} frames (Anny CPU)…")
    frames_data = precompute(bvh_path, frame_indices, model, blender_mats, args.with_mesh)

    print("[4] Génération animation…")
    fig, anim = make_animation(frames_data, bvh_path.stem, args.fps, faces_np)

    print(f"[5] Sauvegarde : {out_path}")
    anim.save(str(out_path), writer="pillow", fps=args.fps, dpi=args.dpi)
    plt.close(fig)

    size_ko = out_path.stat().st_size // 1024
    print(f"\nGIF sauvegardé : {out_path}  ({size_ko} Ko, {len(frames_data)} frames @ {args.fps} fps)")
    print(f"Durée : {len(frames_data)/args.fps:.1f}s")


if __name__ == "__main__":
    main()
