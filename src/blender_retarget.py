"""
Pipeline retarget via Blender headless.

Workflow :
  1. Blender charge Anny_mesh_BVH_rig.blend, supprime breast.L/R,
     retargète le BVH via retarget_bvh, exporte les matrices os par frame en JSON.
  2. Python charge le JSON et l'utilise avec Anny en mode "absolute".

API principale :
    run_blender_retarget(bvh_path, output_json)   # appelle Blender
    load_retarget_json(json_path)                  # charge le JSON → {frame: {bone: np.array 4×4}}
    params_from_frame(frame_data, model)           # frame_data → pose_params (absolute)

Usage rapide :
    uv run python src/blender_retarget.py --bvh data/cmu/data/005/05_01.bvh --frame 100
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT   = Path(__file__).parent.parent

# Cache des poses de repos natives par model (phénotype par défaut = aucun kwargs)
_rest_bone_poses_cache: dict = {}


def _get_rest_bone_poses(model) -> np.ndarray:
    """
    rest_bone_poses d'Anny au phénotype par défaut (aucun kwargs = corps neutre).
    Utilisé comme fallback pour la propagation des os absents du JSON (breast.L/R).
    """
    key = id(model)
    if key not in _rest_bone_poses_cache:
        identity = {label: torch.eye(4).unsqueeze(0) for label in model.bone_labels}
        with torch.no_grad():
            out = model(pose_parameters=identity, pose_parameterization="absolute")
        _rest_bone_poses_cache[key] = out["rest_bone_poses"].squeeze(0).cpu().numpy()
    return _rest_bone_poses_cache[key]
BLEND_FILE     = PROJECT_ROOT / "blender" / "anny_armature.blend"
EXPORT_SCRIPT  = PROJECT_ROOT / "blender" / "retarget_export.py"
OUTPUT_DIR     = PROJECT_ROOT / "output" / "retarget_cache"


# ── Appel Blender ─────────────────────────────────────────────────────────────

def find_blender() -> str:
    """Cherche l'exécutable Blender dans les emplacements courants."""
    candidates = [
        "blender",                              # dans le PATH
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/snap/bin/blender",
        str(Path.home() / "blender" / "blender"),
    ]
    import shutil
    for c in candidates:
        if shutil.which(c):
            return c
    raise FileNotFoundError(
        "Blender introuvable. Ajoutez-le au PATH ou installez-le via snap/apt."
    )


def run_blender_retarget(
    bvh_path: str | Path,
    output_json: str | Path | None = None,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
    blender_bin: str | None = None,
) -> Path:
    """
    Appelle Blender en headless pour retargeter le BVH et exporter les matrices.
    Retourne le chemin du JSON produit.
    """
    bvh_path = Path(bvh_path).resolve()
    if not bvh_path.exists():
        raise FileNotFoundError(f"BVH introuvable : {bvh_path}")

    if output_json is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_json = OUTPUT_DIR / f"{bvh_path.stem}.json"
    output_json = Path(output_json)

    if blender_bin is None:
        blender_bin = find_blender()

    cmd = [
        blender_bin,
        "--background", str(BLEND_FILE),
        "--python",     str(EXPORT_SCRIPT),
        "--",
        str(bvh_path),
        str(output_json),
    ]
    if start is not None:
        cmd.append(str(start))
        if end is not None:
            cmd.append(str(end))
            cmd.append(str(step))

    print(f"[Blender] {' '.join(cmd)}")

    # --- NETTOYAGE DE L'ENVIRONNEMENT POUR BLENDER ---
    # On copie l'environnement actuel du système
    clean_env = os.environ.copy()
    
    # 1. On retire Anaconda du PATH pour ce sous-processus
    if "PATH" in clean_env:
        paths = clean_env["PATH"].split(os.pathsep)
        # On filtre pour enlever tout chemin qui mène vers anaconda3
        cleaned_paths = [p for p in paths if "anaconda3" not in p]
        clean_env["PATH"] = os.pathsep.join(cleaned_paths)
    
    # 2. On supprime PYTHONPATH s'il existe, car il force Blender 
    # à regarder dans les dossiers de modules d'Anaconda
    clean_env.pop("PYTHONPATH", None)
    # -------------------------------------------------

    # On passe clean_env à subprocess
    result = subprocess.run(cmd, capture_output=False, text=True, env=clean_env)
    
    if result.returncode != 0:
        raise RuntimeError(f"Blender a échoué (code {result.returncode})")

    if not output_json.exists():
        raise RuntimeError(f"JSON non généré : {output_json}")

    print(f"[Blender] export terminé → {output_json}")
    return output_json

# ── Chargement JSON ───────────────────────────────────────────────────────────

def load_retarget_json(json_path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    """
    Charge le JSON exporté par retarget_export.py.
    Retourne {frame_idx (int) → {bone_name → np.array(4,4) float32}}.
    """
    with open(json_path) as f:
        raw = json.load(f)
    return {
        int(k): {bone: np.array(v, dtype=np.float32) for bone, v in frame.items()}
        for k, frame in raw.items()
    }


def get_frame_data(
    retarget_data: dict[int, dict[str, np.ndarray]],
    frame_idx: int,
) -> dict[str, np.ndarray]:
    """Retourne les données de la frame la plus proche de frame_idx."""
    frames = list(retarget_data.keys())
    closest = min(frames, key=lambda f: abs(f - frame_idx))
    return retarget_data[closest]


# ── Paramètres Anny ───────────────────────────────────────────────────────────

def params_from_frame(
    frame_data: dict[str, np.ndarray],
    model,
) -> dict[str, torch.Tensor]:
    """
    Convertit un frame_data Blender → pose_params Anny (mode "absolute").

    Les os absents du JSON (breast.L/R supprimés de Blender) héritent de la
    déformation de leur parent plutôt que de rester à l'identité.
    Formule : M_bone = M_parent_anim @ inv(M_parent_rest) @ M_bone_rest
    → deform_bone = deform_parent : le breast suit le thorax sans rotation propre.
    """
    rest = _get_rest_bone_poses(model)
    params: dict[str, torch.Tensor] = {}
    M_array: dict[str, np.ndarray] = {}

    for i, label in enumerate(model.bone_labels):
        if label in frame_data:
            M = frame_data[label]
        else:
            parent_idx = int(model.bone_parents[i])
            if parent_idx >= 0:
                parent_label = model.bone_labels[parent_idx]
                M_parent = M_array.get(parent_label)
                if M_parent is not None:
                    M = M_parent @ np.linalg.inv(rest[parent_idx]) @ rest[i]
                else:
                    M = rest[i]
            else:
                M = rest[i]

        M = M.astype(np.float32)
        M_array[label] = M
        params[label] = torch.from_numpy(M).float().unsqueeze(0)
    return params


# ── Helpers combinés ──────────────────────────────────────────────────────────

def retarget_frame_blender(
    retarget_data: dict[int, dict[str, np.ndarray]],
    frame_idx: int,
    model,
) -> dict[str, torch.Tensor]:
    """Shortcut : frame_idx → pose_params Anny (absolute)."""
    fd = get_frame_data(retarget_data, frame_idx)
    return params_from_frame(fd, model)


# ── CLI de test ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test retarget Blender → Anny")
    parser.add_argument("--bvh",   default="data/cmu/data/005/05_01.bvh")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end",   type=int, default=None)
    parser.add_argument("--step",  type=int, default=1)
    parser.add_argument("--out",   default="output/visualization_blender.png")
    parser.add_argument("--json",  default=None, help="JSON pré-calculé (évite de relancer Blender)")
    parser.add_argument("--gif",   action="store_true", help="Génère un GIF animé au lieu d'une image")
    parser.add_argument("--gif-frames", type=int, default=60, help="Nombre de frames pour le GIF (défaut 60)")
    parser.add_argument("--fps",   type=int, default=12)
    args = parser.parse_args()

    bvh_path = Path(args.bvh)

    # 1. Retarget via Blender (ou charge le cache)
    if args.json and Path(args.json).exists():
        json_path = Path(args.json)
        print(f"[cache] chargement JSON existant : {json_path}")
    else:
        json_path = run_blender_retarget(
            bvh_path,
            start=args.start,
            end=args.end,
            step=args.step,
        )

    # 2. Charge le JSON
    print(f"[load] {json_path}")
    retarget_data = load_retarget_json(json_path)
    frames_available = sorted(retarget_data.keys())
    print(f"  {len(frames_available)} frames disponibles : {frames_available[:5]}…{frames_available[-3:]}")

    # 3. Visualise une frame avec Anny
    from retarget import load_anny_model, generate_mesh
    from bvh_reader import BVHFile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualize import (
        draw_mesh_silhouette, draw_mesh_shaded, draw_skeleton, setup_ax,
        get_anny_joint_positions, ANNY_DISPLAY_BONES,
        CMU_BONES, project_xz,
    )

    print("[model] chargement Anny…")
    model = load_anny_model()

    print(f"[retarget] frame {args.frame}…")
    pose_params = retarget_frame_blender(retarget_data, args.frame, model)
    mesh, output = generate_mesh(model, pose_params, pose_parameterization="absolute")

    bvh = BVHFile(bvh_path)
    joint_pos = bvh.joint_world_positions(min(args.frame, bvh.n_frames - 1))

    # Visualisation
    fig, (ax_bvh, ax_anny) = plt.subplots(1, 2, figsize=(14, 8), facecolor="#0d0d1a")
    fig.suptitle(f"Blender retarget — {bvh_path.stem}  frame {args.frame}",
                 color="white", fontsize=13)

    bvh_2d = project_xz(joint_pos)
    from visualize import draw_bvh_skeleton_annotated
    draw_bvh_skeleton_annotated(ax_bvh, bvh_2d)
    setup_ax(ax_bvh, f"BVH CMU — frame {args.frame}")
    xs = [p[0] for p in bvh_2d.values()]
    ys = [p[1] for p in bvh_2d.values()]
    ax_bvh.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    ax_bvh.set_ylim(min(ys) - 0.3, max(ys) + 0.3)

    verts = mesh.vertices
    faces = model.faces if isinstance(model.faces, np.ndarray) else model.faces.cpu().numpy()
    draw_mesh_shaded(ax_anny, verts, faces)
    anny_pos = get_anny_joint_positions(output, model)
    cx = verts[:, 0].mean()
    anny_2d = {k: np.array([p[0] - cx, p[2]]) for k, p in anny_pos.items()}
    draw_skeleton(ax_anny, anny_2d, ANNY_DISPLAY_BONES,
                  joint_color="white", bone_color="#ff6b35", joint_size=25, linewidth=2)
    setup_ax(ax_anny, "Anny (Blender retarget)")
    ax_anny.set_xlim(verts[:, 0].min() - cx - 0.15, verts[:, 0].max() - cx + 0.15)
    ax_anny.set_ylim(verts[:, 2].min() - 0.1, verts[:, 2].max() + 0.1)

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualize] sauvegardé : {out_path}")

    if args.gif:
        _make_gif(retarget_data, frames_available, model, bvh_path, args)


# ── Helpers projection multi-vues ─────────────────────────────────────────────

# Angles de caméra : (theta_deg, label)
# theta=0 → face (-Y), theta négatif → tourne vers la droite de l'écran
_GIF_VIEWS = [
    (  0,  "Face"),
    (-50,  "¾ avant"),
    (130,  "¾ arrière"),
]


def _rotate_verts_z(verts: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Tourne les vertices de theta_deg autour de Z pour simuler la caméra
    à cet angle depuis l'avant (-Y).  theta=0 → vue de face inchangée.
    Après rotation, draw_mesh_shaded projette correctement (culling ny<0, proj XZ).
    """
    if theta_deg == 0:
        return verts
    rad = np.radians(theta_deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.column_stack([
        c * verts[:, 0] + s * verts[:, 1],
        -s * verts[:, 0] + c * verts[:, 1],
        verts[:, 2],
    ])


def _project_joints_angle(positions: dict, theta_deg: float) -> dict:
    """Projette les joints 3D → 2D (u=horizontal, v=Z) pour l'angle theta_deg."""
    rad = np.radians(theta_deg)
    c, s = np.cos(rad), np.sin(rad)
    return {
        name: np.array([c * p[0] + s * p[1], p[2]])
        for name, p in positions.items()
    }


def _make_gif(retarget_data, frames_available, model, bvh_path, args):
    """
    Génère un GIF animé 3 vues côte à côte : face, ¾ avant, ¾ arrière.
    Bounds globaux stables par vue. Qualité haute (dpi=120).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.animation import FuncAnimation
    from animate_bvh import _ANNY_LEFT_BONES, _ANNY_RIGHT_BONES, _ANNY_CENTER_BONES
    from visualize import get_anny_joint_positions, draw_mesh_shaded, _C_LEFT, _C_RIGHT, _C_CENTER, ANNY_DISPLAY_BONES

    faces          = model.faces if isinstance(model.faces, np.ndarray) else model.faces.cpu().numpy()
    _display_joints = {j for pair in ANNY_DISPLAY_BONES for j in pair}

    n = min(args.gif_frames, len(frames_available))
    indices    = np.linspace(0, len(frames_available) - 1, n, dtype=int)
    frame_list = [frames_available[i] for i in indices]

    print(f"[gif] pré-calcul {len(frame_list)} frames × {len(_GIF_VIEWS)} vues...")

    # Pré-calcul : vertices rotatés + joints projetés par vue
    precomputed = []
    view_xs = {theta: [] for theta, _ in _GIF_VIEWS}
    view_zs = {theta: [] for theta, _ in _GIF_VIEWS}

    for i, f in enumerate(frame_list):
        params = retarget_frame_blender(retarget_data, f, model)
        with __import__("torch").no_grad():
            out = model(pose_parameters=params, pose_parameterization="absolute")
        verts    = out["vertices"].squeeze(0).cpu().numpy()
        anny_pos = get_anny_joint_positions(out, model)

        frame_views = {}
        for theta, _ in _GIF_VIEWS:
            verts_rot  = _rotate_verts_z(verts, theta)
            cx         = float(verts_rot[:, 0].mean())
            joints_2d  = _project_joints_angle(anny_pos, theta)
            joints_2d  = {k: np.array([p[0] - cx, p[1]]) for k, p in joints_2d.items()}
            frame_views[theta] = (verts_rot, cx, joints_2d)
            view_xs[theta].extend(verts_rot[:, 0] - cx)
            view_zs[theta].extend(verts_rot[:, 2])

        precomputed.append((f, frame_views))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(frame_list)}")

    # Limites stables par vue
    view_limits = {}
    for theta, _ in _GIF_VIEWS:
        xs, zs = view_xs[theta], view_zs[theta]
        px = (max(xs) - min(xs)) * 0.05 + 0.04
        pz = (max(zs) - min(zs)) * 0.05 + 0.04
        view_limits[theta] = (
            (min(xs) - px, max(xs) + px),
            (min(zs) - pz, max(zs) + pz),
        )

    BG = "#0d0d1a"
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor=BG)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01, wspace=0.05)
    fig.suptitle(f"Blender retarget — {bvh_path.stem}", color="white", fontsize=11)

    def update(i):
        f, frame_views = precomputed[i]
        for ax, (theta, label) in zip(axes, _GIF_VIEWS):
            ax.cla()
            verts_rot, cx, joints_2d = frame_views[theta]

            draw_mesh_shaded(ax, verts_rot, faces)

            for bones, color in [
                (_ANNY_LEFT_BONES,   _C_LEFT),
                (_ANNY_RIGHT_BONES,  _C_RIGHT),
                (_ANNY_CENTER_BONES, _C_CENTER),
            ]:
                segs = [[joints_2d[a], joints_2d[b]] for a, b in bones
                        if a in joints_2d and b in joints_2d]
                if segs:
                    ax.add_collection(LineCollection(segs, colors=color, linewidths=1.8))

            for name, p in joints_2d.items():
                if name in _display_joints:
                    ax.scatter(p[0], p[1], c="white", s=8, zorder=5)

            xlim, ylim = view_limits[theta]
            ax.set_facecolor(BG)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_title(f"{label}  ·  f{f}", color="white", fontsize=9)
            ax.axis("off")

    anim = FuncAnimation(fig, update, frames=len(precomputed),
                         interval=1000 // args.fps, blit=False)
    gif_path = Path(args.out).with_suffix(".gif")
    print(f"[gif] sauvegarde {gif_path}...")
    anim.save(str(gif_path), writer="pillow", fps=args.fps, dpi=120)
    plt.close(fig)
    print(f"[gif] sauvegardé : {gif_path}")


if __name__ == "__main__":
    main()
