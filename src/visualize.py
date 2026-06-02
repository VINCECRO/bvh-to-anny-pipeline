"""
Rendu image : squelette BVH + mesh Anny + variations morphologiques.

Produit output/visualization.png avec 3 panneaux :
  [A] Squelette BVH seul (stick figure)
  [B] Mesh Anny + squelette superposé
  [C] 4 morphologies Anny côte à côte (âge / corpulence)

Usage :
  uv run python src/visualize.py
  uv run python src/visualize.py --bvh data/cmu/data/013/13_29.bvh --frame 30
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # rendu sans affichage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from bvh_reader import BVHFile
from retarget import (
    CMU_TO_ANNY,
    load_blender_matrices, load_anny_model,
    retarget_frame, generate_mesh,
)


# ── Connexions squelette CMU pour le dessin ────────────────────────────────────
# Chaque tuple : (joint_parent, joint_enfant)
CMU_BONES = [
    # Colonne / torse
    ("Hips", "LowerBack"), ("LowerBack", "Spine"), ("Spine", "Spine1"),
    # Cou / tête
    ("Spine1", "Neck"), ("Neck", "Neck1"), ("Neck1", "Head"),
    # Épaules
    ("Spine1", "LeftShoulder"), ("Spine1", "RightShoulder"),
    # Bras gauche
    ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    # Bras droit
    ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    # Hanche
    ("Hips", "LHipJoint"), ("Hips", "RHipJoint"),
    # Jambe gauche
    ("LHipJoint", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    # Jambe droite
    ("RHipJoint", "RightUpLeg"), ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
]



def project_xz(positions: dict[str, np.ndarray], offset_x=0.0) -> dict[str, np.ndarray]:
    """
    Projection orthographique frontale : X → horizontal, Y → vertical.
    (CMU Mocap : Y est l'axe vertical)
    """
    if not positions:
        return {}
    vals = np.array(list(positions.values()))
    cx = vals[:, 0].mean()
    return {name: np.array([p[0] - cx + offset_x, p[1]]) for name, p in positions.items()}


def get_anny_joint_positions(output, model) -> dict[str, np.ndarray]:
    """
    Extrait les positions des joints Anny.
    Anny utilise Z comme axe vertical → on retourne (X, Z) pour l'affichage.
    """
    bone_poses = output["bone_poses"].squeeze(0).cpu().numpy()  # (N, 4, 4)
    positions = {}
    for i, label in enumerate(model.bone_labels):
        p = bone_poses[i, :3, 3]
        positions[label] = p  # XYZ complet, projection faite à l'affichage
    return positions


# Connexions squelette Anny (sous-ensemble lisible pour l'affichage)
ANNY_DISPLAY_BONES = [
    # Colonne
    ("root", "spine05"), ("spine05", "spine04"), ("spine04", "spine03"),
    ("spine03", "spine02"), ("spine02", "spine01"),
    # Cou / tête
    ("spine01", "neck01"), ("neck01", "neck02"), ("neck02", "neck03"), ("neck03", "head"),
    # Bras gauche
    ("spine01", "clavicle.L"), ("clavicle.L", "shoulder01.L"),
    ("shoulder01.L", "upperarm01.L"), ("upperarm01.L", "upperarm02.L"),
    ("upperarm02.L", "lowerarm01.L"), ("lowerarm01.L", "lowerarm02.L"),
    ("lowerarm02.L", "wrist.L"),
    # Bras droit
    ("spine01", "clavicle.R"), ("clavicle.R", "shoulder01.R"),
    ("shoulder01.R", "upperarm01.R"), ("upperarm01.R", "upperarm02.R"),
    ("upperarm02.R", "lowerarm01.R"), ("lowerarm01.R", "lowerarm02.R"),
    ("lowerarm02.R", "wrist.R"),
    # Jambe gauche
    ("root", "pelvis.L"), ("pelvis.L", "upperleg01.L"), ("upperleg01.L", "upperleg02.L"),
    ("upperleg02.L", "lowerleg01.L"), ("lowerleg01.L", "lowerleg02.L"),
    ("lowerleg02.L", "foot.L"), ("foot.L", "toe1-1.L"),
    # Jambe droite
    ("root", "pelvis.R"), ("pelvis.R", "upperleg01.R"), ("upperleg01.R", "upperleg02.R"),
    ("upperleg02.R", "lowerleg01.R"), ("lowerleg01.R", "lowerleg02.R"),
    ("lowerleg02.R", "foot.R"), ("foot.R", "toe1-1.R"),
]


def draw_skeleton(ax, positions_2d: dict[str, np.ndarray], bones: list[tuple],
                  joint_color="white", bone_color="cyan", joint_size=40, linewidth=2,
                  label=None):
    """Dessine un squelette 2D sur un axe matplotlib."""
    segments = []
    visible = {j for pair in bones for j in pair}
    for a, b in bones:
        if a in positions_2d and b in positions_2d:
            segments.append([positions_2d[a], positions_2d[b]])
    if segments:
        lc = LineCollection(segments, colors=bone_color, linewidths=linewidth, label=label)
        ax.add_collection(lc)
    # N'afficher que les joints présents dans la liste d'os (pas les os faciaux)
    xs = [p[0] for n, p in positions_2d.items() if n in visible]
    ys = [p[1] for n, p in positions_2d.items() if n in visible]
    ax.scatter(xs, ys, c=joint_color, s=joint_size, zorder=5)


# Partition des os CMU par côté pour la coloration
_CMU_LEFT_BONES  = {b for b in CMU_BONES if "Left"  in b[0] or "Left"  in b[1]}
_CMU_RIGHT_BONES = {b for b in CMU_BONES if "Right" in b[0] or "Right" in b[1]}
_CMU_LEFT_BONES  |= {("Hips","LHipJoint"),("LHipJoint","LeftUpLeg")}
_CMU_RIGHT_BONES |= {("Hips","RHipJoint"),("RHipJoint","RightUpLeg")}
_CMU_CENTER_BONES = [b for b in CMU_BONES if b not in _CMU_LEFT_BONES and b not in _CMU_RIGHT_BONES]

_C_LEFT   = "#00e5ff"   # cyan   — côté gauche
_C_RIGHT  = "#ff9500"   # orange — côté droit
_C_CENTER = "#e0e0e0"   # gris clair — colonne, tête


def draw_bvh_skeleton_annotated(ax, positions_2d: dict[str, np.ndarray]):
    """
    Squelette BVH annoté :
      - Côté gauche en cyan, côté droit en orange, colonne en gris
      - Cercle pour la tête, labels sur les joints clés
      - Ligne de sol de référence
    """
    lw = 2.5

    # Segments colorés par côté
    for bones, color in [(_CMU_LEFT_BONES, _C_LEFT),
                         (_CMU_RIGHT_BONES, _C_RIGHT),
                         (_CMU_CENTER_BONES, _C_CENTER)]:
        segs = []
        for a, b in bones:
            if a in positions_2d and b in positions_2d:
                segs.append([positions_2d[a], positions_2d[b]])
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=3))

    # Points joints — taille selon importance
    major = {"Hips", "Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot",
             "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg"}
    for name, p in positions_2d.items():
        size = 80 if name in major else 25
        color = "yellow" if name == "Hips" else "white"
        ax.scatter(p[0], p[1], c=color, s=size, zorder=6)

    # Cercle distinctif pour la tête
    if "Head" in positions_2d:
        hx, hy = positions_2d["Head"]
        ys = [p[1] for p in positions_2d.values()]
        head_r = (max(ys) - min(ys)) * 0.05
        circle = plt.Circle((hx, hy), head_r, color="#e0e0e0", fill=False, linewidth=2, zorder=7)
        ax.add_patch(circle)

    # Labels des joints clés avec indication L/R
    annotations = {
        "Head":       ("HEAD",    (0,  8)),
        "Hips":       ("HIPS",    (5,  0)),
        "LeftHand":   ("L HAND",  (5,  0)),
        "RightHand":  ("R HAND",  (-45, 0)),
        "LeftFoot":   ("L FOOT",  (5,  0)),
        "RightFoot":  ("R FOOT",  (-45, 0)),
        "LeftArm":    ("L ÉPAULE",(5,  0)),
        "RightArm":   ("R ÉPAULE",(-52,0)),
    }
    for jname, (lbl, off) in annotations.items():
        if jname in positions_2d:
            p = positions_2d[jname]
            color = _C_LEFT if "Left" in jname else (_C_RIGHT if "Right" in jname else "yellow")
            ax.annotate(lbl, p, xytext=off, textcoords="offset points",
                        fontsize=7, color=color, fontweight="bold", zorder=8)

    # Ligne de sol : au niveau du pied le plus bas
    ys_feet = [positions_2d[j][1] for j in ("LeftFoot","RightFoot") if j in positions_2d]
    if ys_feet:
        floor_y = min(ys_feet)
        xs_all  = [p[0] for p in positions_2d.values()]
        ax.axhline(floor_y, color="#555", linewidth=1, linestyle="--", alpha=0.6, zorder=1)
        ax.text(min(xs_all), floor_y, "  sol", color="#888", fontsize=7, va="top")

    # Légende côté gauche/droit
    ax.legend(handles=[
        mpatches.Patch(color=_C_LEFT,   label="Côté gauche"),
        mpatches.Patch(color=_C_RIGHT,  label="Côté droit"),
        mpatches.Patch(color=_C_CENTER, label="Colonne / tête"),
    ], loc="lower right", facecolor="#1a1a2e", labelcolor="white", fontsize=7)


def draw_mesh_silhouette(ax, vertices: np.ndarray, faces: np.ndarray,
                         alpha=0.15, color="lightblue"):
    """
    Rendu simplifié du mesh Anny : projection frontale X (horizontal) / Z (vertical).
    Anny utilise Z comme axe vertical.
    Garde seulement les faces orientées vers l'observateur (+Y dans Anny = profondeur).
    """
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # Anny : X = gauche/droite, Z = haut/bas, Y = profondeur
    cx = vertices[:, 0].mean()
    verts_2d = np.column_stack([vertices[:, 0] - cx, vertices[:, 2]])  # X, Z

    patches = []
    for face in faces:
        pts = verts_2d[face]
        v0, v1, v2 = pts[0], pts[1], pts[2]
        # Face culling : normale Y (profondeur) > 0 = visible de face
        cross_z = (v1[0]-v0[0])*(v2[1]-v0[1]) - (v1[1]-v0[1])*(v2[0]-v0[0])
        if cross_z < 0:
            patches.append(Polygon(pts, closed=True))

    if patches:
        pc = PatchCollection(patches, alpha=alpha, facecolor=color, edgecolor="none")
        ax.add_collection(pc)


def draw_mesh_shaded(ax, vertices: np.ndarray, faces: np.ndarray,
                     base_color=(0.55, 0.65, 0.80)):
    """
    Rendu mesh opaque avec ombrage plat (painter's algorithm, entièrement vectorisé).

    Anny : X = gauche/droite, Z = vertical, Y = profondeur (-Y = devant).
    Projection frontale : X horizontal, Z vertical.

    Algorithme :
      1. Calcul vectorisé des normales de face
      2. Back-face culling (normal.Y < 0 → face vers l'observateur)
      3. Tri arrière→avant (Y décroissant)
      4. Ombrage plat : ambient + diffus (dot normal · direction_lumière)
      5. PolyCollection avec couleur par face (pas de boucle Python)
    """
    from matplotlib.collections import PolyCollection

    cx = vertices[:, 0].mean()
    v3 = vertices[faces]          # (F, 3, 3) — triplets de sommets

    # ── Normales vectorisées ──────────────────────────────────────────────────
    e1 = v3[:, 1] - v3[:, 0]
    e2 = v3[:, 2] - v3[:, 0]
    normals  = np.cross(e1, e2)                              # (F, 3)
    norms    = np.linalg.norm(normals, axis=1)               # (F,)
    valid    = norms > 1e-10                                  # (F,) bool
    normals[valid]  /= norms[valid, None]
    normals[~valid]  = 0.0

    # ── Back-face culling : normal.Y < 0 → face vers l'observateur ───────────
    fi = np.where(valid & (normals[:, 1] < 0))[0]
    if len(fi) == 0:
        return

    # ── Tri arrière→avant (Y moyen décroissant) ───────────────────────────────
    order = np.argsort(-v3[fi, :, 1].mean(axis=1))
    fi = fi[order]

    # ── Ombrage diffus + ambient ──────────────────────────────────────────────
    # Lumière depuis le haut-gauche-avant (espace Anny)
    light = np.array([-0.3, -1.0, 0.6], dtype=np.float32)
    light /= np.linalg.norm(light)
    intensity  = np.clip(normals[fi] @ light, 0.0, 1.0)     # (N,)
    brightness = (0.30 + 0.70 * intensity)[:, None]           # (N, 1)
    facecolors = np.clip(np.array(base_color, dtype=np.float32) * brightness, 0.0, 1.0)

    # ── Projection 2D et rendu ────────────────────────────────────────────────
    pts = np.stack([v3[fi, :, 0] - cx, v3[fi, :, 2]], axis=-1)  # (N, 3, 2)
    pc  = PolyCollection(pts, facecolors=facecolors, edgecolors="none", linewidths=0)
    ax.add_collection(pc)


def setup_ax(ax, title: str, bg_color="#1a1a2e"):
    ax.set_facecolor(bg_color)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(True, alpha=0.15, color="white")


def render_morpho_panel(ax, model, phenotype_overrides: dict, label: str,
                        frame_pose_params: dict):
    """
    Panneau C : rendu d'une morphologie particulière avec la pose BVH.
    phenotype_overrides : paramètres phénotypiques à modifier (valeurs 0..1).
    """
    # Paramètres phénotypiques de base (valeurs moyennes)
    pheno = {k: torch.tensor([0.5], dtype=torch.float32) for k in model.phenotype_labels}
    for k, v in phenotype_overrides.items():
        if k in pheno:
            pheno[k] = torch.tensor([float(v)], dtype=torch.float32)

    with torch.no_grad():
        output = model(
            pose_parameters=frame_pose_params,
            phenotype_kwargs=pheno,
            pose_parameterization="absolute",
        )

    verts = output["vertices"].squeeze(0).cpu().numpy()
    faces = model.faces if isinstance(model.faces, np.ndarray) else model.faces.cpu().numpy()

    draw_mesh_silhouette(ax, verts, faces, alpha=0.25, color="lightcoral")

    anny_pos = get_anny_joint_positions(output, model)
    cx = verts[:, 0].mean()
    anny_2d = {k: np.array([p[0] - cx, p[2]]) for k, p in anny_pos.items()}
    draw_skeleton(ax, anny_2d, ANNY_DISPLAY_BONES, joint_color="white",
                  bone_color="orange", joint_size=15, linewidth=1)

    setup_ax(ax, label)
    ax.set_xlim(verts[:, 0].min() - cx - 0.1, verts[:, 0].max() - cx + 0.1)
    ax.set_ylim(verts[:, 2].min() - 0.1, verts[:, 2].max() + 0.1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh", default="data/cmu/data/005/05_01.bvh")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--out", default="output/visualization.png")
    args = parser.parse_args()

    bvh_path = Path(args.bvh)
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)

    print(f"[1] Chargement BVH : {bvh_path.name}, frame {args.frame}")
    bvh = BVHFile(bvh_path)
    frame_idx = min(args.frame, bvh.n_frames - 1)
    joint_pos = bvh.joint_world_positions(frame_idx)

    print("[2] Chargement modèle Anny...")
    model        = load_anny_model()
    blender_mats = load_blender_matrices()

    print("[3] Retargeting BVH → Anny...")
    pose_params = retarget_frame(bvh, frame_idx, model, blender_mats)
    mesh, output = generate_mesh(model, pose_params, pose_parameterization="absolute")

    # ── Figure ────────────────────────────────────────────────────────────────
    print("[4] Rendu image...")
    fig = plt.figure(figsize=(20, 10), facecolor="#0d0d1a")
    fig.suptitle(
        f"GestureAI Pipeline — {bvh_path.stem}  frame {frame_idx}/{bvh.n_frames-1}",
        color="white", fontsize=14, y=0.98
    )

    # Grille : [A][B][C1][C2][C3][C4]  — A et B plus larges
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.2, 1.2, 1, 1],
        height_ratios=[1, 1],
        hspace=0.35, wspace=0.3,
        left=0.04, right=0.98, top=0.93, bottom=0.05,
    )
    ax_bvh  = fig.add_subplot(gs[:, 0])   # Panneau A : squelette BVH
    ax_anny = fig.add_subplot(gs[:, 1])   # Panneau B : mesh Anny + squelette
    ax_m1   = fig.add_subplot(gs[0, 2])   # Panneau C : morpho 1
    ax_m2   = fig.add_subplot(gs[0, 3])   # Panneau C : morpho 2
    ax_m3   = fig.add_subplot(gs[1, 2])   # Panneau C : morpho 3
    ax_m4   = fig.add_subplot(gs[1, 3])   # Panneau C : morpho 4

    # ── A : Squelette BVH ─────────────────────────────────────────────────────
    bvh_2d = project_xz(joint_pos)
    draw_bvh_skeleton_annotated(ax_bvh, bvh_2d)
    setup_ax(ax_bvh, f"Squelette CMU Mocap — {bvh_path.stem} f{frame_idx}")
    xs = [p[0] for p in bvh_2d.values()]
    ys = [p[1] for p in bvh_2d.values()]
    pad_x = max((max(xs) - min(xs)) * 0.25, 0.3)
    pad_y = max((max(ys) - min(ys)) * 0.15, 0.3)
    ax_bvh.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax_bvh.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax_bvh.set_xlabel("← R   X   L →", color="gray", fontsize=8)
    ax_bvh.set_ylabel("Hauteur (u. BVH)", color="gray", fontsize=8)

    # ── B : Mesh Anny + squelette ──────────────────────────────────────────────
    verts = mesh.vertices
    faces = model.faces if isinstance(model.faces, np.ndarray) else model.faces.cpu().numpy()
    draw_mesh_silhouette(ax_anny, verts, faces, alpha=0.2, color="lightblue")

    anny_pos = get_anny_joint_positions(output, model)
    cx = verts[:, 0].mean()
    # Anny : X horizontal, Z vertical
    anny_2d = {k: np.array([p[0] - cx, p[2]]) for k, p in anny_pos.items()}
    draw_skeleton(ax_anny, anny_2d, ANNY_DISPLAY_BONES,
                  joint_color="white", bone_color="#ff6b35", joint_size=25, linewidth=2)
    setup_ax(ax_anny, "Mesh Anny + squelette (pose BVH)")
    ax_anny.set_xlim(verts[:, 0].min() - cx - 0.15, verts[:, 0].max() - cx + 0.15)
    ax_anny.set_ylim(verts[:, 2].min() - 0.1, verts[:, 2].max() + 0.1)

    # Légende
    ax_anny.legend(handles=[
        mpatches.Patch(color="lightblue", alpha=0.6, label="Mesh Anny"),
        mpatches.Patch(color="#ff6b35", label="Squelette Anny"),
    ], loc="lower right", facecolor="#1a1a2e", labelcolor="white", fontsize=7)

    # ── C : 4 variations morphologiques ───────────────────────────────────────
    # Même pose BVH, morphologies différentes
    # pose_params avec batch_size=1 requis pour render_morpho_panel
    morpho_configs = [
        ({"age": 0.15, "gender": 0.8, "weight": 0.4, "muscle": 0.6},
         "Jeune adulte\nFéminin · svelte"),
        ({"age": 0.70, "gender": 0.1, "weight": 0.7, "muscle": 0.4},
         "Adulte mature\nMasculin · corpulent"),
        ({"age": 0.55, "gender": 0.5, "weight": 0.5, "muscle": 0.9},
         "Adulte\nMusclé"),
        ({"age": 0.05, "gender": 0.5, "weight": 0.45, "muscle": 0.5},
         "Enfant"),
    ]

    for ax_m, (pheno_overrides, title) in zip(
        [ax_m1, ax_m2, ax_m3, ax_m4], morpho_configs
    ):
        render_morpho_panel(ax_m, model, pheno_overrides, title, pose_params)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[5] Image sauvegardée : {out_path}")
    print(f"    Taille : {out_path.stat().st_size // 1024} Ko")


if __name__ == "__main__":
    main()
