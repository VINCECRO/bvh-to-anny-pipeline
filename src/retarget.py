"""
Retargeting CMU BVH → Anny.

API publique :
  load_blender_matrices()  — charge blender/anny_rest_matrices_calibrated.json
  load_anny_model()        — crée le modèle Anny
  tpose_params()           — T-pose calibrée via paramétrage "absolute"
  retarget_frame()         — frame animée via paramétrage "rest_relative"
  generate_mesh()          — génère le mesh trimesh depuis des pose_params
  bvh_joint_positions_in_anny_space()  — convertit positions BVH → espace Anny

Conversion de coordonnées BVH → Anny :
  BVH  Y-up : X = droite/gauche, Y = haut, Z = profondeur
  Anny Z-up : X = droite/gauche, Z = haut, Y = profondeur (-Y = devant)
  Matrice _C = Rx(+90°) : Y → Z, Z → -Y
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from bvh_reader import BVHFile

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

BLENDER_JSON = Path(__file__).parent.parent / "blender" / "anny_rest_matrices_calibrated.json"

# Rx(+90°) : BVH Y-up → Anny Z-up
_C = np.array([[1.,  0.,  0.],
               [0.,  0., -1.],
               [0.,  1.,  0.]], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Mapping CMU joints → os Anny
# ─────────────────────────────────────────────────────────────────────────────
# Convention : seul le PREMIER os de chaque liste (os proximal) reçoit la rotation.

CMU_TO_ANNY: dict[str, list[str]] = {
    "Hips":          ["root"],
    "LHipJoint":     ["pelvis.L"],
    "RHipJoint":     ["pelvis.R"],
    "LeftUpLeg":     ["upperleg01.L", "upperleg02.L"],
    "LeftLeg":       ["lowerleg01.L", "lowerleg02.L"],
    "LeftFoot":      ["foot.L"],
    "LeftToeBase":   ["toe1-1.L"],
    "RightUpLeg":    ["upperleg01.R", "upperleg02.R"],
    "RightLeg":      ["lowerleg01.R", "lowerleg02.R"],
    "RightFoot":     ["foot.R"],
    "RightToeBase":  ["toe1-1.R"],
    "LowerBack":     ["spine05"],
    "Spine":         ["spine04", "spine03"],
    "Spine1":        ["spine02", "spine01"],
    "Neck":          ["neck01", "neck02"],
    "Neck1":         ["neck03"],
    "Head":          ["head"],
    "LeftShoulder":  ["clavicle.L", "shoulder01.L"],
    "LeftArm":       ["upperarm01.L", "upperarm02.L"],
    "LeftForeArm":   ["lowerarm01.L", "lowerarm02.L"],
    "LeftHand":      ["wrist.L"],
    "RightShoulder": ["clavicle.R", "shoulder01.R"],
    "RightArm":      ["upperarm01.R", "upperarm02.R"],
    "RightForeArm":  ["lowerarm01.R", "lowerarm02.R"],
    "RightHand":     ["wrist.R"],
    "LeftHandIndex1":  ["finger2-1.L"],
    "LThumb":          ["finger1-1.L"],
    "RightHandIndex1": ["finger2-1.R"],
    "RThumb":          ["finger1-1.R"],
}

# Dictionnaire inverse : os Anny proximal → joint CMU
_ANNY_TO_CMU: dict[str, str] = {
    bones[0]: cmu
    for cmu, bones in CMU_TO_ANNY.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_blender_matrices(json_path: Path = BLENDER_JSON) -> dict[str, np.ndarray]:
    """
    Charge les matrices de T-pose calibrées depuis Blender.
    Format : {bone_name → matrice 4×4 float32} en espace monde Anny (Z-up, mètres).
    """
    with open(json_path) as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


def load_anny_model(device: str = "cpu"):
    """Crée et retourne le modèle Anny (topologie default-notongue)."""
    import anny
    return anny.create_fullbody_model(
        rig="default-notongue",
        topology="default-notongue",
        all_phenotypes=True,
        remove_unattached_vertices=True,
    ).to(dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def bvh_joint_positions_in_anny_space(
    bvh: BVHFile,
    frame_idx: int,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Positions monde des joints BVH converties en espace Anny (Z-up, mètres).
    scale : facteur de conversion BVH-unités → mètres.
    """
    raw = bvh.joint_world_positions(frame_idx)
    return {k: _C @ (v.astype(np.float32) * scale) for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# T-pose
# ─────────────────────────────────────────────────────────────────────────────

def tpose_params(
    model,
    blender_mats: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """
    Paramètres Anny pour la T-pose calibrée Blender.

    Utilise pose_parameterization="absolute" : chaque matrice est la transform
    monde (rotation + translation) de l'os en espace Anny.
    Les os absents du JSON (faciaux, langue…) gardent leur pose de repos native.
    """
    identity = {label: torch.eye(4).unsqueeze(0) for label in model.bone_labels}
    with torch.no_grad():
        rest_out = model(pose_parameters=identity, pose_parameterization="rest_relative")
    rest_native = rest_out["rest_bone_poses"].squeeze(0).cpu().numpy()

    params: dict[str, torch.Tensor] = {}
    for i, label in enumerate(model.bone_labels):
        M = blender_mats[label] if label in blender_mats else rest_native[i]
        params[label] = torch.from_numpy(M.copy()).float().unsqueeze(0)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Retargeting animé
# ─────────────────────────────────────────────────────────────────────────────

def retarget_frame(
    bvh: BVHFile,
    frame_idx: int,
    model,
    blender_mats: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """
    FK retargeting BVH → paramètres absolus Anny (pose_parameterization="absolute").

    Formule pour l'os b (parent p, joint CMU j) :
        R_local_tpose = R_cal[p]^T @ R_cal[b]   ← rotation locale de b dans la T-pose calibrée
        R_anim[b]     = R_anim[p] @ R_local_tpose @ _C @ R_local_bvh[j] @ _C^T

    Traitement top-down (parent avant enfant). Pour les os non mappés, R_local_bvh = I :
    l'os garde son orientation locale T-pose et suit l'animation de son parent.

    Cohérence T-pose : si tous les R_local_bvh = I → résultat = blender_mats (T-pose calibrée ✓).
    À utiliser avec pose_parameterization="absolute".
    """
    frame_data = bvh.frame_as_parse_frame_dict(frame_idx)

    n = len(model.bone_labels)
    R_world_anim: list[np.ndarray] = [np.eye(3, dtype=np.float32)] * n
    pos_world_anim: list[np.ndarray] = [np.zeros(3, dtype=np.float32)] * n
    params: dict[str, torch.Tensor] = {}

    for i, label in enumerate(model.bone_labels):
        parent_idx = int(model.bone_parents[i])

        M_cal = blender_mats.get(label)
        R_cal = M_cal[:3, :3].astype(np.float32) if M_cal is not None else None
        pos_cal = M_cal[:3, 3].astype(np.float32) if M_cal is not None else None

        # Parent calibré et animé
        if parent_idx < 0:
            R_parent = np.eye(3, dtype=np.float32)
            R_parent_cal = np.eye(3, dtype=np.float32)
            pos_parent = np.zeros(3, dtype=np.float32)
            pos_parent_cal = np.zeros(3, dtype=np.float32)
        else:
            R_parent = R_world_anim[parent_idx]
            pos_parent = pos_world_anim[parent_idx]
            parent_label = model.bone_labels[parent_idx]
            M_parent_cal = blender_mats.get(parent_label)
            R_parent_cal = M_parent_cal[:3, :3].astype(np.float32) if M_parent_cal is not None else np.eye(3, np.float32)
            pos_parent_cal = M_parent_cal[:3, 3].astype(np.float32) if M_parent_cal is not None else pos_parent.copy()

        # Rotation locale T-pose (dans le repère du parent calibré)
        R_local_tpose = R_parent_cal.T @ (R_cal if R_cal is not None else np.eye(3, dtype=np.float32))

        # Rotation BVH locale (identité si os non mappé)
        cmu_joint = _ANNY_TO_CMU.get(label)
        R_bvh_local = (frame_data[cmu_joint][:3, :3].astype(np.float32)
                       if (cmu_joint and cmu_joint in frame_data)
                       else np.eye(3, dtype=np.float32))

        # Rotation monde animée
        R_anim = R_parent @ R_local_tpose @ _C @ R_bvh_local @ _C.T
        R_world_anim[i] = R_anim

        # Position monde via FK : offset local T-pose tourné par le parent animé
        if R_cal is not None and pos_cal is not None:
            offset_local = R_parent_cal.T @ (pos_cal - pos_parent_cal)
            pos_anim = pos_parent + R_parent @ offset_local
        else:
            pos_anim = pos_parent.copy()
        pos_world_anim[i] = pos_anim

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R_anim
        M[:3, 3] = pos_anim
        params[label] = torch.from_numpy(M.copy()).float().unsqueeze(0)

    return params


# ─────────────────────────────────────────────────────────────────────────────
# Génération mesh
# ─────────────────────────────────────────────────────────────────────────────

def generate_mesh(
    model,
    pose_params: dict[str, torch.Tensor],
    pose_parameterization: str = "rest_relative",
    phenotype_kwargs: dict | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """
    Génère le mesh Anny depuis des paramètres de pose.
    Retourne (mesh trimesh, output dict Anny).
    """
    pheno = phenotype_kwargs or {}
    with torch.no_grad():
        output = model(
            pose_parameters=pose_params,
            phenotype_kwargs=pheno,
            pose_parameterization=pose_parameterization,
        )
    vertices = output["vertices"].squeeze(0).cpu().numpy()
    faces    = model.faces if isinstance(model.faces, np.ndarray) else model.faces.cpu().numpy()
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), output
