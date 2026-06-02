"""
Parseur BVH direct — n'utilise pas bvhio pour la FK.

Pourquoi : bvhio réaligne les offsets de repos sur son propre axe Y,
ce qui rend RestPose.Position incompatible avec les offsets BVH bruts.
Ce module lit le fichier BVH directement et calcule la FK proprement.

Convention de rotation CMU BVH (tous les joints) :
  CHANNELS 3 Zrotation Yrotation Xrotation
  → rotation intrinsèque ZYX = Rz(az) @ Ry(ay) @ Rx(ax)
  (confirmé par comparaison avec SpaceWorld de bvhio)

Usage :
  from bvh_reader import BVHFile
  bvh = BVHFile('data/cmu/data/005/05_01.bvh')
  positions = bvh.joint_world_positions(frame_idx=1)
  # → dict[str, np.ndarray(3)]  (coordonnées en unités BVH, Y = vertical)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np


# ── Matrices de rotation élémentaires ─────────────────────────────────────────

def _Rx(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def _Ry(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def _Rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

_ROTMAT = {'Xrotation': _Rx, 'Yrotation': _Ry, 'Zrotation': _Rz}


def _euler_to_rotmat(channel_names: list[str], values: list[float]) -> np.ndarray:
    """
    BVH rotation intrinsèque : appliquer chaque rotation autour de l'axe du corps courant.
    Pour [Zrotation, Yrotation, Xrotation] → M = Rz(az) @ Ry(ay) @ Rx(ax).
    """
    R = np.eye(3, dtype=np.float64)
    for ch, val in zip(channel_names, values):
        R = R @ _ROTMAT[ch](val)
    return R


# ── Structure d'un joint ───────────────────────────────────────────────────────

class _Joint:
    __slots__ = ('name', 'parent', 'children', 'offset', 'channels')

    def __init__(self, name: str, parent: str | None, offset: np.ndarray, channels: list[str]):
        self.name     = name
        self.parent   = parent
        self.children: list[str] = []
        self.offset   = offset           # vecteur 3D dans le référentiel du parent au repos
        self.channels = channels         # ex: ['Zrotation', 'Yrotation', 'Xrotation']


# ── Parseur principal ──────────────────────────────────────────────────────────

class BVHFile:
    """
    Lit un fichier BVH et calcule la FK joint-par-joint.

    Attributes
    ----------
    joint_names : list[str]         — ordre depth-first de la hiérarchie
    n_frames    : int
    frame_time  : float             — durée d'une frame en secondes
    """

    def __init__(self, filepath: str | Path):
        self._joints: dict[str, _Joint] = {}
        self._joint_order: list[str] = []
        self._channel_index: list[tuple[str, str]] = []  # [(joint_name, channel), ...]
        self._frames: list[list[float]] = []
        self.frame_time: float = 0.0
        self._parse(Path(filepath))

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, path: Path) -> None:
        text = path.read_text()
        hier, motion = text.split('MOTION', 1)
        self._parse_hierarchy(hier)
        self._parse_motion(motion)

    def _parse_hierarchy(self, text: str) -> None:
        stack: list[str] = []
        cur_name: str | None = None
        end_site_depth = 0  # profondeur de blocs "End Site" ouverts

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line == '{':
                continue

            parts = line.split()
            tag = parts[0].upper()

            if tag in ('ROOT', 'JOINT'):
                name = parts[1]
                parent = stack[-1] if stack else None
                j = _Joint(name=name, parent=parent, offset=np.zeros(3), channels=[])
                if parent:
                    self._joints[parent].children.append(name)
                self._joints[name] = j
                self._joint_order.append(name)
                stack.append(name)
                cur_name = name

            elif tag == 'END':
                # Début d'un bloc "End Site" — on compte la profondeur pour absorber son '}'
                end_site_depth += 1

            elif tag == 'OFFSET' and end_site_depth == 0 and cur_name:
                self._joints[cur_name].offset = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64
                )

            elif tag == 'CHANNELS' and end_site_depth == 0 and cur_name:
                n = int(parts[1])
                channels = parts[2: 2 + n]
                self._joints[cur_name].channels = channels
                for ch in channels:
                    self._channel_index.append((cur_name, ch))

            elif tag == '}':
                if end_site_depth > 0:
                    # Ferme un bloc End Site, pas un joint
                    end_site_depth -= 1
                elif stack:
                    stack.pop()
                    cur_name = stack[-1] if stack else None

    def _parse_motion(self, text: str) -> None:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        # lines[0] = 'Frames: N'
        # lines[1] = 'Frame Time: T'
        # lines[2:] = frame data
        self.n_frames  = int(lines[0].split(':')[1])
        self.frame_time = float(lines[1].split(':')[1])
        for line in lines[2: 2 + self.n_frames]:
            self._frames.append([float(v) for v in line.split()])

    # ── Accès ─────────────────────────────────────────────────────────────────

    @property
    def joint_names(self) -> list[str]:
        return self._joint_order

    def raw_frame(self, frame_idx: int) -> list[float]:
        return self._frames[min(frame_idx, self.n_frames - 1)]

    # ── FK ────────────────────────────────────────────────────────────────────

    def joint_world_positions(self, frame_idx: int) -> dict[str, np.ndarray]:
        """
        Calcule les positions monde de tous les joints pour la frame donnée.

        Retourne dict[joint_name → np.ndarray(3)] en unités BVH (Y = vertical).
        """
        frame = self.raw_frame(frame_idx)

        # Mapper les valeurs de canaux sur les joints
        ch_vals: dict[str, dict[str, float]] = {}
        for i, (jname, ch) in enumerate(self._channel_index):
            ch_vals.setdefault(jname, {})[ch] = frame[i]

        world_pos: dict[str, np.ndarray] = {}
        world_rot: dict[str, np.ndarray] = {}  # matrice 3×3

        def _fk(name: str, parent_pos: np.ndarray, parent_rot: np.ndarray) -> None:
            j = self._joints[name]
            vals = ch_vals.get(name, {})

            # Translation (canaux Xposition/Yposition/Zposition — root seulement)
            translation = np.array([
                vals.get('Xposition', 0.0),
                vals.get('Yposition', 0.0),
                vals.get('Zposition', 0.0),
            ], dtype=np.float64)

            # Rotation locale
            rot_chs = [ch for ch in j.channels if ch.endswith('rotation')]
            rot_vals = [vals.get(ch, 0.0) for ch in rot_chs]
            local_rot = _euler_to_rotmat(rot_chs, rot_vals)

            # Rotation monde = parent_rot @ local_rot
            w_rot = parent_rot @ local_rot

            # Position monde
            if j.parent is None:
                # Root : la translation est directement en espace monde
                w_pos = parent_pos + translation
            else:
                # Joints non-root : offset dans le référentiel monde du parent
                w_pos = parent_pos + parent_rot @ j.offset

            world_pos[name] = w_pos
            world_rot[name] = w_rot

            for child_name in j.children:
                _fk(child_name, w_pos, w_rot)

        root_name = self._joint_order[0]
        _fk(root_name, np.zeros(3), np.eye(3, dtype=np.float64))

        return world_pos

    def joint_world_rotations(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Retourne les matrices de rotation monde 3×3 pour tous les joints."""
        frame = self.raw_frame(frame_idx)
        ch_vals: dict[str, dict[str, float]] = {}
        for i, (jname, ch) in enumerate(self._channel_index):
            ch_vals.setdefault(jname, {})[ch] = frame[i]

        world_rot: dict[str, np.ndarray] = {}

        def _fk(name: str, parent_rot: np.ndarray) -> None:
            j = self._joints[name]
            vals = ch_vals.get(name, {})
            rot_chs  = [ch for ch in j.channels if ch.endswith('rotation')]
            rot_vals = [vals.get(ch, 0.0) for ch in rot_chs]
            local_rot = _euler_to_rotmat(rot_chs, rot_vals)
            w_rot = parent_rot @ local_rot
            world_rot[name] = w_rot
            for child_name in j.children:
                _fk(child_name, w_rot)

        _fk(self._joint_order[0], np.eye(3, dtype=np.float64))
        return world_rot

    def frame_as_parse_frame_dict(self, frame_idx: int) -> dict[str, np.ndarray]:
        """
        Retourne les rotations locales au format compatible avec parse_frame() de parse_bvh.py :
          {joint_name: matrice 4×4 float32 (rotation locale)}
          '_root_position': np.array([x, y, z]) en mètres (÷ 100 pour Anny)

        Remplace l'usage de parse_frame() + bvhio pour le retargeting Anny.
        """
        frame = self.raw_frame(frame_idx)
        ch_vals: dict[str, dict[str, float]] = {}
        for i, (jname, ch) in enumerate(self._channel_index):
            ch_vals.setdefault(jname, {})[ch] = frame[i]

        result: dict[str, np.ndarray] = {}
        for name in self._joint_order:
            j = self._joints[name]
            vals = ch_vals.get(name, {})
            rot_chs  = [ch for ch in j.channels if ch.endswith('rotation')]
            rot_vals = [vals.get(ch, 0.0) for ch in rot_chs]
            local_rot = _euler_to_rotmat(rot_chs, rot_vals).astype(np.float32)
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = local_rot
            result[name] = M

        # Position monde du root en mètres (÷ 100, convention parse_bvh.py)
        root = self._joint_order[0]
        rv = ch_vals.get(root, {})
        result['_root_position'] = np.array([
            rv.get('Xposition', 0.0),
            rv.get('Yposition', 0.0),
            rv.get('Zposition', 0.0),
        ], dtype=np.float32) / 100.0

        return result
