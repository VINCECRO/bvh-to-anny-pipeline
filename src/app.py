"""
Explorateur interactif CMU Mocap → Anny — Approche B (retarget Blender).
Rendu : PyVista + VTK.js (quads natifs, smooth shading).

Usage :
  uv run streamlit run src/app.py
"""

import base64
import sys
from pathlib import Path

import numpy as np
import torch
import streamlit as st
import streamlit.components.v1 as components

ROOT      = Path(__file__).parent.parent
CACHE_DIR = ROOT / "output" / "retarget_cache"
sys.path.insert(0, str(Path(__file__).parent))

from blender_retarget import (
    load_retarget_json,
    params_from_frame,
    get_frame_data,
    run_blender_retarget,
)
from retarget import load_anny_model
from visualize import ANNY_DISPLAY_BONES


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GestureAI — Pose Explorer",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0d0d1a; color: #e0e0e0; }
    [data-testid="stSidebar"]          { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    h1 { color: #00e5ff !important; }
    .stSlider > label { color: #aaa !important; }
    [data-testid="stMetric"] { background: #1a1a2e; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Données ────────────────────────────────────────────────────────────────────

@st.cache_data
def list_subjects() -> dict[str, list[Path]]:
    data_dir = ROOT / "data" / "cmu" / "data"
    result: dict[str, list[Path]] = {}
    for d in sorted(data_dir.iterdir()):
        if d.is_dir():
            files = sorted(d.glob("*.bvh"))
            if files:
                result[d.name] = files
    return result


def cache_path(bvh: Path) -> Path:
    return CACHE_DIR / f"{bvh.stem}.json"


# ── Modèle Anny ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return load_anny_model()


# ── Retarget data ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_retarget(json_path: str) -> dict[int, dict[str, np.ndarray]]:
    return load_retarget_json(json_path)


# ── Mesh Anny ─────────────────────────────────────────────────────────────────

@st.cache_data
def compute_mesh(
    _model,
    _retarget_data,
    json_path: str,
    frame_idx: int,
    age: float,
    gender: float,
    weight: float,
    muscle: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame_data = get_frame_data(_retarget_data, frame_idx)
    params     = params_from_frame(frame_data, _model)

    pheno = {k: torch.tensor([0.5], dtype=torch.float32) for k in _model.phenotype_labels}
    pheno["age"]    = torch.tensor([age],    dtype=torch.float32)
    pheno["gender"] = torch.tensor([gender], dtype=torch.float32)
    pheno["weight"] = torch.tensor([weight], dtype=torch.float32)
    pheno["muscle"] = torch.tensor([muscle], dtype=torch.float32)

    with torch.no_grad():
        out = _model(
            pose_parameters=params,
            phenotype_kwargs=pheno,
            pose_parameterization="absolute",
        )

    verts      = out["vertices"].squeeze(0).cpu().numpy()
    bone_poses = out["bone_poses"].squeeze(0).cpu().numpy()
    return verts, bone_poses


# ── Rendu Three.js → HTML ─────────────────────────────────────────────────────

_LEFT_COLOR   = "#00e5ff"
_RIGHT_COLOR  = "#ff9500"
_CENTER_COLOR = "#e0e0e0"


@st.cache_data
def render_to_html(
    verts: np.ndarray,
    bone_poses: np.ndarray,
    faces: np.ndarray,
    bone_labels: tuple,
    show_mesh: bool,
    show_skeleton: bool,
) -> str:
    """
    Génère un HTML Three.js autonome avec OrbitControls.
    - Quads triangulés (diagonale courte) + normales lissées par trimesh
    - Données encodées en base64 → aucun serveur, aucun thread requis
    """
    import trimesh as _trimesh

    # ── Triangulation (diagonale la plus courte par quad) ─────────────────────
    if faces.shape[1] == 4:
        q  = faces
        d1 = np.linalg.norm(verts[q[:, 0]] - verts[q[:, 2]], axis=1)
        d2 = np.linalg.norm(verts[q[:, 1]] - verts[q[:, 3]], axis=1)
        m  = d1 <= d2
        ta = np.where(m[:, None], q[:, [0, 1, 2]], q[:, [0, 1, 3]])
        tb = np.where(m[:, None], q[:, [0, 2, 3]], q[:, [1, 2, 3]])
        tri = np.vstack([ta, tb])
    else:
        tri = faces

    # ── Normales lissées (vraie moyenne angulaire par sommet) ─────────────────
    tm       = _trimesh.Trimesh(vertices=verts, faces=tri, process=False)
    vnormals = tm.vertex_normals.astype(np.float32)

    # ── Encodage binaire base64 ────────────────────────────────────────────────
    pos_b64  = base64.b64encode(verts.astype(np.float32).tobytes()).decode()
    norm_b64 = base64.b64encode(vnormals.tobytes()).decode()
    idx_b64  = base64.b64encode(tri.astype(np.uint32).tobytes()).decode()

    # ── Caméra ────────────────────────────────────────────────────────────────
    cx = float(verts[:, 0].mean())
    cy = float(verts[:, 1].mean())
    cz = float(verts[:, 2].mean())
    h  = float(verts[:, 2].max() - verts[:, 2].min())

    # ── Squelette JS ──────────────────────────────────────────────────────────
    skel_js = ""
    if show_skeleton:
        bone_list = list(bone_labels)
        bpos = {bone_list[i]: bone_poses[i, :3, 3] for i in range(len(bone_list))}

        for color_hex, side in [(_LEFT_COLOR, ".L"), (_RIGHT_COLOR, ".R"), (_CENTER_COLOR, None)]:
            pairs = [
                (bpos[a].tolist(), bpos[b].tolist()) for a, b in ANNY_DISPLAY_BONES
                if a in bpos and b in bpos and (
                    (side and side in b) or
                    (side is None and ".L" not in b and ".R" not in b)
                )
            ]
            if not pairs:
                continue
            pts_flat = [c for pair in pairs for pt in pair for c in pt]
            skel_js += f"""
{{
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.BufferAttribute(new Float32Array({pts_flat}), 3));
  scene.add(new THREE.LineSegments(sg,
    new THREE.LineBasicMaterial({{color: 0x{color_hex[1:]}}})));
}}"""

    mesh_js = ""
    if show_mesh:
        mesh_js = f"""
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(b64f32("{pos_b64}"),  3));
geo.setAttribute('normal',   new THREE.BufferAttribute(b64f32("{norm_b64}"), 3));
geo.setIndex(new THREE.BufferAttribute(b64u32("{idx_b64}"), 1));
const mat = new THREE.MeshPhongMaterial({{
  color: 0xD4A574, shininess: 25, side: THREE.DoubleSide
}});
scene.add(new THREE.Mesh(geo, mat));"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>body{{margin:0;background:#0d0d1a;overflow:hidden}}canvas{{display:block}}</style>
</head><body>
<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
             "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d1a);

const camera = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.001, 500);
camera.position.set({cx:.4f}, {cy - h*2.2:.4f}, {cz:.4f});
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({{antialias: true}});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set({cx:.4f}, {cy:.4f}, {cz:.4f});
controls.enableDamping = true;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const dl = new THREE.DirectionalLight(0xffffff, 0.75);
dl.position.set(1, -2, 3); scene.add(dl);
const dl2 = new THREE.DirectionalLight(0xffffff, 0.2);
dl2.position.set(-1, 1, 1); scene.add(dl2);

function b64f32(s){{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);}}
function b64u32(s){{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Uint32Array(u.buffer);}}

{mesh_js}
{skel_js}

(function loop(){{requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);}})();
window.addEventListener('resize',()=>{{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);}});
</script></body></html>"""


# ── Interface ──────────────────────────────────────────────────────────────────

def main():
    st.title("GestureAI — Explorateur de poses")

    subjects = list_subjects()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Navigation")

        subject = st.selectbox("Sujet", list(subjects.keys()))

        files      = subjects[subject]
        file_names = [f.name for f in files]

        labeled_names = [
            f"{n}  ✓" if cache_path(files[i]).exists() else n
            for i, n in enumerate(file_names)
        ]
        sel_label = st.selectbox("Fichier BVH", labeled_names,
                                 help="✓ = cache Blender disponible")
        sel_idx  = labeled_names.index(sel_label)
        bvh_path = files[sel_idx]
        json_fp  = cache_path(bvh_path)

        if json_fp.exists():
            with st.spinner("Chargement du cache retarget…"):
                retarget_data    = load_retarget(str(json_fp))
                frames_available = sorted(retarget_data.keys())

            frame = st.slider(
                "Frame",
                frames_available[0],
                frames_available[-1],
                min(100, frames_available[-1]),
                step=1,
            )
            st.caption(f"{len(frames_available)} frames en cache")
        else:
            frame = None
            st.warning("Aucun cache retarget pour ce fichier.")
            if st.button("Générer le cache (Blender, ~30 s)"):
                with st.spinner("Retargeting via Blender…"):
                    run_blender_retarget(str(bvh_path))
                st.success("Cache généré !")
                st.rerun()

        st.divider()
        st.subheader("Morphologie")
        age    = st.slider("Âge",          0.0, 1.0, 0.30, 0.05,
                           help="0 = bébé, 1 = vieillard")
        gender = st.slider("Genre  (M→F)", 0.0, 1.0, 0.50, 0.05,
                           help="0 = masculin, 1 = féminin")
        weight = st.slider("Corpulence",   0.0, 1.0, 0.40, 0.05)
        muscle = st.slider("Musculature",  0.0, 1.0, 0.60, 0.05)

        st.divider()
        st.subheader("Affichage")
        show_mesh     = st.checkbox("Mesh Anny",  value=True)
        show_skeleton = st.checkbox("Squelette",  value=True)

    if frame is None:
        st.info("Sélectionnez un fichier BVH avec un cache retarget (✓), ou générez-le via la sidebar.")
        return

    # ── Chargement modèle ─────────────────────────────────────────────────────
    with st.spinner("Chargement du modèle Anny (première fois ~20 s)…"):
        model = load_model()

    # ── Calcul mesh ───────────────────────────────────────────────────────────
    with st.spinner("Calcul de la pose…"):
        verts, bone_poses = compute_mesh(
            model, retarget_data, str(json_fp), frame,
            age, gender, weight, muscle,
        )

    faces = (
        model.faces if isinstance(model.faces, np.ndarray)
        else model.faces.cpu().numpy()
    )

    # ── Rendu PyVista → VTK.js HTML ──────────────────────────────────────────
    with st.spinner("Rendu PyVista…"):
        html_scene = render_to_html(
            verts, bone_poses, faces,
            tuple(model.bone_labels),
            show_mesh=show_mesh,
            show_skeleton=show_skeleton,
        )

    components.html(html_scene, height=720, scrolling=False)

    # ── Métriques ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sujet",    subject)
    c2.metric("Fichier",  bvh_path.name)
    c3.metric("Frame",    f"{frame} / {frames_available[-1]}")
    c4.metric("Vertices", f"{len(verts):,}")


if __name__ == "__main__":
    main()
