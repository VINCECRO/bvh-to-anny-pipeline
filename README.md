# GestureAI — Pose Dataset Pipeline

Pipeline de génération d'un dataset de poses humaines pour **GestureAI**, un outil pédagogique d'apprentissage du dessin du corps humain.

Pour chaque pose sélectionnée, le pipeline produit deux sorties :
- **Vue pédagogique** — rendu Blender (EEVEE/Cycles) avec overlay Bridgman et vue anatomique
- **Vue photoréaliste** — image générée par Flux.1 Schnell depuis la depth map

---

## Stack technique

| Composant | Outil | Licence |
|---|---|---|
| Poses 3D | CMU Mocap (BVH) | Domaine public |
| Modèle corporel | Anny (NAVER Labs) | Apache 2.0 |
| Rendu 3D | Blender 4.0 headless (`bpy`) | GPL |
| Génération image | Flux.1 Schnell | Apache 2.0 |
| Runtime local | RTX 2070 8 GB (NF4/GGUF) | — |
| Deep learning | PyTorch 2.12 + CUDA 13 | BSD |
| Rotations | roma 1.5.6 | MIT |
| Mesh | trimesh 4.12.2 | MIT |

**Gestionnaire de paquets : [UV](https://docs.astral.sh/uv/)**

---

## Installation

### Prérequis

- Python 3.10+
- UV (`pip install uv`)
- Blender 4.0 installé et accessible dans le PATH (`blender --version`)
- CUDA 13 recommandé pour PyTorch

### Cloner avec les sous-modules

```bash
git clone --recurse-submodules <url-du-repo>
cd Pose_Compute_V2_Anny_x_CMU
```

Si le dépôt est déjà cloné sans sous-modules :

```bash
git submodule update --init
```

### Sparse checkout pour CMU Mocap (sujets utilisés)

```bash
cd data/cmu
git sparse-checkout set data/005 data/013 data/014 data/015 data/060 data/061 data/062 data/063 data/064
git checkout
cd ../..
```

### Addon Blender — retarget_bvh

Installer l'addon `retarget_bvh` dans Blender 4.0 :

```
~/.config/blender/4.0/scripts/addons/retarget_bvh-master/
```

### Installer l'environnement Python

```bash
uv sync
```

---

## Sources de données

### CMU Mocap (`data/cmu/`)

Fichiers BVH domaine public. Sparse checkout limité aux sujets artistiquement intéressants :

| Sujets | Catégorie |
|---|---|
| 005 | Danse |
| 013, 014, 015 | Sports |
| 060–064 | Danse latine |

Source : [github.com/una-dinosauria/cmu-mocap](https://github.com/una-dinosauria/cmu-mocap)

### Modèle Anny (`data/anny/`)

Modèle corporel paramétrique NAVER Labs. Installé comme workspace UV editable.
Topologie utilisée : `default-notongue` (Apache 2.0, assets MakeHuman CC0).

Source : [github.com/naver/anny](https://github.com/naver/anny)

---

## Architecture du pipeline

```
CMU Mocap BVH
    ↓
[1] Parsing BVH → rotations articulaires (bvh_reader.py ✓)
    ↓
[2] Sélection de frames (diversité de poses)
    ↓
[3a] Retargeting Python pur (retarget.py) — T-pose ✓, animation ⚠ en cours
  OU
[3b] Retargeting Blender headless (blender_retarget.py) — ✓ fonctionnel
    ↓
[4] Export mesh → Blender headless (à implémenter)
    → depth map | vue anatomique | overlay Bridgman
    ↓
[5] Flux.1 Schnell ControlNet Depth → photo réaliste (à implémenter)
```

---

## État d'avancement

### Étape 1 — Retargeting BVH → Anny

| Module | Statut |
|---|---|
| `src/bvh_reader.py` — parseur BVH direct, FK manuelle | ✓ |
| `src/retarget.py` — mapping CMU→Anny, T-pose calibrée | ✓ T-pose / ⚠ animation |
| `src/validate_tpose.py` — validation overlay BVH ↔ Anny | ✓ |
| `blender/anny_rest_matrices_calibrated.json` — matrices T-pose | ✓ |
| `blender/retarget_export.py` — script Blender headless | ✓ |
| `src/blender_retarget.py` — wrapper Python → Blender → JSON → Anny | ✓ |
| `blender/anny_armature.blend` — scène Blender avec AnnySkeleton | ✓ |
| Bug GIF : mannequin statique après ~frame 120 | ⚠ à corriger |
| Sélection des 20 frames de test | ☐ |

### Étape 2 — Pipeline Blender

| Module | Statut |
|---|---|
| `src/render_blender.py` — depth map + vue anatomique + overlay | ☐ |
| `src/build_primitives.py` — primitives Bridgman depuis joints Anny | ☐ |

### Étape 3 — Génération Flux

| Module | Statut |
|---|---|
| `src/generate_flux.py` — depth map → Flux.1 Schnell | ☐ |

---

## Commandes

```bash
# Valider la T-pose (BVH frame 0 ↔ Anny calibrée)
uv run python src/validate_tpose.py
uv run python src/validate_tpose.py --bvh data/cmu/data/013/13_01.bvh

# Visualiser une frame animée (retargeting Python)
uv run python src/visualize.py --bvh data/cmu/data/005/05_01.bvh --frame 100

# GIF animé BVH vs Anny (retargeting Python)
uv run python src/animate_bvh.py --bvh data/cmu/data/005/05_01.bvh --start 0 --end 90 --step 3 --fps 12

# Générer le JSON retarget via Blender (~30s, mis en cache)
blender --background blender/anny_armature.blend \
        --python blender/retarget_export.py \
        -- data/cmu/data/005/05_01.bvh output/retarget_cache/05_01.json 0 599 1

# Visualiser une frame via Blender retarget (avec cache JSON)
uv run python src/blender_retarget.py \
    --bvh data/cmu/data/005/05_01.bvh \
    --json output/retarget_cache/05_01.json \
    --frame 100

# GIF animé via Blender retarget
uv run python src/blender_retarget.py \
    --bvh data/cmu/data/005/05_01.bvh \
    --json output/retarget_cache/05_01.json \
    --gif --gif-frames 60 --fps 12

# Explorateur Streamlit interactif
uv run streamlit run src/app.py
```

---

## Structure du projet

```
Pose_Compute_V2_Anny_x_CMU/
├── src/
│   ├── bvh_reader.py           # Parseur BVH direct (pas de bvhio)
│   ├── retarget.py             # Retargeting CMU→Anny (Python pur)
│   ├── blender_retarget.py     # Retargeting CMU→Anny (Blender headless)
│   ├── validate_tpose.py       # Validation T-pose BVH ↔ Anny
│   ├── visualize.py            # Rendu matplotlib squelette + mesh
│   ├── animate_bvh.py          # GIF animé BVH vs Anny
│   ├── app.py                  # Explorateur Streamlit 3D interactif
│   ├── explore_anny_bones.py   # Liste hiérarchie des 152 os Anny
│   └── export_anny_skeleton.py # Export squelette Anny → Blender
├── blender/
│   ├── retarget_export.py                  # Script Blender headless
│   ├── anny_armature.blend                 # Scène avec AnnySkeleton (rig MakeHuman)
│   ├── anny_rest_matrices_calibrated.json  # Matrices T-pose calibrées
│   └── anny_rest_matrices.json             # T-pose native Anny (référence)
├── data/
│   ├── cmu/     # Submodule — CMU Mocap BVH (sparse checkout)
│   └── anny/    # Submodule — Modèle Anny (UV editable)
├── output/
│   └── retarget_cache/   # JSON retarget Blender (~26 Mo/BVH, non versionné)
├── pyproject.toml
└── CLAUDE.md             # Documentation technique détaillée du pipeline
```

---

## Licence

Le code de ce projet est original. Les dépendances externes conservent leurs licences respectives :
- CMU Mocap BVH : domaine public
- Anny (NAVER Labs) : Apache 2.0
- Assets MakeHuman (inclus dans Anny) : CC0
