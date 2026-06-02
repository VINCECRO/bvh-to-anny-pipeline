# GestureAI — Pipeline Dataset : CLAUDE.md

Pipeline de génération d'un dataset de poses humaines pour l'outil pédagogique GestureAI.
Ce document est la référence technique pour implémenter et faire évoluer le pipeline.

---

## Contexte du projet

GestureAI est un outil pédagogique pour apprendre à dessiner le corps humain.
Le dataset alimente le **Mode entraînement** : des poses de référence curatorées et progressives,
accompagnées d'overlays géométriques Bridgman (primitives volumiques) et de vues anatomiques.

### Deux sorties par pose

1. **Vue pédagogique** — rendu Blender (EEVEE ou Cycles) avec overlay Bridgman et vue anatomique
2. **Vue photoréaliste** — image générée par Flux.1 Schnell depuis la depth map

---

## Stack technique

| Composant | Outil | Licence | Rôle |
|---|---|---|---|
| Poses 3D | CMU Mocap (BVH) | Domaine public | Rotations articulaires |
| Modèle corporel | Anny (NAVER Labs) | Apache 2.0 | Mesh 3D pose + morphologie |
| Parsing BVH | `bvh_reader.py` (direct) | — | FK manuelle, pas de dépendance bvhio |
| Rendu 3D | Blender headless (`bpy`) | GPL | Depth map, vues anatomiques, overlay |
| Génération image | Flux.1 Schnell | Apache 2.0 | Photo réaliste |
| Runtime local | RTX 2070 8GB | — | NF4/GGUF quantization |
| Runtime prod | Replicate / fal.ai | — | ~$0.03–0.05/image |
| Mesh utils | `trimesh 4.12.2` | MIT | Export .obj/.ply |
| Deep learning | `PyTorch 2.12+cu130` | BSD | Runtime Anny |
| Visualisation | `matplotlib` + `Agg` | PSF | Rendu image sans affichage |
| Rotations | `roma 1.5.6` | MIT | Conversions quaternion/matrice |

**Gestionnaire de paquets : UV** (`uv sync` pour installer l'environnement).

### Ce qui est exclu et pourquoi

| Outil | Raison |
|---|---|
| SMPL / SMPL-X / STAR | Académique, commercial via Meshcapade (payant) |
| AMASS | Académique (Max-Planck), interdit usage commercial |
| Mixamo | Machine learning interdit (Adobe ToS) |
| Flux.1 Dev | Licence non commerciale |
| Bandai Namco mocap | CC-BY-NC-ND (non commercial) |
| Assets jeux vidéo | Copyright, extraction illégale |

---

## Sources de données

### Poses — CMU Mocap

- **Format** : BVH (Bio Vision Hierarchy) — fichier texte, hiérarchie d'os + rotations Euler par frame
- **Source BVH** : https://github.com/una-dinosauria/cmu-mocap
- **Mirror Kaggle** : https://www.kaggle.com/datasets/kmader/cmu-mocap
- **Site officiel** : http://mocap.cs.cmu.edu (format ASF/AMC natif, moins pratique)
- **Licence** : domaine public — commercial OK, pas de revente des données brutes
- **Données présentes** : `data/cmu/` — sparse checkout git, 228 fichiers BVH, sujets 005, 013, 014, 015, 060–064

Catégories prioritaires pour des poses artistiquement intéressantes :
- Sujets 05, 60+ : danse
- Sujets 13, 14, 15 : sports
- Sujets avec arts martiaux, acrobaties, mouvements expressifs

**Hiérarchie CMU (31 joints) :**
```
Hips (root, 6 DOF : XYZ + rotations)
├── LHipJoint → LeftUpLeg → LeftLeg → LeftFoot → LeftToeBase
├── RHipJoint → RightUpLeg → RightLeg → RightFoot → RightToeBase
└── LowerBack → Spine → Spine1
                         ├── Neck → Neck1 → Head
                         ├── LeftShoulder → LeftArm → LeftForeArm → LeftHand → (doigts)
                         └── RightShoulder → RightArm → RightForeArm → RightHand → (doigts)
```

**Unités BVH** : unités mocap propriétaires (≈ 5–7 mm/unité).
**Échelle** : ~0.059 m/unité (estimée hauteur BVH ↔ hauteur Anny).

**API `bvh_reader.py` — parseur direct (pas de bvhio) :**
```python
from bvh_reader import BVHFile

bvh = BVHFile("data/cmu/data/005/05_01.bvh")
bvh.n_frames                        # nb frames
bvh.joint_world_positions(frame)    # dict {name → np.array(3)}  Y-up, unités BVH
bvh.joint_world_rotations(frame)    # dict {name → np.array(3,3)} rotations monde Y-up
bvh.frame_as_parse_frame_dict(frame)# dict {name → 4×4} rotations locales + _root_position
```

**Convention BVH CMU** : `CHANNELS 3 Zrotation Yrotation Xrotation` → rotation intrinsèque ZYX.

### Morphologie — Anny

- **GitHub** : https://github.com/naver/anny
- **Paper** : https://arxiv.org/abs/2511.03589
- **Licence** : Apache 2.0 (assets MakeHuman CC0)
- **Code présent** : `data/anny/` — installé comme workspace UV editable (`uv add --editable ./data/anny`)

**Important** : ne pas utiliser la topologie `smplx` — non commerciale.
Utiliser uniquement `rig="default-notongue"`, `topology="default-notongue"`.

**API Anny :**
```python
import anny, torch

model = anny.create_fullbody_model(
    rig="default-notongue",
    topology="default-notongue",
    all_phenotypes=True,
    remove_unattached_vertices=True,
).to(dtype=torch.float32, device="cpu")

# model.bone_count    → 152
# model.bone_labels   → liste des 152 noms d'os
# model.bone_parents  → indices parents (−1 pour root)
# model.faces         → triangles du mesh
# model.template_vertices.shape → (13492, 3)
```

**Paramétrisations de pose :**
```python
# rest_relative : delta depuis la pose de repos native
output = model(pose_parameters=params, pose_parameterization="rest_relative")

# absolute : transform monde absolu par os (utilisé pour T-pose calibrée)
output = model(pose_parameters=params, pose_parameterization="absolute")

# output["vertices"]      → tensor (B, 13492, 3)
# output["bone_poses"]    → tensor (B, 152, 4, 4) — transforms monde
# output["rest_bone_poses"] → tensor (B, 152, 4, 4) — transforms repos (LOCAL, relatifs au parent)
```

**Système de coordonnées Anny** : **Z = vertical (haut)**, X = gauche/droite, Y = profondeur (-Y = devant).
Pour afficher correctement : projeter sur (X, Z), pas (X, Y).

**Paramètres phénotypiques** (valeurs 0–1) :
```python
pheno = {k: torch.tensor([0.5]) for k in model.phenotype_labels}
pheno["age"]    = torch.tensor([0.15])  # 0=bébé, 1=vieillard
pheno["gender"] = torch.tensor([0.8])   # 0=masculin, 1=féminin
pheno["weight"] = torch.tensor([0.3])   # 0=mince, 1=obèse
pheno["muscle"] = torch.tensor([0.9])   # 0=flasque, 1=très musclé
```

**Hiérarchie des 152 os Anny (topologie `default-notongue`) :**
```
root
├── pelvis.L → upperleg01.L → upperleg02.L → lowerleg01.L → lowerleg02.L → foot.L → (orteils)
├── pelvis.R → upperleg01.R → upperleg02.R → lowerleg01.R → lowerleg02.R → foot.R → (orteils)
└── spine05 → spine04 → spine03 → spine02 → spine01
                                    ├── breast.L / breast.R
                                    ├── clavicle.L → shoulder01.L → upperarm01.L → upperarm02.L
                                    │               → lowerarm01.L → lowerarm02.L → wrist.L → (doigts)
                                    ├── clavicle.R → (symétrique)
                                    └── neck01 → neck02 → neck03 → head → (os faciaux)
```

---

## Conversion de coordonnées BVH → Anny

**Matrice de conversion `_C = Rx(+90°)` :**
```python
_C = np.array([[1.,  0.,  0.],
               [0.,  0., -1.],
               [0.,  1.,  0.]], dtype=np.float32)
```

- BVH X → Anny X (inchangé)
- BVH Y (haut) → Anny Z (haut) ✓
- BVH Z (profondeur) → Anny -Y (devant)

**Attention** : utiliser `Rx(+90°)` et non `Rx(-90°)`. `Rx(-90°)` inverse l'axe vertical (squelette à l'envers).
Vérification : `_C @ [0, 1, 0]` doit donner `[0, 0, 1]` (up → up).

**Pour les rotations** : `R_anny = _C @ R_bvh @ _C.T` (changement de base).

---

## Calibration T-pose Blender

La T-pose native d'Anny (bras légèrement en avant) ne correspond pas exactement
à la T-pose BVH CMU (bras dans le plan frontal exact). La calibration corrige cet écart.

### Procédure dans Blender

1. Importer l'armature Anny et un fichier BVH CMU dans `blender/Anny_mesh_BVH_rig.blend`
2. Utiliser le plugin **retarget_bvh** pour aligner la T-pose Anny sur la T-pose BVH
3. En **Pose Mode**, sélectionner tous les os et appliquer : `Pose → Apply → Apply Pose as Rest Pose`
4. Exporter les matrices de pose avec ce script :

```python
import bpy
import json

arm_obj = bpy.data.objects["AnnySkeleton"]
corrections = {}

for pbone in arm_obj.pose.bones:
    # pbone.matrix = matrice 4×4 de l'os en Pose Mode (espace armature)
    # Contient la rotation calibrée + position tête de l'os
    M = pbone.matrix
    corrections[pbone.name] = [list(row) for row in M]

with open("/tmp/anny_rest_matrices_calibrated.json", "w") as f:
    json.dump(corrections, f, indent=2)

print(f"{len(corrections)} os exportés")
```

> **Important** : utiliser `pbone.matrix` (pose bone en Pose Mode) et **non** `bone.matrix_local`
> (rest matrix qui ne change pas avec retarget_bvh). Copier le JSON dans `blender/anny_rest_matrices_calibrated.json`.

### Résultat

`blender/anny_rest_matrices_calibrated.json` : 150 os, matrices 4×4 monde (Z-up, mètres).
- Translations = positions tête des os en T-pose calibrée
- Rotations = orientations des os alignées sur la convention CMU BVH

Utilisé par `tpose_params()` avec `pose_parameterization="absolute"`.

---

## Retargeting CMU → Anny

Deux approches ont été explorées. L'approche Python pure est plus propre et plus légère,
mais n'est pas encore validée. L'approche Blender fonctionne mais est plus lourde.

---

### Approche A — Python pur (préférée, à valider) — `src/retarget.py`

**Avantages** : pas de dépendance Blender au runtime, calcul direct en Python/numpy, plus intégrable.
**Statut** : ⚠ T-pose validée, animation avec poses incohérentes non résolues.

```python
from retarget import load_anny_model, load_blender_matrices, tpose_params, retarget_frame, generate_mesh

model        = load_anny_model()
blender_mats = load_blender_matrices()  # anny_rest_matrices_calibrated.json

# T-pose exacte — fonctionne
params = tpose_params(model, blender_mats)
mesh, out = generate_mesh(model, params, pose_parameterization="absolute")

# Frame animée — formule implémentée mais poses encore incohérentes
params = retarget_frame(bvh, frame_idx, model, blender_mats)
mesh, out = generate_mesh(model, params, pose_parameterization="absolute")
```

**Formule actuelle (implémentée, résultat incorrect)** — pour chaque os `b`, traitement top-down :

```
R_local_tpose[b] = R_cal[parent]^T @ R_cal[b]   # rotation locale T-pose
R_world_anim[b]  = R_parent_anim @ R_local_tpose @ _C @ R_local_bvh[j] @ _C^T
pos_anim[b]      = pos_parent_anim + R_parent_anim @ (R_cal[parent]^T @ (pos_cal[b] - pos_cal[parent]))
```

Résultat passé en `pose_parameterization="absolute"`.

**Points bloquants identifiés** :
- `rest_bone_poses` d'Anny contient des transforms **monde** (pas locaux malgré la doc)
- `absolute` mode dans Anny : `bone_transforms = bone_poses @ rest_poses_inv` → les translations doivent être en mètres, cohérentes avec `rest_bone_poses`
- La formule FK ci-dessus donne des poses incohérentes — probablement un problème de convention d'axe ou d'ordre de composition non résolu
- Piste principale : vérifier l'alignement entre la convention d'axe local des os Blender (Y = head→tail) et l'espace dans lequel `R_local_bvh` est défini

---

### Approche B — Blender headless (fonctionnelle, plus lourde) — `src/blender_retarget.py`

**Avantages** : utilise le plugin `retarget_bvh` (MakeHuman config) qui gère correctement la T-pose et les conventions.
**Inconvénients** : dépendance Blender au runtime (~30s/BVH), JSON de 26 Mo par fichier complet.
**Statut** : ✓ Animation fonctionnelle, bug restant sur la durée du GIF (cf. Questions ouvertes).

**Architecture** :
```
BVH → [Blender headless + retarget_bvh] → JSON {frame: {bone: matrice 4×4}} → Anny absolute mode
```

**Script Blender** (`blender/retarget_export.py`, tourne dans Blender via `--python`) :
1. Charge `blender/anny_armature.blend` (AnnySkeleton, rig MakeHuman, 150 os)
2. Supprime `breast.L` / `breast.R` (perturbent l'auto-détection retarget_bvh)
3. Active le mode silencieux retarget_bvh (évite les opérateurs UI en headless)
4. Appelle `mcp.load_and_retarget` — auto-détecte rig MakeHuman sur AnnySkeleton
5. Par frame : `scene.frame_set(frame)` + `view_layer.update()` puis `arm_obj.matrix_world @ pbone.matrix`

**Points techniques critiques** :
- Lire `arm_obj.matrix_world @ pbone.matrix` (espace monde), **pas** `pbone.matrix` seul (espace armature)
- Lire directement depuis `arm_obj.pose.bones` après `view_layer.update()`, **pas** via `evaluated_get(depsgraph)` (snapshot de la frame précédente)
- Ne **pas** re-register `retarget_bvh` depuis le dossier projet (conflit avec l'addon Blender installé dans `~/.config/blender/4.0/scripts/addons/retarget_bvh-master/`)
- Le joint Hips BVH a des canaux XYZ position en unités mocap (~30 unités) — **ce script ne copie pas la translation** car `retarget_bvh` gère déjà ça correctement

**Wrapper Python** (`src/blender_retarget.py`) :
```python
from blender_retarget import run_blender_retarget, load_retarget_json, retarget_frame_blender

# Génère le JSON (appelle Blender, ~30s)
json_path = run_blender_retarget("data/cmu/data/005/05_01.bvh")

# Charge + utilise (rapide, pas de Blender)
data = load_retarget_json(json_path)
params = retarget_frame_blender(data, frame_idx=100, model=model)
mesh, out = generate_mesh(model, params, pose_parameterization="absolute")
```

**Format JSON** : `{"frame": {"bone_name": [[4×4 matrix]], ...}, ...}`
Même espace que `anny_rest_matrices_calibrated.json` → compatible `absolute` mode direct.

**Stockage** : ~26 Mo par BVH complet (600 frames, step=1). À 228 BVH : ~6 Go.
Options : step=3 → ~9 Mo/fichier, ou génération à la volée sans cache.

**Fichier Blender requis** : `blender/anny_armature.blend`
- Contient `AnnySkeleton` (150 os, `breast.L/R` présents mais supprimés à l'exécution)
- Rig détecté comme MakeHuman par retarget_bvh (fingerprint `risorius03.R`)
- Addon `retarget_bvh` installé dans Blender 4.0 (`~/.config/blender/4.0/scripts/addons/retarget_bvh-master/`)

---

### Mapping CMU (31 joints) → Anny (152 os)

Seul le **premier os** (proximal) de chaque segment reçoit la rotation.

| Joint CMU | Os Anny ciblés |
|---|---|
| Hips | root |
| LHipJoint / RHipJoint | pelvis.L / pelvis.R |
| LeftUpLeg / RightUpLeg | upperleg01.L+02.L / .R |
| LeftLeg / RightLeg | lowerleg01.L+02.L / .R |
| LeftFoot / RightFoot | foot.L / foot.R |
| LeftToeBase / RightToeBase | toe1-1.L / toe1-1.R |
| LowerBack | spine05 |
| Spine | spine04, spine03 |
| Spine1 | spine02, spine01 |
| Neck / Neck1 | neck01+02 / neck03 |
| Head | head |
| LeftShoulder / RightShoulder | clavicle.L+shoulder01.L / .R |
| LeftArm / RightArm | upperarm01.L+02.L / .R |
| LeftForeArm / RightForeArm | lowerarm01.L+02.L / .R |
| LeftHand / RightHand | wrist.L / wrist.R |
| LThumb / RThumb | finger1-1.L / finger1-1.R |
| LeftHandIndex1 / RightHandIndex1 | finger2-1.L / finger2-1.R |

---

## Architecture du pipeline

```
CMU Mocap BVH
    ↓
[1] Parsing BVH → rotations articulaires 3D (numpy)        [bvh_reader.py ✓]
    ↓
[2] Sélection de frames (diversité de poses)
    ↓
[3a] Anny : mesh 3D — APPROCHE A (Python, à valider)       [retarget.py — T-pose ✓, animation ⚠]
   OU
[3b] Anny : mesh 3D — APPROCHE B (Blender, fonctionnelle)  [blender_retarget.py ✓]
    pose = matrices os depuis retarget_bvh Blender
    morphologie = échantillonnage aléatoire phénotypes
    → vertices (N, 3) + faces (M, 3)
    ↓
[4] Export mesh → Blender headless (bpy)                   [à implémenter]
    ↓
┌─────────────────────────────────────────────────────┐
│                    Blender                          │
│  depth_map │ vue anatomique │ overlay Bridgman      │
└─────────────────────────────────────────────────────┘
          ↓
[5] Flux.1 Schnell (ControlNet Depth)                      [à implémenter]
    → photo_realiste.png
```

---

## Checklist d'implémentation

### Étape 1 : Retargeting BVH → Anny
- [x] `src/bvh_reader.py` : parseur BVH direct, FK manuelle
- [x] `src/retarget.py` : mapping CMU→Anny, T-pose calibrée
- [x] `src/validate_tpose.py` : validation overlay BVH ↔ Anny (T-pose ✓)
- [x] `blender/anny_rest_matrices_calibrated.json` : matrices T-pose calibrées
- [x] `blender/retarget_export.py` : script Blender headless, retarget via retarget_bvh
- [x] `src/blender_retarget.py` : wrapper Python → Blender → JSON → Anny absolute
- [x] `blender/anny_armature.blend` : scène Blender avec AnnySkeleton (rig MakeHuman)
- [ ] **Bug GIF durée** : le mannequin ne bouge que jusqu'à ~frame 120 sur 600 — à investiguer (cf. Questions ouvertes)
- [ ] **Approche A (Python)** : résoudre les poses incohérentes de `retarget_frame()` — priorité dès que possible car plus propre que Blender
- [ ] Sélection des 20 frames de test (diversité de poses)

### Étape 2 : Pipeline Blender
- [ ] `src/render_blender.py` : script bpy headless → depth map + vue anatomique + overlay
- [ ] `src/build_primitives.py` : joints Anny → primitives Bridgman (analytique)

### Étape 3 : Génération Flux
- [ ] `src/generate_flux.py` : depth map → Flux.1 Schnell → photo réaliste

### Critères de validation
- [ ] Animation Anny cohérente avec BVH (pas de croisement de membres, orientations correctes)
- [ ] Overlay Bridgman visuellement cohérent avec le mesh
- [ ] Depth map lisible et non ambiguë
- [ ] Photo Flux fidèle à la pose sur les poses sans occlusion majeure

---

## Questions ouvertes

**⚠ BUG PRIORITAIRE — GIF : mannequin statique après ~frame 120**
Le JSON exporté par Blender contient bien 600 frames avec des valeurs différentes par frame.
Pourtant dans le GIF (`blender_retarget.py --gif`), le mannequin ne bouge que jusqu'à ~frame 120
puis reste statique. Le problème est probablement dans la sélection de frames du GIF
(les 60 frames échantillonnées depuis 600 concentrent les premières) ou dans le rendu Anny
sur les frames tardives. À investiguer : comparer les matrices JSON frames 120 vs 300 vs 500,
vérifier si Anny produit des résultats identiques pour ces frames dans `generate_mesh`.

**⚠ Approche A (Python pur) — retargeting à corriger**
La T-pose fonctionne. L'animation produit des poses incohérentes malgré plusieurs tentatives.
La formule FK top-down dans `retarget_frame()` (utilisant `_C @ R_local_bvh @ _C^T` et les matrices
calibrées Blender comme T-pose de référence) ne donne pas de résultat correct.
Piste principale : le problème vient peut-être de la convention d'axe local des os Blender
(Y-axis = head→tail) qui ne correspond pas à la convention BVH des rotations locales.
Cette approche reste préférable à l'approche Blender pour la production (légère, sans dépendance).

**Qualité du mapping CMU → Anny**
Le mapping propage la même rotation sur les deux os d'un segment double (ex: upperleg01+02).
Une amélioration : interpoler ou répartir la rotation. À valider sur poses extrêmes (splits, acrobaties).

**Gestion des occlusions pour Flux**
Sur les poses avec membres croisés, Flux peut halluciner.
Stratégie : masques Blender pour détecter les occlusions + scoring de la génération.

**Cohérence prompt Flux / phénotypes Anny**
Si Anny génère un mesh corpulent, le prompt Flux doit le refléter.
Prévoir un mapping phénotypes → tokens textuels.

---

## Structure du projet

```
Pose_Compute_V2_Anny_x_CMU/
├── src/
│   ├── bvh_reader.py           # Parseur BVH direct (pas de bvhio)
│   ├── retarget.py             # Retargeting CMU→Anny approche A (Python pur, T-pose ✓, animation ⚠)
│   ├── blender_retarget.py     # Retargeting CMU→Anny approche B (Blender headless, ✓)
│   ├── validate_tpose.py       # Validation T-pose BVH ↔ Anny
│   ├── visualize.py            # Rendu matplotlib (squelette + mesh + morphos)
│   ├── animate_bvh.py          # GIF animé comparaison BVH / Anny (approche A)
│   ├── app.py                  # Explorateur Streamlit 3D interactif
│   ├── explore_anny_bones.py   # Liste hiérarchie des 152 os Anny
│   └── export_anny_skeleton.py # Export squelette Anny → Blender
├── blender/
│   ├── retarget_export.py               # Script Blender headless (tourne via --python)
│   ├── anny_armature.blend              # Scène Blender avec AnnySkeleton (rig MakeHuman)
│   ├── Anny_mesh_BVH_rig.blend          # Ancienne scène Blender calibration T-pose
│   ├── anny_rest_matrices_calibrated.json  # T-pose calibrée (utilisée par approche A et B)
│   ├── anny_rest_matrices.json          # T-pose native Anny (référence)
│   └── anny_skeleton.json               # Hiérarchie + positions os Anny
├── output/
│   └── retarget_cache/                  # JSON retarget Blender (~26 Mo/BVH)
│       ├── 05_01.json
│       └── ...
├── data/
│   ├── cmu/data/              # BVH CMU Mocap (sparse checkout git)
│   │   ├── 005/               # Danse (20 fichiers)
│   │   ├── 013/, 014/, 015/   # Sports
│   │   └── 060/–064/          # Danse latine
│   └── anny/                  # Modèle Anny (workspace UV editable)
├── pyproject.toml             # UV project (Python 3.10)
└── CLAUDE.md
```

## Commandes utiles

```bash
# Installer l'environnement
uv sync

# Valider la T-pose (BVH frame 0 ↔ Anny calibrée) — Approche A
uv run python src/validate_tpose.py
uv run python src/validate_tpose.py --bvh data/cmu/data/013/13_01.bvh

# Visualisation frame animée — Approche A (animation incohérente)
uv run python src/visualize.py --bvh data/cmu/data/005/05_01.bvh --frame 100

# GIF animé — Approche A (BVH vs Anny)
uv run python src/animate_bvh.py --bvh data/cmu/data/005/05_01.bvh --start 0 --end 90 --step 3 --fps 12

# ─── Approche B (Blender headless) ────────────────────────────────────────────

# Générer le JSON retarget pour un BVH (~30s, cache dans output/retarget_cache/)
blender --background blender/anny_armature.blend \
        --python blender/retarget_export.py \
        -- data/cmu/data/005/05_01.bvh output/retarget_cache/05_01.json 0 599 1

# Ou via le wrapper Python (appelle Blender automatiquement si pas de cache)
uv run python src/blender_retarget.py --bvh data/cmu/data/005/05_01.bvh

# Visualiser une frame (avec cache JSON existant)
uv run python src/blender_retarget.py \
    --bvh data/cmu/data/005/05_01.bvh \
    --json output/retarget_cache/05_01.json \
    --frame 100

# Générer un GIF animé (60 frames, bounds automatiques)
uv run python src/blender_retarget.py \
    --bvh data/cmu/data/005/05_01.bvh \
    --json output/retarget_cache/05_01.json \
    --gif --gif-frames 60 --fps 12

# ─── Autres outils ──────────────────────────────────────────────────────────

# Explorateur Streamlit interactif
uv run streamlit run src/app.py

# Lister les os Anny
uv run python src/explore_anny_bones.py
```
