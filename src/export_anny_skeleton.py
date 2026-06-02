"""
Exporte le squelette Anny (rest pose) vers :
  - output/anny_skeleton.json   : positions tête/queue + hiérarchie de chaque os
  - output/blender_add_armature.py : script Blender pour recréer l'armature

Usage :
    uv run python src/export_anny_skeleton.py

Puis dans Blender (headless) :
    blender --background --python output/blender_add_armature.py

Ou depuis le Scripting tab de Blender : ouvrir et Run Script.
"""

import json
from pathlib import Path

import numpy as np
import torch
import anny


def export_skeleton(out_dir: Path = Path("output")) -> None:
    out_dir.mkdir(exist_ok=True)

    print("Chargement modèle Anny (rest pose)...")
    model = anny.create_fullbody_model(
        rig="default-notongue",
        topology="default-notongue",
        remove_unattached_vertices=True,
    ).to(dtype=torch.float32, device="cpu")
    print(f"  {model.bone_count} os, {model.template_vertices.shape[0]} vertices")

    rest_params = {label: torch.eye(4).unsqueeze(0) for label in model.bone_labels}
    with torch.no_grad():
        out = model(pose_parameters=rest_params)

    bone_poses = out["bone_poses"].squeeze(0).cpu().numpy()  # (152, 4, 4)
    parents = [int(p) for p in model.bone_parents]
    labels = model.bone_labels

    # Head = world translation de l'os
    heads = bone_poses[:, :3, 3]  # (152, 3)

    # Construire la liste des enfants par os
    children: dict[int, list[int]] = {i: [] for i in range(len(labels))}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    # Tail = tête du premier enfant ; pour les os feuilles → axe Y local × 5 cm
    # (dans Blender les os pointent le long de leur Y local, head → tail)
    tails = []
    for i in range(len(labels)):
        if children[i]:
            tail = heads[children[i][0]].tolist()
        else:
            local_y = bone_poses[i, :3, 1]  # colonne Y = direction de l'os
            tail = (heads[i] + local_y * 0.05).tolist()
        tails.append(tail)

    bones = [
        {
            "id": i,
            "name": labels[i],
            "parent": parents[i],
            "head": heads[i].tolist(),
            "tail": tails[i],
        }
        for i in range(len(labels))
    ]

    json_path = out_dir / "anny_skeleton.json"
    with open(json_path, "w") as f:
        json.dump({"bones": bones}, f, indent=2)
    print(f"Squelette exporté : {json_path}")

    _write_blender_script(out_dir)


def _write_blender_script(out_dir: Path) -> None:
    script = '''\
"""
Script Blender : recrée l\'armature Anny depuis anny_skeleton.json.

Usage headless :
    blender --background --python output/blender_add_armature.py

Usage interactif (Scripting tab) :
    Ajuster JSON_PATH si besoin, puis Run Script.
"""
import sys
import json
from pathlib import Path

import bpy
import mathutils

JSON_PATH = Path(__file__).parent / "anny_skeleton.json"
BLEND_OUT = Path(__file__).parent / "anny_armature.blend"

with open(JSON_PATH) as f:
    bones = json.load(f)["bones"]

# Scène vide
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

arm_data = bpy.data.armatures.new("AnnySkeleton")
arm_obj  = bpy.data.objects.new("AnnySkeleton", arm_data)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
arm_obj.select_set(True)

bpy.ops.object.mode_set(mode="EDIT")
edit_bones = arm_data.edit_bones

created: dict[int, bpy.types.EditBone] = {}
for b in bones:
    eb = edit_bones.new(b["name"])
    eb.head = mathutils.Vector(b["head"])
    eb.tail = mathutils.Vector(b["tail"])
    created[b["id"]] = eb

for b in bones:
    pid = b["parent"]
    if pid >= 0:
        created[b["id"]].parent = created[pid]
        created[b["id"]].use_connect = False

bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT.resolve()))
print(f"Armature sauvegardée : {BLEND_OUT}")
'''
    script_path = out_dir / "blender_add_armature.py"
    script_path.write_text(script)
    print(f"Script Blender      : {script_path}")
    print()
    print("Étapes suivantes :")
    print("  1. uv run python src/export_anny_skeleton.py   ← génère le JSON")
    print("  2. blender --background --python output/blender_add_armature.py")
    print("     → crée output/anny_armature.blend")
    print("  3. Ouvrir anny_armature.blend dans Blender pour calibrer")


if __name__ == "__main__":
    export_skeleton()
