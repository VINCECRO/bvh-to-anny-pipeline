"""
Script Blender : recrée l'armature Anny depuis anny_skeleton.json.

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
