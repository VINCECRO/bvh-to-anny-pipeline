"""
blender/retarget_export.py
Script Blender headless : retarget BVH → matrices os Anny (JSON par frame).

Principe :
  1. Charge anny_armature.blend (AnnySkeleton déjà configuré rig MakeHuman).
  2. Supprime breast.L / breast.R.
  3. Active AnnySkeleton + appelle mcp.load_and_retarget (retarget_bvh).
  4. Par frame : scene.frame_set + view_layer.update()
     puis lit arm_obj.matrix_world @ pbone.matrix
     (même formule que pour anny_rest_matrices_calibrated.json).

Usage :
    blender --background blender/anny_armature.blend \
            --python blender/retarget_export.py \
            -- <bvh_path> <output_json> [start] [end] [step]
"""

import sys
import json
import bpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ── Arguments ─────────────────────────────────────────────────────────────────

def parse_args():
    if "--" not in sys.argv:
        print("Usage: blender --background anny_armature.blend --python retarget_export.py -- bvh output [start] [end] [step]")
        sys.exit(1)
    args = sys.argv[sys.argv.index("--") + 1:]
    bvh_path = Path(args[0]).resolve()
    out_path  = Path(args[1])
    start     = int(args[2]) if len(args) > 2 else None
    end       = int(args[3]) if len(args) > 3 else None
    step      = int(args[4]) if len(args) > 4 else 1
    return bvh_path, out_path, start, end, step


# ── Addon retarget_bvh ────────────────────────────────────────────────────────

def enable_retarget_bvh():
    """
    L'addon retarget_bvh est déjà installé dans Blender (retarget_bvh-master).
    On active seulement le mode silencieux pour éviter les opérateurs UI en headless.
    NE PAS re-register — ça crée un conflit avec la version déjà chargée.
    """
    for mod_name in ("retarget_bvh", "retarget_bvh_master"):
        try:
            mod = __import__(mod_name + ".utils", fromlist=["setSilentMode"])
            mod.setSilentMode(True)
            print(f"[addon] {mod_name} mode silencieux activé")
            return
        except Exception:
            pass
    print("[addon] setSilentMode non trouvé, opérateur mcp toujours disponible")


# ── Armature ──────────────────────────────────────────────────────────────────

def find_anny_armature():
    """Retourne l'armature avec le plus d'os (= AnnySkeleton)."""
    arms = [(len(o.data.bones), o) for o in bpy.data.objects if o.type == 'ARMATURE']
    if not arms:
        return None
    arm = max(arms, key=lambda x: x[0])[1]
    print(f"[anny] {arm.name}  ({len(arm.data.bones)} os)")
    return arm


def remove_breast_bones(arm_obj):
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_obj.data.edit_bones
    removed = [n for n in ('breast.L', 'breast.R') if n in eb]
    for n in removed:
        eb.remove(eb[n])
    bpy.ops.object.mode_set(mode='OBJECT')
    if removed:
        print(f"[anny] os supprimés : {removed}")


# ── Retarget ──────────────────────────────────────────────────────────────────

def retarget(arm_obj, bvh_path):
    """Active AnnySkeleton et lance mcp.load_and_retarget."""
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    print(f"[retarget] {bvh_path.name} → {arm_obj.name}")
    bpy.ops.mcp.load_and_retarget(
        'EXEC_DEFAULT',
        filepath=str(bvh_path),
        useAutoTarget=True,
        useBendPositive=False,
        useSimplify=False,
        useTimeScale=False,
        useDefaultSS=False,  # désactive le sous-échantillonnage fps (évite de perdre 4/5 des frames sur BVH 120fps)
        ssFactor=1,
    )
    print("[retarget] terminé")


# ── Export ────────────────────────────────────────────────────────────────────

def export_frames(arm_obj, out_path, start, end, step):
    """
    Par frame : scene.frame_set + view_layer.update()
    puis lecture directe arm_obj.matrix_world @ pbone.matrix.
    Même espace que anny_rest_matrices_calibrated.json → absolute mode ✓.
    NOTE : on lit pbone.matrix DIRECTEMENT (pas evaluated_get) après update —
    evaluated_get() peut retourner un snapshot de la frame précédente.
    """
    scene = bpy.context.scene
    # Utilise la plage réelle de l'action pour éviter d'exporter des frames
    # au-delà des keyframes (ce qui produirait une pose gelée sur la dernière frame).
    act = arm_obj.animation_data.action if arm_obj.animation_data else None
    if start is None:
        start = int(act.frame_range[0]) if act else scene.frame_start
    if end is None:
        end = int(act.frame_range[1]) if act else scene.frame_end
    if act:
        print(f"[export] plage action détectée : frames {int(act.frame_range[0])}–{int(act.frame_range[1])}")

    frame_range = range(start, end + 1, step)
    print(f"[export] {len(frame_range)} frames [{start}:{end}:{step}]")

    world_mat = arm_obj.matrix_world
    results   = {}

    for i, frame in enumerate(frame_range):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        frame_data = {}
        for pbone in arm_obj.pose.bones:
            M = world_mat @ pbone.matrix
            frame_data[pbone.name] = [list(row) for row in M]

        results[str(frame)] = frame_data

        if i % 50 == 0 or i == len(frame_range) - 1:
            print(f"  frame {frame} ({i+1}/{len(frame_range)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f)
    size_kb = out_path.stat().st_size // 1024
    print(f"[export] {len(results)} frames → {out_path}  ({size_kb} Ko)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    bvh_path, out_path, start, end, step = parse_args()
    print(f"=== retarget_export : {bvh_path.name} ===  Blender {bpy.app.version_string}")

    enable_retarget_bvh()

    arm_obj = find_anny_armature()
    if arm_obj is None:
        print("ERREUR : aucune armature dans la scène")
        sys.exit(1)

    remove_breast_bones(arm_obj)
    retarget(arm_obj, bvh_path)
    export_frames(arm_obj, out_path, start, end, step)
    print("=== terminé ===")


main()
