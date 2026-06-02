"""
Liste tous les os d'Anny et leur hiérarchie parente.
Utile pour construire le mapping CMU → Anny.

Lancer une seule fois pour comprendre la structure.
"""

import torch
import anny

model = anny.create_fullbody_model(
    rig="default-notongue",
    topology="default-notongue",
    remove_unattached_vertices=True,
).to(dtype=torch.float32, device="cpu")

print(f"Nombre d'os Anny : {model.bone_count}")
print(f"Nombre de vertices : {model.template_vertices.shape[0]}")
print()

# bone_parents[i] = index du parent de l'os i (-1 si racine)
parents = model.bone_parents
labels = model.bone_labels

print("Index | Nom de l'os              | Parent")
print("-" * 55)
for i, label in enumerate(labels):
    parent_idx = parents[i]
    parent_name = labels[parent_idx] if parent_idx >= 0 else "ROOT"
    print(f"  {i:3d} | {label:25s} | {parent_name}")
