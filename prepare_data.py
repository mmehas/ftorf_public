import os
import zipfile
import shutil

target_dir = "data"

# Step 1: Download F-ToRF data
# Please download the `real_scenes.zip` and `synthetic_scenes.zip` files from 
# https://1drv.ms/f/c/4dd35d8ee847a247/EsiF6mb15ZlKlTZmg8N_OIcBCaQGUmWWVNOldMTaRsQXeQ?e=eIy7Rz
# to the data/ directory.

real_scenes_zip = "data/real_scenes.zip"
with zipfile.ZipFile(real_scenes_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(target_dir, "real_scenes"))
ftorf_real_scenes = ["baseball", "fan", "jacks1", "pillow", "target1"]
for scene in ftorf_real_scenes:
    with zipfile.ZipFile(os.path.join(target_dir, "real_scenes", f"{scene}.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(target_dir, "real_scenes", f"{scene}"))
    os.remove(os.path.join(target_dir, "real_scenes", f"{scene}.zip"))


synthetic_scenes_zip = "data/synthetic_scenes.zip"
with zipfile.ZipFile(synthetic_scenes_zip, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(target_dir, "synthetic_scenes"))
for folder in os.listdir(os.path.join(target_dir, "synthetic_scenes")):
    if folder.startswith("occlusion_"):
        folder_path = os.path.join(target_dir, "synthetic_scenes", folder)
        shutil.rmtree(folder_path)
        print(f"Removed folder: {folder_path}")