import os
import shutil

# paths
train_txt = r"/data/van/Dance/Bailando_new/data/motorica/train_files.txt"
test_txt = r"/data/van/Dance/Bailando_new/data/motorica/test_files.txt"
json_dir = r"/data/van/Dance/Bailando_new/data/motorica/features"   # folder where all json files are stored
train_out = r"/data/van/Dance/Bailando_new/data/motorica/data_split/train"
test_out = r"/data/van/Dance/Bailando_new/data/motorica/data_split/test"

# make sure output dirs exist
os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

def copy_files(list_file, out_dir):
    with open(list_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    
    for name in names:
        json_file = os.path.join(json_dir, f"{name}.json")
        if os.path.exists(json_file):
            shutil.copy(json_file, out_dir)
            print(f"Copied {json_file} -> {out_dir}")
        else:
            print(f"Warning: {json_file} not found")

# copy train and test
copy_files(train_txt, train_out)
copy_files(test_txt, test_out)

print("✅ Done.")
