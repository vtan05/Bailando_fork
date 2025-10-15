import os

def rename_npy_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".npy") and "-retargeted" in filename:
            old_path = os.path.join(folder_path, filename)
            new_filename = filename.replace("-retargeted", "")
            new_path = os.path.join(folder_path, new_filename)
            
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    folder = "/data/van/Dance/Bailando_new/data/motorica_npy/motion"  # change this to your folder path
    rename_npy_files(folder)
