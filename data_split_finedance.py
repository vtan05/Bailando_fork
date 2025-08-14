import os
import shutil

def get_finedance_split():
    all_list = [str(i).zfill(3) for i in range(1, 212)]

    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193",
                 "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]

    train_list = [x for x in all_list if x not in test_list]

    return ignor_list, train_list, test_list

def copy_files(source_dir, output_dir):
    ignor_list, train_list, test_list = get_finedance_split()

    # Create output directories if they don't exist
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for sample_id in train_list:
        if sample_id in ignor_list:
            continue
        src = os.path.join(source_dir, f"{sample_id}.json")
        dst = os.path.join(train_dir, f"{sample_id}.json")
        if os.path.exists(src):
            shutil.copy(src, dst)

    for sample_id in test_list:
        if sample_id in ignor_list:
            continue
        src = os.path.join(source_dir, f"{sample_id}.json")
        dst = os.path.join(test_dir, f"{sample_id}.json")
        if os.path.exists(src):
            shutil.copy(src, dst)

if __name__ == "__main__":
    source_directory = "/data/van/Dance/Bailando_new/data/finedance/features_22jointsv2"          # Replace with your source data directory
    output_directory = "/data/van/Dance/Bailando_new/data/finedance/data_split"  # Replace with your target directory
    copy_files(source_directory, output_directory)
