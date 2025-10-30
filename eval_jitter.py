import os
import numpy as np
from tqdm import tqdm


def list_npy_in_dir(dir):
    '''
    List all files in dir.

    Args:
        dir: directory to retreive file names from
    '''

    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(f"{dir}/{f}")]
    files.sort()
    return files

def measure_jitter(joint_pos, fps):
    jitter = (joint_pos[3:] - 3 * joint_pos[2:-1] + 3 * joint_pos[1:-2] - joint_pos[:-3]) * (fps ** 3) 
    jitter = np.linalg.norm(jitter, axis=2) # [297, 19]
    jitter = jitter.mean() 
    return jitter

def load_pred_position(npy_path):
    obj = np.load(npy_path, allow_pickle=True)

    # If it's a 0-d object array that wraps a dict, unwrap it
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()

    # Expect a dict with 'pred_position'
    if isinstance(obj, dict):
        if 'pred_position' not in obj:
            raise KeyError(f"'pred_position' not found in {npy_path}. Keys: {list(obj.keys())}")
        arr = obj['pred_position']
    else:
        # Fallback: file directly stored an array
        arr = obj

    arr = np.asarray(arr, dtype=np.float32)

    # Canonicalize to (T, J, 3)
    if arr.ndim == 2 and (arr.shape[1] % 3 == 0):   # (T, 3J)
        T, C = arr.shape
        J = C // 3
        arr = arr.reshape(T, J, 3)
    elif arr.ndim == 3:
        if arr.shape[-1] == 3:
            pass
        elif arr.shape[0] == 3:        # (3, T, J) -> (T, J, 3)
            arr = np.moveaxis(arr, 0, -1)
        elif arr.shape[1] == 3:        # (T, 3, J) -> (T, J, 3)
            arr = np.moveaxis(arr, 1, -1)
        else:
            raise ValueError(f"Unexpected 3D shape {arr.shape} in {npy_path}")
    else:
        raise ValueError(f"Unexpected shape {arr.shape} in {npy_path}")

    return arr  # (T, J, 3) float32

def measure_jitter_npy(dir:str, fps:int):
    print('Computing jitter metric:')
    file_list = list_npy_in_dir(dir)
    total_jitter = np.zeros([len(file_list)]) # one jitter metric for one motion data

    jitter_bar = tqdm(range(len(file_list)))
    for i in jitter_bar:
        fname = file_list[i]
        full_data_dir = f"{dir}/{fname}"

        joint_pos = load_pred_position(full_data_dir)     # ndarray (T, J, 3)
        print(joint_pos.shape)
        jitter = measure_jitter(joint_pos, fps)
        total_jitter[i] = jitter

    jitter_mean = total_jitter.mean()
    print(f"Total mean of jitter of {len(file_list)} motions: {jitter_mean}")


if __name__ == "__main__":

    fps = 30 
    data_dir = r"/data/van/Dance/Bailando_new/experiments/actor_critic/eval/pkl/ep000010"
    measure_jitter_npy(data_dir, fps)
    