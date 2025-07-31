import os
import sys
import json
import shutil
import pickle as pkl
import numpy as np
import pandas as pd
import joblib as jl
import torch
import argparse
import essentia
import librosa
from essentia.standard import *
from sklearn.pipeline import Pipeline
from extractor import FeatureExtractor
from scipy.signal import resample
from smplx_fk import SMPLX_Skeleton
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)


# -------------------------
# ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--motion_dir', '-orig', default=r"/data/van/Dance/Bailando_new/data/finedance/motion/",
                    help="Path where original motion files (in npy format) are stored")
parser.add_argument('--dest_dir', '-dest', default=r"/data/van/Dance/Bailando_new/data/finedance/features",
                    help="Path where extracted motion features will be stored")
parser.add_argument('--music_dir', type=str, default=r"/data/van/Dance/Bailando_new/data/finedance/music_wav",
                    help="Path to music .wav files")
parser.add_argument('--fps', type=int, default=30)
args = parser.parse_args()

os.makedirs(args.dest_dir, exist_ok=True)

extractor = FeatureExtractor()

def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax


def set_on_ground(root_pos, local_q_156, smplx_model):
    # root_pos = root_pos[:, :] - root_pos[:1, :]
    floor_height = 0
    length = root_pos.shape[0]
    # model_q = model_q.view(b*s, -1)
    # model_x = model_x.view(-1, 3)
    positions = smplx_model.forward(local_q_156, root_pos)
    positions = positions.view(length, -1, 3)   # bxt, j, 3
    
    l_toe_h = positions[0, 10, 1] - floor_height
    r_toe_h = positions[0, 11, 1] - floor_height
    if abs(l_toe_h - r_toe_h) < 0.02:
        height = (l_toe_h + r_toe_h)/2
    else:
        height = min(l_toe_h, r_toe_h)
    root_pos[:, 1] = root_pos[:, 1] - height

    return root_pos, local_q_156


def process_motion(motion_path):
    import numpy as np
    import torch

    data = np.load(motion_path)

    # Step 1: Parse root position and 6D local rotation
    if data.shape[1] == 315:
        root_pos = data[:, :3]
        local_q = data[:, 3:]
    elif data.shape[1] == 319:
        root_pos = data[:, 4:7]
        local_q = data[:, 7:]
    else:
        raise ValueError(f"Unexpected input shape: {data.shape}")

    # Step 2: Initialize FK skeleton model
    smplx_model = SMPLX_Skeleton()

    # Step 3: Convert to tensor and reshape
    root_pos = torch.Tensor(root_pos)                          # [T, 3]
    local_q = torch.Tensor(local_q).view(-1, 52, 6)            # [T, 52, 6]
    local_q = ax_from_6v(local_q)                              # [T, 52, 3]
    length = root_pos.shape[0]

    # Step 4: Set root on ground (optional normalization)
    local_q_156 = local_q.view(length, -1)                     # [T, 156]
    root_pos, local_q_156 = set_on_ground(root_pos, local_q_156, smplx_model)

    # Step 5: Forward kinematics (full 52-joint output)
    positions = smplx_model.forward(local_q_156, root_pos)     # [T, 52, 3]

    # Step 6: Extract only 21 FineDance joints
    smplx_to_21 = [
        0,             # pelvis
        1, 4, 7, 10,   # left hip, knee, ankle, foot
        2, 5, 8, 11,   # right hip, knee, ankle, foot
        3, 6,          # spine1, spine2
        12, 15,        # neck, head
        13, 14,        # left_collar, right_collar
        16, 18, 20,    # left shoulder, elbow, wrist
        17, 19, 21     # right shoulder, elbow, wrist
    ]
    positions = positions[:, smplx_to_21, :]         # [T, 21, 3]
    positions = positions.reshape(positions.shape[0], -1)  # [T, 63]
    return positions.numpy()


# -------------------------
# AUDIO FEATURE EXTRACTION
# -------------------------
def extract_acoustic_feature(audio, sr):
    melspe_db = extractor.get_melspectrogram(audio, sr)
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr, octave=7 if sr==sr else 5)
    onset_env = extractor.get_onset_strength(audio_percussive, sr)
    tempogram = extractor.get_tempogram(onset_env, sr)
    onset_beat = extractor.get_onset_beat(onset_env, sr)[0]
    onset_env = onset_env.reshape(1, -1)
    feature = np.concatenate([mfcc, mfcc_delta, chroma_cqt, onset_env, onset_beat, tempogram], axis=0)
    return feature.transpose(1, 0)

def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    loader = essentia.standard.MonoLoader(filename=file_path, sampleRate=sr)
    audio = np.array(loader()).T
    return extract_acoustic_feature(audio, sr)

# -------------------------
# ALIGNMENT
# -------------------------
def align_frames(music_feat, motion_feat):
    # Align music to motion by resampling
    motion_len = len(motion_feat)
    music_feat_resampled = resample(music_feat, motion_len, axis=0)
    print(f"[INFO] Resampled music from {len(music_feat)} to {motion_len}")
    return music_feat_resampled, motion_feat

# -------------------------
# SAVE JSON
# -------------------------
def save_json(file_id, music_feat, motion_feat):
    sample_dict = {
        'id': file_id,
        'music_array': music_feat.tolist(),
        'motion_array': motion_feat.tolist()
    }
    with open(os.path.join(args.dest_dir, f"{file_id}.json"), 'w') as f:
        json.dump(sample_dict, f)

# -------------------------
# MAIN
# -------------------------
if __name__ == '__main__':
    motion_files = sorted([f for f in os.listdir(args.motion_dir) if f.endswith('.npy')])
    music_files = sorted([f for f in os.listdir(args.music_dir) if f.endswith('.wav')])
    file_ids = sorted(set(os.path.splitext(f)[0] for f in motion_files) & set(os.path.splitext(f)[0] for f in music_files))

    print(f"[INFO] Found {len(file_ids)} matching Motion + WAV files.")

    max_motion_frames = 0
    max_music_frames = 0

    for file_id in file_ids:
        print(f"\n>>> Processing {file_id}")
        motion_path = os.path.join(args.motion_dir, f"{file_id}.npy")
        music_path = os.path.join(args.music_dir, f"{file_id}.wav")

        motion_feat = process_motion(motion_path)
        print(f"[INFO] Motion features: {motion_feat.shape}")
        if motion_feat is None:
            continue

        music_feat = process_audio(music_path)
        print(f"[INFO] Music features: {music_feat.shape}")

        music_aligned, motion_aligned = align_frames(music_feat, motion_feat)
        save_json(file_id, music_aligned, motion_aligned)

        # Track max lengths
        max_motion_frames = max(max_motion_frames, motion_aligned.shape[0])
        max_music_frames = max(max_music_frames, music_aligned.shape[0])
        print("\n" + "="*50)
        print(f"[SUMMARY] Max motion frames: {max_motion_frames}")
        print(f"[SUMMARY] Max music frames: {max_music_frames}")
        print("="*50)