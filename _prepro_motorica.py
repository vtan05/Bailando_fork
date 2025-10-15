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
import numpy as np
from essentia.standard import *
from scipy.signal import resample
from extractor import FeatureExtractor
from smplx_fk import SMPLX_Skeleton
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)

# -------------------------
# ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--npz_dir', '-orig', default=r"/data/van/Dance/Bailando_new/data/motorica/smpl",
                    help="Path where original motion files (in npz format) are stored")
parser.add_argument('--dest_dir', '-dest', default=r"/data/van/Dance/Bailando_new/data/motorica/features",
                    help="Path where extracted motion features will be stored")
parser.add_argument('--music_dir', type=str, default=r"/data/van/Dance/Bailando_new/data/motorica/wav",
                    help="Path to music .wav files")
parser.add_argument('--fps', type=int, default=30)
args = parser.parse_args()

os.makedirs(args.dest_dir, exist_ok=True)

extractor = FeatureExtractor()
# -------------------------
# CONSTANTS
# -------------------------
smplx_to_22 = [
        0,   # pelvis
        1,   # left_hip
        2,   # right_hip
        3,   # spine1
        4,   # left_knee
        5,   # right_knee
        6,   # spine2
        7,   # left_ankle
        8,   # right_ankle
        9,   # spine3
        10,  # left_foot
        11,  # right_foot
        12,  # neck
        13,  # left_collar
        14,  # right_collar
        15,  # head
        16,  # left_shoulder
        17,  # right_shoulder
        18,  # left_elbow
        19,  # right_elbow
        20,  # left_wrist
        21   # right_wrist
    ]


def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat


def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat

# -------------------------
# MOTION PROCESSING
# -------------------------
def process_npz(file_path, file_id):
    data = np.load(file_path)
    rots = np.squeeze([data["poses"]]) # axis-angle format
    rots = torch.Tensor(rots)
    trans = np.squeeze(data["trans"]) 
    trans = torch.Tensor(trans)
   
    # rots = ax_to_6v(rots)
    # rots_quats = quat_from_6v(rots)
    # positions = offsets[None].repeat(len(rots_quats), axis=0)

    smplx_model = SMPLX_Skeleton()
    length = trans.shape[0]
    local_q = rots.view(length, -1)         
    positions = smplx_model.forward(local_q, trans) 

    positions = positions[:, smplx_to_22, :]         # [T, 22, 3]
    positions = positions.reshape(positions.shape[0], -1)  # [T, 66]
    print(f"[INFO] Reduced to 22 joints, shape {positions.shape}")
    return positions

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
    npz_files = sorted([f for f in os.listdir(args.npz_dir) if f.endswith('.npz')])
    music_files = sorted([f for f in os.listdir(args.music_dir) if f.endswith('.wav')])
    file_ids = sorted(set(os.path.splitext(f)[0] for f in npz_files) & set(os.path.splitext(f)[0] for f in music_files))

    print(f"[INFO] Found {len(file_ids)} matching npz + WAV files.")

    max_motion_frames = 0
    max_music_frames = 0

    for file_id in file_ids:
        print(f"\n>>> Processing {file_id}")
        npz_path = os.path.join(args.npz_dir, f"{file_id}.npz")
        music_path = os.path.join(args.music_dir, f"{file_id}.wav")

        motion_feat = process_npz(npz_path, file_id)
        print(f"[INFO] Motion features: {motion_feat.shape}")

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
