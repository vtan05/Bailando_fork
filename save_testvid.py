import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ========= Config =========
KEYPOINTS_PATH = r"/data/van/Dance/Bailando_new/experiments/sep_vqvae/vis/pkl/ep000500/211.json.pkl.npy"
FPS = 30
MAX_SECONDS = 30
CONNECT_EDGES = True           # set False for dots-only
OUT_VIDEO = "finedance_vis.mp4"
OUT_STILL = "frame0_debug.png"

# ========= Load =========
raw = np.load(KEYPOINTS_PATH, allow_pickle=True).item()
pos = raw["pred_position"].reshape((-1, 22, 3)).astype(np.float32) * 100  # (T,22,3)

# Root/global translation if available
root = raw.get("root_translation", raw.get("trans", None))
if root is not None:
    T = min(len(pos), len(root))
    pos = pos[:T]
    pos += root[:T, None, :]
    print("✅ Added root motion")
else:
    print("⚠️ No root motion found (local coords)")

# Trim to MAX_SECONDS
T = min(len(pos), FPS * MAX_SECONDS)
pos = pos[:T]

# Z-up: swap Y/Z
P = pos[..., [0, 2, 1]]

# ========= Skeleton edges (SMPL-X style, arms from chest via clavicles) =========
# Index legend in your order:
# 0 pelvis
# 1 L_hip   2 R_hip
# 3 spine1  6 spine2  9 spine3 (chest)  12 neck  15 head
# 4 L_knee  7 L_ankle 10 L_foot
# 5 R_knee  8 R_ankle 11 R_foot
# 13 L_collar 16 L_shoulder 18 L_elbow 20 L_wrist
# 14 R_collar 17 R_shoulder 19 R_elbow 21 R_wrist
EDGES = [
    # Left leg
    (0,1), (1,4), (4,7), (7,10),
    # Right leg
    (0,2), (2,5), (5,8), (8,11),
    # Spine
    (0,3), (3,6), (6,9), (9,12), (12,15),
    # Arms
    (9,13), (13,16), (16,18), (18,20),   # left
    (9,14), (14,17), (17,19), (19,21)    # right
]

# ========= Bone-length sanity check =========
def check_bones(P, edges, name="Z-up"):
    bones = []
    for (i, j) in edges:
        d = np.linalg.norm(P[:, i] - P[:, j], axis=-1)  # (T,)
        bones.append(d)
    bones = np.array(bones)  # (B,T)
    med_each = np.median(bones, axis=1)
    global_med = med_each.mean() + 1e-8
    ratios = bones.mean(axis=1) / global_med
    bad = [(edges[k], float(ratios[k]), float(med_each[k])) for k in range(len(edges)) if ratios[k] > 2.2]
    print(f"[{name}] median bone length ≈ {global_med:.2f}")
    if bad:
        print("Suspicious edges (>2.2× median):")
        for e, r, m in sorted(bad, key=lambda x: -x[1]):
            print(f"  {e}: {r:.2f}× (edge median {m:.2f})")
    else:
        print("No suspicious edges found.")

check_bones(P, EDGES)

# ========= Axis limits (global, once) =========
mins = P.min(axis=(0,1))
maxs = P.max(axis=(0,1))
span = maxs - mins
pad = max(50.0, 0.2 * float(span.max()))
x_min, x_max = float(mins[0]-pad), float(maxs[0]+pad)
y_min, y_max = float(mins[1]-pad), float(maxs[1]+pad)
z_min, z_max = float(mins[2]-pad), float(maxs[2]+pad)

# ========= Plot helpers =========
def draw_dots(ax, joints):
    ax.scatter(joints[:,0], joints[:,1], joints[:,2], s=25)
    for idx, (x, y, z) in enumerate(joints):
        ax.text(x, y, z, str(idx), fontsize=8)

def draw_edges(ax, joints, edges):
    for (i, j) in edges:
        xs = [joints[i,0], joints[j,0]]
        ys = [joints[i,1], joints[j,1]]
        zs = [joints[i,2], joints[j,2]]
        ax.plot(xs, ys, zs, linewidth=2, marker='o')

# ========= Save frame-0 debug still =========
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
ax.view_init(elev=30, azim=140)
ax.set_title("Frame 0 — dots + indices")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
draw_dots(ax, P[0])
plt.savefig(OUT_STILL, dpi=170); plt.close(fig)
print(f"🖼 Saved: {os.path.abspath(OUT_STILL)}")

# ========= Animate and save =========
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
writer = FFMpegWriter(fps=FPS, bitrate=2000)

with writer.saving(fig, OUT_VIDEO, dpi=100):
    for t in range(len(P)):
        ax.clear()
        J = P[t]
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)
        ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])
        ax.view_init(elev=30, azim=140)
        ax.set_title(f"Frame {t+1}/{len(P)}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        draw_dots(ax, J)
        if CONNECT_EDGES:
            draw_edges(ax, J, EDGES)
        writer.grab_frame()

print(f"✅ Video saved to: {os.path.abspath(OUT_VIDEO)}")
