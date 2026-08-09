"""
Use SIFT (scale+rotation invariant) for Phase 1, then SuperPoint+LightGlue
for Phase 2, all on the large-canvas yaw-corrected photo #8.

SIFT index: index/sift_index_z19.pkl
"""

import sys, os, pickle, bisect
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
sys.path.insert(0, str(Path(__file__).parent))

from localize import _get_models, lightglue_match
from correct_tilt import correction_homography

PHOTO      = Path(__file__).parent / 'photo_20260705/original/25.0431990_121.4743276-NOFIX_NOFIX__NAm__59.9m.png'
TRUE_TX, TRUE_TY = 439053, 224452
ZOOM       = 19
TILE_DIR_P = Path(__file__).parent.parent / 'gpsless_mapping/tiles'
SIFT_INDEX = Path(__file__).parent / 'gpsless_SIFT/index/sift_index_z19.pkl'

# Load SIFT index
print(f'Loading SIFT index: {SIFT_INDEX} …')
with open(SIFT_INDEX, 'rb') as f:
    sift_index = pickle.load(f)
print(f'  {len(sift_index)} tiles, {sum(len(e["descs"]) for e in sift_index):,} keypoints')

# Check index entry structure
e0 = sift_index[0]
print(f'  Entry keys: {list(e0.keys())}')
print(f'  Desc shape: {e0["descs"].shape}  dtype={e0["descs"].dtype}')

# Build SIFT FLANN index
tile_offsets = [0]
for e in sift_index:
    tile_offsets.append(tile_offsets[-1] + len(e['descs']))
all_descs = np.vstack([e['descs'] for e in sift_index]).astype(np.float32)
print(f'  Total descriptors: {len(all_descs):,}')

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
flann.add([all_descs])
flann.train()
print('  FLANN ready.\n')


def sift_flann_vote(img_bgr, ratio=0.75):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=2000)
    kps, descs = sift.detectAndCompute(gray, None)
    if descs is None or len(kps) < 5:
        return [], 0
    raw  = flann.knnMatch(descs.astype(np.float32), k=2)
    good = [m for m, n in raw
            if len((m,n)) == 2 and m.distance < ratio * n.distance]
    votes = defaultdict(int)
    for m in good:
        ti = bisect.bisect_right(tile_offsets, m.trainIdx) - 1
        votes[ti] += 1
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    return ranked, len(kps)


def large_canvas_correct(img, pitch_deg, roll_deg, yaw_deg):
    h, w = img.shape[:2]
    H = correction_homography(img.shape, pitch_deg, roll_deg, yaw_deg)
    corners = np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]],
                       dtype=np.float64).T
    mapped = H @ corners;  mapped /= mapped[2:3]
    min_x, max_x = mapped[0].min(), mapped[0].max()
    min_y, max_y = mapped[1].min(), mapped[1].max()
    T = np.array([[1,0,-min_x],[0,1,-min_y],[0,0,1]], dtype=np.float64)
    return cv2.warpPerspective(img, T@H,
                               (int(np.ceil(max_x-min_x)), int(np.ceil(max_y-min_y))),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# Load models
print('Loading SuperPoint + LightGlue …')
_get_models()

orig = cv2.imread(str(PHOTO))
print(f'Original: {orig.shape[1]}×{orig.shape[0]}\n')

print('=' * 65)
# Test the best correction angles
for yaw, pitch in [(-118, -10), (-118, 0), (-110, 0), (-125, -10)]:
    corrected = large_canvas_correct(orig, pitch_deg=pitch, roll_deg=0, yaw_deg=yaw)
    ranked, n_kp = sift_flann_vote(corrected)

    if not ranked:
        print(f'yaw={yaw} pitch={pitch}: no SIFT votes')
        continue

    top_ti, top_v = ranked[0]
    top_tx = sift_index[top_ti]['tile_x']
    top_ty = sift_index[top_ti]['tile_y']

    true_rank, true_votes = None, 0
    for r, (ti, v) in enumerate(ranked):
        if sift_index[ti]['tile_x'] == TRUE_TX and sift_index[ti]['tile_y'] == TRUE_TY:
            true_rank, true_votes = r+1, v
            break

    print(f'yaw={yaw:+d}° pitch={pitch:+d}°  kp={n_kp}  '
          f'top=({top_tx},{top_ty}) v={top_v}  '
          f'true rank=#{true_rank or "N/A"} votes={true_votes}')

    # Phase 2: try top-5 FLANN tiles with SuperPoint+LightGlue
    best_result = None
    for r2, (ti2, _) in enumerate(ranked[:20]):
        tx2 = sift_index[ti2]['tile_x']
        ty2 = sift_index[ti2]['tile_y']
        res = lightglue_match(corrected, tx2, ty2, ZOOM, TILE_DIR_P,
                              radius=1, min_inliers=6)
        if res:
            lat2, lon2, inl2, *_ = res
            if best_result is None or inl2 > best_result[2]:
                best_result = (lat2, lon2, inl2, tx2, ty2, r2+1)

    if best_result:
        lat2, lon2, inl2, btx, bty, brank = best_result
        print(f'  → Phase 2 best: tile({btx},{bty}) rank#{brank} '
              f'inliers={inl2}  est=({lat2:.5f},{lon2:.5f})')
        # True GPS: 25.0431990, 121.4743276
        dlat = abs(lat2 - 25.0431990) * 111000
        dlon = abs(lon2 - 121.4743276) * 111000 * 0.906
        print(f'  → Error: Δlat={dlat:.0f}m  Δlon={dlon:.0f}m  '
              f'total≈{(dlat**2+dlon**2)**0.5:.0f}m')
    else:
        # Also try forced on true tile
        res_true = lightglue_match(corrected, TRUE_TX, TRUE_TY, ZOOM, TILE_DIR_P,
                                   radius=1, min_inliers=4)
        forced = f'inliers={res_true[2]}' if res_true else 'FAIL'
        print(f'  → Phase 2 top-20 all failed.  Forced true tile: {forced}')
    print()

torch.cuda.empty_cache()
