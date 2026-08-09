"""
Use radius=2 in LightGlue Phase 2 so the composite covers 5×5 tiles.
The true tile (439053,224452) was only 1-2 tiles away from the best Phase 1
candidates — a larger composite should capture it.
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

with open(SIFT_INDEX, 'rb') as f:
    sift_index = pickle.load(f)
tile_offsets = [0]
for e in sift_index:
    tile_offsets.append(tile_offsets[-1] + len(e['descs']))
all_descs = np.vstack([e['descs'] for e in sift_index]).astype(np.float32)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
flann.add([all_descs])
flann.train()


def sift_vote(img_bgr, top_n=20, ratio=0.75):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=2000)
    _, descs = sift.detectAndCompute(gray, None)
    if descs is None:
        return []
    raw  = flann.knnMatch(descs.astype(np.float32), k=2)
    good = [m for m, n in raw if len((m,n))==2 and m.distance < ratio * n.distance]
    votes = defaultdict(int)
    for m in good:
        ti = bisect.bisect_right(tile_offsets, m.trainIdx) - 1
        votes[ti] += 1
    return sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


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


print('Loading SuperPoint + LightGlue …')
_get_models()
orig = cv2.imread(str(PHOTO))

corrected = large_canvas_correct(orig, pitch_deg=-10, roll_deg=0, yaw_deg=-118)
ranked = sift_vote(corrected, top_n=30)

print(f'SIFT Phase 1 — top 10 tiles:')
for r, (ti, v) in enumerate(ranked[:10]):
    tx, ty = sift_index[ti]['tile_x'], sift_index[ti]['tile_y']
    d = abs(tx-TRUE_TX)+abs(ty-TRUE_TY)
    print(f'  #{r+1}: ({tx},{ty}) votes={v}  Δ={d}')

print(f'\nPhase 2 with radius=2 on top-30 SIFT candidates:')
best = None
for r, (ti, _) in enumerate(ranked):
    tx, ty = sift_index[ti]['tile_x'], sift_index[ti]['tile_y']
    res = lightglue_match(corrected, tx, ty, ZOOM, TILE_DIR_P,
                          radius=2, min_inliers=8)
    if res:
        lat, lon, inl, *_ = res
        d = abs(tx-TRUE_TX)+abs(ty-TRUE_TY)
        dlat = abs(lat-25.0431990)*111000
        dlon = abs(lon-121.4743276)*111000*0.906
        err  = (dlat**2+dlon**2)**0.5
        print(f'  rank#{r+1} tile({tx},{ty}) Δ={d}  inliers={inl}'
              f'  est=({lat:.5f},{lon:.5f})  err={err:.0f}m')
        if best is None or inl > best[0]:
            best = (inl, lat, lon, err, tx, ty, r+1)

print()
if best:
    inl, lat, lon, err, tx, ty, rank = best
    print(f'BEST RESULT: tile({tx},{ty}) rank#{rank}')
    print(f'  inliers={inl}  est=({lat:.7f}, {lon:.7f})')
    print(f'  true=  (25.0431990, 121.4743276)')
    print(f'  error = {err:.0f} m')
else:
    # Try forced true tile with radius=2
    res = lightglue_match(corrected, TRUE_TX, TRUE_TY, ZOOM, TILE_DIR_P,
                          radius=2, min_inliers=4)
    if res:
        lat, lon, inl, *_ = res
        dlat = abs(lat-25.0431990)*111000
        dlon = abs(lon-121.4743276)*111000*0.906
        err  = (dlat**2+dlon**2)**0.5
        print(f'All Phase 2 failed.  Forced true tile r=2: inliers={inl}  err={err:.0f}m')
    else:
        print('All failed including forced true tile.')

torch.cuda.empty_cache()
