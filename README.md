# GPS-less Drone Navigation: SuperPoint + LightGlue

Replaces the SIFT-based pipeline in `gpsless_mapping` with a neural feature extractor
(SuperPoint) and a neural matcher (LightGlue), running on the Jetson Orin GPU.

---

## Why the upgrade

| | SIFT (gpsless_mapping) | SuperPoint + LightGlue (this folder) |
|---|---|---|
| Feature extractor | Hand-crafted gradient histogram | Neural net, pretrained on real images |
| Descriptor size | 128-dim float32 | 256-dim float32, L2-normalised |
| Matching step | BFMatcher + manual Lowe's ratio test | LightGlue neural matcher — no manual threshold |
| GPU support | No | Yes (Jetson Orin CUDA) |
| Cross-domain robustness | OK | Better (handles viewpoint + lighting change between satellite tile and drone photo) |

LightGlue replaces the entire BFMatcher + ratio-test step. You give it two feature
dicts and it returns only high-confidence verified pairs directly.

---

## Files

| File | Purpose |
|---|---|
| `config.py` | All tunable settings (zoom, max keypoints, thresholds) |
| `tile_utils.py` | GPS ↔ tile pixel math (same as gpsless_mapping) |
| `build_index.py` | Extract SuperPoint features from every tile → `index/sp_index_z{zoom}.pkl` |
| `localize.py` | Localize a drone photo using SuperPoint + LightGlue |
| `index/sp_index_z19.pkl` | Pre-built index — 660 tiles, 337,859 keypoints, 334 MB |

---

## Installation

LightGlue is not on PyPI; install from source:

```bash
git clone https://github.com/cvg/LightGlue.git
pip install kornia
cp -r LightGlue/lightglue ~/.local/lib/python3.10/site-packages/
```

Other dependencies are already present on the Jetson (`torch`, `opencv-python`, `numpy`).

---

## Quick start

### Build / rebuild the index

Only needed when you download new tiles or change `SP_MAX_KEYPOINTS` in `config.py`.
Tiles are read from `../gpsless_mapping/tiles` by default.

```bash
cd /home/scott/claude-project/gpsless_superpoint
python3 build_index.py
```

Output example:
```
Device        : cuda
Tiles found   : 660 (zoom 19)
Max kp/tile   : 512
Index saved → index/sp_index_z19.pkl
  Tiles indexed  : 660  (skipped 0)
  Total keypoints: 337,859
  Index file size: 334.0 MB
```

### Localize a drone photo

```bash
python3 localize.py /path/to/drone_photo.jpg
python3 localize.py /path/to/drone_photo.jpg --show   # also saves result_tile.png
```

Output example:
```
Query image    : drone.jpg  (1280×960 px)
Query features : 847 keypoints  (SuperPoint, device=cuda)
Loading index from index/sp_index_z19.pkl ...
Tile index     : 660 tiles
Total train kp : 337,859  — building FLANN ...
Good matches   : 312 (after ratio test)

Top candidate tiles:
  tile (439041,224461)  votes=87
  ...

LightGlue match on 3×3 grid centred (439041,224461) ...

=======================================================
  GPS Location : 25.0412345,  121.4523456
  RANSAC inliers: 42  |  centre tile (439041,224461) z19
  Google Maps  : https://maps.google.com/?q=25.0412345,121.4523456
=======================================================
```

---

## Algorithm

The same two-phase structure as the SIFT version is kept so the logic is easy to compare.

### Phase 1 — Coarse vote (FLANN)

1. SuperPoint extracts up to `SP_QUERY_MAX_KEYPOINTS` (1024) keypoints from the query photo.
2. Their 256-dim descriptors are matched against all 337k pooled tile descriptors using FLANN (L2, same as the SIFT version — SuperPoint descriptors are L2-normalised so L2 distance works correctly).
3. Each match votes for the tile it came from. Top 20 tiles advance to Phase 2.

### Phase 2 — Fine match (SuperPoint + LightGlue)

1. The winning tile and its 8 neighbours are stitched into a 3×3 composite (768×768 px at zoom 19).
2. SuperPoint extracts up to `SP_COMPOSITE_MAX_KEYPOINTS` (2048) keypoints from the composite.
3. **LightGlue** matches the query features against the composite features. Unlike ratio-test matching, LightGlue is a Transformer-based neural network that attends across both images simultaneously and outputs only high-confidence verified pairs — no manual threshold needed.
4. RANSAC homography is estimated from the verified pairs.
5. The query image centre is projected through the homography into the composite pixel space, then converted to GPS (lat, lon).

---

## Tuning

All parameters are in `config.py`. The most useful ones:

| Parameter | Default | Effect |
|---|---|---|
| `SP_MAX_KEYPOINTS` | 512 | Keypoints per tile in the index. More = larger index file, better coarse vote. |
| `SP_QUERY_MAX_KEYPOINTS` | 1024 | Keypoints extracted from the query photo. |
| `SP_COMPOSITE_MAX_KEYPOINTS` | 2048 | Keypoints extracted from the 3×3 stitched composite. |
| `SP_KEYPOINT_THRESHOLD` | 0.005 | Detector confidence cutoff. Lower = more keypoints. |
| `TOP_CANDIDATES` | 20 | How many top-voted tiles Phase 2 tries before giving up. |
| `MIN_INLIERS` | 6 | Minimum RANSAC inliers to accept a GPS fix. |

If localisation fails:
- Lower `SP_KEYPOINT_THRESHOLD` (e.g. `0.001`) to detect more features.
- Raise `SP_MAX_KEYPOINTS` and rebuild the index.
- Make sure the drone photo was taken roughly nadir (straight down) over the tile area.

---

## Hardware notes (Jetson Orin)

- SuperPoint model: ~5 MB weights, cached at `~/.cache/torch/hub/checkpoints/superpoint_v1.pth`
- LightGlue model: ~45 MB weights, cached at `~/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth`
- Both models are loaded once and reused for all queries in a session.
- The `NvMapMemAllocInternalTagged` warnings printed by the Jetson CUDA driver are harmless — they appear on first allocation and do not affect results.
- Peak GPU memory during localization: ~500 MB (well within Orin limits).
