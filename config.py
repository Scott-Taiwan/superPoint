# ── Area coverage ────────────────────────────────────────────────────────────
TAIPEI_BBOX = {
    'lat_min': 25.02,
    'lat_max': 25.09,
    'lon_min': 121.43,
    'lon_max': 121.51,
}

ZOOM_LEVEL = 19
TILE_SIZE = 256  # px per tile (standard slippy-map)

# Reuse tiles already downloaded by gpsless_mapping
TILE_DIR = '../gpsless_mapping/tiles'
INDEX_DIR = 'index'

# ESRI World Imagery — free satellite tiles, no API key required
# Note: ESRI URL order is z/y/x (not z/x/y like OSM)
TILE_SERVER_URL = (
    'https://server.arcgisonline.com/ArcGIS/rest/services/'
    'World_Imagery/MapServer/tile/{z}/{y}/{x}'
)

# ── SuperPoint settings ───────────────────────────────────────────────────────
# Max keypoints extracted per tile when building the index.
# More → better coverage; fewer → smaller index file and faster Phase 1 vote.
# Rule of thumb: 256–512 for zoom 19 (256 px tile), up to 1024 for zoom 17.
SP_MAX_KEYPOINTS = 512

# Detector confidence threshold (0 keeps everything, 0.01 keeps strong corners).
SP_KEYPOINT_THRESHOLD = 0.005

# Max keypoints when running SuperPoint on the QUERY image during localization.
# Can be higher than SP_MAX_KEYPOINTS because it only runs once per query.
SP_QUERY_MAX_KEYPOINTS = 1024

# Max keypoints when running SuperPoint on the 3×3 stitched composite.
SP_COMPOSITE_MAX_KEYPOINTS = 2048

# ── Phase 1 coarse vote ───────────────────────────────────────────────────────
# Lowe's ratio test threshold — same as SIFT version.
# Only used for the FLANN coarse vote; LightGlue handles its own filtering.
MATCH_RATIO = 0.75

# How many top-voted tiles to try fine matching on.
TOP_CANDIDATES = 20

# ── Phase 2 fine match ────────────────────────────────────────────────────────
# Minimum RANSAC inliers to accept a location fix.
MIN_INLIERS = 12
