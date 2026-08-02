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
TILE_DIR = '../gpsless_mapping/tiles'   # default (ESRI) — used when no interactive selection
INDEX_DIR = 'index'

# ESRI World Imagery — free satellite tiles, no API key required
# Note: ESRI URL order is z/y/x (not z/x/y like OSM)
TILE_SERVER_URL = (
    'https://server.arcgisonline.com/ArcGIS/rest/services/'
    'World_Imagery/MapServer/tile/{z}/{y}/{x}'
)

# ── Tile sources (used by build_index.py) ─────────────────────────────────────
TILE_SOURCES = {
    'esri': {
        'name': 'ESRI World Imagery',
        'desc': '全球衛星影像，免費，無需帳號',
        'url' : TILE_SERVER_URL,
        'dir' : '../gpsless_mapping/tiles',
        'index': 'index/sp_index_z19_esri.pkl',
    },
    'tgos': {
        'name': 'Taiwan TGOS 正射影像',
        'desc': '台灣政府航拍正射影像，台灣本島解析度較高，免費',
        'url' : ('https://wmts.nlsc.gov.tw/wmts/PHOTO2/default/'
                 'EPSG:3857/{z}/{y}/{x}'),
        'dir' : '../gpsless_mapping/tiles_tgos',
        'index': 'index/sp_index_z19_tgos.pkl',
    },
}


def choose_tile_source() -> dict:
    """Interactively ask the user which tile source to use.
    Returns the selected source dict from TILE_SOURCES."""
    keys = list(TILE_SOURCES.keys())
    print('\n請選擇 Tile 來源：')
    for i, key in enumerate(keys, 1):
        src = TILE_SOURCES[key]
        print(f'  {i}) {src["name"]:30s} — {src["desc"]}')
    while True:
        choice = input(f'\n請輸入編號 (1–{len(keys)}): ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            return TILE_SOURCES[keys[int(choice) - 1]]
        print(f'請輸入 1 到 {len(keys)} 之間的數字')

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
