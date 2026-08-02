# 程式檔案清單

## 一、SuperPoint + LightGlue 定位系統（主程式）
路徑根目錄：`/home/scott/claude-project/gpsless_superpoint/`

| 檔案 | 完整路徑 | 功能說明 |
|------|----------|----------|
| `config.py` | `gpsless_superpoint/config.py` | 所有參數設定：飛行區域 bbox、zoom level、keypoint 數量、tile 來源定義（`TILE_SOURCES`）、互動選擇函式（`choose_tile_source()`） |
| `build_index.py` | `gpsless_superpoint/build_index.py` | 對每張衛星 Tile 執行 SuperPoint 提取特徵點與描述子，建立搜尋索引（`.pkl`）供定位使用。執行時詢問使用 ESRI 或 TGOS 圖資 |
| `localize.py` | `gpsless_superpoint/localize.py` | 核心定位程式。輸入一張無人機空拍影像，執行兩階段比對（Phase 1：FLANN 投票粗定位；Phase 2：LightGlue 精定位），輸出估算 GPS 座標 |
| `tile_utils.py` | `gpsless_superpoint/tile_utils.py` | Tile 座標與 GPS 互換工具函式：`deg2tile()`、`tile2deg()`、`pixel_to_latlon()`、`bbox_to_tiles()`、`meters_per_pixel()` |
| `batch_test.py` | `gpsless_superpoint/batch_test.py` | 批次測試工具：對資料夾內所有照片執行 `localize()`，計算每張的定位誤差，輸出統計表 |
| `run_test.sh` | `gpsless_superpoint/run_test.sh` | 設定 Jetson 必要環境變數（`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`）並執行 batch_test.py |
| `requirements.txt` | `gpsless_superpoint/requirements.txt` | Python 套件依賴清單（lightglue、opencv、torch 等） |
| `README.md` | `gpsless_superpoint/README.md` | 專案說明文件 |
| `NOTES_zh.md` | `gpsless_superpoint/NOTES_zh.md` | 開發注意事項（中文）：NvMap 崩潰、尺度對齊、Homography Bug 修正等 10 項重點 |
| `操作流程.md` | `gpsless_superpoint/操作流程.md` | 完整操作流程與指令（中文）：從下載圖資到取得定位結果的逐步說明 |
| `program_list.md` | `gpsless_superpoint/program_list.md` | 本檔案：所有程式路徑與功能對照表 |
| `.gitignore` | `gpsless_superpoint/.gitignore` | Git 排除清單：排除大型索引檔、LightGlue 套件、Tile 圖資、編譯產物 |

---

## 二、衛星圖資下載（Tile 下載工具）
路徑根目錄：`/home/scott/claude-project/gpsless_mapping/`

| 檔案 | 完整路徑 | 功能說明 |
|------|----------|----------|
| `download_tiles.py` | `gpsless_mapping/download_tiles.py` | 從 ESRI 或 TGOS 伺服器下載衛星 Tile 到本地端。執行時互動詢問來源（或用 `--source esri/tgos` 直接指定）。ESRI 存到 `tiles/`，TGOS 存到 `tiles_tgos/` |
| `config.py` | `gpsless_mapping/config.py` | 台北 bbox、zoom level、`TILE_SOURCES` 定義、`choose_tile_source()` 函式 |
| `tile_utils.py` | `gpsless_mapping/tile_utils.py` | 與 gpsless_superpoint 版本相同的 Tile 座標工具函式 |

---

## 三、無人機飛行程式（C++，執行於 Jetson Orin Nano）

### 3-1 SuperPoint 版本（最新）
| 檔案 | 完整路徑 | 功能說明 |
|------|----------|----------|
| `drone_localize.cpp` | `gpsless_superpoint/gpsless_SIFT/shot_estimate/drone_localize.cpp` | 無人機主飛行迴圈：GStreamer CSI 相機擷取、MAVLink 讀取 GPS / Attitude、Attitude Guard（傾斜 >5° 跳過拍照）、呼叫 Python `localize.py`、結果回傳 Pixhawk |
| `run1.sh` | `gpsless_superpoint/gpsless_SIFT/shot_estimate/run1.sh` | 執行腳本：設定環境、啟動 drone_localize |

### 3-2 gpsless_mapping 版本（同步維護）
| 檔案 | 完整路徑 | 功能說明 |
|------|----------|----------|
| `drone_localize.cpp` | `gpsless_mapping/shot_estimate/drone_localize.cpp` | 與上方相同，同步維護的副本 |

---

## 四、舊版 SIFT 系統（參考用）
路徑根目錄：`/home/scott/claude-project/gpsless_superpoint/gpsless_SIFT/`

| 檔案 | 完整路徑 | 功能說明 |
|------|----------|----------|
| `localize.py` | `gpsless_SIFT/localize.py` | SIFT 版本定位程式（已被 SuperPoint 版取代，保留作對照） |
| `build_index.py` | `gpsless_SIFT/build_index.py` | SIFT 版本 Index 建立 |
| `download_tiles.py` | `gpsless_SIFT/download_tiles.py` | 舊版 Tile 下載（功能與 gpsless_mapping 版相同） |
| `cuda_bf_matcher.cu` | `gpsless_SIFT/cuda_bf_matcher.cu` | CUDA BF Matcher 加速器（SIFT 用） |
| `cuda_bf_matcher.h` | `gpsless_SIFT/cuda_bf_matcher.h` | CUDA BF Matcher header |
| `gps_check.py` | `gpsless_SIFT/gps_check.py` | 計算兩 GPS 座標間距離的工具 |

---

## 五、資料目錄

| 目錄 | 完整路徑 | 說明 |
|------|----------|------|
| `tiles/` | `gpsless_mapping/tiles/19/{x}/{y}.png` | 下載的 ESRI 衛星 Tile（zoom 19，660 張，涵蓋台北） |
| `tiles_tgos/` | `gpsless_mapping/tiles_tgos/19/{x}/{y}.png` | 下���的 TGOS 正射影像 Tile（下載後存放位置） |
| `index/` | `gpsless_superpoint/index/` | SuperPoint 特徵索引檔 |
| `index/sp_index_z19_esri.pkl` | `gpsless_superpoint/index/sp_index_z19_esri.pkl` | ESRI 來源的 SuperPoint Index（165.7 MB，660 tiles，159,914 keypoints） |
| `index/sp_index_z19_tgos.pkl` | `gpsless_superpoint/index/sp_index_z19_tgos.pkl` | TGOS 來源�� SuperPoint Index（建立後存放位置） |
| `photo_20260705/original/` | `gpsless_superpoint/photo_20260705/original/` | 2026-07-05 F450 無人機飛行測試照片（13 張，含定位結果對照） |
| `photo_without_legs/` | `gpsless_superpoint/photo_without_legs/` | 早期測試照片（收起起落架前後對照） |

---

## 六、依賴套件（Python）

| 套件 | 安裝方式 | 用途 |
|------|----------|------|
| `lightglue` | `pip install lightglue` | SuperPoint 特徵提取 + LightGlue 神經網路匹配 |
| `torch` | NVIDIA 自訂 wheel（Jetson 專用） | GPU 推論 |
| `opencv-python` | 系統內建（`/usr/lib/python3/dist-packages`，4.5.4，含 GStreamer） | ���像處理、Homography、FLANN |
| `numpy` | `pip install numpy` | 矩陣運算 |
| `tqdm` | `pip install tqdm` | 進度條顯示 |
| `requests` | `pip install requests` | Tile 下載 HTTP 請求 |

> ⚠️ **注意**：執行相機擷取相關程式時，必須使用系統 OpenCV（`/usr/lib/python3/dist-packages`），不能使用 pip 安裝的版本（無 GStreamer 支援）。
