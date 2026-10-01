"""Stage 3: face detection, landmarks, head pose and tracking for every frame on a 25 fps grid.

For each video writes data/faces/<id>.npz (per-detection arrays) and data/faces/<id>.json (summary).
`bvsr.cut` uses these to make face-centred clips and to reject clips with no face, several faces,
small faces or large head rotation.

Frame grid
    The video is decoded once with `setpts=PTS-STARTPTS,fps=25`, so frame i is the picture shown at
    video time i/25 s. The wav used for ASR/alignment starts at the audio stream's start, so a word
    at wav time t sits on frame round((t - offset) * 25), where offset = video_start - audio_start
    (read with ffprobe and stored in the summary; 0 for most YouTube files).

    python -m bvsr.faces                  # all downloaded videos without face data
    python -m bvsr.faces --workers 4      # videos in parallel (CPU; each worker ~150 MB RAM)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .common import DATA, RAW, ROOT, ensure_dirs

FACES = DATA / "faces"
MODEL = ROOT / "models" / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FPS = 25
NOSE_TIP = 1                        # MediaPipe face-mesh index used as the face centre (LRW aligns nose centres)


# --- video I/O on the 25 fps grid -------------------------------------------------

def probe_video(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,start_time",
         "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    streams = json.loads(out)["streams"]
    v = next(s for s in streams if s["codec_type"] == "video")
    a = next((s for s in streams if s["codec_type"] == "audio"), None)
    v_start = float(v.get("start_time") or 0)
    a_start = float(a.get("start_time") or 0) if a else v_start
    return {"width": int(v["width"]), "height": int(v["height"]), "offset": round(v_start - a_start, 6)}


def iter_frames(path: Path, width: int, height: int):
    """Yield (index, RGB frame) on the 25 fps grid. `bvsr.cut` decodes with the identical command,
    so frame indices match between stages."""
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
           "-vf", f"setpts=PTS-STARTPTS,fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = width * height * 3
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=size * 4)
    try:
        i = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            i += 1
    finally:
        # The caller may stop early (all clips written); don't leave ffmpeg blocked on a full pipe.
        if p.poll() is None:
            p.kill()
        p.stdout.close()
        p.wait()


# --- per-frame measurements -------------------------------------------------------

def lips_indices() -> np.ndarray:
    from mediapipe.tasks.python import vision

    return np.array(sorted({i for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS for i in (c.start, c.end)}))


def head_pose(matrix: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from MediaPipe's facial transformation matrix.

    Checked on real footage: yaw tracks the nose's horizontal offset within the face outline
    (r = 0.86, same sign), so the convention is right for filtering by |yaw| / |pitch|.
    """
    import cv2

    pitch, yaw, roll = cv2.RQDecomp3x3(np.asarray(matrix)[:3, :3])[0]
    return float(pitch), float(yaw), float(roll)


def mouth_sharpness(gray: np.ndarray, lips_xy: np.ndarray) -> float:
    """Variance of the Laplacian over the mouth region rescaled to 96 px wide (higher = sharper)."""
    import cv2

    x0, y0 = lips_xy.min(0)
    x1, y1 = lips_xy.max(0)
    w = max(x1 - x0, 1.0)
    pad = 0.25 * w
    x0, x1 = int(max(0, x0 - pad)), int(min(gray.shape[1], x1 + pad))
    y0, y1 = int(max(0, y0 - pad)), int(min(gray.shape[0], y1 + pad))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    roi = cv2.resize(roi, (96, max(1, round(96 * roi.shape[0] / roi.shape[1]))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def frame_hist(img: np.ndarray) -> np.ndarray:
    """Normalised HSV colour histogram of a downscaled frame (for shot-cut detection)."""
    import cv2

    small = cv2.resize(img, (160, max(1, round(160 * img.shape[0] / img.shape[1]))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    import cv2

    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))  # 0 = identical, 1 = disjoint


# --- tracking -----------------------------------------------------------------------

def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Greedy IoU tracker. A track ends after `max_gap` frames without a match, so shot cuts and
    jumps between people start a new track instead of joining two faces."""

    def __init__(self, min_iou: float, max_gap: int):
        self.min_iou, self.max_gap = min_iou, max_gap
        self.active: dict[int, tuple[int, np.ndarray]] = {}  # track id -> (last frame, last box)
        self.next_id = 0

    def cut(self) -> None:
        """Shot boundary: never continue a track across it, even if the face is in the same place."""
        self.active = {}

    def update(self, frame: int, boxes: list[np.ndarray]) -> list[int]:
        self.active = {t: v for t, v in self.active.items() if frame - v[0] <= self.max_gap + 1}
        pairs = sorted(((iou(b, v[1]), t, j) for t, v in self.active.items() for j, b in enumerate(boxes)), reverse=True)
        ids: list[int | None] = [None] * len(boxes)
        used: set[int] = set()
        for score, t, j in pairs:
            if score < self.min_iou:
                break
            if t in used or ids[j] is not None:
                continue
            ids[j], used = t, used | {t}
        for j in range(len(boxes)):
            if ids[j] is None:
                ids[j], self.next_id = self.next_id, self.next_id + 1
            self.active[ids[j]] = (frame, boxes[j])
        return ids  # type: ignore[return-value]


# --- driver -------------------------------------------------------------------------

def model_sha256() -> str:
    return hashlib.sha256(MODEL.read_bytes()).hexdigest()


def analyze(video: Path, cfg: dict) -> tuple[dict, dict]:
    """Per-frame face measurements for any video file. Returns (arrays, summary)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    info = probe_video(video)
    W, H = info["width"], info["height"]
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=cfg["max_faces"],
        min_face_detection_confidence=cfg["det_conf"],
        min_face_presence_confidence=cfg["det_conf"],
        min_tracking_confidence=cfg["det_conf"],
        output_facial_transformation_matrixes=True,
    )
    lips = lips_indices()
    tracker = Tracker(cfg["min_iou"], cfg["max_gap"])
    rows: dict[str, list] = {k: [] for k in ("frame", "track", "box", "pose", "sharp", "lmk")}
    shot_dist: list[float] = []      # histogram distance to the previous frame (0 for frame 0)
    cuts: list[int] = []             # frames that start a new shot
    prev_hist = None
    t0 = time.time()
    n_frames = 0
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, img in iter_frames(video, W, H):
            n_frames = i + 1
            hist = frame_hist(img)
            dist = hist_distance(prev_hist, hist) if prev_hist is not None else 0.0
            prev_hist = hist
            shot_dist.append(dist)
            if dist > cfg["cut_threshold"]:
                cuts.append(i)
                tracker.cut()
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)), i * 1000 // FPS)
            if not res.face_landmarks:
                tracker.update(i, [])
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            pts_all, boxes = [], []
            for face in res.face_landmarks:
                pts = np.array([[q.x * W, q.y * H] for q in face], dtype=np.float32)
                pts_all.append(pts)
                boxes.append(np.concatenate([pts.min(0), pts.max(0)]))
            ids = tracker.update(i, boxes)
            for pts, box, tid, mat in zip(pts_all, boxes, ids, res.facial_transformation_matrixes):
                rows["frame"].append(i)
                rows["track"].append(tid)
                rows["box"].append(box)
                rows["pose"].append(head_pose(mat))
                rows["sharp"].append(mouth_sharpness(gray, pts[lips]))
                rows["lmk"].append(pts)

    arrays = {
        "frame": np.array(rows["frame"], np.int32),
        "track": np.array(rows["track"], np.int32),
        "box": np.array(rows["box"], np.float32).reshape(-1, 4),        # x0, y0, x1, y1 of all 478 landmarks
        "pose": np.array(rows["pose"], np.float32).reshape(-1, 3),      # pitch, yaw, roll (deg)
        "sharp": np.array(rows["sharp"], np.float32),
        "lmk": np.array(rows["lmk"], np.float32).reshape(-1, 478, 2),   # pixel coords in the source frame
        "cuts": np.array(cuts, np.int32),                               # first frame of each new shot
        "shot_dist": np.array(shot_dist, np.float32),                   # per-frame histogram distance
    }
    tracks = {}
    for t in np.unique(arrays["track"]):
        f = arrays["frame"][arrays["track"] == t]
        tracks[int(t)] = {"first": int(f.min()), "last": int(f.max()), "n": int(len(f))}
    summary = {
        "fps": FPS, "n_frames": n_frames, "width": W, "height": H, "offset": info["offset"],
        "n_detections": int(len(arrays["frame"])),
        "cuts": [int(c) for c in cuts],
        "frames_with_face": int(len(np.unique(arrays["frame"]))),
        "tracks": tracks, "config": cfg, "model_sha256": cfg["model_sha256"],
        "seconds": round(time.time() - t0, 1),
    }
    return arrays, summary


def process_video(vid: str, cfg: dict) -> dict:
    arrays, summary = analyze(RAW / vid / f"{vid}.mp4", cfg)
    summary = {"video_id": vid, **summary}
    FACES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FACES / f"{vid}.npz", **arrays)
    (FACES / f"{vid}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                   help="videos processed in parallel (default: half the CPU cores, max 4)")
    p.add_argument("--max-faces", type=int, default=4, help="faces per frame the landmarker may return")
    p.add_argument("--det-conf", type=float, default=0.5)
    p.add_argument("--min-iou", type=float, default=0.3, help="box overlap needed to continue a track")
    p.add_argument("--max-gap", type=int, default=2, help="missed frames a track survives")
    p.add_argument("--cut-threshold", type=float, default=0.35,
                   help="Bhattacharyya distance between consecutive frame histograms that marks a shot cut")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    if not MODEL.exists():
        raise SystemExit(f"missing {MODEL}; download it with:\n  curl -L -o {MODEL} {MODEL_URL}")
    ids = args.ids or sorted(d.name for d in RAW.iterdir() if (d / f"{d.name}.mp4").exists())
    if not args.overwrite:
        ids = [v for v in ids if not (FACES / f"{v}.npz").exists()]
    if not ids:
        print("[faces] nothing to do")
        return
    cfg = {"max_faces": args.max_faces, "det_conf": args.det_conf, "min_iou": args.min_iou,
           "max_gap": args.max_gap, "cut_threshold": args.cut_threshold, "model_sha256": model_sha256()}
    workers = min(args.workers, len(ids))
    print(f"[faces] {len(ids)} videos, {workers} worker(s)", flush=True)
    if workers == 1:
        results = (process_video(v, cfg) for v in ids)
    else:
        pool = ProcessPoolExecutor(workers)
        futures = [pool.submit(process_video, v, cfg) for v in ids]
        results = (f.result() for f in as_completed(futures))
    for s in results:
        print(f"[faces] {s['video_id']}: {s['frames_with_face']}/{s['n_frames']} frames with a face, "
              f"{len(s['tracks'])} tracks, {s['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
