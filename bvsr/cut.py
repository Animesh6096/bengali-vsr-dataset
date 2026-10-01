"""Stage 4: cut face-centred, LRW-style word clips and LRS-style sentence clips.

Needs stage 2 (data/align/<id>.json) and stage 3 (data/faces/<id>.npz).

Every clip is built from the same 25 fps frame grid that stage 3 analysed, so frame k of a
clip is exactly global frame `frame_start + k`, its landmarks are the ones measured on that
frame, and its audio is the matching 1/25 s slices of the 16 kHz wav.

Word clips (LRW): 29 frames, 256x256, face centred on the nose tip, word in the middle.
Sentence clips (LRS3): one per speech chunk, 224x224.

A candidate is rejected (and logged with the reason) if it crosses a shot cut, the face is
missing for more than a couple of frames, another face is on screen, the face is small, or
the head is turned too far. See `--help` for thresholds.

    python -m bvsr.cut stats          # word counts from the alignment -> data/word_counts.tsv
    python -m bvsr.cut words          # + --vocab file.txt to cut only chosen words
    python -m bvsr.cut sentences
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import ALIGN, CLIPS, DATA, RAW, SAMPLE_RATE, ensure_dirs
from .faces import FACES, FPS, NOSE_TIP, iter_frames

WORD_FRAMES = 29                         # LRW clip length (1.16 s)
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS   # 640 audio samples per video frame


# --- alignment input ----------------------------------------------------------------

def iter_words(min_score: float, min_dur: float, max_dur: float, min_chars: int, vids: list[str] | None = None):
    files = [ALIGN / f"{v}.json" for v in vids] if vids else sorted(ALIGN.glob("*.json"))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for ci, chunk in enumerate(data["chunks"]):
            for wi, w in enumerate(chunk["words"]):
                d = w["end"] - w["start"]
                if len(w["label"]) < min_chars or not (min_dur <= d <= max_dur) or w["score"] < min_score:
                    continue
                yield data["video_id"], ci, wi, w, chunk


def stats(args: argparse.Namespace) -> None:
    counts: Counter[str] = Counter()
    videos: dict[str, set[str]] = defaultdict(set)
    for vid, _, _, w, _ in iter_words(args.min_score, args.min_dur, args.max_dur, args.min_chars):
        counts[w["label"]] += 1
        videos[w["label"]].add(vid)
    out = DATA / "word_counts.tsv"
    with out.open("w", encoding="utf-8") as f:
        f.write("word\tcount\tn_videos\n")
        for word, n in counts.most_common():
            f.write(f"{word}\t{n}\t{len(videos[word])}\n")
    print(f"[stats] {sum(counts.values())} usable word tokens, {len(counts)} types -> {out}")
    for word, n in counts.most_common(args.show):
        print(f"  {word}\t{n}\t({len(videos[word])} videos)")


# --- face data ----------------------------------------------------------------------

class FaceData:
    def __init__(self, vid: str):
        d = np.load(FACES / f"{vid}.npz")
        self.summary = json.loads((FACES / f"{vid}.json").read_text(encoding="utf-8"))
        self.n_frames = self.summary["n_frames"]
        self.offset = self.summary["offset"]        # video_start - audio_start (s)
        self.width, self.height = self.summary["width"], self.summary["height"]
        self.frame, self.track, self.box = d["frame"], d["track"], d["box"]
        self.pose, self.sharp, self.lmk = d["pose"], d["sharp"], d["lmk"]
        self.cuts = d["cuts"]
        self.shot_dist = d["shot_dist"]
        self.span = {int(t): (int(self.frame[self.track == t].min()), int(self.frame[self.track == t].max()))
                     for t in np.unique(self.track)}

    def frame_of(self, t: float) -> float:
        """Audio-timeline seconds -> (fractional) index on the 25 fps grid."""
        return (t - self.offset) * FPS

    def time_of(self, i: int) -> float:
        return i / FPS + self.offset

    def rows(self, i0: int, i1: int) -> np.ndarray:
        return np.where((self.frame >= i0) & (self.frame < i1))[0]


@dataclass
class Filters:
    max_missing: int            # frames of the main face that may be missing (interpolated)
    min_other_frames: int       # a second face counts if seen on at least this many frames
    allow_multi_face: bool
    min_face: float             # px, width of the landmark box in the source frame
    max_yaw: float
    max_pitch: float
    pose_stat: str              # "max" (words) or "p95" (long sentences)
    max_frame_change: float     # histogram distance between consecutive frames inside the clip


def check_face(fd: FaceData, i0: int, i1: int, f: Filters) -> tuple[int | None, str, dict]:
    """Pick the face for frames [i0, i1) and test it. Returns (track or None, reason, metrics)."""
    n = i1 - i0
    idx = fd.rows(i0, i1)
    if len(idx) == 0:
        return None, "no_face", {}
    counts = Counter(fd.track[idx].tolist())
    main, seen = counts.most_common(1)[0]
    m = idx[fd.track[idx] == main]
    pose_abs = np.abs(fd.pose[m])
    agg = np.max if f.pose_stat == "max" else (lambda a, axis: np.percentile(a, 95, axis=axis))
    width = fd.box[m, 2] - fd.box[m, 0]
    others = [t for t, c in counts.items() if t != main and c >= f.min_other_frames]
    metrics = {
        "track": int(main),
        "missing_frames": int(n - seen),
        "other_faces": len(others),
        "face_width_min": round(float(width.min()), 1),
        "face_width_median": round(float(np.median(width)), 1),
        "pitch_abs": round(float(agg(pose_abs[:, 0], axis=0)), 1),
        "yaw_abs": round(float(agg(pose_abs[:, 1], axis=0)), 1),
        "roll_abs": round(float(agg(pose_abs[:, 2], axis=0)), 1),
        "sharpness_median": round(float(np.median(fd.sharp[m])), 1),
        "frame_change_max": round(float(fd.shot_dist[i0 + 1:i1].max()), 3) if n > 1 else 0.0,
    }
    first, last = fd.span[main]
    if np.any((fd.cuts > i0) & (fd.cuts < i1)):
        return None, "shot_cut", metrics
    if first > i0 or last < i1 - 1:
        # Tracks end at cuts and when the face jumps, so the face must be tracked across the whole clip.
        return None, "track_break", metrics
    if metrics["frame_change_max"] > f.max_frame_change:
        return None, "unstable_frames", metrics
    if metrics["missing_frames"] > f.max_missing:
        return None, "face_missing", metrics
    if others and not f.allow_multi_face:
        return None, "multiple_faces", metrics
    if metrics["face_width_min"] < f.min_face:
        return None, "small_face", metrics
    if metrics["yaw_abs"] > f.max_yaw:
        return None, "yaw", metrics
    if metrics["pitch_abs"] > f.max_pitch:
        return None, "pitch", metrics
    return int(main), "ok", metrics


def track_landmarks(fd: FaceData, track: int, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
    """Landmarks (n, 478, 2) of one track on frames [i0, i1); gaps linearly interpolated."""
    m = np.where((fd.track == track) & (fd.frame >= i0) & (fd.frame < i1))[0]
    have = fd.frame[m] - i0
    n = i1 - i0
    out = np.empty((n, 478, 2), np.float32)
    interp = np.ones(n, bool)
    interp[have] = False
    src = fd.lmk[m]
    for k in range(n):
        if not interp[k]:
            out[k] = src[np.searchsorted(have, k)]
            continue
        j = np.searchsorted(have, k)
        if j == 0:
            out[k] = src[0]
        elif j == len(have):
            out[k] = src[-1]
        else:
            a, b = have[j - 1], have[j]
            w = (k - a) / (b - a)
            out[k] = (1 - w) * src[j - 1] + w * src[j]
    return out, interp


def crop_boxes(lmk: np.ndarray, scale: float, smooth: int) -> np.ndarray:
    """(n, 3) crop boxes cx, cy, side in source pixels.

    Centre = nose tip, smoothed over `smooth` frames to remove landmark jitter while still
    following head motion. Side = scale x median face size, constant within a clip (like LRW,
    which has little or no scale change inside a clip)."""
    centre = lmk[:, NOSE_TIP].astype(np.float64)
    if smooth > 1 and len(centre) > 1:
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        padded = np.pad(centre, ((pad, smooth - 1 - pad), (0, 0)), mode="edge")
        centre = np.stack([np.convolve(padded[:, d], k, mode="valid") for d in range(2)], 1)
    size = np.median(np.maximum(lmk[..., 0].max(1) - lmk[..., 0].min(1), lmk[..., 1].max(1) - lmk[..., 1].min(1)))
    side = np.full((len(lmk), 1), scale * size)
    return np.concatenate([centre, side], 1).astype(np.float32)


def int_box(cx: float, cy: float, side: float) -> tuple[int, int, int]:
    """Pixel box (x0, y0, side) actually cropped; landmarks are mapped with the same numbers."""
    return int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))


def crop(img: np.ndarray, cx: float, cy: float, side: float, out: int) -> tuple[np.ndarray, float]:
    """Square crop centred at (cx, cy), zero-padded outside the frame, resized to out x out.
    Returns the crop and the fraction of it that fell outside the frame."""
    import cv2

    x0, y0, s = int_box(cx, cy, side)
    h, w = img.shape[:2]
    l, t, r, b = max(0, -x0), max(0, -y0), max(0, x0 + s - w), max(0, y0 + s - h)
    patch = img[max(0, y0):min(h, y0 + s), max(0, x0):min(w, x0 + s)]
    if l or t or r or b:
        patch = cv2.copyMakeBorder(patch, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    pad_frac = 1 - (s - l - r) * (s - t - b) / (s * s)
    interp = cv2.INTER_AREA if s > out else cv2.INTER_CUBIC
    return cv2.resize(patch, (out, out), interpolation=interp), pad_frac


def encode(frames: np.ndarray, audio: np.ndarray, dst: Path, crf: int) -> None:
    import soundfile as sf

    dst.parent.mkdir(parents=True, exist_ok=True)
    n, h, w, _ = frames.shape
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        sf.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
               "-i", str(wav), "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "64k", "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst)]
        subprocess.run(cmd, input=frames.tobytes(), check=True)


def audio_slice(wav: np.ndarray, fd: FaceData, i0: int, n: int) -> np.ndarray:
    s0 = int(round(fd.time_of(i0) * SAMPLE_RATE))
    out = np.zeros(n * SAMPLES_PER_FRAME, np.float32)
    a, b = max(0, s0), min(len(wav), s0 + len(out))
    if b > a:
        out[a - s0:b - s0] = wav[a:b]
    return out


# --- job planning -----------------------------------------------------------------

@dataclass
class Job:
    kind: str            # "word" | "sentence"
    i0: int
    i1: int
    track: int
    dst: Path
    meta: dict
    size: int


def plan_words(vid: str, fd: FaceData, args, vocab: set[str] | None) -> tuple[list[Job], list[dict]]:
    f = Filters(args.max_missing, args.min_other_frames, args.allow_multi_face, args.min_face,
                args.max_yaw, args.max_pitch, "max", args.max_frame_change)
    jobs, rejects = [], []
    per_video: Counter[str] = Counter()
    for _, ci, wi, w, chunk in iter_words(args.min_score, args.min_dur, args.max_dur, args.min_chars, [vid]):
        label = w["label"]
        if vocab is not None and label not in vocab:
            continue
        mid = (w["start"] + w["end"]) / 2
        i0 = round(fd.frame_of(mid)) - WORD_FRAMES // 2      # frame 14 of 29 is the word's midpoint
        i1 = i0 + WORD_FRAMES
        base = {"video_id": vid, "word": label, "frame_start": i0, "word_start": w["start"]}
        if i0 < 0 or i1 > fd.n_frames:
            rejects.append({**base, "reason": "video_edge"})
            continue
        if per_video[label] >= args.max_per_video:
            rejects.append({**base, "reason": "per_video_cap"})
            continue
        track, reason, metrics = check_face(fd, i0, i1, f)
        if track is None:
            rejects.append({**base, "reason": reason, **metrics})
            continue
        # Frames whose sampling instant i/25 (+offset) falls inside the word; a word shorter
        # than one frame gets the frame nearest its midpoint.
        fs = math.ceil(fd.frame_of(w["start"]) - 1e-6) - i0
        fe = math.floor(fd.frame_of(w["end"]) + 1e-6) - i0
        if fs > fe:
            fs = fe = WORD_FRAMES // 2
        fs, fe = max(0, fs), min(WORD_FRAMES - 1, fe)
        meta = {
            "word": label, "surface": w["word"], "video_id": vid,
            "fps": FPS, "n_frames": WORD_FRAMES, "size": args.word_size,
            "frame_start": i0, "frame_end": i1 - 1,             # global 25 fps frame indices in the video
            "clip_start": round(fd.time_of(i0), 4),               # seconds on the audio/alignment timeline
            "clip_end": round(fd.time_of(i1), 4),
            "word_start": w["start"], "word_end": w["end"], "duration": round(w["end"] - w["start"], 3),
            "start_frame": fs, "end_frame": fe,                   # word frames inside the clip (inclusive, 0-based)
            "align_score": w["score"], "chunk": ci, "index": wi, "transcript_source": chunk["source"],
            "face": metrics,
        }
        dst = CLIPS / "words" / label / f"{vid}_{i0:06d}.mp4"
        jobs.append(Job("word", i0, i1, track, dst, meta, args.word_size))
        per_video[label] += 1
    return jobs, rejects


def plan_sentences(vid: str, fd: FaceData, args) -> tuple[list[Job], list[dict]]:
    f = Filters(args.max_missing_frac, args.min_other_frames, args.allow_multi_face, args.min_face,
                args.max_yaw, args.max_pitch, "p95", args.max_frame_change)
    data = json.loads((ALIGN / f"{vid}.json").read_text(encoding="utf-8"))
    jobs, rejects = [], []
    for ci, c in enumerate(data["chunks"]):
        i0 = math.ceil(fd.frame_of(c["start"]))
        i1 = min(fd.n_frames, math.floor(fd.frame_of(c["end"])) + 1)
        base = {"video_id": vid, "chunk": ci, "frame_start": i0}
        dur = (i1 - i0) / FPS
        if not c["words"] or c["score"] is None or c["score"] < args.min_score:
            rejects.append({**base, "reason": "low_align_score"})
            continue
        if not (args.min_len <= dur <= args.max_len):
            rejects.append({**base, "reason": "length"})
            continue
        f.max_missing = int(args.max_missing_frac * (i1 - i0))
        track, reason, metrics = check_face(fd, i0, i1, f)
        if track is None:
            rejects.append({**base, "reason": reason, **metrics})
            continue
        t0 = fd.time_of(i0)
        words = [{"word": w["word"], "label": w["label"],
                  "start": round(w["start"] - t0, 3), "end": round(w["end"] - t0, 3),
                  "start_frame": max(0, math.ceil(fd.frame_of(w["start"]) - 1e-6) - i0),
                  "end_frame": min(i1 - i0 - 1, math.floor(fd.frame_of(w["end"]) + 1e-6) - i0),
                  "score": w["score"]} for w in c["words"]]
        meta = {"video_id": vid, "chunk": ci, "text": c["text"], "transcript_source": c["source"],
                "fps": FPS, "n_frames": i1 - i0, "size": args.sentence_size,
                "frame_start": i0, "frame_end": i1 - 1,
                "clip_start": round(t0, 4), "clip_end": round(fd.time_of(i1), 4),
                "align_score": c["score"], "words": words, "face": metrics}
        jobs.append(Job("sentence", i0, i1, track, CLIPS / "sentences" / vid / f"{ci:05d}.mp4", meta, args.sentence_size))
    return jobs, rejects


# --- rendering ------------------------------------------------------------------------

def render(vid: str, fd: FaceData, jobs: list[Job], args) -> None:
    """One decode pass over the video; each frame is cropped into every clip that covers it."""
    import soundfile as sf

    if not jobs:
        return
    wav, sr = sf.read(RAW / vid / f"{vid}.wav", dtype="float32")
    assert sr == SAMPLE_RATE
    jobs = sorted(jobs, key=lambda j: j.i0)
    prep: dict[int, tuple] = {}
    nxt, active = 0, []
    last_needed = max(j.i1 for j in jobs) - 1
    for i, img in iter_frames(RAW / vid / f"{vid}.mp4", fd.width, fd.height):
        while nxt < len(jobs) and jobs[nxt].i0 <= i:
            j = jobs[nxt]
            # Buffers exist only while a clip is open: a few clips overlap at any time, not all of them.
            lmk, interp = track_landmarks(fd, j.track, j.i0, j.i1)
            prep[nxt] = (lmk, interp, crop_boxes(lmk, args.crop_scale, args.smooth),
                         np.empty((j.i1 - j.i0, j.size, j.size, 3), np.uint8), np.zeros(j.i1 - j.i0))
            active.append(nxt)
            nxt += 1
        done = []
        for a in active:
            j = jobs[a]
            lmk, interp, boxes, frames, pads = prep[a]
            k = i - j.i0
            cx, cy, side = boxes[k]
            frames[k], pads[k] = crop(img, cx, cy, side, j.size)
            if i == j.i1 - 1:
                done.append(a)
        for a in done:
            active.remove(a)
            finish(jobs[a], prep.pop(a), wav, fd, args)
        if i >= last_needed:
            break


def finish(job: Job, prep, wav: np.ndarray, fd: FaceData, args) -> None:
    lmk, interp, boxes, frames, pads = prep
    n = job.i1 - job.i0
    ib = np.array([int_box(*b) for b in boxes], np.float32)          # x0, y0, side actually cropped
    lmk_crop = (lmk - ib[:, None, :2]) * (job.size / ib[:, None, 2:3])
    encode(frames, audio_slice(wav, fd, job.i0, n), job.dst, args.crf)
    np.savez_compressed(job.dst.with_suffix(".npz"),
                        landmarks=lmk_crop.astype(np.float16),        # (n, 478, 2) in clip pixels
                        crop_box=boxes,                               # (n, 3) cx, cy, side in source pixels
                        interpolated=interp)                          # (n,) landmarks interpolated, not detected
    meta = dict(job.meta)
    meta["face"] = {**meta["face"], "crop_scale": args.crop_scale,
                    "interpolated_frames": int(interp.sum()), "pad_fraction_max": round(float(pads.max()), 3)}
    job.dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if job.kind == "sentence":
        # LRS3-style text: transcript, then "WORD START END SCORE" (seconds from clip start, aligner score).
        lines = [f"Text:  {meta['text']}", f"Conf:  {meta['align_score']}", "", "WORD START END SCORE"]
        lines += [f"{w['word']} {w['start']:.2f} {w['end']:.2f} {w['score']:.3f}" for w in meta["words"]]
        job.dst.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_video(vid: str, kind: str, args, vocab) -> tuple[str, int, list[dict]]:
    fd = FaceData(vid)
    jobs, rejects = plan_words(vid, fd, args, vocab) if kind == "words" else plan_sentences(vid, fd, args)
    if not args.overwrite:
        jobs = [j for j in jobs if not j.dst.exists()]
    render(vid, fd, jobs, args)
    return vid, len(jobs), rejects


def run(args: argparse.Namespace, kind: str) -> None:
    vocab = None
    if kind == "words" and args.vocab:
        vocab = {l.split("\t")[0].strip() for l in Path(args.vocab).read_text(encoding="utf-8").splitlines() if l.strip()}
    vids = sorted(f.stem for f in ALIGN.glob("*.json") if (FACES / f"{f.stem}.npz").exists())
    if args.ids:
        vids = [v for v in vids if v in args.ids]
    skipped = sorted({f.stem for f in ALIGN.glob("*.json")} - set(vids))
    if skipped:
        print(f"[{kind}] skipping {len(skipped)} aligned video(s) without face data (run bvsr.faces): {skipped[:5]}")
    if args.clean:
        for v in vids:
            for p in ((CLIPS / "words").glob(f"*/{v}_*") if kind == "words" else [CLIPS / "sentences" / v]):
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    workers = min(args.workers, max(1, len(vids)))
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(process_video, vids, [kind] * len(vids), [args] * len(vids), [vocab] * len(vids)))
    else:
        results = [process_video(v, kind, args, vocab) for v in vids]

    log = CLIPS / f"rejects_{kind}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    reasons: Counter[str] = Counter()
    written = 0
    with log.open("w", encoding="utf-8") as fh:
        for vid, n, rejects in results:
            written += n
            for r in rejects:
                reasons[r["reason"]] += 1
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = written + sum(reasons.values())
    summary = {"kind": kind, "videos": len(vids), "candidates": total, "written": written,
               "rejected": dict(reasons.most_common()), "args": {k: v for k, v in vars(args).items() if k != "func"}}
    (CLIPS / f"summary_{kind}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{kind}] {len(vids)} videos: {total} candidates -> {written} clips written")
    for r, n in reasons.most_common():
        print(f"  rejected {r:16s} {n}")
    print(f"  log: {log}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def word_filters(sp):
        # Aligner scores are mean log-probs. In testing, a hallucinated word scored -3.98 and real words -0.02..-0.75.
        sp.add_argument("--min-score", type=float, default=-1.0)
        sp.add_argument("--min-dur", type=float, default=0.12)
        sp.add_argument("--max-dur", type=float, default=1.0, help="LRW clips are 1.16 s, longer words get cut")
        sp.add_argument("--min-chars", type=int, default=2, help="min Unicode code points in the label")

    def face_filters(sp):
        sp.add_argument("--ids", nargs="*")
        sp.add_argument("--min-face", type=float, default=100, help="min face width in source px (LRW-Persian: 100)")
        sp.add_argument("--max-yaw", type=float, default=30, help="deg (LRW-Persian: 30)")
        sp.add_argument("--max-pitch", type=float, default=40, help="deg (LRW-Persian: 40)")
        sp.add_argument("--min-other-frames", type=int, default=3,
                        help="a second face counts only if seen on this many frames (drops 1-frame false detections)")
        sp.add_argument("--allow-multi-face", action="store_true",
                        help="keep clips with other faces (only once active-speaker detection exists)")
        sp.add_argument("--max-frame-change", type=float, default=0.25,
                        help="max histogram distance between consecutive frames (rejects fades/flashes)")
        sp.add_argument("--crop-scale", type=float, default=1.6, help="crop side = scale x face-landmark extent")
        sp.add_argument("--smooth", type=int, default=5, help="frames to average the crop centre over")
        sp.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better, bigger)")
        sp.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
        sp.add_argument("--overwrite", action="store_true", help="re-render clips that already exist")
        sp.add_argument("--clean", action="store_true", help="delete this kind's existing clips for these videos first")

    ps = sub.add_parser("stats")
    word_filters(ps)
    ps.add_argument("--show", type=int, default=40)
    ps.set_defaults(func=stats)

    pw = sub.add_parser("words")
    word_filters(pw)
    face_filters(pw)
    pw.add_argument("--vocab", help="file with one word per line (first column of word_counts.tsv works)")
    pw.add_argument("--max-per-video", type=int, default=5, help="clips of one word per video")
    pw.add_argument("--max-missing", type=int, default=2, help="face frames that may be missing (interpolated)")
    pw.add_argument("--word-size", type=int, default=256, help="output px (LRW: 256)")
    pw.set_defaults(func=None)

    pc = sub.add_parser("sentences")
    face_filters(pc)
    pc.add_argument("--min-score", type=float, default=-1.0)
    pc.add_argument("--min-len", type=float, default=1.0)
    pc.add_argument("--max-len", type=float, default=20.0)
    pc.add_argument("--max-missing-frac", type=float, default=0.05, help="fraction of face frames that may be missing")
    pc.add_argument("--sentence-size", type=int, default=224, help="output px (LRS3: 224)")
    pc.set_defaults(func=None)

    args = p.parse_args()
    ensure_dirs()
    if args.cmd == "stats":
        stats(args)
    else:
        run(args, args.cmd)


if __name__ == "__main__":
    main()
