"""Screen candidate channels before bulk download: how much of their footage is usable?

For each channel (or single video) in a links file, take a few recent videos, download only a
short section of each at low resolution, and measure with the same face analysis as stage 3:
  - share of frames with exactly one face that passes the clip filters
    (width >= 100 px at 720p-equivalent, |yaw| <= 30, |pitch| <= 40),
  - share of frames with several faces / no face,
  - speech share (Silero VAD) and frames that are both speech and a usable single face,
  - shot cuts per minute,
  - the licence YouTube reports (Creative Commons or standard).
It also saves a contact sheet per video so a person can check who is on screen.

    python -m bvsr.screen plan links/candidate_channels.txt --per-channel 3
    python -m bvsr.screen fetch            # polite, resumable section downloads
    python -m bvsr.screen analyze          # CPU, --workers N
    python -m bvsr.screen report           # data/screen/channels.tsv + data/screen/report.html

The links file format is the one `bvsr.download expand` reads; text after '#' is kept as the
channel's label ("name | type | region | licence").
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import subprocess
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .common import DATA, SAMPLE_RATE, append_jsonl, ensure_dirs, read_jsonl

SCREEN = DATA / "screen"
PLAN = SCREEN / "plan.jsonl"
LOG = SCREEN / "fetch_log.jsonl"
RAWS = SCREEN / "raw"
METRICS = SCREEN / "metrics"
SHEETS = SCREEN / "sheets"


def read_links_with_labels(path: str) -> list[tuple[str, str]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        url, _, label = line.partition("#")
        out.append((url.strip(), label.strip()))
    return out


# --- plan ---------------------------------------------------------------------------

def plan(args: argparse.Namespace) -> None:
    import yt_dlp

    ensure_dirs()
    SCREEN.mkdir(parents=True, exist_ok=True)
    planned = {r["id"] for r in read_jsonl(PLAN)}
    ydl = yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "quiet": True, "no_warnings": True,
                            "ignoreerrors": True, "sleep_interval_requests": 1.0,
                            "playlist_items": f"1:{args.scan}"})
    for url, label in read_links_with_labels(args.links):
        name = label.split("|")[0].strip() or url
        info = ydl.extract_info(url, download=False)
        if not info:
            print(f"[plan] could not read {url}")
            continue
        entries = [e for e in (info.get("entries") or [info]) if e and e.get("id")]
        ok = [e for e in entries
              if (e.get("duration") or 0) >= args.min_duration and (e.get("duration") or 0) <= args.max_duration
              and e.get("live_status") not in ("is_live", "is_upcoming", "post_live", "was_live")
              and "/shorts/" not in (e.get("url") or "")]
        if not ok and entries and "entries" not in info:   # single video URL without duration in flat mode
            ok = entries
        # Spread the picks over the scanned list instead of taking the newest few (often the same show).
        k = min(args.per_channel, len(ok))
        picks = [ok[round(i * (len(ok) - 1) / max(1, k - 1))] for i in range(k)] if k else []
        for e in picks:
            if e["id"] in planned:
                continue
            append_jsonl(PLAN, {"id": e["id"], "channel": name, "label": label, "source": url,
                                "title": e.get("title"), "duration": e.get("duration")})
            planned.add(e["id"])
        print(f"[plan] {name}: {len(picks)} of {len(ok)} eligible videos")
    print(f"[plan] {len(planned)} videos planned -> {PLAN}")


# --- fetch --------------------------------------------------------------------------

def section_bounds(duration: float | None, length: float) -> tuple[float, float]:
    """Skip intros: start about a quarter into the video, at least 60 s in."""
    if not duration:
        return 60.0, 60.0 + length
    start = min(max(60.0, 0.25 * duration), max(0.0, duration - length - 10))
    return start, start + length


def fetch(args: argparse.Namespace) -> None:
    import yt_dlp
    from yt_dlp.utils import DownloadError, download_range_func

    from .download import _classify

    RAWS.mkdir(parents=True, exist_ok=True)
    last = {r["id"]: r["status"] for r in read_jsonl(LOG)}          # latest status per video
    done = {v for v, st in last.items() if st in ("ok", "unavailable")}
    todo = [r for r in read_jsonl(PLAN) if r["id"] not in done]
    print(f"[fetch] {len(todo)} sections to download ({len(done)} done)")
    recent: deque[float] = deque()
    rl_hits = 0
    for n, r in enumerate(todo, 1):
        now = time.time()
        while recent and now - recent[0] > 3600:
            recent.popleft()
        if len(recent) >= args.per_hour:
            wait = 3600 - (now - recent[0]) + 5
            print(f"[fetch] hourly cap reached, sleeping {wait / 60:.1f} min", flush=True)
            time.sleep(wait)
        start, end = section_bounds(r.get("duration"), args.section)
        opts = {
            "format": "bv*+ba/b",
            "format_sort": [f"res:{args.height}", "vcodec:h264", "acodec:m4a", "ext:mp4:m4a"],
            "merge_output_format": "mp4",
            "download_ranges": download_range_func(None, [(start, end)]),
            "outtmpl": str(RAWS / "%(id)s.%(ext)s"),
            "writeinfojson": True,
            "sleep_interval": args.sleep_min, "max_sleep_interval": args.sleep_max,
            "sleep_interval_requests": 0.75,
            "retries": 5, "fragment_retries": 5,
            "quiet": True, "no_warnings": True, "noprogress": True,
        }
        print(f"[fetch] ({n}/{len(todo)}) {r['channel'][:30]} {r['id']} {start:.0f}-{end:.0f}s", flush=True)
        t_start = time.time()
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={r['id']}"])
        except DownloadError as e:
            status = _classify(str(e))
            append_jsonl(LOG, {"id": r["id"], "status": status, "error": str(e)[:300], "t": time.time()})
            if status == "rate_limited":
                rl_hits += 1
                if rl_hits >= 3:
                    print("[fetch] rate-limited 3 times; stopping. Re-run later to resume.")
                    return
                backoff = min(1800 * 2 ** (rl_hits - 1), 4 * 3600)
                print(f"[fetch]   rate-limited, backing off {backoff / 60:.0f} min", flush=True)
                time.sleep(backoff)
            else:
                print(f"[fetch]   {status}: {str(e)[:120]}")
            continue
        took = time.time() - t_start
        recent.append(time.time())
        video = RAWS / f"{r['id']}.mp4"
        got = _duration(video)
        if got < 0.8 * (end - start):
            # Seen while YouTube was throttling: ffmpeg ends early and leaves a short file.
            video.unlink(missing_ok=True)
            append_jsonl(LOG, {"id": r["id"], "status": "truncated", "got_s": got, "took_s": round(took), "t": time.time()})
            print(f"[fetch]   truncated ({got:.0f} s of {end - start:.0f} s); will retry next run", flush=True)
        else:
            rl_hits = 0
            append_jsonl(LOG, {"id": r["id"], "status": "ok", "section": [start, end], "took_s": round(took), "t": time.time()})
        if took > args.max_seconds:
            # No explicit rate-limit message, but a crawl means we're being throttled: stop politely.
            print(f"[fetch] section took {took / 60:.0f} min (> {args.max_seconds / 60:.0f}); likely throttled. "
                  "Stopping; re-run later to resume.", flush=True)
            return


def _duration(path: Path) -> float:
    if not path.exists():
        return 0.0
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


# --- analyze ------------------------------------------------------------------------

def contact_sheet(video: Path, dst: Path, n: int = 8, width: int = 180) -> None:
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(video)], capture_output=True, text=True).stdout.strip() or 0)
    if dur <= 0:
        return
    step = dur / n
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"fps=1/{step:.3f},scale={width}:-2,tile={n}x1", "-frames:v", "1", "-q:v", "4", str(dst)], check=False)


def speech_mask(video: Path, n_frames: int, fps: int) -> np.ndarray:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                          "-f", "s16le", "-"], capture_output=True).stdout
    wav = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    mask = np.zeros(n_frames, bool)
    if len(wav) == 0:
        return mask
    ts = get_speech_timestamps(torch.from_numpy(wav), load_silero_vad(), sampling_rate=SAMPLE_RATE, return_seconds=True)
    for t in ts:
        mask[int(t["start"] * fps):min(n_frames, int(math.ceil(t["end"] * fps)))] = True
    return mask


def analyze_one(vid: str, cfg: dict, filt: dict) -> dict:
    from .faces import FPS, analyze

    video = RAWS / f"{vid}.mp4"
    arrays, summary = analyze(video, cfg)
    n = summary["n_frames"]
    scale = 720.0 / min(summary["width"], summary["height"])     # face sizes as if the video were 720p
    faces_per_frame = np.bincount(arrays["frame"], minlength=n)[:n]
    one = faces_per_frame == 1
    valid = np.zeros(n, bool)
    if len(arrays["frame"]):
        width = (arrays["box"][:, 2] - arrays["box"][:, 0]) * scale
        ok = (width >= filt["min_face"]) & (np.abs(arrays["pose"][:, 1]) <= filt["max_yaw"]) \
            & (np.abs(arrays["pose"][:, 0]) <= filt["max_pitch"])
        valid[arrays["frame"][ok]] = True
    valid &= one
    speech = speech_mask(video, n, FPS)
    info_path = RAWS / f"{vid}.info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    widths = (arrays["box"][:, 2] - arrays["box"][:, 0]) * scale if len(arrays["frame"]) else np.array([0.0])
    m = {
        "id": vid, "seconds": round(n / FPS, 1), "height": summary["height"], "width": summary["width"],
        "one_face": round(float(one.mean()), 3) if n else 0.0,
        "multi_face": round(float((faces_per_frame >= 2).mean()), 3) if n else 0.0,
        "no_face": round(float((faces_per_frame == 0).mean()), 3) if n else 0.0,
        "valid_face": round(float(valid.mean()), 3) if n else 0.0,
        "speech": round(float(speech.mean()), 3) if n else 0.0,
        "usable": round(float((valid & speech).mean()), 3) if n else 0.0,
        "cuts_per_min": round(len(arrays["cuts"]) / max(n / FPS / 60, 1e-6), 2),
        "face_width_median_720p": round(float(np.median(widths)), 1),
        "yaw_abs_median": round(float(np.median(np.abs(arrays["pose"][:, 1]))), 1) if len(arrays["frame"]) else None,
        "license": info.get("license") or "Standard YouTube License",
        "language_tag": info.get("language"), "title": info.get("title"),
        "channel_reported": info.get("channel"), "uploader_id": info.get("uploader_id"),
    }
    SHEETS.mkdir(parents=True, exist_ok=True)
    contact_sheet(video, SHEETS / f"{vid}.jpg")
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / f"{vid}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def analyze_all(args: argparse.Namespace) -> None:
    from .faces import MODEL, model_sha256

    last = {r["id"]: r["status"] for r in read_jsonl(LOG)}
    ok = [v for v, st in last.items() if st == "ok"]
    todo = [v for v in ok if (RAWS / f"{v}.mp4").exists() and (args.overwrite or not (METRICS / f"{v}.json").exists())]
    if not todo:
        print("[analyze] nothing to do")
        return
    if not MODEL.exists():
        raise SystemExit(f"missing {MODEL}; see README Setup")
    cfg = {"max_faces": 4, "det_conf": 0.5, "min_iou": 0.3, "max_gap": 2, "cut_threshold": 0.35, "model_sha256": model_sha256()}
    filt = {"min_face": args.min_face, "max_yaw": args.max_yaw, "max_pitch": args.max_pitch}
    print(f"[analyze] {len(todo)} sections, {args.workers} worker(s)", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(analyze_one, v, cfg, filt): v for v in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                m = f.result()
                print(f"[analyze] ({i}/{len(todo)}) {m['id']} usable={m['usable']:.2f} one_face={m['one_face']:.2f} "
                      f"multi={m['multi_face']:.2f} speech={m['speech']:.2f} cuts/min={m['cuts_per_min']}", flush=True)
            except Exception as e:  # one broken file must not stop the batch
                print(f"[analyze] ({i}/{len(todo)}) {futs[f]} FAILED: {e}", flush=True)


# --- report -------------------------------------------------------------------------

def report(args: argparse.Namespace) -> None:
    plan_rows = {r["id"]: r for r in read_jsonl(PLAN)}
    by_channel: dict[str, list[dict]] = {}
    for f in sorted(METRICS.glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        p = plan_rows.get(m["id"], {})
        by_channel.setdefault(p.get("channel", "?"), []).append({**m, "label": p.get("label", "")})
    keys = ["usable", "valid_face", "one_face", "multi_face", "no_face", "speech", "cuts_per_min", "face_width_median_720p"]
    rows = []
    for ch, ms in by_channel.items():
        agg = {k: round(float(np.mean([m[k] for m in ms])), 3) for k in keys}
        parts = [x.strip() for x in ms[0]["label"].split("|")]
        rows.append({"channel": ch, "type": parts[1] if len(parts) > 1 else "", "region_guess": parts[2] if len(parts) > 2 else "",
                     "n_videos": len(ms), "cc_videos": sum("Creative" in m["license"] for m in ms),
                     "language_tags": ",".join(sorted({str(m["language_tag"]) for m in ms})), **agg,
                     "ids": ",".join(m["id"] for m in ms), "host_gender_notes": ""})
    rows.sort(key=lambda r: -r["usable"])
    cols = list(rows[0].keys()) if rows else []
    tsv = SCREEN / "channels.tsv"
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    out = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           "<title>Channel screening</title><style>",
           ":root{--bg:#fafafa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e6}",
           "@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#ececef;--muted:#9a9aa2;--line:#2e2e34}}",
           "body{margin:0;font:14px/1.45 system-ui,'Noto Sans Bengali',sans-serif;background:var(--bg);color:var(--fg)}",
           "main{max-width:1500px;margin:0 auto;padding:20px 16px}table{border-collapse:collapse;width:100%}",
           "td,th{border-bottom:1px solid var(--line);padding:6px;text-align:left;vertical-align:top}",
           ".m{color:var(--muted);font-size:12px} img{max-width:100%;display:block;margin:2px 0}</style></head><body><main>",
           "<h1>Channel screening</h1><p class='m'>usable = share of frames with exactly one face passing the clip filters "
           "(≥100 px at 720p-equivalent, |yaw| ≤ 30°, |pitch| ≤ 40°) <i>and</i> speech (VAD). Averaged over the sampled sections. "
           "Contact sheets show 8 evenly spaced frames per section. Gender and front-facing must be judged by eye.</p>",
           "<table><tr><th>channel</th><th>usable</th><th>one face</th><th>multi</th><th>speech</th><th>cuts/min</th><th>face px</th><th>CC</th><th>sections</th></tr>"]
    for r in rows:
        sheets = "".join(f"<img src='sheets/{html.escape(i)}.jpg' alt='{html.escape(i)}' loading='lazy'>"
                         f"<div class='m'>{html.escape(i)}</div>" for i in r["ids"].split(","))
        out.append(f"<tr><td><b>{html.escape(r['channel'])}</b><div class='m'>{html.escape(r['type'])} · {html.escape(r['region_guess'])} · lang {html.escape(r['language_tags'])}</div></td>"
                   f"<td>{r['usable']:.2f}</td><td>{r['one_face']:.2f}</td><td>{r['multi_face']:.2f}</td><td>{r['speech']:.2f}</td>"
                   f"<td>{r['cuts_per_min']}</td><td>{r['face_width_median_720p']:.0f}</td><td>{r['cc_videos']}/{r['n_videos']}</td><td style='width:60%'>{sheets}</td></tr>")
    out.append("</table></main></body></html>")
    (SCREEN / "report.html").write_text("\n".join(out), encoding="utf-8")
    print(f"[report] {len(rows)} channels -> {tsv}, {SCREEN / 'report.html'}")
    for r in rows:
        print(f"  {r['usable']:.2f}  one={r['one_face']:.2f} multi={r['multi_face']:.2f} speech={r['speech']:.2f} "
              f"cuts/min={r['cuts_per_min']:5.2f} face={r['face_width_median_720p']:4.0f}px  {r['channel']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("links")
    pp.add_argument("--per-channel", type=int, default=3)
    pp.add_argument("--scan", type=int, default=30, help="newest videos listed per channel to pick from")
    pp.add_argument("--min-duration", type=float, default=300)
    pp.add_argument("--max-duration", type=float, default=3 * 3600)
    pf = sub.add_parser("fetch")
    pf.add_argument("--section", type=float, default=120, help="seconds downloaded per video")
    pf.add_argument("--height", type=int, default=480)
    pf.add_argument("--sleep-min", type=float, default=10)
    pf.add_argument("--sleep-max", type=float, default=20)
    pf.add_argument("--per-hour", type=int, default=150)
    pf.add_argument("--max-seconds", type=float, default=300, help="stop the run if one section takes longer (throttling)")
    pa = sub.add_parser("analyze")
    pa.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    pa.add_argument("--min-face", type=float, default=100)
    pa.add_argument("--max-yaw", type=float, default=30)
    pa.add_argument("--max-pitch", type=float, default=40)
    pa.add_argument("--overwrite", action="store_true")
    sub.add_parser("report")
    args = p.parse_args()
    {"plan": plan, "fetch": fetch, "analyze": analyze_all, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
