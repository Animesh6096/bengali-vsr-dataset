"""Stage 1: expand a links file into a manifest and download videos politely.

Usage:
    python -m bvsr.download expand links/bengali_news.txt
    python -m bvsr.download fetch  [--max-videos 20] [--cookies-from-browser firefox]

The links file holds one URL per line (video, playlist, or channel `/videos` tab).
Anything after '#' is a comment. Every video lands in data/raw/<id>/.
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import yt_dlp
from yt_dlp.utils import DownloadError

from .common import ARCHIVE, DOWNLOAD_LOG, MANIFEST, RAW, append_jsonl, ensure_dirs, extract_wav, read_jsonl

# Messages YouTube returns when a session/IP is being throttled. Seeing one means back off, not retry.
RATE_LIMIT_MARKERS = (
    "Sign in to confirm you",
    "HTTP Error 429",
    "rate-limited",
    "This content isn't available, try again later",
)
# Permanent per-video failures: log and move on.
PERMANENT_MARKERS = (
    "Private video",
    "Video unavailable",
    "members-only",
    "Join this channel",
    "This live event",
    "Premieres in",
    "age-restricted",
    "Sign in to confirm your age",
)


def read_links(path: str) -> list[str]:
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                urls.append(line)
    return urls


def _flatten(info: dict, source: str):
    """Yield leaf video entries from a (possibly nested) playlist/channel info dict."""
    if info.get("_type") in ("playlist", "multi_video") or "entries" in info:
        for entry in info.get("entries") or []:
            if entry:
                yield from _flatten(entry, source)
    else:
        yield info


def expand(args: argparse.Namespace) -> None:
    ensure_dirs()
    known = {r["id"] for r in read_jsonl(MANIFEST)}
    opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "sleep_interval_requests": args.sleep_requests,
        "ignoreerrors": True,
    }
    added = skipped = 0
    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in read_links(args.links):
            info = ydl.extract_info(url, download=False)
            if not info:
                print(f"[expand] could not read {url}")
                continue
            for e in _flatten(info, url):
                vid = e.get("id")
                if not vid or vid in known:
                    continue
                dur = e.get("duration")
                live = e.get("live_status")
                if live in ("is_live", "is_upcoming", "post_live"):
                    skipped += 1
                    continue
                # Flat entries sometimes lack duration; keep them and let `fetch` check later.
                if dur is not None and not (args.min_duration <= dur <= args.max_duration):
                    skipped += 1
                    continue
                if "/shorts/" in (e.get("url") or ""):
                    skipped += 1
                    continue
                rec = {
                    "id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": e.get("title"),
                    "channel": e.get("channel") or e.get("uploader") or info.get("channel") or info.get("uploader"),
                    "channel_id": e.get("channel_id") or info.get("channel_id"),
                    "duration": dur,
                    "source": url,
                    "tag": args.tag,
                }
                append_jsonl(MANIFEST, rec)
                known.add(vid)
                added += 1
    print(f"[expand] added {added} videos, skipped {skipped} (duration/live/shorts). manifest: {MANIFEST}")


def _classify(msg: str) -> str:
    if any(m in msg for m in RATE_LIMIT_MARKERS):
        return "rate_limited"
    if any(m in msg for m in PERMANENT_MARKERS):
        return "unavailable"
    return "error"


def _ydl_opts(args: argparse.Namespace) -> dict:
    opts = {
        # <=720p is enough for lip ROIs (LRS3 face crops are 224x224) and keeps disk use sane.
        # Prefer H.264: AV1/VP9 decode poorly (or not at all) in OpenCV/MediaPipe builds used later.
        "format": "bv*+ba/b",
        "format_sort": [f"res:{args.max_height}", "vcodec:h264", "acodec:m4a", "ext:mp4:m4a"],
        "merge_output_format": "mp4",
        "outtmpl": str(RAW / "%(id)s" / "%(id)s.%(ext)s"),
        "writeinfojson": True,
        "writesubtitles": True,               # human-made Bengali subs, if any, are a better transcript than ASR
        "writeautomaticsub": args.auto_subs,
        "subtitleslangs": ["bn", "bn-.*"],
        "subtitlesformat": "vtt",
        "download_archive": str(ARCHIVE),
        # Same values as yt-dlp's `-t sleep` preset.
        "sleep_interval": args.sleep_min,
        "max_sleep_interval": args.sleep_max,
        "sleep_interval_requests": args.sleep_requests,
        "sleep_interval_subtitles": 5,
        "retries": 10,
        "fragment_retries": 10,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if args.limit_rate:
        opts["ratelimit"] = args.limit_rate
    if args.cookies:
        opts["cookiefile"] = args.cookies
    if args.cookies_from_browser:
        opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    return opts


def fetch(args: argparse.Namespace) -> None:
    ensure_dirs()
    manifest = read_jsonl(MANIFEST)
    done = {r["id"] for r in read_jsonl(DOWNLOAD_LOG) if r["status"] in ("ok", "unavailable")}
    todo = [r for r in manifest if r["id"] not in done]
    if args.tag:
        todo = [r for r in todo if r.get("tag") == args.tag]
    if args.max_videos:
        todo = todo[: args.max_videos]
    print(f"[fetch] {len(todo)} videos to download ({len(done)} already done)")

    recent: deque[float] = deque()     # download timestamps within the last hour
    consecutive_rl = 0
    with yt_dlp.YoutubeDL(_ydl_opts(args)) as ydl:
        for i, rec in enumerate(todo, 1):
            # Hourly cap, kept below YouTube's ~300 videos/hour guest-session limit.
            now = time.time()
            while recent and now - recent[0] > 3600:
                recent.popleft()
            if len(recent) >= args.per_hour:
                wait = 3600 - (now - recent[0]) + 5
                print(f"[fetch] hourly cap {args.per_hour} reached, sleeping {wait / 60:.1f} min")
                time.sleep(wait)

            vid = rec["id"]
            print(f"[fetch] ({i}/{len(todo)}) {vid} {rec.get('title') or ''}")
            try:
                ydl.download([rec["url"]])
            except DownloadError as e:
                msg = str(e)
                status = _classify(msg)
                append_jsonl(DOWNLOAD_LOG, {"id": vid, "status": status, "error": msg[:500], "t": time.time()})
                if status != "rate_limited":
                    print(f"[fetch]   {status}: {msg[:160]}")
                    continue
                consecutive_rl += 1
                if consecutive_rl >= args.max_rate_limit_hits:
                    print("[fetch] repeatedly rate-limited; stopping. Resume later (progress is saved).")
                    return
                backoff = min(args.backoff * 2 ** (consecutive_rl - 1), 4 * 3600)
                print(f"[fetch]   rate-limited ({consecutive_rl}x), backing off {backoff / 60:.0f} min")
                time.sleep(backoff)
                continue

            consecutive_rl = 0
            recent.append(time.time())
            video = RAW / vid / f"{vid}.mp4"
            if not video.exists():
                # Already in the download archive from an earlier run, or merged under another extension.
                cands = sorted((RAW / vid).glob(f"{vid}.*"))
                video = next((c for c in cands if c.suffix in (".mp4", ".mkv", ".webm")), None)
            if video is None:
                append_jsonl(DOWNLOAD_LOG, {"id": vid, "status": "error", "error": "no video file after download"})
                continue
            wav = RAW / vid / f"{vid}.wav"
            if not wav.exists():
                extract_wav(video, wav)
            append_jsonl(DOWNLOAD_LOG, {"id": vid, "status": "ok", "video": str(video.relative_to(RAW)), "t": time.time()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("expand", help="resolve links (videos/playlists/channels) into data/manifest.jsonl")
    pe.add_argument("links")
    pe.add_argument("--tag", default=None, help="label stored with each video, e.g. 'news' or a channel name")
    pe.add_argument("--min-duration", type=float, default=60)
    pe.add_argument("--max-duration", type=float, default=3 * 3600)
    pe.add_argument("--sleep-requests", type=float, default=0.75)
    pe.set_defaults(func=expand)

    pf = sub.add_parser("fetch", help="download videos listed in the manifest")
    pf.add_argument("--max-videos", type=int, default=0, help="0 = all")
    pf.add_argument("--tag", default=None, help="only fetch videos with this tag")
    pf.add_argument("--max-height", type=int, default=720)
    pf.add_argument("--auto-subs", action="store_true", help="also save YouTube auto-generated Bengali captions")
    pf.add_argument("--sleep-min", type=float, default=10)
    pf.add_argument("--sleep-max", type=float, default=20)
    pf.add_argument("--sleep-requests", type=float, default=0.75)
    pf.add_argument("--per-hour", type=int, default=150, help="max videos per rolling hour")
    pf.add_argument("--limit-rate", default=None, help="bandwidth cap, e.g. 5M")
    pf.add_argument("--backoff", type=float, default=1800, help="first back-off in seconds after a rate-limit")
    pf.add_argument("--max-rate-limit-hits", type=int, default=3)
    pf.add_argument("--cookies", default=None, help="Netscape cookies.txt (see README before using)")
    pf.add_argument("--cookies-from-browser", default=None, help="e.g. firefox, chrome")
    pf.set_defaults(func=fetch)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
