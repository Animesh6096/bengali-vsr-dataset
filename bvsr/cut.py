"""Stage 4: cut LRW-style word clips and LRS-style sentence clips from aligned videos.

Word clips follow LRW: 29 frames at 25 fps (1.16 s), target word centred, metadata
records the word's duration so start/end frames can be recovered.

    python -m bvsr.cut stats                          # word counts -> data/word_counts.tsv
    python -m bvsr.cut words --vocab data/vocab.txt   # cut clips for chosen words
    python -m bvsr.cut sentences

NOTE: clips are full-frame for now. Face tracking + active-speaker filtering (stage 3)
will add a per-frame speaking-face box; until then use single-speaker sources.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import ALIGN, CLIPS, DATA, RAW, ensure_dirs, probe_duration, run_ffmpeg

FPS = 25
WORD_FRAMES = 29                     # LRW clip length
WORD_SPAN = WORD_FRAMES / FPS        # 1.16 s


def conf_ok(score: float | None, min_score: float) -> bool:
    return score is not None and score >= min_score


def iter_words(min_score: float, min_dur: float, max_dur: float, min_chars: int):
    for f in sorted(ALIGN.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for ci, chunk in enumerate(data["chunks"]):
            for wi, w in enumerate(chunk["words"]):
                d = w["end"] - w["start"]
                if len(w["label"]) < min_chars or not (min_dur <= d <= max_dur) or not conf_ok(w["score"], min_score):
                    continue
                yield data["video_id"], ci, wi, w


def stats(args: argparse.Namespace) -> None:
    counts: Counter[str] = Counter()
    videos: dict[str, set[str]] = defaultdict(set)
    for vid, _, _, w in iter_words(args.min_score, args.min_dur, args.max_dur, args.min_chars):
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


def _video_path(vid: str) -> Path:
    for ext in (".mp4", ".mkv", ".webm"):
        p = RAW / vid / f"{vid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(vid)


def cut_clip(src: Path, start: float, dur: float, dst: Path, frames: int | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -ss before -i seeks fast; with re-encoding ffmpeg still decodes up to the exact timestamp.
    vf = f"fps={FPS}"
    if frames:
        # Reading exactly `dur` can round to frames-1; read a little extra, cap video frames, trim audio.
        args = ["-ss", f"{start:.3f}", "-t", f"{dur + 0.2:.3f}", "-i", str(src), "-vf", vf,
                "-frames:v", str(frames), "-af", f"atrim=duration={dur:.3f}"]
    else:
        args = ["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-vf", vf]
    args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-ac", "1", "-ar", "16000", str(dst)]
    run_ffmpeg(args)


def words(args: argparse.Namespace) -> None:
    vocab = None
    if args.vocab:
        vocab = {l.split("\t")[0].strip() for l in Path(args.vocab).read_text(encoding="utf-8").splitlines() if l.strip()}
    per_video: Counter[tuple[str, str]] = Counter()
    durations: dict[str, float] = {}
    n = 0
    for vid, ci, wi, w in iter_words(args.min_score, args.min_dur, args.max_dur, args.min_chars):
        label = w["label"]
        if vocab is not None and label not in vocab:
            continue
        # Cap repeats of one word from one video (e.g. a news anchor's catch-phrase) to keep speaker diversity.
        if per_video[(label, vid)] >= args.max_per_video:
            continue
        mid = (w["start"] + w["end"]) / 2
        start = mid - WORD_SPAN / 2
        if vid not in durations:
            durations[vid] = probe_duration(_video_path(vid))
        if start < 0 or start + WORD_SPAN > durations[vid]:
            continue
        name = f"{vid}_{int(w['start'] * 1000):08d}"
        dst = CLIPS / "words" / label / f"{name}.mp4"
        if not dst.exists():
            cut_clip(_video_path(vid), start, WORD_SPAN, dst, frames=WORD_FRAMES)
        meta = {
            "word": label, "surface": w["word"], "video_id": vid, "chunk": ci, "index": wi,
            "word_start": w["start"], "word_end": w["end"], "clip_start": round(start, 3),
            "duration": round(w["end"] - w["start"], 3), "score": w["score"],
            # Frames of the clip that contain the word (LRW gives duration; we store both).
            "start_frame": max(0, round((w["start"] - start) * FPS)),
            "end_frame": min(WORD_FRAMES - 1, round((w["end"] - start) * FPS)),
        }
        dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        per_video[(label, vid)] += 1
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"[words] wrote {n} clips under {CLIPS / 'words'}")


def sentences(args: argparse.Namespace) -> None:
    n = 0
    for f in sorted(ALIGN.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        vid = data["video_id"]
        for ci, c in enumerate(data["chunks"]):
            dur = c["end"] - c["start"]
            if not c["words"] or not conf_ok(c["score"], args.min_score) or not (args.min_len <= dur <= args.max_len):
                continue
            dst = CLIPS / "sentences" / vid / f"{ci:05d}.mp4"
            if not dst.exists():
                cut_clip(_video_path(vid), c["start"], dur, dst)
            # LRS3-style text file: transcript, then one word per line with times relative to the clip.
            lines = [f"Text:  {c['text']}", f"Conf:  {c['score']}", f"Source: {c['source']}", "",
                     "WORD START END SCORE"]
            lines += [f"{w['word']} {w['start'] - c['start']:.2f} {w['end'] - c['start']:.2f} {w['score']:.3f}"
                      for w in c["words"]]
            dst.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n += 1
    print(f"[sentences] wrote {n} clips under {CLIPS / 'sentences'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def word_filters(sp):
        # Aligner scores are mean log-probs; -1.0 ~ 37% avg per-frame prob. Tune on a hand-checked sample.
        sp.add_argument("--min-score", type=float, default=-1.0)
        sp.add_argument("--min-dur", type=float, default=0.12)
        sp.add_argument("--max-dur", type=float, default=1.0, help="LRW clips are 1.16 s, longer words get cut")
        sp.add_argument("--min-chars", type=int, default=2, help="min Unicode code points in the label")

    ps = sub.add_parser("stats")
    word_filters(ps)
    ps.add_argument("--show", type=int, default=40)
    ps.set_defaults(func=stats)

    pw = sub.add_parser("words")
    word_filters(pw)
    pw.add_argument("--vocab", help="file with one word per line (first column of word_counts.tsv works)")
    pw.add_argument("--max-per-video", type=int, default=5)
    pw.add_argument("--limit", type=int, default=0)
    pw.set_defaults(func=words)

    pc = sub.add_parser("sentences")
    pc.add_argument("--min-score", type=float, default=-1.0)
    pc.add_argument("--min-len", type=float, default=1.0)
    pc.add_argument("--max-len", type=float, default=20.0, help="VAD chunks are <= --max-chunk (20 s) anyway")
    pc.set_defaults(func=sentences)

    args = p.parse_args()
    ensure_dirs()
    args.func(args)


if __name__ == "__main__":
    main()
