"""Build a local HTML page to browse the generated clips (for spot-checking labels).

    python -m bvsr.viewer            # writes data/viewer.html, open it in a browser

The page links to clips under data/clips/ by relative path, so it only works next to the
data folder and never embeds or uploads any video.
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict

from .common import CLIPS, DATA

CSS = """
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e3e3e6; --accent:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#141416; --fg:#ececef; --muted:#9a9aa2; --card:#1e1e22; --line:#2e2e34; --accent:#ff6b5b; } }
body { margin:0; font:15px/1.5 system-ui, "Noto Sans Bengali", sans-serif; background:var(--bg); color:var(--fg); }
main { max-width:1200px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
.muted { color:var(--muted); font-size:13px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px; }
.card video { width:100%; border-radius:4px; background:#000; aspect-ratio:1/1; object-fit:contain; }
.word { font-size:20px; font-weight:600; }
.sent { display:grid; grid-template-columns:minmax(200px, 320px) 1fr; gap:16px; margin-bottom:16px; }
.sent video { width:100%; border-radius:6px; background:#000; }
.words span { display:inline-block; margin:2px; padding:2px 6px; border:1px solid var(--line); border-radius:4px; cursor:pointer; font-size:15px; }
.words span.low { color:var(--accent); border-color:var(--accent); }
.words span small { color:var(--muted); font-size:11px; margin-left:4px; }
@media (max-width:600px) { .sent { grid-template-columns:1fr; } }
"""

JS = """
document.querySelectorAll('.words span').forEach(s => s.addEventListener('click', () => {
  const v = s.closest('.sent').querySelector('video');
  v.currentTime = parseFloat(s.dataset.t); v.play();
  setTimeout(() => v.pause(), (parseFloat(s.dataset.e) - parseFloat(s.dataset.t)) * 1000 + 150);
}));
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-words", type=int, default=40, help="word classes to show (most frequent first)")
    p.add_argument("--per-word", type=int, default=6)
    p.add_argument("--low-score", type=float, default=-1.0, help="highlight words scoring below this")
    args = p.parse_args()

    by_word: dict[str, list[dict]] = defaultdict(list)
    for meta_path in sorted((CLIPS / "words").glob("*/*_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["_clip"] = meta_path.with_suffix(".mp4").relative_to(DATA).as_posix()
        by_word[meta["word"]].append(meta)
    words = sorted(by_word, key=lambda w: -len(by_word[w]))[: args.max_words]
    n_clips = sum(len(v) for v in by_word.values())

    out = [f"<!doctype html><html lang='bn'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>BVSR sample</title><style>{CSS}</style></head><body><main>",
           "<h1>Bengali VSR dataset: sample output</h1>",
           f"<p class='muted'>{n_clips} word clips across {len(by_word)} word classes. "
           f"Word clips follow LRW: 29 frames at 25 fps (1.16 s), 256x256, face centred, word in the middle; "
           f"'frames' marks where the aligner places the word, yaw is the largest head turn in the clip. "
           f"Showing the {len(words)} most frequent words.</p>",
           "<h2>Word-level clips (LRW style)</h2>"]
    for w in words:
        out.append(f"<h3 class='word'>{html.escape(w)} <span class='muted'>({len(by_word[w])} clips)</span></h3><div class='grid'>")
        for m in by_word[w][: args.per_word]:
            out.append(
                f"<div class='card'><video src='{html.escape(m['_clip'])}' controls loop muted preload='metadata'></video>"
                f"<div class='muted'>{m['video_id']} @ {m['word_start']:.2f}s<br>"
                f"dur {m['duration']:.2f}s · frames {m['start_frame']}–{m['end_frame']} · score {m['align_score']:.2f}"
                f" · yaw {m['face']['yaw_abs']:.0f}°</div></div>")
        out.append("</div>")

    out.append("<h2>Sentence-level clips (LRS style)</h2>"
               "<p class='muted'>Click a word to play just that word. Red = low aligner confidence.</p>")
    for meta_path in sorted((CLIPS / "sentences").glob("*/*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        spans = []
        for w in meta["words"]:
            low = " class='low'" if w["score"] < args.low_score else ""
            spans.append(f"<span{low} data-t='{w['start']:.2f}' data-e='{w['end']:.2f}'>{html.escape(w['word'])}"
                         f"<small>{w['start']:.2f}s</small></span>")
        clip = meta_path.with_suffix(".mp4").relative_to(DATA).as_posix()
        out.append(f"<div class='sent'><video src='{html.escape(clip)}' controls preload='metadata'></video>"
                   f"<div><div class='muted'>{meta_path.parent.name}/{meta_path.stem} · {meta['n_frames']} frames"
                   f" · mean score {meta['align_score']}</div>"
                   f"<div class='words'>{''.join(spans)}</div></div></div>")

    out.append(f"</main><script>{JS}</script></body></html>")
    dst = DATA / "viewer.html"
    dst.write_text("\n".join(out), encoding="utf-8")
    print(f"[viewer] {dst}")


if __name__ == "__main__":
    main()
