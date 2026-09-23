# bvsr: Bengali VSR dataset builder

Builds LRW-style word clips and LRS-style sentence clips from YouTube. See [PLAN.md](PLAN.md) for the research background and roadmap.

## Status (v0.1)
| Stage | What it does | Status |
|---|---|---|
| 1 `bvsr.download` | links (videos/playlists/channels) -> manifest -> rate-limited downloads, H.264 <=720p + 16 kHz wav | working |
| 2 `bvsr.transcribe` | Silero VAD -> Bengali Whisper ASR -> MMS forced alignment (word start/end/confidence) | working |
| 3 face / active speaker | face tracking, TalkNet active-speaker check, mouth ROI crop | **not started** |
| 4 `bvsr.cut` | LRW-style word clips (29 frames @ 25 fps, word centred) + LRS-style sentence clips | working (full-frame, no crop yet) |
| 5-7 | vocabulary selection, speaker-disjoint splits, human verification | not started |

Tested end to end on one 3:53 Bengali talking-head video on an M2 MacBook (8 GB): 66 s for ASR + alignment,
peak ~2.2 GB RAM, 569 aligned words -> 422 word clips (296 word classes) + 15 sentence clips.

## Setup
```bash
brew install ffmpeg deno          # deno: JS runtime yt-dlp needs for YouTube
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run
```bash
# 1. links -> manifest -> downloads (data/raw/<id>/)
.venv/bin/python -m bvsr.download expand links/my_links.txt --tag news
.venv/bin/python -m bvsr.download fetch --max-videos 20

# 2. VAD + Bengali ASR + word alignment (data/align/<id>.json)
.venv/bin/python -m bvsr.transcribe              # --device cuda on the GPU box, --use-subs to prefer human subtitles

# 4. clips
.venv/bin/python -m bvsr.cut stats               # data/word_counts.tsv
.venv/bin/python -m bvsr.cut words --vocab data/vocab.txt
.venv/bin/python -m bvsr.cut sentences

# browse the output (local page, links to the clips on disk)
.venv/bin/python -m bvsr.viewer && open data/viewer.html
```

## Output format
```
data/align/<video>.json                      chunks with text + per-word {word, label, start, end, score}
data/clips/words/<word>/<video>_<ms>.mp4     29 frames @ 25 fps, word centred
data/clips/words/<word>/<video>_<ms>.json    {word, video_id, word_start, word_end, duration, score, start_frame, end_frame}
data/clips/sentences/<video>/<n>.mp4 + .txt  LRS3-style: transcript + "WORD START END SCORE" lines
```
`score` is the aligner's mean log-probability per word (0 = certain). The default filter `--min-score -1.0`
drops words ASR hallucinated (in testing a made-up word scored -3.98 vs -0.02 to -0.75 for real ones).

Downloaded videos and generated clips stay in `data/` (git-ignored): the footage belongs to its creators.

Every stage can be resumed: finished videos are skipped when a stage runs again.

## Rate limits
The defaults stay well below YouTube's limit of about 300 videos/hour for guest sessions: a 10–20 s sleep between videos, 0.75 s between requests, at most 150 videos per hour, and an exponential back-off on `Sign in to confirm you're not a bot` or HTTP 429. PLAN.md §4 covers cookies and PO tokens. If you use cookies, use a dedicated account, because yt-dlp warns that accounts can be banned.

## License
Code: MIT (see [LICENSE](LICENSE)). The license covers this code only, not any video or data it downloads.
