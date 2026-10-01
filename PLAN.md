# LRW-BN: an in-the-wild Bengali lip-reading dataset built from YouTube

Goal: build a word-level (LRW-style) and sentence-level (LRS-style) Bengali visual speech
recognition dataset from YouTube, using a pipeline that also works for other languages.

Why: our thesis (*A Transfer Learning Framework for Cross-Script VSR*) got 79.77% on LRW-AR
but only 45.83% on LipBengal, and pointed to data scarcity as the main cause. The thesis
recommended at least 100 samples per word. (Its "20 vs 85 samples per word" figures don't match
the datasets' published sizes; see RESEARCH_LOG §2 before quoting them.) Existing Bengali datasets
are prompted recordings: BenAV (50 words, 128 speakers) and LipBengal (150 students of one
institution, 92% male). No in-the-wild, word-level Bengali VSR dataset was found (RESEARCH_LOG §7).

---

## 1. How existing datasets were built

| Dataset | Language | Source | How labels and clips were made | Scale |
|---|---|---|---|---|
| **LRW** (Chung & Zisserman 2016) | English | BBC TV | Subtitles (OCR'd from broadcast bitmaps) force-aligned with the Penn Phonetics Lab Forced Aligner, checked against IBM Watson STT. Speaking face chosen by a linear SVM on the frequency spectrum of mouth openness. Face cropped around the mouth from landmarks. Clips are **29 frames (1.16 s), 256×256, 25 fps**, word in the middle; metadata gives the word duration. Vocabulary: the 500 most frequent words of **5–10 characters**. Test set checked by hand. | 500 words; 800–1000 train, 50 val and 50 test clips per word; split **by broadcast date**, with a one-week gap before test |
| **LRS2 / LRS3** | English | BBC / TED + TEDx on YouTube | Same VGG pipeline (shot detection, face tracking, forced alignment, SyncNet). LRS3 is released for research under CC BY 4.0; copyright stays with the video owners. | LRS3: 400+ h, 5,594 talks |
| **LRW-1000 / CAS-VSR-W1k** | Mandarin | 26 TV sources, 51 programs, 500+ h raw | "Naturally distributed": the number of clips per class follows real word frequency, so classes are imbalanced | 1,000 classes, 718,018 clips, 2,000+ speakers |
| **LRW-AR** | Arabic | News programs on YouTube | Automated pipeline (few details published) | 100 words, 20,000 clips, 36 speakers |
| **LRW-Persian** (2025) | Persian | 67 TV programs, 1,989 h | **VOSK** ASR. Words kept if ≥4 characters and confidence >0.9, and clips are under 1.5 s. **TalkNet** active-speaker detection. Face ≥100×100 px, DeepFace >0.75, MobileNetV2 mask filter, pose filter (\|yaw\|>30° or \|pitch\|>40° removed), MediaPipe FaceLandmarker metadata. Vocabulary = top 2,500 words per program, intersected across channels, then pruned by hand. **Program-disjoint** train/test split. | 743 words, 414,308 clips, split 78/22 |
| **VoxCeleb / AVSpeech** | multi | YouTube | Released as **YouTube IDs + timestamps**, not as media files | — |

What we take from these:
1. **Labels come from audio.** Every in-the-wild dataset gets its words from subtitles or ASR, forced-aligns them to the audio, and cuts the video at those timestamps.
2. **Active-speaker detection is required.** On-screen faces are often not the person speaking (B-roll, reaction shots, voice-over). LRW used a mouth-openness frequency classifier; the later LRS pipeline (CVPR 2017) used SyncNet; LRW-Persian used TalkNet.
3. **Splits must not share speakers or programs.** LRW split by date and LRW-Persian by program. A random split lets the same anchor appear in train and test, which inflates results.
4. **Release metadata, not media** (VoxCeleb / AVSpeech model). This limits copyright exposure.

## 2. Bengali-specific tool choices

| Need | Choice | Why |
|---|---|---|
| ASR | `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` (Apache-2.0, 0.8B) | Strongest off-the-shelf model in three 2026 papers on the DL Sprint 4.0 Bengali-Loop long-form benchmark (YouTube, mostly drama): **34.07% WER**, vs 50.67% for Hishab TITU-BN; fine-tuned systems reached 24–27%. About 1 word in 3 is wrong, so labels need human checking. It can be swapped with `--asr-model`. |
| Better transcripts | Human-made `bn` YouTube subtitles when present (`--use-subs`) | LRW itself was built from subtitles |
| Word timestamps | MMS forced aligner (`MahmoudAshraf/mms-300m-1130-forced-aligner`, 1130 languages) via `ctc-forced-aligner`, with **uroman** romanisation | torchaudio's `forced_align` / `MMS_FA` were deprecated in 2.8 and removed in 2.9, so we use the standalone package |
| VAD / chunking | Silero VAD | Keeps ASR input under 20 s and removes silence and music, which reduces Whisper hallucinations |
| Active speaker | TalkNet-ASD (MIT) or LR-ASD; SyncNet confidence as a cross-check | TalkNet: LRW-Persian. SyncNet: the LRS pipeline (CVPR 2017). |
| Face / lips | MediaPipe FaceLandmarker (+ FAN fallback) | Same stack as the thesis preprocessing (ROI = 1.5 × mouth width, 1.8 × mouth height) |
| Speaker IDs for splits | Face-embedding clustering (InsightFace/ArcFace) | Needed for speaker-disjoint splits |
| Label normalisation | NFC, strip ZWJ/ZWNJ and punctuation, Bengali letters only | Matches the thesis finding that Unicode normalisation matters a lot for Bengali |

## 3. Pipeline

```
links.txt ─► [1 download] ─► data/raw/<id>/{mp4,wav,info.json,bn.vtt}
                 │
                 ▼
           [2 VAD → ASR/subs → MMS forced alignment] ─► data/align/<id>.json
                 │
                 ▼
           [3 face track → quality filters → (active speaker ◄── NEXT)]
                 │
                 ▼
           [4 cut] ─► clips/words/<word>/<id>_<ms>.mp4 (29 f, centred)
                  └─► clips/sentences/<id>/<n>.mp4 + LRS3-style .txt
                 │
                 ▼
           [5 vocab selection] ─► [6 speaker/channel-disjoint splits] ─► [7 human check of test set] ─► [8 release]
```

| # | Stage | Status | Notes |
|---|---|---|---|
| 0 | **Source curation** | you | Talking-head content first: news anchors, talk shows, lectures, speeches, interviews, vlogs. Include both Bangladeshi and West Bengal channels for dialect coverage. Track the channel for each video. |
| 1 | Download (`bvsr.download`) | **done** | Expands playlists and channels, filters by duration, skips live videos and Shorts, stays under the rate limit (§4), and can resume |
| 2 | Transcribe + align (`bvsr.transcribe`) | **done** | Per-word start, end and log-prob score |
| 3a | Faces (`bvsr.faces`) | **done** | Per frame on a 25 fps grid: MediaPipe 478 landmarks, head pose, mouth sharpness, IoU face tracks, histogram shot cuts. Clips are rejected on shot cut, broken track, more than one face, face <100 px, or \|yaw\|>30° / \|pitch\|>40° (LRW-Persian limits). |
| 3b | Active speaker | next | TalkNet (or SyncNet confidence): keep a word only if the tracked face is the one speaking. Needed before multi-person sources (news panels, talk shows) are used; until then multi-face clips are rejected. |
| 4 | Cut clips (`bvsr.cut`) | **done** | Face-centred 256×256 word clips (LRW) and 224×224 sentence clips (LRS3), with per-frame landmarks and crop boxes. A mouth ROI (96×96) is **preprocessing**, left to users or a separate tool, since LRW does not ship mouth crops either. |
| 5 | Vocabulary | todo | Use `cut stats` counts. Candidate rules: ≥2–3 graphemes, ≥N distinct videos/channels per word, and a target of **500 words × ≥200 clips** (LRW-style, balanced). A second, naturally distributed release like LRW-1000 is optional. Decide how to handle homophenes (e.g. কলম / গরম in the thesis). |
| 6 | Splits | todo | Channel-disjoint (or at least speaker-disjoint) val/test, with a clip budget per word. Record age, gender and pose metadata as LRW-Persian does. |
| 7 | Verification | todo | Bengali speakers check every **test** clip, using a small review UI: play the clip, then accept, reject or correct the label. Measure ASR/alignment precision on a 500-clip sample to tune `--min-score`. |
| 8 | Release | todo | Publish YouTube IDs, timestamps, labels, crop boxes and scripts. Share clips only on request under a research licence, and set up a takedown address. Get university ethics approval, because faces are biometric data. |

### How much video we need (estimate)
LRW-Persian got 414k clips from 1,989 h, about 208 usable clips per hour of source. If
Bengali yields about the same, **500 words × 200 clips = 100k clips needs about 500 h** of
source video, or roughly 1,500 videos of 20 min each. At 720p that is about 0.5 TB of raw
video (estimate). The slow parts will be **GPU time for ASR and active-speaker detection**,
not the download.

### Compute
The pipeline runs on this Mac (M2, 8 GB, MPS) for development only. Bulk ASR, alignment
and TalkNet should run on the lab GPU (the RTX 3060 from the thesis or better). Everything
takes `--device cuda`.

## 4. YouTube rate limits

Facts, from the yt-dlp wiki and source as of Sept 2026:

- **Guest session: about 300 videos/hour** (about 1,000 webpage/player requests/hour).
- **Logged-in session: about 2,000 videos/hour** (about 4,000 requests/hour), but *"By using your account with yt-dlp, you run the risk of it being banned (temporarily or permanently)."*
- yt-dlp's `-t sleep` preset is `--sleep-requests 0.75 --sleep-interval 10 --max-sleep-interval 20 --sleep-subtitles 5`.
- Since **yt-dlp 2025.11.12**, full YouTube support needs an external JS runtime. **Deno** is enabled by default and is installed here.
- Some IPs get "Sign in to confirm you're not a bot". yt-dlp then needs a **PO token**, which it cannot generate. The usual fix is the `bgutil-ytdlp-pot-provider` plugin.

What `bvsr.download` does:
- It uses the `-t sleep` values, a rolling **150 videos/hour** cap (half the guest limit), and a download archive so a run can stop and resume.
- On rate-limit messages (`Sign in to confirm…`, HTTP 429) it **backs off exponentially** (30 min, then 60, …) and stops after 3 hits. Progress is kept in `data/download_log.jsonl`.
- It downloads ≤720p only, one file per video.

Can we bypass the limit? At our scale we don't need to: about 1,500 videos at 150/hour is
about 10 hours spread over a few days. If we do hit limits:
1. Keep yt-dlp updated (`pip install -U yt-dlp`). Most "blocks" are YouTube changes that a newer release already handles.
2. Install the PO-token provider plugin if you see the bot check.
3. Use cookies from a **dedicated project Google account**, not a personal one (the ban risk is on the account). This raises the limit to about 2,000/hour. Export cookies from a private window as the yt-dlp FAQ describes.
4. Run from the university network, ideally with IT's approval, and spread downloads over days.

I won't set up rotating proxy pools or multiple accounts to get around blocks. That breaks
YouTube's Terms of Service, gets the IP or account banned faster, and is hard to defend in a
paper's ethics section. Downloading for research datasets is normally justified through
research exceptions, releasing IDs only, and ethics approval. Check with the university
before publishing.

## 5. Next steps
1. **You:** collect 20–50 links (channels or playlists) of Bengali talking-head content in `links/`.
2. Pilot: download about 20 videos → transcribe → hand-check 200 word clips → tune `--min-score` and the VAD settings.
3. Build stage 3 (TalkNet + MediaPipe crop) and test it on the pilot.
4. Scale up on the GPU and apply the vocab, split and verification rules.
5. Once Bengali works, add languages by changing `--asr-model` and the aligner `language=` ISO code.
