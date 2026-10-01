# Research log: Bengali in-the-wild lip-reading dataset (LRW-BN)

This is the project's memory. It records **what was decided and why, what was measured, what was
verified and how, and what is still open**. It is written so that a later work session, a
co-author, or the person writing the paper can pick up without the original conversations.

**Rule: update this file in the same commit as any change to the pipeline.** Add a changelog entry
(§1), update the affected decision (§3) and add any new numbers (§4).

Contents:
1. Changelog
2. Project summary
3. Design decisions (with rationale and rejected alternatives)
4. Experiments and measurements
5. Verification done
6. Known limitations and open issues
7. Related work (verified facts, with sources)
8. Paper notes: draft outline and evidence still needed
9. Reproducibility: versions, model revisions, hashes
10. Bugs found and fixed (engineering notes)

---

## 1. Changelog

| Date | Version | Change |
|---|---|---|
| 2026-09-24 | v0.1 | Stages 1 (download), 2 (VAD → ASR → forced alignment) and 4 (full-frame word/sentence clips) built and tested on one video. Local viewer. Repo published: github.com/Animesh6096/bengali-vsr-dataset (MIT, code only). |
| 2026-09-24 | v0.1 | Full README. |
| 2026-10-01 | v0.2 | Stage 3a: `bvsr.faces` (MediaPipe landmarks, head pose, tracking, shot-cut detection). `bvsr.cut` rewritten: face-centred 256×256 word clips and 224×224 sentence clips rendered on a shared 25 fps grid, per-frame landmarks and crop boxes, rejection log. MediaPipe pinned to 0.10.35. This research log and CLAUDE.md added. |
| 2026-10-01 | docs | Checked three open facts against primary sources (§5, §7): LRW sample format and framing, LipBengal details, and the ASR papers. Corrected: LipBengal is not "controlled" (prompted speech by 150 students); LRW chose its speaker with a mouth-openness classifier, not SyncNet; tugstugi claim narrowed to the Bengali-Loop benchmark (34.07% WER). |
| 2026-10-01 | v0.2.1 | Source research: 37 candidate channels found and checked by metadata only (`docs/SOURCES.md`, `links/candidate_channels.txt`). Copyright tiers, balance targets and a vocabulary plan written down (D15–D17). Novelty check: BenAV and MultiVSR added to related work. Thesis's "20 vs 85 samples per word" flagged as inconsistent with published dataset sizes (§2). `download expand --max-per-link` added. |

---

## 2. Project summary

**Goal.** Build a word-level (LRW-style) and sentence-level (LRS3-style) Bengali visual speech
recognition dataset from YouTube, with a pipeline that can later be reused for other languages.
The target is publication of the dataset as a paper.

**Motivation.** The team's thesis, *A Transfer Learning Framework for Cross-Script Visual Speech
Recognition* (Karim, Bhattacharjee, Fahad, Khan; BRAC University, Dec 2025), reported:
- 79.77% top-1 accuracy on LRW-AR (WER 20.23%), but only 45.83% on LipBengal (WER 54.19%);
- top-10: 94.64% on LRW-AR and 74.73% on LipBengal;
- data scarcity as the main cause;
- a recommendation of at least 100 samples per word.

**Caution: the thesis's "85 samples per word (LRW-AR)" and "20 samples per word (LipBengal)"
(§4.6, §5.4) don't match the published dataset sizes.** LRW-AR is 20,000 clips for 100 words, i.e.
200 per word (160 per word in its 16,000-clip training split, thesis Table 3.2). LipBengal has 150
speakers each recording up to 503 words, allowing up to about 150 per word. The thesis also calls
LipBengal's vocabulary both "503 words" and "1,000 words". The figures may describe the subset
actually used in training. **Recount from the training data before citing them** (clips per class in
the LRW-AR and LipBengal folders that were used).

LipBengal (Sahed et al., *Data in Brief* 2025) was recorded by 150 undergraduate students of one
institution (MIST; 92% male) reading prompted words on phone cameras: 720p, 30 fps, released as PNG
frames, up to 503 words per speaker, 363,150 utterances, 54 classes. Its abstract calls the conditions
"diverse and uncontrolled", but the speech is prompted, not natural. No in-the-wild, word-level
Bengali VSR dataset is publicly available, as far as we found. The search and what is still
unverified are in §7 ("Novelty check").

**People.** Animesh Bhattacharjee (repository owner); thesis supervisor Dr. Aniqua Nusrat Zereen,
co-supervisor Md Nafiz Ishtiaque Mahee (BRAC University).

---

## 3. Design decisions

Each entry gives the decision, why it was made, and what was considered and rejected. The code
location is in brackets.

### D1. Source and release model
- **Decision:** collect from YouTube. Release **YouTube IDs, frame ranges, crop boxes and labels**, not the videos, following VoxCeleb and AVSpeech. Clips go to researchers only on request, with a takedown contact. Ethics approval is needed before release, because faces and voices are personal data.
- **Why:** VoxCeleb is distributed as YouTube URLs with utterance timestamps. LRS3 is released for research under CC BY 4.0, with copyright staying with the video owners.
- **Consequence:** the repository holds code only. `data/` is git-ignored, and so are any samples from third-party videos.

### D2. Downloader [`bvsr/download.py`]
- **yt-dlp**, with these defaults:
  - 10–20 s random sleep between videos and 0.75 s between requests (yt-dlp's `-t sleep` preset, read from yt-dlp's `options.py`);
  - at most **150 videos per rolling hour**;
  - an exponential back-off starting at 30 min on rate-limit messages, stopping after 3 hits;
  - a download archive so runs can resume.
- **YouTube limits** (yt-dlp wiki): about 300 videos/hour for guest sessions and about 2,000/hour for accounts. Using an account risks a temporary or permanent ban.
- **We do not bypass limits** with proxy pools or multiple accounts. That violates YouTube's terms and is hard to defend in an ethics section. At our scale, about 1,500 videos, the limit isn't a bottleneck.
- **Format:** `-S res:720,vcodec:h264,acodec:m4a`. The first test download picked **AV1 at 360×640**; AV1 decodes poorly in OpenCV/MediaPipe builds, and the resolution was low. With the sort order, the same video came as H.264 720×1280.
- **Since yt-dlp 2025.11.12**, full YouTube support needs an external JavaScript runtime. Deno is enabled by default.
- **PO tokens** ("Sign in to confirm you're not a bot") can be handled with the `bgutil-ytdlp-pot-provider` plugin if needed. It is not installed.

### D3. Voice activity detection [`bvsr/transcribe.py: vad_chunks`]
- **Silero VAD** (pip `silero-vad`; model bundled, no download).
- **Settings:** `max_speech_duration_s=20`, `min_silence_duration_ms=300`, `speech_pad_ms=100`.
- **Chunking:** speech regions are merged into chunks of ≤20 s when the gap is under 1 s. Chunks shorter than 1 s are dropped.
- **Why:** Whisper needs input under 30 s. Removing silence and music reduces hallucinated text.

### D4. ASR [`bvsr/transcribe.py: ASR`]
- **Model:** `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` (Apache-2.0; Whisper-medium, 0.8 B params per the model card).
- **Evidence (read directly, 2026-10-01):** three papers from the Kaggle DL Sprint 4.0 competition, all on its **Bengali-Loop** long-form benchmark. Bangla-WhisperDiar describes it as 191 YouTube recordings, 158.6 h, about 792k words from 11 channels, mostly Bangla drama plus news and entertainment.
  - WhisperAlign (Chowdhury et al., arXiv 2603.04809): tugstugi as released **34.07% WER**, Hishab TITU-BN 50.67%, their fine-tune 27.00%.
  - Bangla-WhisperDiar (Bhuiyan et al., arXiv 2605.08214): tugstugi 34.07%, LoRA fine-tune 31.32%, their system 24.41%, TITU-BN 50.67%.
  - Jahan et al. (arXiv 2603.03158): tugstugi with their segmentation 0.380 WER (private leaderboard) vs about 0.79 for WhisperX-based variants.
- **Claim we can make:** the strongest off-the-shelf model *among those tested in these papers*, on one long-form YouTube benchmark. Not "the best Bengali ASR". **About 1 word in 3 is wrong**, which is why label checking (§8) is essential.
- **Size disagreement:** the papers give 307M and ~764M parameters. Whisper-medium is about 769M, which matches the model card's 0.8B; use that.
- **Candidates to compare on our pilot:** the fine-tuned systems above, if their weights are released (not checked).
- **Run settings:**
  - Whisper is called directly (`WhisperForConditionalGeneration.generate`, `language="bn"`, `max_new_tokens=200`), not through the `pipeline` helper (see §10, B1).
  - fp16 on CUDA/MPS, fp32 on CPU.
  - Batch size 2.
- **Alternative:** human Bengali subtitles when present (`--use-subs`; checked against `info.json` so auto-captions are excluded). LRW used broadcast subtitles.

### D5. Forced alignment [`bvsr/transcribe.py: Aligner`]
- **Model:** MMS forced aligner `MahmoudAshraf/mms-300m-1130-forced-aligner` (1,130 languages), run through the `ctc-forced-aligner` package. Bengali text is romanised with **uroman**.
- **Why not torchaudio:** `torchaudio.functional.forced_align` and `MMS_FA` were deprecated in 2.8 and removed in 2.9 (torchaudio docs).
- **Per-chunk alignment:** each chunk's audio is padded by 0.25 s on each side. One bad chunk then can't derail a whole video.
- **Numbers dropped:** words with digits are removed before alignment, because uroman can't romanise digits as spoken words. Neighbouring words absorb the time.
- **Score:** the mean log-probability of the word's aligned frames (emissions are `log_softmax`, checked in the package source). 0 means certain.
- **Threshold `--min-score -1.0`:** in the TTS test, the hallucinated word scored −3.98 while real words scored −0.02 to −0.75 (§4.1). *To do: tune it on the human-checked pilot (§8).*
- **Resolution:** the MMS model stride is 320 samples at 16 kHz, i.e. 20 ms.

### D6. Memory design (8 GB laptop) [`bvsr/transcribe.py`]
- **Two phases:** ASR for all pending videos (writing `data/asr/`), then free the model, then alignment for all videos (writing `data/align/`). Only one large model is in memory at a time.
- **Why:** with both models loaded on MPS, the first run stalled for 40 min at about 5% CPU on the 8 GB M2. After the change, a 3:53 video took 66 s with a peak of about 2.2 GB.
- **CPU stages run alongside:** faces and clip cutting use the CPU and about 200–500 MB each, so they can run in parallel with ASR on the GPU. `--workers` parallelises across videos.

### D7. Word clip length: 29 frames, word centred, context included [`bvsr/cut.py`]
- **Decision:** 29 frames at 25 fps (1.16 s), centred on the word's midpoint (frame 14 = midpoint). Neighbouring words are deliberately included.
- **Why:** this is the LRW definition. The LRW page says all videos are 29 frames with the word in the middle, and the word duration is given in metadata. It gives a fixed-length input, matches LRW-pretrained models (including the thesis model), and keeps the lip movement that starts before the sound and continues after it.
- **Context share:** the median word lasted 0.24 s (about 6 frames) in the pilot, so about 23 of 29 frames are context. The metadata gives the word's frames (`start_frame`, `end_frame`) so models can mask or trim.

### D8. Frame grid and timing [`bvsr/faces.py: iter_frames`, `bvsr/cut.py`]
- **Grid:** every stage decodes with the same command, `ffmpeg -vf setpts=PTS-STARTPTS,fps=25`, so global frame *i* is the picture at video time *i*/25. Stage 3 measurements and stage 4 pixels therefore refer to identical frames.
- **Audio/video offset:** a wav time *t* maps to frame `(t − offset) × 25`, where `offset = video_start − audio_start`. The offset is read with ffprobe and stored per video; it was 0.0 in the pilot.
- **Word frames:** a frame belongs to the word if its sampling instant lies inside [word_start, word_end]. A word shorter than one frame gets the frame at its midpoint.
- **Audio:** a clip's audio is exactly `n_frames × 640` samples of the 16 kHz wav, starting at the first frame's time.

### D9. Face analysis [`bvsr/faces.py`]
- **Model:** MediaPipe Face Landmarker (FaceMesh-V2, 478 landmarks, BlazeFace short-range detector inside), `num_faces=4`, all confidence thresholds 0.5, VIDEO running mode.
- **Version:** **mediapipe 0.10.35.** 1.0.1 aborts on macOS (`graph_service.h ... Service is unavailable`, from `DrishtiMetalHelper` in `TensorsToDetectionsCalculator`), even with `Delegate.CPU`.
- **Head pose:** `cv2.RQDecomp3x3` on the facial transformation matrix gives (pitch, yaw, roll). Validated by correlating yaw with the nose tip's horizontal offset inside the face outline (landmarks 1, 234, 454) over 200 frames: **r = 0.86, same sign**.
- **Tracking:** greedy IoU matching on landmark bounding boxes (IoU ≥ 0.3). A track survives 2 missed frames and is force-ended at every detected shot cut.
- **Shot cuts:** Bhattacharyya distance between HSV histograms (32 hue × 16 saturation bins, frame downscaled to 160 px wide) of consecutive frames; a cut is declared above **0.35**. This is similar in spirit to the LRW/LRS pipeline's colour-histogram shot detection. Calibration is in §4.3.
- **Mouth sharpness:** variance of the Laplacian over the lip-landmark box padded by 25%, resized to 96 px wide. It is recorded but not yet used as a filter, because no threshold has been calibrated.

### D10. Clip filters [`bvsr/cut.py: check_face`]
Thresholds marked † come from LRW-Persian (arXiv 2510.22716): face ≥100×100 px, and |yaw| > 30° or |pitch| > 40° excluded.

| Filter | Words | Sentences | Why |
|---|---|---|---|
| shot cut inside clip | reject | reject | two shots under one label; the crop would jump |
| main face track must cover the whole clip | required | required | tracks end at cuts and face jumps |
| frame-to-frame change > 0.25 inside clip | reject | reject | fades and flashes. Non-cut frames have a 99.9th percentile of 0.215 (§4.3) |
| face missing | ≤ 2 of 29 frames | ≤ 5% of frames | interpolated linearly |
| other face seen on ≥ 3 frames | reject | reject | no active-speaker check yet; ≥3 frames because a 1-frame false detection on hands was seen (§4.3) |
| face width (landmark box) | ≥ 100 px † | ≥ 100 px † | |
| \|yaw\| | ≤ 30° † (max over frames) | ≤ 30° † (95th percentile) | |
| \|pitch\| | ≤ 40° † (max) | ≤ 40° † (95th percentile) | |
| alignment score | ≥ −1.0 per word | ≥ −1.0 chunk mean | D5 |
| word duration | 0.12–1.0 s | — | a word longer than 1.16 s can't fit |
| label length | ≥ 2 code points | — | |
| clips of one word per video | ≤ 5 | — | speaker diversity |

### D11. Crop [`bvsr/cut.py: crop_boxes, crop`]
- **Square crop centred on the nose tip** (landmark 1). The centre is smoothed with a 5-frame moving average.
- **Size:** side = **1.6 × median landmark extent**, constant within a clip. Parts outside the frame are filled with black, and the padded fraction is recorded.
- **Output:** resized to **256 px for words (LRW size)** and **224 px for sentences (LRS3 size)**, with INTER_AREA when shrinking and INTER_CUBIC when enlarging.
- **Checked against the official LRW sample** (`AFTERNOON.mp4` from the LRW page, 2026-10-01):
  - Format: 256×256, 25 fps, 29 frames, 1.16 s, MPEG-4 Part 2 video; AAC 16 kHz mono, 1.16 s.
  - Framing, measured with our MediaPipe landmarker on all 29 frames: the face fills **0.616** of the crop (implied scale **1.62**); nose at (0.496, 0.504); mouth centre at (0.50, 0.61).
  - Our clips (5 measured): face fills 0.619–0.635 (scale 1.58–1.62); nose at about (0.50, 0.50); mouth y 0.61–0.62. **Our framing matches LRW's**, based on n = 1 LRW sample.
  - The sample's `.txt` holds Disk reference, Channel, Program start, Clip start and Duration only, no frame indices. Ours adds frame indices, face metrics and landmarks.
- **A conflict to note in the paper:** the LRW paper (Chung & Zisserman, ACCV 2016, §3 Stage 5) says the face is cropped with the mouth centred, using the landmark registration. Zhang et al. 2020 (arXiv 2003.03206) say LRW is loosely registered by nose centres. The released sample has the **nose** at the centre and the mouth lower. We centre on the nose, which matches the sample.
- **Mouth ROI is preprocessing, not dataset content.** LRW does not ship mouth crops; the common recipe is a 96×96 crop from lip landmarks and random 88×88 crops in training. We store landmarks per frame so users can crop the mouth without detecting faces again. The thesis pipeline (ROI = 1.5 × mouth width, 1.8 × mouth height) can be applied directly.
- **Landmark storage:** landmarks are mapped with the **same integer box** used for the pixels, so they are pixel-exact (see §10, B5).

### D12. Labels [`bvsr/common.py: bn_normalize`]
- Unicode NFC; zero-width characters (U+200B, U+200C, U+200D, U+FEFF) removed; only the Bengali block U+0980–U+09FF kept.
- The danda (U+0964) is in the Devanagari block, so it is removed too.
- The surface form as transcribed is kept as `surface`.
- *Open:* homophenes (e.g. কলম / গরম from the thesis) and dialect spelling variants are not merged.

### D13. Encoding
- **Video:** libx264, CRF 18, preset medium, yuv420p, 25 fps.
- **Audio:** AAC 64 kb/s, 16 kHz mono. The AAC encoder adds its usual priming delay, which mp4 edit lists compensate.
- **Metadata:** `.json` per clip; `.npz` per clip with `landmarks` (float16, clip pixels), `crop_box` (cx, cy, side in source pixels) and `interpolated` flags.
- **Sentences** also get an LRS3-style `.txt`. Its last column is the **aligner** score, labelled `SCORE`. It is not the active-speaker score that LRS3's `ASDSCORE` column holds.

### D14. Vocabulary and splits (not built yet)
Options for the supervisor to decide:
- **Balanced (LRW):** e.g. 500 words × ≥200 clips.
- **Naturally distributed (LRW-1000).**

LRW-Persian took the top 2,500 words per program, intersected across channels and pruned by hand
to 743 words. Splits should be **speaker- or channel-disjoint**: LRW split by broadcast date,
LRW-Persian by program.

### D15. Source selection [`docs/SOURCES.md`]
- **Talking-head content from official channels, mixed genres:** news, talk shows, podcasts and interviews, lectures/TEDx, vlogs. Prior datasets also mix broadcast genres (LRW: BBC programmes; LRW-1000: news and conversational programmes; LRW-Persian: 67 programmes). What keeps the visual format consistent is the face filters (D10), not the genre.
- **No re-upload channels:** the search found many, e.g. channels re-posting BBC bulletins or radio news. A licence shown by a non-owner is invalid, and re-uploads would duplicate footage.
- **Radio-as-video has no visible speaker,** so it is excluded.
- **YouTube's language tag is unreliable** (Prothom Alo tagged `hi`; EKHON TV and The Talk Show `en`), so a per-video language-ID check is needed. MultiVSR used a VoxLingua-trained model.
- **Candidate channels:** 37, checked by metadata on 2026-10-01: licence, language tag, followers, resolution of 1–2 long videos each. **Not screened visually.** Region is a guess from names and titles.
- **Screening (proposed, needs a go-ahead):** a 2-minute 480p section of 3 videos per channel, run through `bvsr.faces`; rank channels by the share of frames with one frontal face ≥100 px, cuts per minute, and speech share.
- **Per-link cap:** `download expand --max-per-link N` (default 50) takes the newest N videos per channel or playlist, via yt-dlp `playlist_items` (`playlistend` is deprecated). Tested on one channel with N = 5.

### D16. Copyright tiers
- **Facts (checked 2026-10-01):**
  - YouTube offers the Standard YouTube License or CC BY (YouTube Help, "License types on YouTube"). CC BY allows sharing and adapting with attribution.
  - TEDx talks are CC BY-NC-ND (TEDx licence page, e.g. TEDxTokyo; TED Talks usage policy), so no derivatives: cut clips can't be shared.
  - yt-dlp's `license` field is set from the "License" metadata row, which appears for CC videos; it is empty for standard ones (yt-dlp `_video.py`, around line 4510).
- **Tier A** (CC BY and uploaded by the owner): clips can be released with attribution. **Tier B** (standard or TEDx): metadata only, following VoxCeleb, AVSpeech and MultiVSR.
- **Search result:** only 3 of 37 candidate channels are CC BY (one podcast, two religious-lecture channels). Tier B will dominate.
- **Not legal advice;** confirm with the university.
- **Other platforms (checked 2026-10-01):** Wikimedia Commons (free licences only; about 20 Bengali videos), Internet Archive (2,509 Bengali video items, but licence tags unreliable), AVSpeech (YouTube IDs, no language labels), TED/TEDx (CC BY-NC-ND). Details in `docs/SOURCES.md` §3b.
- **LRW and LRS2 access requires a signed Data Sharing agreement with BBC R&D**, for non-commercial academic research (LRS2 page). Permission from channels is the cleanest route to releasing clips.

### D17. Vocabulary selection plan (to run after the pilot)
- **Prior rules:**
  - LRW: the 500 most frequent words of 5–10 characters; ≥800 training and ≥40 val/test occurrences each (paper).
  - LRW-1000: naturally distributed.
  - LRW-AR: 100 words.
  - LRW-Persian: top 2,500 per programme, intersected and pruned to 743.
- **Plan:**
  - Count clips, distinct speakers and distinct channels per word.
  - Length rule by **median aligned duration** (~0.25–0.9 s) plus ≥3 grapheme clusters, instead of a character count, because Bengali conjuncts and vowel signs don't map to Latin letters.
  - Require spread: e.g. ≥20 speakers and ≥3 channels per word.
  - Exclude numbers and news-cycle proper names. Decide about English loanwords (exclude, or keep as a flagged subset).
  - Report homophene groups using the thesis viseme mapping.
  - Main benchmark: balanced, e.g. 500 words × ≥200 clips. Optionally also a naturally distributed release.
  - Final list reviewed by Bengali speakers.
- **Balance targets:**
  - ≥35–40% female clips. Existing Bengali sets are 92% male (LipBengal) and 107 of 128 speakers male (BenAV).
  - Both Bangladesh and West Bengal.
  - Per-speaker and per-channel caps.
  - Gender **annotated by hand per speaker**, not inferred from names or classifiers.

---

## 4. Experiments and measurements

Machine: Apple M2 MacBook Air, 8 GB RAM, macOS; Python 3.12.10 (versions in §9).

### 4.1 TTS sanity test (2026-09-24)
- **Input:** macOS Bengali voice "Piya" reading 4 sentences written by us, over a grey video (8.5 s). Text: আমি বাংলাদেশে থাকি। আজকে আবহাওয়া খুব ভালো। আমাদের দেশের মানুষ খুব পরিশ্রমী। শিক্ষা আমাদের সবার অধিকার।
- **ASR output errors:** ভালো→ভাল, শিক্ষা→শিখা, and an inserted word যা.
- **Alignment scores:** real words −0.02 to −0.75; inserted যা **−3.98**.
- **Cost:** ASR 21 s, peak 2.5 GB; alignment 11 s, 0.5 GB.

### 4.2 Pilot video (2026-09-24 and 2026-10-01)
- **Video:** YouTube `4jrdEHp5afY`, channel "10 Minute School", 233 s, one speaker, studio, portrait. Downloaded as H.264 720×1280 at 29.97 fps.
- **VAD:** 15 chunks. ASR 51 s, alignment 7 s, total 66 s, peak 2.2 GB.
- **Words aligned:** 569. Score percentiles (10, 25, 50, 75, 90): −0.86, −0.38, −0.14, −0.06, −0.02. 521 pass −1.0. Median word duration 0.24 s.
- **`cut stats`:** 486 usable tokens, 297 types. Only 39 types occur ≥3 times, all from one speaker.
- **Observed ASR errors:** English loanwords seem error-prone (e.g. labels মক্কটেজ and ভায়া look like misheard English). Not measured yet.

### 4.3 Faces on the pilot video (2026-10-01)
- **Speed:** probe of 200 frames: 8.5 ms per frame (CPU, XNNPACK), a face on 200/200 frames. Full run: 5,834 frames, a face on **5,562 (95.3%)**, 58 s without and 89 s with shot detection (the second run may have been slowed by other load; not investigated). Peak RAM about 314 MB.
- **Face width (px):** 5th / 50th / 95th percentile 197 / 224 / 256.
- **Pose (5th / 50th / 95th percentile):** yaw −24.7 / −16.0 / −7.0°; pitch −14.1 / −2.4 / 6.9°. The speaker faces slightly off-camera.
- **Mouth sharpness (same percentiles):** 73 / 132 / 278.
- **Tracks before shot detection:** 7. Breaks at frames 1777, 2850, 3443 and 5236 were **zoom cuts** (checked visually at 1776/1777). Frame 2672 had a 1-frame false face on the speaker's **hands**. At the cut frames, MediaPipe briefly returned two boxes for the same person (a stale tracked box plus a new detection).
- **Shot-cut calibration** (Bhattacharyya distance between consecutive frames):

  | Kind of frame | Distance |
  |---|---|
  | non-cut frames (50th / 99th / 99.9th percentile) | 0.032 / 0.087 / 0.215 |
  | zoom cuts (same scene) | 0.119–0.167, below threshold, but the tracker breaks there because the face box jumps |
  | scene changes and a flash/light-leak transition (frames 108–123, 277, 300–308) | 0.425–0.99 |
  | detected cuts at threshold 0.35 | 17 frames (the transition counts as several) |
  | tracks after shot detection | 20 |

- **Burned-in captions:** in this video they appear below the mouth. Other videos may cover the mouth; no check exists yet.

### 4.4 Clip cutting on the pilot video (2026-10-01)
- **Words:** 470 candidates → **386 clips** (50 s, 213 MB). Rejected: 39 yaw > 30°, 26 per-video cap, 10 shot cut, 5 track break, 3 face missing, 1 video edge. The full log is in `data/clips/rejects_words.jsonl` and `summary_words.json`.
- **Sentences:** 15 candidates → **8 clips** (12.5 s, 482 MB). Rejected: 3 track break, 2 shot cut, 1 yaw, 1 low alignment score.
- The earlier full-frame version (v0.1) gave 422 word clips. The difference comes from the new face and cut filters.

---

## 5. Verification done

| What | How | Result |
|---|---|---|
| Clip format | ffprobe with frame counting on 40 random word clips | all 256×256, 29 frames, audio 1.160 s |
| Format and framing vs LRW | official LRW sample probed; face extent and nose/mouth position measured with the same landmarker on it and on 5 of our clips | same format; face fills 0.616 (LRW) vs 0.619–0.635 (ours) of the crop, nose centred in both (D11) |
| Landmark ↔ pixel alignment | stored lip landmarks and nose tip drawn back onto all 29 frames of a clip | on the lips and nose in every frame |
| Word timing ↔ lips | 29-frame strips with the word frames marked, two clips (v0.1 and v0.2) | mouth moves on the marked frames (visual only) |
| Pose sign and convention | yaw vs nose-offset cue, 200 frames | r = 0.86 |
| Shot cuts | frames on both sides of the detected cuts, by eye | zoom cuts and a flash transition confirmed |
| Hallucination filter | TTS sentence with a known transcript | invented word scored −3.98, filtered |
| Viewer links | every `src` in viewer.html checked on disk | 147/147 resolve |

**Not verified yet (needed for the paper):** label accuracy on real speech (§8), word-boundary
accuracy against human marks, and audio–video sync beyond visual checks (SyncNet offsets). LRW
framing was checked against only one official sample.

---

## 6. Known limitations and open issues

1. **No active-speaker detection yet** (stage 3b; TalkNet or SyncNet). Multi-face clips are rejected, which will lose most news and talk-show footage.
2. **ASR labels are not human-verified.** The ASR model has 34.07% WER on the Bengali-Loop benchmark (D4), and English loanwords look error-prone. Our alignment-score filter removes some errors but not substituted words that are actually spoken similarly. A pilot human check of about 200 random clips is planned.
3. **Long sentences that cross a shot cut are dropped.** LRS splits them at the cut instead.
4. **The BlazeFace short-range detector** misses small or far faces. Those would mostly fail the 100 px rule anyway, but wide shots are under-represented.
5. **No occlusion check** (burned-in captions, microphones, hands over the mouth).
6. **Mouth sharpness is recorded but not used as a filter.**
7. **Only one video tested** (one speaker, studio). Dialect, gender and recording-condition coverage depend on source curation.
8. **Homophenes and spelling variants are not merged in labels.**
9. **Numbers spoken as words are not aligned** (D5).
10. **Faces processing time varied (58 s vs 89 s) between two runs;** not investigated.
11. **Candidate channels not screened visually;** gender mix and front-facing share unknown (D15).
12. **No language-ID check;** YouTube's language tag is unreliable (D15).
13. **Novelty claim partly unverified:** BenAV's recording conditions and the 2022 "Lip Reading Bengali Words" paper (ACM, doi 10.1145/3579654.3579677; page returned HTTP 403) still need checking (§7).
14. **The thesis's samples-per-word figures need recounting** from the training data (§2).

---

## 7. Related work (facts checked against the sources listed)

| Dataset | Facts used | Source |
|---|---|---|
| LRW (Chung & Zisserman, ACCV 2016) | 500 words; 29 frames (1.16 s), word in the middle; duration in metadata; 800–1000 train / 50 val / 50 test per word; split by broadcast date | https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html |
| LRW paper (read directly) | subtitles from broadcast bitmaps by OCR; Penn Phonetics Lab Forced Aligner; checked against IBM Watson STT; speaking face chosen by a linear SVM on the frequency spectrum of mouth openness; face cropped with the mouth centred; vocabulary = 500 most frequent words of 5–10 characters; time-disjoint sets with a one-week gap before test; test set checked by hand; model input 112×112 | https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16/chung16.pdf |
| LRW sample clip | 256×256, 25 fps, 29 frames, AAC 16 kHz mono; metadata = disk reference, channel, program start, clip start, duration | https://www.robots.ox.ac.uk/~vgg/data/lip_reading/data/AFTERNOON.mp4 |
| LRS pipeline (Chung et al., CVPR 2017) | Penn Phonetics Lab Forced Aligner; errors filtered with IBM Watson STT; SyncNet for sync and choosing the speaking face; shot detection, face detection and tracking | https://openaccess.thecvf.com/content_cvpr_2017/papers/Chung_Lip_Reading_Sentences_CVPR_2017_paper.pdf |
| LRS3 | TED/TEDx, 400+ h; CC BY 4.0 for research, copyright stays with owners | https://mm.kaist.ac.kr/datasets/lip_reading/ |
| LRW-1000 / CAS-VSR-W1k | 1,000 classes, 718,018 samples, 2,000+ speakers; 26 sources, 51 programs, 500+ h; naturally distributed | https://arxiv.org/abs/1810.06990 |
| LRW-AR | 100 words, 20,000 videos, 36 speakers; YouTube news; automated pipeline | https://crns-smartvision.github.io/lrwar/ |
| LRW-Persian (2025) | 1,989 h from 67 TV programs; VOSK ASR; words ≥4 chars, confidence > 0.9, < 1.5 s; TalkNet; face ≥100×100; DeepFace > 0.75; mask filter; \|yaw\| > 30° / \|pitch\| > 40° removed; MediaPipe FaceLandmarker; top 2,500 per program intersected and pruned to 743 words; 414,308 clips; program-disjoint 78/22 split | https://arxiv.org/abs/2510.22716 |
| VoxCeleb | released as YouTube URLs + timestamps, CC BY 4.0 | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ |
| BenAV (Pondit, Rukon, Das, Kabir; ICONIP 2021) | 50 words; 128 speakers (107 male, 21 female); 26,300 utterances; 7.3 h; baselines 98.70% / 82.5% in two configurations. **Recording conditions not stated** in the abstract or repo page | https://researchoutput.csu.edu.au/en/publications/benav-a-bengali-audio-visual-corpus-for-visual-speech-recognition/ ; https://github.com/AnikNicks/BenAV-A-New-Bengali-Audio-Visual-Corpus |
| MultiVSR (Prajwal, Hegde, Zisserman; VGG 2025) | ~12k h, 13 languages (English, Portuguese, Spanish, Russian, German, French, Japanese, Italian, Mandarin, Polish, Dutch, Catalan, Turkish); **no Bengali**. Pipeline: AVSpeech YouTube IDs, 25 fps, S3FD faces, ≥96×96, SyncNet offset ≤±10 frames, VoxLingua language ID, WhisperX (Whisper large-v3) word alignment; languages chosen with ≥100 h | https://www.robots.ox.ac.uk/~vgg/publications/2025/Prajwal25/prajwal25.pdf |
| TEDx licence | CC BY-NC-ND: share with attribution; no commercial use; no derivatives | https://www.tedxtokyo.com/creative-commons/ ; https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy |
| YouTube licences | Standard YouTube License or CC BY; CC BY allows reuse with attribution | https://support.google.com/youtube/answer/2797468 |
| LRS2 access | videos + metadata for non-commercial academic research; users sign a Data Sharing agreement with BBC R&D | https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html |
| Wikimedia Commons licensing | only free licences or public domain; NC and ND not allowed | https://commons.wikimedia.org/wiki/Commons:Licensing |
| AVSpeech | ~4,700 h, ~290k YouTube videos, multilingual, IDs + timestamps, no language labels | https://looking-to-listen.github.io/avspeech/ |
| Internet Archive Bengali video | 2,509 items (mediatype:movies, Bengali tag), 581 with a CC or public-domain tag; tags set by uploaders, many clearly not owners | archive.org advancedsearch, 2026-10-01 |
| LipBengal (Sahed et al., Data in Brief 58:111254, 2025; doi:10.1016/j.dib.2024.111254) | 150 speakers (MIST undergraduates; 92% male, 8% female); up to 503 words each; 363,150 utterances; 54 classes; phone cameras at 720p 30 fps on the MIST campus and a dormitory hall; released as 720×1280 PNG frames; abstract says "diverse and uncontrolled conditions". The "40 speakers" in an earlier search summary was wrong. | PubMed 39845145; full text PMC11750490 |
| Bengali ASR on Bengali-Loop (DL Sprint 4.0) | tugstugi as released 34.07% WER; Hishab TITU-BN 50.67%; fine-tuned systems 24.41–27.00% | arXiv 2603.04809, 2605.08214, 2603.03158 |
| ROI study (Zhang et al. 2020) | LRW faces loosely registered by nose centres, little or no scale change in a clip (conflicts with the LRW paper's "mouth centred"; the official sample has the nose at the centre, D11) | https://arxiv.org/abs/2003.03206 |
| MediaPipe Face Landmarker | bundle = BlazeFace short-range + FaceMesh-V2 (478 points) + blendshapes | https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker |
| yt-dlp limits | ~300 videos/h guest, ~2,000/h account, ban risk; `-t sleep`; PO tokens; JS runtime needed since 2025.11.12 | https://github.com/yt-dlp/yt-dlp/wiki/Extractors, https://github.com/yt-dlp/yt-dlp/issues/15012 |

### Novelty check (2026-10-01)
Searched for Bengali/Bangla lip-reading and audio-visual datasets, and for multilingual VSR
corpora that might include Bengali.

**Found:**
- **BenAV (2021):** 50 words, 128 speakers. Recording conditions not stated in the sources checked; its ~99% baseline suggests controlled recordings (inference, unverified).
- **LipBengal (2025):** prompted words, 150 students, phone cameras.
- **"Lip Reading Bengali Words" (ACM 2022):** dataset unknown; the page was not accessible.
- **MultiVSR (2025):** no Bengali. LRS3, LRW-1000, LRW-AR and LRW-Persian are other languages.

**Claim we can make now:** "to our knowledge, the first word-level Bengali lip-reading dataset of
natural, in-the-wild speech". Not "large-scale" until we have the numbers. Before submission, read
the BenAV and the ACM 2022 papers, and repeat the search on Google Scholar and IEEE Xplore.

**MultiVSR is the closest methodological work** (automatic YouTube pipeline, SyncNet filtering,
WhisperX labels). Expect reviewers to compare against it.

---

## 8. Paper notes

**Working title:** *LRW-BN: a large-scale in-the-wild Bengali lip-reading dataset* (placeholder).

**Possible contributions:**
1. The first in-the-wild, word-level Bengali VSR dataset, with a sentence-level part.
2. An open, reproducible pipeline that runs on a laptop and can be ported to other languages by changing the ASR model and the aligner's ISO code.
3. Baselines, including transfer from English LRW using the thesis framework.
4. An analysis of Bengali-specific issues: script normalisation, homophenes, English loanwords.

**Draft outline and the evidence each part still needs:**

| Section | Have | Still needed |
|---|---|---|
| Collection (sources, curation) | downloader, rate-limit policy | channel list; hours per channel, dialect and gender balance |
| Labelling (VAD → ASR → alignment) | method, TTS test, pilot scores | **human-checked label accuracy** on random clips; word-boundary error vs human marks; threshold chosen from that data |
| Visual processing | faces, cuts, filters, calibration (§4.3) | active-speaker detection; sync check (SyncNet offset distribution) |
| Statistics | pilot funnel (§4.4) | full funnel; words × clips; speakers; duration and pose histograms |
| Splits | plan (D14) | speaker/channel-disjoint splits; a verified test set |
| Baselines | thesis model and results on LipBengal / LRW-AR | train and test on the new dataset; top-1 / top-10 |
| Ethics | release plan (D1) | ethics approval; licence; takedown process |

**Every rejected clip is logged with its measurements** (`rejects_*.jsonl`), so the per-filter
funnel table and threshold ablations can be produced without re-running the pipeline.

---

## 9. Reproducibility

| Component | Version / revision |
|---|---|
| Python | 3.12.10 |
| yt-dlp | 2026.8.19 (+ deno 2.9.7) |
| ffmpeg | 9.0.2 |
| torch / transformers / accelerate | 2.14.0 / 5.17.0 / 1.15.0 |
| silero-vad | 6.2.3 |
| ctc-forced-aligner / uroman | 0.3.0 / 1.3.1.1 |
| mediapipe / opencv-contrib-python | 0.10.35 / 5.0.0.93 |
| numpy / soundfile | 2.5.3 / 0.14.0 |
| ASR model | `bengaliAI/tugstugi_bengaliai-asr_whisper-medium`, HF snapshot `da605cc1bd2f60a18d8e440e977ddfa921a88e63` |
| Aligner model | `MahmoudAshraf/mms-300m-1130-forced-aligner`, HF snapshot `49402e9577b1158620820667c218cd494cc44486` |
| Face model | `face_landmarker.task` (float16/latest), SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`, also recorded per video in `data/faces/<id>.json` |

All thresholds used for a run are stored in `data/clips/summary_*.json` (`args`) and
`data/faces/<id>.json` (`config`).

*To do before release:* pin the HF model revisions in code (`revision=`) and pin all package
versions in `requirements.txt`.

---

## 10. Bugs found and fixed

| ID | Problem | Fix |
|---|---|---|
| B1 | The transformers ASR pipeline pops keys from dict inputs, so a retry failed with "needs to contain a raw key" | call Whisper `generate` directly with arrays |
| B2 | ASR and the aligner loaded together on MPS stalled for 40 min on 8 GB | two-phase design (D6) |
| B3 | Word clips had 28 frames instead of 29 (cutting with `-t 1.16` can round down) | read a slightly longer span and cap at 29 frames; later replaced by grid rendering |
| B4 | The downloader picked AV1 at 360p | H.264 format sort (D2) |
| B5 | Landmarks were mapped with unrounded box coordinates while pixels used the rounded box | shared `int_box` |
| B6 | All clip frame buffers were allocated up front (~2.4 GB for 422 clips) | allocate only while a clip is open |
| B7 | A lambda in argparse defaults couldn't be pickled for worker processes | dispatch on `args.cmd` |
| B8 | The sentence `.txt` header said `ASDSCORE` (LRS3's active-speaker score) for our aligner score | renamed to `SCORE` |
| B9 | MediaPipe 1.0.1 aborts on macOS | pinned 0.10.35 |
| B10 | A one-frame false face (hands) would count as a second face | require ≥3 frames |
