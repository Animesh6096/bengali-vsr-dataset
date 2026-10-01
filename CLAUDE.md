# Project instructions: bengali-vsr-dataset

Pipeline that builds an LRW-style Bengali lip-reading dataset from YouTube. The goal is a dataset
paper, so correctness and traceability matter more than speed.

## Read first
1. `docs/RESEARCH_LOG.md`: every decision and its reason, measurements, verification, open issues, paper notes.
2. `README.md`: user-facing description, usage, output format.
3. `PLAN.md`: original research plan and roadmap.

## After every change to the pipeline
In the same commit:
- `docs/RESEARCH_LOG.md`: add a changelog line (§1), update the affected decision (§3), record new measurements (§4), verification (§5), open issues (§6) or bugs (§10).
- `README.md`: update it if commands, defaults, output format or status changed.
- `PLAN.md`: update its stage table if a stage's status changed.

## Rules
- **Verify before asserting.** Measure or read the source before writing a number or claim into the docs, and mark anything unverified as such. These numbers end up in a paper.
- **Never commit data:** videos, clips, transcripts and samples from third-party videos stay in `data/` or zips, which are git-ignored. The repo holds code and docs only.
- **Rate limits:** keep the downloader's polite defaults. Don't add proxy rotation or multi-account workarounds.
- **Dev machine:** Apple M2, 8 GB RAM. Keep only one large model in memory per process (two-phase transcribe), and check peak RSS (`/usr/bin/time -l`) after changes. Bulk runs go to a CUDA GPU (`--device cuda`).
- **Frame grid:** all stages use `ffmpeg -vf setpts=PTS-STARTPTS,fps=25` (`bvsr.faces.iter_frames`). Don't introduce another decode path for clip frames, or landmarks and frames will drift apart.
- **MediaPipe is pinned to 0.10.35** (1.0.x aborts on macOS).

## Environment
```bash
.venv/bin/python -m bvsr.<download|transcribe|faces|cut|viewer> --help
```
Python 3.12 venv in `.venv/`; ffmpeg and deno from Homebrew; face model in `models/` (git-ignored, see README).
