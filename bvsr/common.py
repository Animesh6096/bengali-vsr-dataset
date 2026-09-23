"""Shared paths and helpers for the Bengali VSR dataset pipeline."""
from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # raw/<video_id>/<video_id>.mp4 / .wav / .info.json / subs
ASR_DIR = DATA / "asr"        # asr/<video_id>.json    (VAD chunks + ASR text, phase 1)
ALIGN = DATA / "align"        # align/<video_id>.json  (chunks + word timestamps, phase 2)
CLIPS = DATA / "clips"        # clips/words/<word>/..., clips/sentences/<video_id>/...
MANIFEST = DATA / "manifest.jsonl"
DOWNLOAD_LOG = DATA / "download_log.jsonl"
ARCHIVE = DATA / "archive.txt"

SAMPLE_RATE = 16000


def ensure_dirs() -> None:
    for d in (DATA, RAW, ASR_DIR, ALIGN, CLIPS):
        d.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    subprocess.run(cmd, check=True)


def extract_wav(video: Path, wav: Path) -> None:
    """16 kHz mono PCM, the input format for VAD, ASR and the aligner."""
    run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)])


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


# --- Bengali text normalisation -------------------------------------------------

_ZW = re.compile("[​‌‍﻿]")           # zero-width chars (ZWNJ/ZWJ vary between keyboards)
_NON_BN = re.compile("[^ঀ-৿]")                  # anything outside the Bengali block
_BN_DIGITS = re.compile("[০-৯0-9]")


def bn_normalize(word: str) -> str:
    """Canonical label form of a word: NFC, no zero-width chars, Bengali letters only.

    Punctuation (danda, quotes, commas) and Latin/other characters are stripped, so
    'বাংলাদেশ।' and 'বাংলাদেশ' map to the same class.
    """
    word = unicodedata.normalize("NFC", word)
    word = _ZW.sub("", word)
    word = _NON_BN.sub("", word)  # danda (U+0964) lives in the Devanagari block, so it goes too
    return word


def has_digits(word: str) -> bool:
    return bool(_BN_DIGITS.search(word))


def free_memory() -> None:
    """Drop cached GPU/MPS allocations after deleting a model."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
