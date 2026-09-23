"""Stage 2: VAD -> Bengali ASR -> word-level forced alignment.

Runs in two phases so only one large model is in memory at a time (matters on 8-16 GB machines):
  phase 1  VAD + Whisper ASR for every pending video  -> data/asr/<id>.json   (ASR model then freed)
  phase 2  MMS forced alignment for every ASR file    -> data/align/<id>.json

data/align/<id>.json:
    {"video_id", "asr_model", "chunks": [
        {"start", "end", "text", "source": "asr"|"subs", "score",
         "words": [{"word", "label", "start", "end", "score"}]}]}

Times are seconds in the original video. `score` is the aligner's mean log-probability
(closer to 0 = more confident); `label` is the normalised class name used for LRW-style folders.

Usage:
    python -m bvsr.transcribe                         # both phases, all pending videos
    python -m bvsr.transcribe --phase asr --ids abc123
    python -m bvsr.transcribe --device cpu --fp32
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .common import ALIGN, ASR_DIR, RAW, SAMPLE_RATE, bn_normalize, ensure_dirs, free_memory, has_digits, pick_device

DEFAULT_ASR = "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"


def load_wav(vid: str) -> np.ndarray:
    wav_path = RAW / vid / f"{vid}.wav"
    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"{wav_path} is {sr} Hz"
    return audio


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# --- VAD chunking ----------------------------------------------------------------

_VAD = None


def vad_chunks(wav: np.ndarray, max_len: float, min_len: float = 1.0) -> list[tuple[float, float]]:
    """Speech regions from Silero VAD, merged greedily into chunks of at most `max_len` seconds."""
    global _VAD
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _VAD is None:
        _VAD = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(wav), _VAD, sampling_rate=SAMPLE_RATE, return_seconds=True,
        max_speech_duration_s=max_len, min_silence_duration_ms=300, speech_pad_ms=100,
    )
    chunks: list[list[float]] = []
    for t in ts:
        s, e = float(t["start"]), float(t["end"])
        if chunks and e - chunks[-1][0] <= max_len and s - chunks[-1][1] < 1.0:
            chunks[-1][1] = e
        else:
            chunks.append([s, e])
    return [(s, e) for s, e in chunks if e - s >= min_len]


# --- ASR -------------------------------------------------------------------------

class ASR:
    """Whisper called directly (no pipeline): chunks are already <30 s, so one forward pass each."""

    def __init__(self, model_name: str, device: str, fp32: bool = False):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        # fp16 halves memory (medium: ~3 GB -> ~1.5 GB). CPU matmuls in fp16 are slow, so CPU stays fp32.
        self.dtype = torch.float32 if (fp32 or device == "cpu") else torch.float16
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=self.dtype).to(device).eval()
        self.gen_kwargs = {"language": "bn", "task": "transcribe"}

    @torch.inference_mode()
    def __call__(self, segments: list[np.ndarray]) -> list[str]:
        feats = self.processor(segments, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        feats = feats.to(self.device, self.dtype)
        try:
            ids = self.model.generate(feats, max_new_tokens=200, **self.gen_kwargs)
        except (ValueError, TypeError):
            # Some fine-tuned checkpoints lack the multilingual language tokens; they only speak Bengali anyway.
            self.gen_kwargs = {}
            ids = self.model.generate(feats, max_new_tokens=200)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


# --- Subtitles (human-made captions beat ASR when they exist) -------------------

_VTT_TIME = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d+)\s+-->\s+(\d+):(\d\d):(\d\d)\.(\d+)")


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def read_vtt(path: Path) -> list[tuple[float, float, str]]:
    cues, cur, text = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        m = _VTT_TIME.search(line)
        if m:
            cur, text = (_secs(*m.groups()[:4]), _secs(*m.groups()[4:])), []
        elif cur and line.strip():
            text.append(re.sub(r"<[^>]+>", "", line).strip())
        elif cur and not line.strip():
            if text:
                cues.append((cur[0], cur[1], " ".join(text)))
            cur = None
    return cues


def manual_subs(vid: str) -> Path | None:
    """Human subtitles only. yt-dlp names auto captions the same way, so check the info.json."""
    info_path = RAW / vid / f"{vid}.info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    langs = [k for k in (info.get("subtitles") or {}) if k == "bn" or k.startswith("bn-")]
    for lang in langs:
        p = RAW / vid / f"{vid}.{lang}.vtt"
        if p.exists():
            return p
    return None


# --- Forced alignment ------------------------------------------------------------

class Aligner:
    def __init__(self, device: str):
        from ctc_forced_aligner import load_alignment_model

        # MMS forced aligner (1130 languages); Bengali goes through uroman romanisation.
        self.model, self.tokenizer = load_alignment_model(device, dtype=torch.float32)
        self.device = device

    def __call__(self, audio: np.ndarray, text: str) -> list[dict]:
        from ctc_forced_aligner import generate_emissions, get_alignments, get_spans, postprocess_results, preprocess_text

        # Numbers are spoken as words but written as digits; uroman can't align them, so drop them here.
        words = [w for w in text.split() if bn_normalize(w) and not has_digits(w)]
        if not words:
            return []
        wav = torch.from_numpy(audio).to(self.model.dtype).to(self.device)
        with torch.inference_mode():
            emissions, stride = generate_emissions(self.model, wav, batch_size=1)
        tokens, text_starred = preprocess_text(" ".join(words), romanize=True, language="ben")
        if not any(t != "<star>" and t.strip() for t in tokens):
            return []
        segments, scores, blank = get_alignments(emissions, tokens, self.tokenizer)
        spans = get_spans(tokens, segments, blank)
        return postprocess_results(text_starred, spans, stride, scores)


# --- Phase 1: VAD + ASR ----------------------------------------------------------

def run_asr(ids: list[str], args: argparse.Namespace, device: str) -> None:
    todo = [v for v in ids if args.overwrite or not (ASR_DIR / f"{v}.json").exists()]
    if not todo:
        return
    asr = None
    for n, vid in enumerate(todo, 1):
        t0 = time.time()
        subs = manual_subs(vid) if args.use_subs else None
        if subs:
            units = [{"start": s, "end": e, "text": t, "source": "subs"} for s, e, t in read_vtt(subs)]
            model_name = None
        else:
            if asr is None:  # lazy: a batch where every video has subtitles never loads Whisper
                print(f"[asr] loading {args.asr_model} on {device}", flush=True)
                asr = ASR(args.asr_model, device, args.fp32)
            audio = load_wav(vid)
            spans = vad_chunks(audio, max_len=args.max_chunk)
            texts: list[str] = []
            for i in range(0, len(spans), args.asr_batch):
                batch = spans[i : i + args.asr_batch]
                texts += asr([audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)] for s, e in batch])
                print(f"\r[asr] {vid}: {min(i + args.asr_batch, len(spans))}/{len(spans)} chunks", end="", flush=True)
            print()
            units = [{"start": round(s, 3), "end": round(e, 3), "text": t, "source": "asr"}
                     for (s, e), t in zip(spans, texts)]
            model_name = args.asr_model
        write_json(ASR_DIR / f"{vid}.json", {"video_id": vid, "asr_model": model_name, "units": units})
        print(f"[asr] ({n}/{len(todo)}) {vid}: {len(units)} chunks, {time.time() - t0:.0f}s", flush=True)
    del asr
    free_memory()


# --- Phase 2: forced alignment ---------------------------------------------------

def align_video(vid: str, aligner: Aligner) -> dict:
    asr_out = json.loads((ASR_DIR / f"{vid}.json").read_text(encoding="utf-8"))
    audio = load_wav(vid)
    total = len(audio) / SAMPLE_RATE
    pad = 0.25
    chunks = []
    for u in asr_out["units"]:
        s, e, text = u["start"], u["end"], u["text"]
        if not text:
            continue
        s0, e0 = max(0.0, s - pad), min(total, e + pad)
        try:
            words = aligner(audio[int(s0 * SAMPLE_RATE) : int(e0 * SAMPLE_RATE)], text)
        except Exception as ex:  # a bad chunk (e.g. hallucinated text longer than audio) must not kill the video
            print(f"  align failed {vid} {s:.1f}-{e:.1f}: {ex}")
            continue
        out_words = [
            {"word": w["text"], "label": bn_normalize(w["text"]),
             "start": round(s0 + w["start"], 3), "end": round(s0 + w["end"], 3), "score": round(w["score"], 4)}
            for w in words
        ]
        chunks.append({
            "start": s, "end": e, "text": text, "source": u["source"],
            "score": round(float(np.mean([w["score"] for w in out_words])), 4) if out_words else None,
            "words": out_words,
        })
    return {"video_id": vid, "asr_model": asr_out["asr_model"], "chunks": chunks}


def run_align(ids: list[str], args: argparse.Namespace, device: str) -> None:
    todo = [v for v in ids if (ASR_DIR / f"{v}.json").exists()
            and (args.overwrite or not (ALIGN / f"{v}.json").exists())]
    if not todo:
        return
    print(f"[align] loading MMS aligner on {device}", flush=True)
    aligner = Aligner(device)
    for n, vid in enumerate(todo, 1):
        t0 = time.time()
        result = align_video(vid, aligner)
        write_json(ALIGN / f"{vid}.json", result)
        n_words = sum(len(c["words"]) for c in result["chunks"])
        print(f"[align] ({n}/{len(todo)}) {vid}: {len(result['chunks'])} chunks, {n_words} words, "
              f"{time.time() - t0:.0f}s", flush=True)
    del aligner
    free_memory()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", help="video ids (default: every downloaded video)")
    p.add_argument("--phase", choices=["all", "asr", "align"], default="all")
    p.add_argument("--asr-model", default=DEFAULT_ASR)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--fp32", action="store_true", help="load Whisper in fp32 (uses 2x memory)")
    p.add_argument("--asr-batch", type=int, default=2, help="chunks per Whisper forward pass; raise on a GPU")
    p.add_argument("--max-chunk", type=float, default=20.0, help="max seconds per VAD chunk sent to ASR")
    p.add_argument("--use-subs", action="store_true", help="use human Bengali subtitles instead of ASR when present")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    ids = args.ids or sorted(d.name for d in RAW.iterdir() if (d / f"{d.name}.wav").exists())
    device = pick_device(args.device)
    if args.phase in ("all", "asr"):
        run_asr(ids, args, device)
    if args.phase in ("all", "align"):
        run_align(ids, args, device)
    print("[transcribe] done")


if __name__ == "__main__":
    main()
