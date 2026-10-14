from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import torch
from typing import Optional, List, Dict, Any
import os
import io
import subprocess
import tempfile
import re
import sys
import numpy as np

DSMER_ROOT = Path(__file__).resolve().parents[3]
if str(DSMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DSMER_ROOT))

from dsmr.models import HMEMNet
from dsmr.utils.config import ExperimentConfig

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None

app = FastAPI(title="HMEM Inference")
MODEL: Optional[torch.nn.Module] = None
TOKENIZER = None
TEXT_BACKBONE = None
CFG = None
MODEL_SOURCE = None
TEXT_BACKBONE_SOURCE = None
TRANSCRIBER = None
EMOTION_LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
MAX_TEXT_LENGTH = 128


def emotion_label(emotion_id: int) -> str:
    if 0 <= emotion_id < len(EMOTION_LABELS):
        return EMOTION_LABELS[emotion_id]
    return "unknown"


class PredictRequest(BaseModel):
    conversation_id: str
    speaker_id: str
    text: Optional[str]


def sanitize_transcript(text: Optional[str]) -> str:
    """Reject garbage ASR output such as punctuation-only strings before use."""
    if text is None:
        return ""
    cleaned = " ".join(str(text).replace("\n", " ").split())
    if not cleaned:
        return ""
    if re.fullmatch(r"[\s\.,;:!?\-]+", cleaned):
        return ""
    alpha_chars = sum(ch.isalpha() for ch in cleaned)
    if alpha_chars == 0:
        return ""
    if alpha_chars / max(len(cleaned.replace(" ", "")), 1) < 0.2:
        return ""
    return cleaned


def split_text_into_turns(text: str, max_words: int = 24) -> List[str]:
    """Split a long transcript into conversation-like utterance chunks.

    This is a lightweight segmentation layer that keeps the model compatible with
    multi-speaker or multi-turn input without requiring a full diarization stack.
    """
    if text is None:
        return []
    cleaned = sanitize_transcript(text)
    if not cleaned:
        return []
    segments = re.split(r"(?<=[.!?])\s+|(?<=\n)\s*", cleaned)
    segments = [seg.strip() for seg in segments if seg and seg.strip()]
    if len(segments) <= 1:
        words = cleaned.split()
        if len(words) <= max_words:
            return [cleaned]
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words) if i < len(words)]

    return segments


def _extract_wav_from_media_bytes(data: bytes, suffix: str = ".bin") -> tuple[np.ndarray, int]:
    """Decode uploaded audio/video into a mono 16 kHz waveform."""
    source_path = None
    wav_path = None
    try:
        if isinstance(data, np.ndarray):
            wav = np.asarray(data, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav.astype(np.float32), 16000

        if not data:
            return np.zeros(0, dtype=np.float32), 16000

        try:
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as source:
                source.write(data)
                source_path = source.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
                wav_path = out.name
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            wav, sample_rate = sf.read(wav_path, dtype="float32")
            wav = np.asarray(wav)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav.astype(np.float32), int(sample_rate)
        except Exception:
            if len(data) % 4 == 0:
                return np.frombuffer(data, dtype=np.float32).astype(np.float32), 16000
            raise
    finally:
        for path in (source_path, wav_path):
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass


def try_pyannote_diarization(audio_data: Optional[bytes | np.ndarray], sample_rate: int = 16000) -> List[tuple[float, float, Optional[str]]]:
    """Attempt real diarization with pyannote if the dependency is available.

    This is intentionally non-fatal: if the package or credentials are unavailable,
    the function returns an empty list so the app falls back to the robust energy
    segmentation path already in use.
    """
    if audio_data is None:
        return []

    try:
        from pyannote.audio import Pipeline
    except Exception:
        return []

    try:
        if isinstance(audio_data, np.ndarray):
            wav = np.asarray(audio_data, dtype=np.float32).reshape(-1)
            if wav.size == 0:
                return []
        else:
            wav, sr = _extract_wav_from_media_bytes(audio_data)
            if wav.size == 0:
                return []
            if sr != sample_rate:
                import torchaudio
                wav_t = torch.from_numpy(wav).float()
                wav = torchaudio.functional.resample(wav_t, sr, sample_rate).numpy().astype(np.float32)

        wav_path = None
        try:
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            sf.write(wav_path, wav.astype(np.float32), sample_rate)

            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                return []

            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
            diarization = pipeline(wav_path)
            segments: List[tuple[float, float, Optional[str]]] = []
            iter_tracks = getattr(diarization, "itertracks", None) or getattr(diarization, "iter_tracks", None)
            if iter_tracks is None:
                return []
            for turn, _, speaker in iter_tracks(yield_label=True):
                start = float(turn.start)
                end = float(turn.end)
                if end > start:
                    segments.append((start, end, str(speaker) if speaker is not None else None))
            return segments
        finally:
            if wav_path:
                try:
                    Path(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        return []


def segment_audio_for_diarization(audio_data: Optional[bytes | np.ndarray], sample_rate: int = 16000, min_silence_sec: float = 0.35, min_segment_sec: float = 0.7) -> List[tuple[float, float, Optional[str]]]:
    """Split raw audio into speech-like ranges using pyannote when available else energy-based fallback."""
    if audio_data is None:
        return []

    diarized = try_pyannote_diarization(audio_data, sample_rate=sample_rate)
    if diarized:
        return diarized

    if isinstance(audio_data, np.ndarray):
        wav = np.asarray(audio_data, dtype=np.float32).reshape(-1)
        sr = sample_rate
    else:
        wav, sr = _extract_wav_from_media_bytes(audio_data)

    if wav.size == 0:
        return []

    if sr != sample_rate:
        import torchaudio
        wav_t = torch.from_numpy(wav).float()
        wav = torchaudio.functional.resample(wav_t, sr, sample_rate).numpy().astype(np.float32)

    frame_size = max(1, int(0.02 * sample_rate))
    if wav.size < frame_size:
        return [(0.0, max(len(wav) / sample_rate, 0.1), None)]

    energy = np.square(np.abs(wav))
    frame_energy = np.array([energy[i:i + frame_size].mean() for i in range(0, len(energy) - frame_size + 1, frame_size)], dtype=np.float32)
    if frame_energy.size == 0:
        return [(0.0, max(len(wav) / sample_rate, 0.1), None)]

    noise_floor = float(np.percentile(frame_energy, 25))
    signal_floor = float(np.percentile(frame_energy, 75))
    threshold = float(max(noise_floor * 1.5, signal_floor * 0.35, 1e-6))
    active = frame_energy >= threshold
    if not active.any():
        return [(0.0, max(len(wav) / sample_rate, 0.1), None)]

    ranges: List[tuple[float, float, Optional[str]]] = []
    start_idx = None
    for idx, flag in enumerate(active):
        if flag and start_idx is None:
            start_idx = idx
        elif not flag and start_idx is not None:
            end_idx = idx
            if (end_idx - start_idx) * frame_size / sample_rate >= min_segment_sec:
                ranges.append((start_idx * frame_size / sample_rate, end_idx * frame_size / sample_rate, None))
            start_idx = None

    if start_idx is not None:
        end_idx = len(active)
        if (end_idx - start_idx) * frame_size / sample_rate >= min_segment_sec:
            ranges.append((start_idx * frame_size / sample_rate, end_idx * frame_size / sample_rate, None))

    if not ranges:
        return [(0.0, max(len(wav) / sample_rate, 0.1), None)]

    merged: List[tuple[float, float, Optional[str]]] = []
    for start, end, _ in ranges:
        if not merged or start - merged[-1][1] > min_silence_sec:
            merged.append((start, end, None))
        else:
            merged[-1] = (merged[-1][0], end, None)
    return merged


def build_multispeaker_turns(text: Optional[str], audio_data: Optional[bytes], video_data: Optional[bytes]) -> List[Dict[str, Any]]:
    """Build a speaker-aware turn list using diarization when audio is available."""
    transcript = (text or "").strip()
    media_bytes = audio_data or video_data
    diarized_segments = []
    if media_bytes:
        diarized_segments = segment_audio_for_diarization(media_bytes)

    turns = split_text_into_turns(transcript)
    if diarized_segments:
        segment_count = len(diarized_segments)
        if segment_count == 1 and turns:
            selected = [" ".join(turns)]
        else:
            selected = turns[:segment_count]
            selected.extend(f"Segment {idx + 1}" for idx in range(len(selected), segment_count))
        output: List[Dict[str, Any]] = []
        speaker_ids: Dict[str, int] = {}
        for index, turn in enumerate(selected):
            text_value = turn.strip() if isinstance(turn, str) else str(turn).strip()
            if not text_value:
                text_value = f"Speaker segment {index + 1}"
            start, end, diarized_speaker = diarized_segments[index]
            if diarized_speaker:
                if diarized_speaker not in speaker_ids:
                    speaker_ids[diarized_speaker] = len(speaker_ids)
                speaker_id = speaker_ids[diarized_speaker]
                speaker_source = "pyannote"
                speaker_label = diarized_speaker
            else:
                speaker_id = index % 8
                speaker_source = "energy_fallback"
                speaker_label = f"Segment {index + 1}"
            output.append({
                "speaker_id": speaker_id,
                "speaker_label": speaker_label,
                "speaker_source": speaker_source,
                "text": text_value,
                "start": start,
                "end": end,
            })
        return output

    if not turns:
        turns = ["Uploaded media segment"] if (audio_data or video_data) else [""]

    if len(turns) == 1:
        return [{"speaker_id": 0, "speaker_label": None, "speaker_source": "text_turn_fallback", "text": turns[0].strip()}]

    output = []
    for index, turn in enumerate(turns):
        if not turn.strip():
            continue
        output.append({"speaker_id": index % 8, "speaker_label": None, "speaker_source": "text_turn_fallback", "text": turn.strip()})
    return output


@app.on_event("startup")
def load_model() -> None:
    global MODEL, TOKENIZER, TEXT_BACKBONE, CFG, MODEL_SOURCE, TEXT_BACKBONE_SOURCE
    # load experiment config when available
    cfg_path = Path(__file__).resolve().parents[3] / "dsmr" / "configs" / "experiment.yaml"
    if cfg_path.exists():
        CFG = ExperimentConfig.from_yaml(cfg_path)
    else:
        CFG = None

    # prefer a TorchScript model if provided via env
    default_model_path = Path(__file__).resolve().parents[4] / "outputs" / "exports" / "hmemnet_ts.pt"
    model_path = os.environ.get("HMEM_TS_PATH", str(default_model_path))
    ckpt_path = os.environ.get("HMEM_CKPT_PATH")

    if model_path and Path(model_path).exists():
        try:
            MODEL = torch.jit.load(model_path, map_location="cpu")
            MODEL.eval()
            MODEL_SOURCE = str(Path(model_path).resolve())
        except Exception:
            MODEL = None

    if MODEL is None:
        # otherwise instantiate Python model and try to load checkpoint
        cfg_kwargs = {}
        if CFG is not None:
            cfg_kwargs = dict(audio_dim=CFG.audio_dim, text_dim=CFG.text_dim, vision_dim=CFG.vision_dim, hidden_dim=CFG.hidden_dim, num_speakers=CFG.num_speakers)
        MODEL = HMEMNet(**cfg_kwargs)
        MODEL.eval()
        MODEL_SOURCE = "python-model"
        if ckpt_path and Path(ckpt_path).exists():
            try:
                data = torch.load(str(ckpt_path), map_location="cpu")
                state = data.get("model_state", data)
                MODEL.load_state_dict(state)
            except Exception:
                pass

    # tokenizer / text backbone (optional)
    pretrained_text = CFG.pretrained_text if CFG is not None else None
    if AutoTokenizer is not None and pretrained_text is not None:
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(pretrained_text)
            if AutoModel is not None:
                try:
                    text_backbone_path = Path(model_path).resolve().parent / "text_backbone" if model_path else None
                    text_source = str(text_backbone_path) if text_backbone_path and text_backbone_path.exists() else pretrained_text
                    TEXT_BACKBONE = AutoModel.from_pretrained(text_source)
                    TEXT_BACKBONE.eval()
                    TEXT_BACKBONE_SOURCE = text_source
                except Exception:
                    TEXT_BACKBONE = None
                    TEXT_BACKBONE_SOURCE = None
        except Exception:
            TOKENIZER = None


def _audio_features_from_bytes(data: bytes, target_T: int) -> torch.Tensor:
    """Decode uploaded audio/video audio and pool it to model time steps."""
    source_path = None
    temporary_wav = None
    try:
        import soundfile as sf
        try:
            wav, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".media", delete=False) as source:
                source.write(data)
                source_path = source.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                temporary_wav = wav_file.name
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000", temporary_wav], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            wav, sample_rate = sf.read(temporary_wav, dtype="float32")
        wav = np.asarray(wav)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        waveform = torch.from_numpy(wav.copy())
        if sample_rate != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return torch.nn.functional.adaptive_avg_pool1d(waveform.view(1, 1, -1), target_T).transpose(1, 2)
    except Exception:
        return torch.zeros(1, target_T, 1)
    finally:
        if source_path:
            Path(source_path).unlink(missing_ok=True)
        if temporary_wav:
            Path(temporary_wav).unlink(missing_ok=True)


def _empty_pitch_summary() -> Dict[str, Any]:
    return {"times": [], "values": [], "mean_f0": None, "min_f0": None, "max_f0": None, "sample_rate": 16000}


def _extract_pitch_for_media(data: Optional[bytes | np.ndarray], sample_rate: int = 16000) -> Dict[str, Any]:
    """Estimate the fundamental frequency (F0) track from uploaded audio/video data."""
    if not data:
        return _empty_pitch_summary()

    try:
        import librosa
    except Exception:
        return {**_empty_pitch_summary(), "error": "librosa unavailable"}

    try:
        wav, sr = _extract_wav_from_media_bytes(data)
        if wav.size == 0:
            return _empty_pitch_summary()
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if sr != sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        if wav.size == 0 or np.max(np.abs(wav)) < 1e-6:
            return _empty_pitch_summary()

        f0 = librosa.yin(wav, fmin=50, fmax=500, sr=sample_rate)
        valid = np.isfinite(f0) & (f0 > 0)
        if not np.any(valid):
            return _empty_pitch_summary()

        values = f0[valid].astype(float)
        times = np.arange(len(f0), dtype=float)[valid] / sample_rate
        stats = {
            "times": [float(t) for t in times.tolist()],
            "values": [float(v) for v in values.tolist()],
            "mean_f0": float(np.mean(values)),
            "min_f0": float(np.min(values)),
            "max_f0": float(np.max(values)),
            "sample_rate": int(sample_rate),
        }
        return stats
    except Exception:
        return {**_empty_pitch_summary(), "error": "pitch extraction failed"}


def _transcribe_bytes(data: bytes, suffix: str) -> str:
    """Convert uploaded media to WAV and transcribe it with Whisper on demand."""
    global TRANSCRIBER
    if not data or AutoModel is None:
        return ""
    source_path = None
    wav_path = None
    try:
        from transformers import pipeline

        if TRANSCRIBER is None:
            model_name = os.environ.get("HMEM_ASR_MODEL", "openai/whisper-tiny")
            TRANSCRIBER = pipeline("automatic-speech-recognition", model=model_name, device=-1)
        with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as source:
            source.write(data)
            source_path = source.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav:
            wav_path = wav.name
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000", wav_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        result = TRANSCRIBER(wav_path)
        return str(result.get("text", "")).strip()
    except Exception:
        return ""
    finally:
        for path in (source_path, wav_path):
            if path:
                try:
                    Path(path).unlink()
                except OSError:
                    pass


def _video_from_bytes(data: bytes) -> torch.Tensor:
    """Decode one center-ish frame from an uploaded video using FFmpeg."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            handle.write(data)
            path = handle.name
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=224:224", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(224, 224, 3)
        return torch.from_numpy(np.transpose(frame, (2, 0, 1)).copy()).float().div(255).unsqueeze(0)
    except Exception:
        return torch.zeros(1, 3, 224, 224)
    finally:
        try:
            Path(path).unlink()
        except Exception:
            pass


@app.post("/predict")
async def predict(
    conversation_id: str = Form("demo"),
    speaker_id: str = Form("0"),
    text: Optional[str] = Form(None),
    audio: UploadFile = File(None),
    video: UploadFile = File(None),
):
    audio_data = await audio.read() if audio is not None else b""
    video_data = await video.read() if video is not None else b""
    transcript = sanitize_transcript(text)
    input_modalities = {
        "text": bool(transcript),
        "audio": bool(audio_data),
        "video": bool(video_data),
    }
    pitch_summary = _extract_pitch_for_media(audio_data or video_data) if (audio_data or video_data) else _empty_pitch_summary()
    transcript_source = "user" if transcript else None
    if audio_data or video_data:
        upload = audio if audio_data else video
        generated_transcript = _transcribe_bytes(audio_data or video_data, Path(upload.filename or "upload.bin").suffix)
        generated_transcript = sanitize_transcript(generated_transcript)
        if generated_transcript:
            transcript = generated_transcript
            transcript_source = "asr"
        elif not transcript:
            transcript = "Uploaded media segment"
            transcript_source = "fallback"

    req = PredictRequest(conversation_id=conversation_id, speaker_id=speaker_id, text=transcript or None)
    if req.text and not audio_data and not video_data:
        turns = [{
            "speaker_id": int(req.speaker_id) if req.speaker_id.isdigit() else 0,
            "speaker_label": None,
            "speaker_source": "provided",
            "text": req.text,
        }]
    else:
        turns = build_multispeaker_turns(req.text, audio_data or None, video_data or None)
    if len(turns) > 1:
        audio_batch = []
        text_batch = []
        vision_batch = []
        speaker_batch = []
        for turn in turns:
            turn_text = turn["text"]
            if turn_text and TOKENIZER is not None and TEXT_BACKBONE is not None:
                tok = TOKENIZER(turn_text, truncation=True, padding="max_length", max_length=MAX_TEXT_LENGTH, return_tensors="pt")
                with torch.no_grad():
                    outputs = TEXT_BACKBONE(**{k: v for k, v in tok.items()})
                    text_feats = outputs.last_hidden_state
            else:
                text_dim = CFG.text_dim if CFG is not None else 768
                text_feats = torch.zeros(1, 32, text_dim)

            target_T = text_feats.size(1)
            audio_feats = _audio_features_from_bytes(audio_data, target_T) if audio_data else torch.zeros(1, target_T, 1)
            vision_feats = _video_from_bytes(video_data) if video_data else torch.zeros(1, 3, 224, 224)
            audio_batch.append(audio_feats)
            text_batch.append(text_feats)
            vision_batch.append(vision_feats)
            speaker_batch.append(int(turn["speaker_id"]))

        audio_batch_tensor = torch.cat(audio_batch, dim=0)
        text_batch_tensor = torch.cat(text_batch, dim=0)
        vision_batch_tensor = torch.cat([v for v in vision_batch], dim=0)
        speaker_ids = torch.tensor(speaker_batch, dtype=torch.long)

        with torch.no_grad():
            output = MODEL(audio_batch_tensor, text_batch_tensor, vision_batch_tensor, speaker_ids)
            if isinstance(output, tuple):
                logits = output[0]
                memory_weights = output[1] if len(output) > 1 else None
            else:
                logits = output
                memory_weights = None
            probs = torch.softmax(logits.mean(dim=1), dim=-1)
            topv, topi = probs.topk(1, dim=-1)
            best_index = int(torch.argmax(topv, dim=0).item())
            emotion_idx = int(topi[best_index, 0].item())
            confidence = float(topv[best_index, 0].item())

        segment_results = []
        for i, turn in enumerate(turns):
            turn_probs = torch.softmax(logits[i].mean(dim=0), dim=-1)
            turn_confidence, turn_idx = torch.max(turn_probs, dim=-1)
            segment_results.append({
                "speaker_id": int(turn["speaker_id"]),
                "speaker_label": turn.get("speaker_label"),
                "speaker_source": turn.get("speaker_source", "text_turn_fallback"),
                "text": turn["text"],
                "emotion": int(turn_idx.item()),
                "emotion_label": emotion_label(int(turn_idx.item())),
                "confidence": float(turn_confidence.item()),
                **({"start": turn["start"], "end": turn["end"]} if "start" in turn and "end" in turn else {}),
            })

        speaker_sources = {item["speaker_source"] for item in segment_results}
        return {
            "emotion": emotion_idx,
            "emotion_label": emotion_label(emotion_idx),
            "confidence": confidence,
            "transcript": transcript,
            "transcript_source": transcript_source,
            "input_modalities": input_modalities,
            "pitch": pitch_summary,
            "segments": segment_results,
            "speaker_source": next(iter(speaker_sources)) if len(speaker_sources) == 1 else "mixed",
            "speaker_count": len({item["speaker_id"] for item in segment_results}),
            "memory_weights": memory_weights.tolist() if isinstance(memory_weights, torch.Tensor) else memory_weights,
        }

    # Single utterance fallback for the original UI flow.
    if req.text and TOKENIZER is not None and TEXT_BACKBONE is not None:
        tok = TOKENIZER(req.text, truncation=True, padding="max_length", max_length=MAX_TEXT_LENGTH, return_tensors="pt")
        with torch.no_grad():
            outputs = TEXT_BACKBONE(**{k: v for k, v in tok.items()})
            text_feats = outputs.last_hidden_state  # [1, T, D]
    else:
        text_dim = CFG.text_dim if CFG is not None else 768
        text_feats = torch.zeros(1, 32, text_dim)

    target_T = text_feats.size(1)
    if audio is not None:
        audio_feats = _audio_features_from_bytes(audio_data, target_T)
    else:
        audio_feats = torch.zeros(1, target_T, 1)

    vision_feats = _video_from_bytes(video_data) if video is not None else torch.zeros(1, 3, 224, 224)
    single_turn = turns[0] if turns else {}
    if single_turn.get("speaker_source") in {"pyannote", "energy_fallback"}:
        model_speaker_id = int(single_turn["speaker_id"])
        speaker_source = single_turn["speaker_source"]
        speaker_label = single_turn.get("speaker_label")
    else:
        model_speaker_id = int(req.speaker_id) if req.speaker_id.isdigit() else 0
        speaker_source = "provided"
        speaker_label = None

    with torch.no_grad():
        output = MODEL(audio_feats, text_feats, vision_feats, torch.tensor([model_speaker_id]))
        if isinstance(output, tuple):
            logits = output[0]
            memory_weights = output[1] if len(output) > 1 else None
        else:
            logits = output
            memory_weights = None
        probs = torch.softmax(logits.mean(dim=1), dim=-1)
        topv, topi = probs.topk(1, dim=-1)
        emotion_idx = int(topi[0, 0].item())
        confidence = float(topv[0, 0].item())

    single_segment = {
        "speaker_id": model_speaker_id,
        "speaker_label": speaker_label,
        "speaker_source": speaker_source,
        "text": single_turn.get("text", transcript or ""),
        "emotion": emotion_idx,
        "emotion_label": emotion_label(emotion_idx),
        "confidence": confidence,
    }
    if "start" in single_turn and "end" in single_turn:
        single_segment.update(start=single_turn["start"], end=single_turn["end"])

    return {
        "emotion": emotion_idx,
        "emotion_label": emotion_label(emotion_idx),
        "confidence": confidence,
        "transcript": transcript,
        "transcript_source": transcript_source,
        "input_modalities": input_modalities,
        "pitch": pitch_summary,
        "speaker_source": speaker_source,
        "segments": [single_segment],
        "memory_weights": memory_weights.tolist() if isinstance(memory_weights, torch.Tensor) else memory_weights,
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "unavailable",
        "model_source": MODEL_SOURCE,
        "text_backbone_source": TEXT_BACKBONE_SOURCE,
    }


frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
