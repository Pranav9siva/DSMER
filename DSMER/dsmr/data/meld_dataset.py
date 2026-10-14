from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import warnings
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import load_dataset, Audio as HF_Audio, Video as HF_Video
from transformers import AutoTokenizer

try:
    import torchaudio
except Exception:  # pragma: no cover - optional dependency
    torchaudio = None

try:
    import torchvision.io as tv_io
    import torchvision.transforms as T
except Exception:  # pragma: no cover - optional dependency
    tv_io = None
    T = None


class MELDConversationDataset(Dataset):
    """Hugging Face `datasets` backed MELD conversation dataset.

    This loader tries to use the HF `datasets` library to load `meld` (or another
    dataset name passed in) and extract text, audio and optionally video features.

    Behavior:
    - Text: tokenized with a `transformers` tokenizer (returns `input_ids` and `attention_mask`).
    - Audio: returns a 1-D tensor waveform resampled/padded to `audio_max_samples`.
    - Video: if present, extracts a center frame and returns a (3, H, W) tensor resized to 224x224.

    The class is robust to missing modalities and falls back to zero tensors when
    a modality is not available in the HF dataset.
    """

    EMOTION_TO_ID = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "joy": 3,
        "neutral": 4,
        "sadness": 5,
        "surprise": 6,
    }

    def __init__(
        self,
        dataset_name: str = "meld",
        split: str = "train",
        tokenizer_name: str = "bert-base-uncased",
        max_text_len: int = 128,
        audio_max_seconds: float = 5.0,
        audio_sample_rate: int = 16000,
        processed_dir: Optional[str] = None,
        media_root: Optional[str] = None,
        audio_root: Optional[str] = None,
        video_root: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer_name = tokenizer_name
        self.max_text_len = max_text_len
        self.audio_max_seconds = audio_max_seconds
        self.audio_sample_rate = audio_sample_rate

        self.local_csv_dir = Path.cwd() / "dataset" / "MELD"
        self.repo_root = Path(__file__).resolve().parents[3]
        self.repo_csv_dir = self.repo_root / "dataset" / "MELD"

        if self.dataset_name.lower() == "meld" and self._has_local_split(self.split):
            self.dataset = self._load_local_split(self.split)
        else:
            try:
                if "+" in self.split:
                    raise Exception("merged splits not supported on Hub; use local CSVs")
                self.dataset = load_dataset(self.dataset_name, split=self.split)
            except Exception:
                self.dataset = self._load_local_split(self.split, raise_on_missing=True)

        # find likely text/audio/video/label columns
        self.text_column = self._find_text_column()
        self.audio_column = self._find_feature_column(HF_Audio)
        self.video_column = self._find_feature_column(HF_Video)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # audio max length in samples
        self.audio_max_samples = int(self.audio_sample_rate * self.audio_max_seconds)

        # processed features directory (per-sample .npz files)
        self.processed_dir = Path(processed_dir) if processed_dir else (Path.cwd() / "dataset" / "processed" / self.dataset_name)
        self.processed_available = self.processed_dir.exists()
        self.media_root = Path(media_root) if media_root else None
        self.audio_root = Path(audio_root) if audio_root else None
        self.video_root = Path(video_root) if video_root else None
        self._media_index: Optional[Dict[str, Path]] = None

        # transforms for video frames
        if T is not None:
            self.video_transform = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])
        else:
            self.video_transform = None

        # build label mapping if labels are strings
        self.label_map = self._build_label_map()
        self.speaker_column = self._find_speaker_column()
        self.speaker_map = self._build_speaker_map() or {}

    def _build_media_index(self) -> Dict[str, Path]:
        """Index media once so per-row lookup does not repeatedly scan the archive."""
        if self._media_index is not None:
            return self._media_index
        roots = [root for root in (self.media_root, self.audio_root, self.video_root) if root is not None]
        index: Dict[str, Path] = {}
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".mp4", ".avi", ".mkv", ".webm"}:
                    index.setdefault(path.stem.lower(), path)
        self._media_index = index
        return index

    def _resolve_media_path(self, example: Dict, suffixes: tuple[str, ...]) -> Optional[Path]:
        """Resolve a MELD utterance to a media file using its dialogue/utterance IDs."""
        dialogue_id = example.get("Dialogue_ID")
        utterance_id = example.get("Utterance_ID")
        if dialogue_id is None or utterance_id is None:
            return None
        stem = f"dia{int(dialogue_id)}_utt{int(utterance_id)}".lower()
        roots = [root for root in (self.audio_root, self.video_root, self.media_root) if root is not None]
        for root in roots:
            for split_dir in (root / self.split, root):
                for suffix in suffixes:
                    candidate = split_dir / f"{stem}{suffix}"
                    if candidate.exists():
                        return candidate
        for key, path in self._build_media_index().items():
            if key == stem and path.suffix.lower() in suffixes:
                return path
        return None

    def _load_local_audio(self, path: Path) -> Optional[torch.Tensor]:
        if path.suffix.lower() == ".mp4":
            try:
                result = subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(self.audio_sample_rate), "-f", "f32le", "pipe:1"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                waveform = torch.from_numpy(np.frombuffer(result.stdout, dtype=np.float32).copy())
                return self._pad_or_truncate_audio(waveform)
            except (OSError, subprocess.SubprocessError, ValueError):
                return None
        try:
            import soundfile as sf
            array, sample_rate = sf.read(str(path), dtype="float32")
            waveform = torch.from_numpy(np.asarray(array))
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)
            if sample_rate != self.audio_sample_rate and torchaudio is not None:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.audio_sample_rate)
            return self._pad_or_truncate_audio(waveform)
        except Exception:
            return None

    def _load_local_video(self, path: Path) -> Optional[torch.Tensor]:
        """Decode one resized center frame using torchvision or the FFmpeg CLI."""
        if tv_io is not None and hasattr(tv_io, "read_video"):
            try:
                frames, _, _ = tv_io.read_video(str(path), pts_unit="sec")
                if frames.shape[0] > 0:
                    frame = frames[frames.shape[0] // 2].numpy()
                    return self.video_transform(frame) if self.video_transform is not None else torch.from_numpy(np.transpose(frame, (2, 0, 1))).float() / 255.0
            except Exception:
                pass
        try:
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=224:224", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(224, 224, 3)
            return torch.from_numpy(np.transpose(frame, (2, 0, 1)).copy()).float() / 255.0
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def _find_text_column(self) -> Optional[str]:
        candidates = ["Utterance", "utterance", "text", "sentence", "dialogue", "utterance_text"]
        for c in candidates:
            if c in self.dataset.column_names:
                return c
        # fallback: pick first string column
        for c in self.dataset.column_names:
            if str(self.dataset.features[c]).lower().startswith("value") or "string" in str(self.dataset.features[c]).lower():
                return c
        return None

    def _find_speaker_column(self) -> Optional[str]:
        candidates = ["Speaker", "speaker", "speaker_id", "SpeakerID", "speakerID"]
        for c in candidates:
            if c in self.dataset.column_names:
                return c
        return None

    def _build_speaker_map(self) -> Optional[Dict[str, int]]:
        if not self.speaker_column or self.speaker_column not in self.dataset.column_names:
            return None
        values = list({self.dataset[i][self.speaker_column] for i in range(len(self.dataset))})
        if all(isinstance(v, str) for v in values):
            uniq = sorted(set(values))
            return {v: i for i, v in enumerate(uniq)}
        return None

    def _find_feature_column(self, feature_type):
        for name, feat in self.dataset.features.items():
            if isinstance(feat, feature_type):
                return name
        return None

    def _build_label_map(self) -> Optional[Dict[str, int]]:
        if "Emotion" in self.dataset.column_names:
            return dict(self.EMOTION_TO_ID)
        # try to discover label column and mapping
        candidates = ["label", "emotion", "Emotion", "labels"]
        for c in candidates:
            if c in self.dataset.column_names:
                vals = list({self.dataset[i][c] for i in range(len(self.dataset))})
                if all(isinstance(v, str) for v in vals):
                    uniq = sorted(set(v.lower() for v in vals))
                    return {v: i for i, v in enumerate(uniq)}
        return None

    def _has_local_split(self, split: str) -> bool:
        if "+" in split:
            return all((self.local_csv_dir / f"{part}_sent_emo.csv").exists() or (self.repo_csv_dir / f"{part}_sent_emo.csv").exists() for part in split.split("+"))
        return (self.local_csv_dir / f"{split}_sent_emo.csv").exists() or (self.repo_csv_dir / f"{split}_sent_emo.csv").exists()

    def _load_local_split(self, split: str, raise_on_missing: bool = False):
        files = []
        parts = split.split("+") if "+" in split else [split]
        for part in parts:
            local_path = self.local_csv_dir / f"{part}_sent_emo.csv"
            repo_path = self.repo_csv_dir / f"{part}_sent_emo.csv"
            if local_path.exists():
                files.append(str(local_path))
            elif repo_path.exists():
                files.append(str(repo_path))
            elif raise_on_missing:
                raise RuntimeError(f"Missing local MELD split CSV for '{part}' at {local_path} or {repo_path}")
        if not files:
            raise RuntimeError(f"No local MELD CSV files found for split '{split}'")
        if "+" in split:
            return load_dataset("csv", data_files={"train": files}, split="train")
        return load_dataset("csv", data_files={split: files[0]}, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def _pad_or_truncate_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        waveform = waveform.float()
        n = waveform.shape[-1]
        if n > self.audio_max_samples:
            waveform = waveform[: self.audio_max_samples]
        elif n < self.audio_max_samples:
            pad = self.audio_max_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # if processed artifacts exist, load them
        from pathlib import Path

        processed_file = Path(self.processed_dir) / f"{self.split}_{index}.npz"
        if processed_file.exists():
            try:
                data = np.load(processed_file)
                text_input_ids = torch.from_numpy(data["input_ids"]).long()
                text_attention_mask = torch.from_numpy(data["attention_mask"]).long()
                audio_tensor = torch.from_numpy(data["audio"]).float()
                vision_tensor = torch.from_numpy(data["vision"]).float()
                label_id = int(data["label"])
                speaker_id = int(data["speaker_id"]) if "speaker_id" in data else 0
                return {
                    "text_input_ids": text_input_ids,
                    "text_attention_mask": text_attention_mask,
                    "audio": audio_tensor,
                    "vision": vision_tensor,
                    "label": torch.tensor(label_id, dtype=torch.long),
                    "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
                    "audio_available": torch.tensor(bool(np.any(audio_tensor)), dtype=torch.bool),
                    "video_available": torch.tensor(bool(np.any(vision_tensor)), dtype=torch.bool),
                }
            except Exception:
                # fall back to live decoding
                pass

        example = self.dataset[index]

        # Text
        text = example.get(self.text_column, "") if self.text_column else ""
        tok = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_text_len, return_tensors="pt"
        )
        text_input_ids = tok["input_ids"].squeeze(0)
        text_attention_mask = tok["attention_mask"].squeeze(0)

        # Audio: prefer locally resolved MELD media when configured.
        local_audio = self._resolve_media_path(example, (".wav", ".flac", ".mp3")) if self.audio_root or self.media_root else None
        if local_audio is None and self.media_root is not None and self.audio_root is None:
            local_audio = self._resolve_media_path(example, (".mp4", ".avi", ".mkv", ".webm"))
        if local_audio is not None:
            audio_tensor = self._load_local_audio(local_audio)
            if audio_tensor is None:
                audio_tensor = torch.zeros(self.audio_max_samples, dtype=torch.float32)
        elif self.audio_column and self.audio_column in example and example[self.audio_column] is not None:
            audio_obj = example[self.audio_column]
            if isinstance(audio_obj, dict) and "array" in audio_obj:
                arr = audio_obj["array"]
                sr = audio_obj.get("sampling_rate", None)
                waveform = torch.from_numpy(np.array(arr))
                if sr and sr != self.audio_sample_rate and torchaudio is not None:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.audio_sample_rate)
                # ensure 1D
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=1)
                audio_tensor = self._pad_or_truncate_audio(waveform)
            else:
                # fallback: zero audio
                audio_tensor = torch.zeros(self.audio_max_samples, dtype=torch.float32)
        else:
            audio_tensor = torch.zeros(self.audio_max_samples, dtype=torch.float32)

        # Video: prefer locally resolved MELD media when configured.
        local_video = self._resolve_media_path(example, (".mp4", ".avi", ".mkv", ".webm")) if self.video_root or self.media_root else None
        if local_video is not None:
            decoded_video = self._load_local_video(local_video)
            vision_tensor = decoded_video if decoded_video is not None else torch.zeros(3, 224, 224, dtype=torch.float32)
        elif self.video_column and self.video_column in example and example[self.video_column] is not None and tv_io is not None:
            try:
                vid = example[self.video_column]
                # datasets video feature may provide 'path' or 'array'
                if isinstance(vid, dict) and "array" in vid:
                    frames = np.array(vid["array"])
                    # take center frame
                    mid = frames.shape[0] // 2
                    frame = frames[mid]
                    if self.video_transform is not None:
                        vision_tensor = self.video_transform(frame)
                    else:
                        vision_tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float() / 255.0
                else:
                    # try reading from path
                    path = vid.get("path") if isinstance(vid, dict) else None
                    if path:
                        frames, _, _ = tv_io.read_video(path, pts_unit="sec")
                        mid = frames.shape[0] // 2
                        frame = frames[mid].numpy()
                        if self.video_transform is not None:
                            vision_tensor = self.video_transform(frame)
                        else:
                            vision_tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float() / 255.0
                    else:
                        vision_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
            except Exception:
                vision_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
        else:
            vision_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        # Labels
        label = None
        for cand in ("label", "emotion", "Emotion", "labels"):
            if cand in example:
                label = example[cand]
                break
        if isinstance(label, str):
            if self.label_map and label.lower() in self.label_map:
                label_id = self.label_map[label.lower()]
            else:
                label_id = 0
        elif isinstance(label, (int, float)):
            label_id = int(label)
        else:
            label_id = -1

        speaker_id = 0
        if self.speaker_column and self.speaker_column in example:
            speaker_value = example[self.speaker_column]
            if isinstance(speaker_value, str):
                speaker_id = self.speaker_map.get(speaker_value, self.speaker_map.get(speaker_value.lower(), 0))
            elif isinstance(speaker_value, (int, float)):
                speaker_id = int(speaker_value)

        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "audio": audio_tensor,
            "vision": vision_tensor,
            "label": torch.tensor(label_id, dtype=torch.long),
            "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
            "audio_available": torch.tensor(bool(torch.any(audio_tensor)), dtype=torch.bool),
            "video_available": torch.tensor(bool(torch.any(vision_tensor)), dtype=torch.bool),
        }
