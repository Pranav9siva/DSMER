import asyncio
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import deployment.backend.api.app as api_app

from deployment.backend.api.app import (
    build_multispeaker_turns,
    split_text_into_turns,
    segment_audio_for_diarization,
    try_pyannote_diarization,
    sanitize_transcript,
)


def test_split_text_into_turns_handles_long_dialogue():
    text = "Hello there! I am very happy today. Why are you angry? This is okay."
    turns = split_text_into_turns(text)

    assert len(turns) >= 2
    assert all(turn.strip() for turn in turns)


def test_build_multispeaker_turns_assigns_unique_speakers():
    turns = build_multispeaker_turns("I am happy. I am angry. I am sad.", None, None)

    assert len(turns) >= 3
    speakers = [turn["speaker_id"] for turn in turns]
    assert len(set(speakers)) >= 2
    assert all(turn["text"].strip() for turn in turns)


def test_segment_audio_for_diarization_detects_multiple_speech_bursts():
    sample_rate = 16000
    sr = np.arange(0, 5, 1 / sample_rate)
    tone_a = 2.5 * np.sin(2 * np.pi * 220 * sr[:sample_rate])
    silence = np.zeros(sample_rate // 2)
    tone_b = 3.0 * np.sin(2 * np.pi * 330 * sr[:sample_rate])
    audio = np.concatenate([silence, tone_a, silence, tone_b, silence])
    segments = segment_audio_for_diarization(audio, sample_rate=sample_rate)

    assert len(segments) >= 2
    assert all(end > start and speaker is None for start, end, speaker in segments)


def test_build_multispeaker_turns_preserves_diarized_speaker_identity(monkeypatch):
    monkeypatch.setattr(
        api_app,
        "segment_audio_for_diarization",
        lambda *_args, **_kwargs: [
            (0.0, 1.0, "SPEAKER_00"),
            (1.0, 2.0, "SPEAKER_01"),
            (2.0, 3.0, "SPEAKER_00"),
        ],
    )
    turns = api_app.build_multispeaker_turns("First turn. Second turn. Third turn.", b"audio", None)

    assert [turn["speaker_id"] for turn in turns] == [0, 1, 0]
    assert [turn["speaker_label"] for turn in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert all(turn["speaker_source"] == "pyannote" for turn in turns)


def test_try_pyannote_diarization_preserves_track_labels(monkeypatch):
    class FakeSegment:
        start = 0.25
        end = 0.8

    class FakeAnnotation:
        def itertracks(self, yield_label=False):
            assert yield_label
            return iter([(FakeSegment(), 0, "SPEAKER_00")])

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, _model_name, token=None):
            assert token == "test-token"
            return cls()

        def __call__(self, _wav_path):
            return FakeAnnotation()

    pyannote_module = ModuleType("pyannote")
    audio_module = ModuleType("pyannote.audio")
    audio_module.Pipeline = FakePipeline
    pyannote_module.audio = audio_module
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    segments = api_app.try_pyannote_diarization(np.zeros(1600, dtype=np.float32))

    assert len(segments) == 1
    assert segments[0][0] == 0.25
    assert abs(segments[0][1] - 0.8) < 1e-6
    assert segments[0][2] == "SPEAKER_00"


def test_predict_returns_emotion_label_and_input_provenance(monkeypatch):
    class FakeTokenizer:
        max_length = None

        def __call__(self, *_args, **_kwargs):
            self.max_length = _kwargs["max_length"]
            return {
                "input_ids": torch.ones((1, self.max_length), dtype=torch.long),
                "attention_mask": torch.ones((1, self.max_length), dtype=torch.long),
            }

    class FakeTextBackbone:
        def __call__(self, input_ids, **_kwargs):
            return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 768)))

    class FakeModel:
        def __call__(self, *_args):
            logits = torch.zeros((1, 32, 7))
            logits[:, :, 3] = 1.0
            return logits

    tokenizer = FakeTokenizer()
    monkeypatch.setattr(api_app, "TOKENIZER", tokenizer)
    monkeypatch.setattr(api_app, "TEXT_BACKBONE", FakeTextBackbone())
    monkeypatch.setattr(api_app, "MODEL", FakeModel())
    monkeypatch.setattr(api_app, "CFG", SimpleNamespace(text_dim=768))

    utterance = "I feel happy. I am pleased."
    result = asyncio.run(api_app.predict("test-conversation", "0", utterance, None, None))

    assert result["emotion"] == 3
    assert result["emotion_label"] == "joy"
    assert result["input_modalities"] == {"text": True, "audio": False, "video": False}
    assert result["transcript_source"] == "user"
    assert tokenizer.max_length == 128
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == utterance


def test_predict_includes_pitch_summary_for_uploaded_audio(monkeypatch):
    class FakeUpload:
        filename = "sample.wav"

        async def read(self):
            return b"audio-bytes"

    class FakeModel:
        def __call__(self, *_args):
            logits = torch.zeros((1, 32, 7))
            logits[:, :, 3] = 1.0
            return logits

    monkeypatch.setattr(api_app, "TOKENIZER", None)
    monkeypatch.setattr(api_app, "TEXT_BACKBONE", None)
    monkeypatch.setattr(api_app, "MODEL", FakeModel())
    monkeypatch.setattr(api_app, "CFG", SimpleNamespace(text_dim=768))
    monkeypatch.setattr(api_app, "segment_audio_for_diarization", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        api_app,
        "_extract_pitch_for_media",
        lambda *_args, **_kwargs: {
            "times": [0.0, 0.5, 1.0],
            "values": [120.0, 130.0, 140.0],
            "mean_f0": 130.0,
            "min_f0": 120.0,
            "max_f0": 140.0,
        },
    )

    result = asyncio.run(api_app.predict("pitch-test", "0", None, FakeUpload(), None))

    assert result["pitch"]["mean_f0"] == 130.0
    assert result["pitch"]["max_f0"] == 140.0


def test_try_pyannote_diarization_returns_empty_without_package():
    segments = try_pyannote_diarization(None, sample_rate=16000)

    assert segments == []


def test_sanitize_transcript_rejects_punctuation_only_output():
    cleaned = sanitize_transcript(". . . . .")

    assert cleaned == ""
