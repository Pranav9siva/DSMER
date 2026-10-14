from typing import Optional

def preprocess_audio(wav_path: Optional[str] = None) -> None:
    """Basic audio preprocessing placeholder.

    Real pipeline should: load audio, resample, normalize, detect speech regions,
    and optionally extract spectrogram features.
    For now this function is a no-op that signals where to implement extraction.
    """
    print("Audio preprocessing placeholder - implement resampling / feature extraction here")
