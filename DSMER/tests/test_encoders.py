import torch

from dsmr.models.encoders.audio_encoder import AudioEncoder
from dsmr.models.encoders.text_encoder import TextEncoder
from dsmr.models.encoders.vision_encoder import VisionEncoder


def test_encoder_interfaces_still_work():
    audio_encoder = AudioEncoder(input_dim=16, hidden_dim=8)
    text_encoder = TextEncoder(input_dim=16, hidden_dim=8)
    vision_encoder = VisionEncoder(input_dim=16, hidden_dim=8)

    audio_inputs = torch.randn(2, 4, 16)
    text_inputs = torch.randn(2, 4, 16)
    vision_inputs = torch.randn(2, 3, 32, 32)

    audio_out = audio_encoder(audio_inputs)
    text_out = text_encoder(text_inputs)
    vision_out = vision_encoder(vision_inputs)

    assert audio_out.shape == (2, 4, 8)
    assert text_out.shape == (2, 4, 8)
    assert vision_out.shape == (2, 1, 8)
