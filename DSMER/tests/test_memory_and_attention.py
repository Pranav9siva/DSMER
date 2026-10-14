import torch
from dsmr.models.memory.memory_manager import HierarchicalMemoryManager
from dsmr.models.memory.speech_attention import MemoryGuidedSpeechAttention


def test_memory_and_attention_shapes():
    B, T, F = 2, 8, 1024
    hidden = 256
    audio = torch.randn(B, T, F)
    context = torch.randn(B, T, hidden)
    speaker_ids = torch.tensor([0, 1])

    mem_mgr = HierarchicalMemoryManager(memory_dim=hidden, num_speakers=4)
    person, dialogue, conversation, person_att = mem_mgr(context, speaker_ids, update=False)
    assert person.shape[0] == B
    assert dialogue.shape[0] == B
    assert conversation.shape[0] == B

    mgsa = MemoryGuidedSpeechAttention(feature_dim=F, memory_dim=hidden, attention_dim=hidden)
    attended, attn_weights = mgsa(audio, person, dialogue, conversation)
    assert attended.shape == (B, T, F)
    assert attn_weights.shape == (B, T)


def test_variable_length_attention_masking():
    B, T, F = 2, 6, 256
    hidden = 64
    audio = torch.randn(B, T, F)
    person = torch.randn(B, hidden)
    dialogue = torch.randn(B, T, hidden)
    conversation = torch.randn(B, T, hidden)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)

    mgsa = MemoryGuidedSpeechAttention(feature_dim=F, memory_dim=hidden, attention_dim=hidden)
    attended, attn_weights = mgsa(audio, person, dialogue, conversation, attention_mask=mask)

    assert attended.shape == (B, T, F)
    assert attn_weights.shape == (B, T)
    assert torch.allclose(attn_weights[0, 4:], torch.zeros_like(attn_weights[0, 4:]))
    assert torch.allclose(attn_weights[1, 3:], torch.zeros_like(attn_weights[1, 3:]))
