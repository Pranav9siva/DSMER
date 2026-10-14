from pathlib import Path
import tempfile

import torch

from dsmr.models.memory.conversation_memory import ConversationMemory
from dsmr.models.memory.dialogue_memory import DialogueMemory
from dsmr.models.memory.memory_controller import AdaptiveMemoryController
from dsmr.models.memory.memory_fusion import MemoryGuidedFusion
from dsmr.models.memory.memory_manager import HierarchicalMemoryManager
from dsmr.models.memory.person_memory import PersonMemory


def test_person_memory_batch_inference_and_checkpointing():
    person_memory = PersonMemory(memory_dim=8, num_speakers=2, update_rate=0.2)
    embeddings = torch.randn(3, 8)
    speaker_ids = torch.tensor([0, 1, 0])

    retrieved, weights, bank = person_memory(embeddings, speaker_ids, update=False)
    assert retrieved.shape == (3, 8)
    assert weights.shape == (3, 1)
    assert bank.shape == (2, 8)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "person_memory.pt"
        person_memory.save_checkpoint(checkpoint_path)
        loaded = PersonMemory(memory_dim=8, num_speakers=2)
        loaded.load_checkpoint(checkpoint_path)
        assert torch.allclose(loaded.memory_bank, person_memory.memory_bank)


def test_dialogue_memory_handles_masked_history():
    dialogue_memory = DialogueMemory(memory_dim=8, max_turns=6)
    embeddings = torch.randn(2, 4, 8)
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    history = dialogue_memory(embeddings, attention_mask=attention_mask)
    assert history.shape == (2, 4, 8)
    assert dialogue_memory.retrieve().shape == (2, 6, 8)


def test_conversation_memory_updates_and_resets():
    conversation_memory = ConversationMemory(memory_dim=8, max_steps=4)
    embeddings = torch.randn(2, 3, 8)
    state = conversation_memory(embeddings)
    assert state.shape == (2, 8)

    reset_state = conversation_memory.reset_memory()
    assert reset_state.shape == (1, 8)


def test_conversation_and_dialogue_memories_produce_gradients():
    dialogue_memory = DialogueMemory(memory_dim=8, max_turns=6)
    conversation_memory = ConversationMemory(memory_dim=8, max_steps=4)
    embeddings = torch.randn(2, 3, 8, requires_grad=True)

    dialogue_output = dialogue_memory(embeddings)
    conversation_output = conversation_memory(embeddings)
    loss = dialogue_output.mean() + conversation_output.mean()
    loss.backward()

    dialogue_grad = sum(p.grad.abs().sum().item() for p in dialogue_memory.parameters() if p.grad is not None)
    conversation_grad = sum(p.grad.abs().sum().item() for p in conversation_memory.parameters() if p.grad is not None)

    assert dialogue_grad > 0.0
    assert conversation_grad > 0.0


def test_memory_manager_exposes_hierarchy_api():
    manager = HierarchicalMemoryManager(memory_dim=8, num_speakers=2)
    context = torch.randn(2, 4, 8)
    speaker_ids = torch.tensor([0, 1])

    person, dialogue, conversation, person_attention = manager(context, speaker_ids, update=False)
    assert person.shape == (2, 8)
    assert dialogue.shape == (2, 4, 8)
    assert conversation.shape == (2, 4, 8)
    assert person_attention.shape == (2, 1)

    merged = manager.merge_memories(person, dialogue, conversation)
    assert merged.shape == (2, 8)

    scores, indices = manager.rank_memories(torch.randn(2, 8), top_k=2)
    assert scores.shape == (2, 2)
    assert indices.shape == (2, 2)


def test_memory_controller_and_fusion_shape_contract():
    controller = AdaptiveMemoryController(hidden_dim=8)
    fusion = MemoryGuidedFusion(hidden_dim=8)

    context = torch.randn(2, 4, 8)
    person_memory = torch.randn(2, 8)
    dialogue_memory = torch.randn(2, 4, 8)
    conversation_memory = torch.randn(2, 4, 8)

    weights = controller(context, person_memory, dialogue_memory, conversation_memory)
    assert weights.shape == (2, 3)

    fused = fusion(context, person_memory.unsqueeze(1).expand(-1, 4, -1), dialogue_memory, conversation_memory, memory_weights=weights)
    assert fused.shape == (2, 4, 8)
