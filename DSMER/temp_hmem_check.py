import torch
from dsmr.models.hmem_net import HMEMNet

model = HMEMNet()
audio = torch.randn(2, 32, 1024)
text = torch.randn(2, 32, 768)
vision = torch.randn(2, 3, 224, 224)
spk = torch.tensor([0, 1], dtype=torch.long)
logits, weights, memories, fused = model(audio, text, vision, spk)
print('ok', logits.shape, weights.shape, len(memories), fused.shape)
