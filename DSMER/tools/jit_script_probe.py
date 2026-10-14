import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dsmr.models.hmem_net import HMEMNet

class Exportable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = HMEMNet(audio_dim=1, text_dim=768, vision_dim=64, hidden_dim=256, num_speakers=100)
    def forward(self, audio, text, vision, speaker_ids):
        logits, _, _, _, _ = self.model(audio, text, vision, speaker_ids, update=False, use_memories=False)
        return logits

m = Exportable().eval()
audio = torch.randn(1, 8, 1)
text = torch.randn(1, 8, 768)
vision = torch.randn(1, 3, 224, 224)
speaker = torch.tensor([0])

print('about to script')
try:
    scripted = torch.jit.script(m)
    print('scripted ok')
    scripted.save('DSMER/exports/probe_scripted.pt')
    print('saved')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('FAILED', type(e).__name__, e)
