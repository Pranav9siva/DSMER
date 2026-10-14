# HMEM-Net

HMEM-Net is a research-oriented multimodal emotion recognition architecture centered on a hierarchical emotional memory system.

## Core idea

The model introduces three memory modules:

1. Person Memory for long-term speaker emotional traits.
2. Dialogue Memory for short-term conversational dynamics.
3. Conversation Memory for global semantic and emotional history.

## Modules

- Encoders: audio, text, vision
- Memory controller: adaptive gating over memory sources
- Memory fusion: memory-guided multimodal fusion
- Memory update engine: adaptive updates with different rates
- Emotion head: prediction over discrete emotion classes

## Usage

```python
from dsmr.models import HMEMNet

model = HMEMNet()
```
