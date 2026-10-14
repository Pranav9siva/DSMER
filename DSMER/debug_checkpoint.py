"""
Debug checkpoint loading and architecture mismatch.
"""
import torch
from pathlib import Path

checkpoint = Path("outputs/checkpoint_epoch_2.pth")
print(f"Checkpoint exists: {checkpoint.exists()}")

if checkpoint.exists():
    print(f"\nLoading: {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    
    print(f"Checkpoint type: {type(ckpt)}")
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    
    print(f"\nState dict keys ({len(state)}): ")
    for k in list(state.keys())[:10]:
        print(f"  {k}: {state[k].shape if hasattr(state[k], 'shape') else type(state[k])}")
    
    # Check if this looks like a trained model
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    if len(state) == 0:
        print("❌ EMPTY CHECKPOINT - No weights saved!")
    elif len(state) < 20:
        print("⚠️  VERY FEW weights - Might be incomplete")
    else:
        print("✅ Checkpoint has weights")
    
    # Check if it's a BiLSTM or LSTM-based model
    if any("lstm" in k.lower() or "gru" in k.lower() for k in state.keys()):
        print("⚠️  Uses LSTM/GRU - might not be multimodal model")
    
    # Check audio encoder dimensions
    audio_keys = [k for k in state.keys() if "audio" in k.lower()]
    if audio_keys:
        print(f"✅ Has audio encoder ({len(audio_keys)} layers)")
        for k in audio_keys[:3]:
            print(f"   {k}: {state[k].shape}")
    else:
        print("❌ No audio encoder found")
