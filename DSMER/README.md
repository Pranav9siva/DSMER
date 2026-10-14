# DSMER

A research-oriented multimodal emotion recognition project scaffold centered on HMEM-Net, a hierarchical multi-scale emotional memory architecture.

## Highlights

- Modular PyTorch implementation
- Memory-based multimodal reasoning with person, dialogue, and conversation memory
- Config-driven experiments and ablation studies
- Training and evaluation entry points

## Structure

- research/: research and experimentation code
- deployment/: deployment, APIs, docker, k8s, and model store
- shared/: shared configs, utilities, and schemas
- dsmr/: compatibility package for the current training and evaluation scripts

## MELD Raw Media

The annotation CSVs are stored under `dataset/MELD/`. The raw multimedia archive is approximately 10.1 GiB and contains nested train, dev, and test MP4 archives. Download and extract it with PowerShell:

```powershell
$env:PYTHONPATH = "DSMER"
python DSMER/tools/download_meld.py --raw
python DSMER/tools/extract_meld_media.py --archive dataset/MELD/raw/MELD.Raw.tar.gz --out dataset/MELD/media --split all
```

The archive uses filenames such as `dia0_utt0.mp4`, which are matched to CSV rows using `Dialogue_ID` and `Utterance_ID`. Audio is decoded from each MP4 with FFmpeg at mono 16 kHz; a resized center video frame is used for vision input. FFmpeg must be installed and available on `PATH`.

Validate coverage before training:

```powershell
python DSMER/tools/validate_meld_media.py --media-root dataset/MELD/media/train --split train
python DSMER/tools/validate_meld_media.py --media-root dataset/MELD/media/dev --split dev
python DSMER/tools/validate_meld_media.py --media-root dataset/MELD/media/test --split test
```

The current archive has complete train and test annotation coverage. The dev release contains one annotation without a matching video and a few extra media files; the validator reports these explicitly. Pass a media root to feature extraction to persist real modality data:

```powershell
python DSMER/tools/extract_features.py --dataset meld --split train --media-root dataset/MELD/media/train --out dataset/processed/meld_real
```
