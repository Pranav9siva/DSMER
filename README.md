# MultiModel Speech Recognition

Multimodal emotion recognition for MELD conversations. The HMEMNet pipeline combines transcript, audio, video, speaker, and conversation-memory signals to classify each utterance into anger, disgust, fear, joy, neutral, sadness, or surprise.

## Requirements

- Python 3.11 or newer
- MELD CSV files and extracted media under `dataset/MELD/`
- FFmpeg on `PATH` for decoding local audio/video
- NVIDIA GPU recommended; training falls back to CPU when CUDA is unavailable

The MELD dataset, trained checkpoints, and exported model files are not included in this repository. Download the dataset using the instructions in [DSMER/README.md](DSMER/README.md), then train the model to generate the artifacts used for evaluation and inference.

## Setup

From the workspace root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r DSMER\requirements.txt
```

The training run expects the MELD media at `dataset\MELD\media`. Supply another location with `--media_root` if needed.

## Train

```powershell
.\.venv\Scripts\python.exe DSMER\train.py --media_root dataset\MELD\media --batch_size 1
```

Batch size 1 is recommended for GPUs with 4 GB VRAM. Training evaluates on the dev split, selects the best checkpoint by dev macro-F1, and exports the paired model artifacts:

- `outputs\checkpoint_best.pth`
- `outputs\training_results.json`
- `outputs\exports\hmemnet_ts.pt` and `hmemnet.onnx`
- `outputs\exports\text_backbone\` (fine-tuned DistilBERT)

## Evaluate

Evaluate the selected checkpoint and its matching text backbone on MELD test data:

```powershell
.\.venv\Scripts\python.exe DSMER\tools\evaluate.py `
	--ckpt outputs\checkpoint_best.pth `
	--split test `
	--batch_size 1 `
	--media_root dataset\MELD\media `
	--text_backbone_path outputs\exports\text_backbone
```

Latest run selected epoch 9 of 14 by dev macro-F1. Dev accuracy was 0.5681 and dev macro-F1 was 0.4097. On the 2,610-example test split, accuracy was 0.5935, weighted F1 was 0.5866, and macro-F1 was 0.4016.
Per-class test results:

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| Anger | 0.412 | 0.365 | 0.387 |
| Disgust | 0.207 | 0.279 | 0.238 |
| Fear | 0.091 | 0.080 | 0.085 |
| Joy | 0.592 | 0.520 | 0.554 |
| Neutral | 0.731 | 0.791 | 0.759 |
| Sadness | 0.338 | 0.250 | 0.287 |
| Surprise | 0.483 | 0.520 | 0.501 |

Fear is the weakest class and neutral is the strongest. The full report and confusion matrix are saved to `outputs\evaluation_test.json`.

## Run the Web App

```powershell
.\.venv\Scripts\python.exe DSMER\deployment\backend\api\app.py
```

Open `http://127.0.0.1:8000`. The app loads `outputs\exports\hmemnet_ts.pt` and the matching text backbone. Check the active artifact paths at `http://127.0.0.1:8000/health`. If you retrain while the app is running, restart the API to load the new files.

Prediction responses retain the numeric `emotion` field and include `emotion_label`, `input_modalities`, and `transcript_source`. Multi-turn results also include per-turn emotion labels, speaker source, and diarization timestamps when available. Pyannote speaker labels require an authorized `HF_TOKEN`; otherwise the API marks its energy/text segmentation fallback.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest DSMER\tests -q
```
