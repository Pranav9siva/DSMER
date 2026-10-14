# Deployment skeleton

This folder contains a minimal FastAPI inference skeleton and tools to export models to TorchScript/ONNX.

Quick start:

1. Train the model from the workspace root. Training saves the best dev macro-F1 checkpoint and automatically exports TorchScript, ONNX, and the matching fine-tuned text backbone under `outputs/`:

```powershell
python DSMER/train.py --media_root dataset\MELD\media --batch_size 1
```

Use `--batch_size 1` on GPUs with limited VRAM. To export an existing checkpoint manually:

```bash
python DSMER/tools/export_model.py --checkpoint outputs/checkpoint_best.pth --outdir outputs/exports
```

2. Install deployment dependencies and run the FastAPI app (development) from the workspace root:

```powershell
pip install -r DSMER/requirements.txt
python DSMER/deployment/backend/api/app.py
```

The API automatically uses `outputs/exports/hmemnet_ts.pt` and loads the matching fine-tuned text backbone from `outputs/exports/text_backbone`. Override the TorchScript model with `HMEM_TS_PATH` when needed. The `/health` response reports both loaded artifact paths.

Start it from the repository root:

```powershell
Set-Location DSMER
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn deployment.backend.api.app:app --host 127.0.0.1 --port 8000
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

The `/predict` endpoint accepts multipart form fields `conversation_id`, `speaker_id`, and `text`, plus `audio` and `video` uploads. When media is uploaded, the API transcribes it with `openai/whisper-tiny` and returns the generated `transcript` together with the predicted MELD emotion index and confidence. The Whisper model downloads on first media request and is cached locally. Set `HMEM_ASR_MODEL` to use another Transformers ASR model.
