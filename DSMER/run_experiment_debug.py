import sys
import traceback
import os
from pathlib import Path
sys.path.insert(0, r'D:\MultiModel_speech_Recognitaion\DSMER')
print('DEBUG RUNNER: start')
print('PYTHON:', sys.executable)
print('CWD:', os.getcwd())
try:
    from dsmr.utils.config import ExperimentConfig
    from dsmr.models.hmem_net import HMEMNet
    from dsmr.trainers import HMEMTrainer
    print('Imported core modules')
    config_path = Path(__file__).resolve().parent / 'dsmr' / 'configs' / 'experiment.yaml'
    cfg = ExperimentConfig.from_yaml(config_path)
    print('Loaded config:', cfg.name)
    print('Pretrained flags:', cfg.pretrained_audio, cfg.pretrained_text, cfg.pretrained_vision)
    print('Fine-tune:', cfg.fine_tune)
    print('Instantiating HMEMNet (this may download pretrained weights)')
    model = HMEMNet(
        audio_dim=cfg.audio_dim,
        text_dim=cfg.text_dim,
        vision_dim=cfg.vision_dim,
        hidden_dim=cfg.hidden_dim,
        num_speakers=cfg.num_speakers,
        pretrained_audio=cfg.pretrained_audio,
        pretrained_text=cfg.pretrained_text,
        pretrained_vision=cfg.pretrained_vision,
        fine_tune=cfg.fine_tune,
    )
    print('Model instantiated')
    trainer = HMEMTrainer(model=model, learning_rate=cfg.learning_rate, device=cfg.device)
    print('Trainer created')
    # run a single epoch from run_experiment logic
    from dsmr.data import MELDConversationDataset
    data_dir = Path(cfg.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parent / 'dsmr' / data_dir
    tokenizer_name = cfg.pretrained_text or 'bert-base-uncased'
    dataset = MELDConversationDataset(dataset_name='meld', split='train', tokenizer_name=tokenizer_name)
    subset_size = getattr(cfg, 'debug_subset_size', 0) or 0
    if subset_size and subset_size > 0:
        from torch.utils.data import Subset
        subset_size = min(len(dataset), int(subset_size))
        dataset = Subset(dataset, list(range(subset_size)))
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    print('Starting training loop (1 epoch)')
    import torch
    # simple small-loop that adapts MELDConversationDataset outputs to trainer.train_step
    for batch_idx, batch_samples in enumerate(loader):
        # DataLoader may return a dict of tensors or a list of sample dicts
        if isinstance(batch_samples, dict):
            text_input_ids = batch_samples['text_input_ids'].to(trainer.device)
            bs, seq = text_input_ids.shape
            text_features = torch.randn(bs, seq, cfg.text_dim, device=trainer.device)

            # synthesize audio features aligned to text token length for debug
            audio = torch.randn(bs, seq, cfg.audio_dim, device=trainer.device)

            vision = batch_samples['vision'].to(trainer.device)
            labels = batch_samples.get('label', batch_samples.get('labels')).to(trainer.device)
            speaker_ids = batch_samples.get('speaker_id', batch_samples.get('speaker_ids', torch.zeros(bs, dtype=torch.long))).to(trainer.device)
        else:
            text_input_ids = torch.stack([s['text_input_ids'] for s in batch_samples]).to(trainer.device)
            bs, seq = text_input_ids.shape
            text_features = torch.randn(bs, seq, cfg.text_dim, device=trainer.device)

            # synthesize audio features aligned to text token length for debug
            audio = torch.randn(bs, seq, cfg.audio_dim, device=trainer.device)

            vision = torch.stack([s['vision'] for s in batch_samples]).to(trainer.device)
            labels = torch.stack([s['label'] for s in batch_samples]).to(trainer.device)
            speaker_ids = torch.stack([s.get('speaker_id', torch.tensor(0, dtype=torch.long)) for s in batch_samples]).to(trainer.device)

        train_batch = {
            'audio_features': audio,
            'text_features': text_features,
            'vision_features': vision,
            'speaker_ids': speaker_ids,
            'labels': labels,
        }
        loss = trainer.train_step(train_batch)
        print('Batch', batch_idx, 'loss', loss.item())
        if batch_idx >= 2:
            break
    print('Training finished')
except Exception as e:
    traceback.print_exc()
    print('DEBUG RUNNER: failed', e)
print('DEBUG RUNNER: end')
