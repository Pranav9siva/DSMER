import sys
sys.path.insert(0, r'D:\MultiModel_speech_Recognitaion\DSMER')
from dsmr.utils.config import ExperimentConfig
from pathlib import Path
cfg = ExperimentConfig.from_yaml('dsmr/configs/experiment.yaml')
out = Path(__file__).resolve().parent / 'debug_inspect.txt'
with open(out,'w',encoding='utf-8') as f:
    f.write(f'fine_tune={cfg.fine_tune}\n')
    f.write(f'pretrained_audio={cfg.pretrained_audio}\n')
    f.write(f'pretrained_text={cfg.pretrained_text}\n')
    f.write(f'pretrained_vision={cfg.pretrained_vision}\n')
    f.write(f'batch_size={cfg.training.batch_size}\n')

# try loading dataset
from dsmr.data import MELDConversationDataset
data_dir = Path(cfg.paths.data_dir)
if not data_dir.is_absolute():
    data_dir = Path(__file__).resolve().parent / data_dir
try:
    ds = MELDConversationDataset(csv_path=data_dir / 'MELD' / 'train_sent_emo.csv')
    f.write(f'dataset_len={len(ds)}\n')
except Exception as e:
    f.write('dataset_error='+str(e)+'\n')
print('wrote', out)
