from pathlib import Path

log_path = Path(__file__).resolve().with_name('test_write.log')
with log_path.open('w', encoding='utf-8') as f:
    f.write('ok\n')
print(f'wrote to {log_path}')
