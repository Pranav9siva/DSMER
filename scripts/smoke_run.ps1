$repoRoot = Resolve-Path "$PSScriptRoot\.."
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Python executable not found at $python. Create venv at repo root named .venv first."
    exit 1
}

Write-Output "Running smoke training using repo .venv Python: $python"
& $python "DSMER\train.py" --dataset meld --split train
