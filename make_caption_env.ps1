# Makes the small Python environment Time Cube's captions run in (%LOCALAPPDATA%\TimeCube\caption-env).
# It sees the system Python's packages (PyTorch and the rest) and adds only transformers 4.57, which Qwen3-VL needs;
# the system Python keeps its own transformers, so other apps aren't touched. About 130 MB, a one-off.
# -Python <path to python.exe> picks the Python to build on (the default is the "python" on your PATH).
param([string]$Python = "")
if (-not $Python) { $Python = (Get-Command python).Source }

$envDir = Join-Path $env:LOCALAPPDATA "TimeCube\caption-env"
if (-not (Test-Path "$envDir\Scripts\python.exe")) {
    & $Python -m venv --system-site-packages $envDir
}
& "$envDir\Scripts\python.exe" -m pip install --disable-pip-version-check "transformers>=4.57,<4.58"
& "$envDir\Scripts\python.exe" -c "import transformers; from transformers import Qwen3VLForConditionalGeneration; print('caption environment ready: transformers', transformers.__version__)"
