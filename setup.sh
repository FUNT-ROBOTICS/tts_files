#!/bin/bash
set -Eeuo pipefail

WORKDIR="$HOME/qwen3-tts"
REPO_DIR="$WORKDIR/repo"

echo "=== 1. System dependencies ==="
apt-get update
apt-get install -y ffmpeg sox git wget curl ca-certificates build-essential

echo "=== 2. Python 3.12 ==="
apt-get update
apt-get install -y python3.12 python3.12-venv python3.12-dev libpython3.12-dev

echo "=== 3. CUDA Toolkit 13.0 ==="
if ! command -v nvcc >/dev/null 2>&1; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
    apt-get install -y cuda-toolkit-13-0
    rm -f cuda-keyring_1.1-1_all.deb
fi

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "--- nvcc version ---"
nvcc --version

echo "=== 4. uv + virtual environment ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

source "$HOME/.local/bin/env" 2>/dev/null || source "$HOME/.cargo/env" 2>/dev/null || true

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12 --seed
else
    echo "Virtual environment already exists."
fi

source .venv/bin/activate

echo "=== 5. Installing vLLM ==="
uv pip install -v vllm==0.25.0 --torch-backend=cu130

echo "=== 6. Clone/update vLLM-Omni ==="
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone https://github.com/vllm-project/vllm-omni.git "$REPO_DIR"
else
    echo "Repository already exists."
fi

cd "$REPO_DIR"
uv pip install -e .
cd "$WORKDIR"

echo "=== 7. Verify installation ==="
python -c "import torch; assert torch.cuda.is_available(); print('Torch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0))"
python -c "import vllm; print('vLLM OK')"
python -c "import vllm_omni; print('vLLM-Omni OK')"

echo "=== 8. Extra packages ==="
uv pip install requests pysbd pydub
mkdir -p outputs

echo "=== 9. Configure ~/.bashrc ==="

if ! grep -q "qwen3-tts/.venv/bin/activate" "$HOME/.bashrc" 2>/dev/null; then
cat >> "$HOME/.bashrc" <<'EOF'

# --- Qwen3-TTS / vLLM-Omni ---
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source "$HOME/qwen3-tts/.venv/bin/activate"
EOF
fi

cat <<'EOF'

=========================================================
SETUP COMPLETE

Project directory:
  ~/qwen3-tts

Repository:
  ~/qwen3-tts/repo

To start the server:

cd ~/qwen3-tts/repo

vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager

=========================================================
EOF
