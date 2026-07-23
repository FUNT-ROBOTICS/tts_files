#!/bin/bash
set -Eeuo pipefail

WORKDIR="$HOME/qwen3-tts"
REPO_DIR="$WORKDIR/repo"

echo "========================================================="
echo " Qwen3-TTS + vLLM-Omni Setup (RunPod PyTorch Template)"
echo "========================================================="

echo
echo "=== 1. Install system packages ==="

apt-get update
apt-get install -y \
    ffmpeg \
    sox \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential

echo
echo "=== 2. Activate RunPod Python Environment ==="

source /opt/venv/bin/activate

echo "Python: $(python --version)"

echo
echo "=== 3. Install uv (if needed) ==="

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    hash -r
fi

echo "uv version:"
uv --version

echo
echo "=== 4. Create Project Directory ==="

mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo
echo "=== 5. Install vLLM ==="

uv pip install -v vllm==0.25.0 --torch-backend=cu130

echo
echo "=== 6. Clone / Update vLLM-Omni ==="

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone https://github.com/vllm-project/vllm-omni.git "$REPO_DIR"
else
    echo "Repository already exists."
    cd "$REPO_DIR"
    git pull
fi

echo
echo "=== 7. Install vLLM-Omni ==="

cd "$REPO_DIR"
uv pip install -e .

echo
echo "=== 8. Install Extra Packages ==="

uv pip install \
    requests \
    pysbd \
    pydub

echo
echo "=== 9. Verify Installation ==="

python - <<'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA :", torch.version.cuda)
print("GPU  :", torch.cuda.get_device_name(0))
print("CUDA Available:", torch.cuda.is_available())
EOF

python -c "import vllm; print('✓ vLLM OK')"
python -c "import vllm_omni; print('✓ vLLM-Omni OK')"

echo
echo "========================================================="
echo "               SETUP COMPLETE"
echo "========================================================="
echo
echo "Activate environment:"
echo
echo "source /opt/venv/bin/activate"
echo
echo "Start server:"
echo
echo "cd ~/qwen3-tts/repo"
echo
echo "vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \\"
echo "    --omni \\"
echo "    --trust-remote-code \\"
echo "    --host 0.0.0.0 \\"
echo "    --port 8091 \\"
echo "    --gpu-memory-utilization 0.90 \\"
echo "    --enforce-eager"
echo
echo "========================================================="
