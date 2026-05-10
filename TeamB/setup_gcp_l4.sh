#!/usr/bin/env bash
# setup_gcp_l4.sh — ncu profiling environment for GCP L4
# Usage: bash setup_gcp_l4.sh
# After: source ~/uai_env/bin/activate

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
die()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# STEP 0 — GPU check
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
  || die "nvidia-smi failed — not a GPU instance"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME"

# STEP 1 — Detect CUDA version
CUDA_VER_FULL=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ,) \
  || die "nvcc not found — install CUDA toolkit first"
CUDA_MAJOR=$(echo "$CUDA_VER_FULL" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VER_FULL" | cut -d. -f2)
CUDA_SHORT="${CUDA_MAJOR}${CUDA_MINOR}"
CUDA_HOME="/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}"
[[ -d "$CUDA_HOME" ]] || CUDA_HOME="/usr/local/cuda"
ok "CUDA: $CUDA_VER_FULL  |  CUDA_HOME: $CUDA_HOME"

# STEP 2 — Fix system CUDA library paths (prevents libtorch_cuda.so resolution errors)
CUDA_LIB_DIR="${CUDA_HOME}/lib64"
if [[ -d "$CUDA_LIB_DIR" ]]; then
  echo "$CUDA_LIB_DIR" | sudo tee /etc/ld.so.conf.d/cuda-l4.conf > /dev/null
  sudo ldconfig
  ok "ldconfig updated: $CUDA_LIB_DIR"
else
  warn "$CUDA_LIB_DIR not found — skipping ldconfig"
fi
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# STEP 3 — System packages
info "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
  build-essential git wget curl unzip \
  python3 python3-pip python3-venv python3-dev \
  libssl-dev libffi-dev > /dev/null
ok "System packages installed"

# STEP 4 — CUPTI (required by PyTorch Profiler for GPU kernel timing)
info "Installing CUPTI..."
sudo apt-get install -y -qq "cuda-cupti-${CUDA_MAJOR}-${CUDA_MINOR}" 2>/dev/null \
  || sudo apt-get install -y -qq libcupti-dev 2>/dev/null \
  || warn "CUPTI install failed — kernel timing may fall back to CPU"
ok "CUPTI done"

# STEP 5 — Python venv
VENV_DIR="$HOME/uai_env"
info "Creating venv at $VENV_DIR..."
python3 -m venv "$VENV_DIR" --system-site-packages 2>/dev/null || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel --quiet
ok "venv ready"

# STEP 6 — PyTorch with matching CUDA build
info "Installing PyTorch (CUDA ${CUDA_MAJOR}.${CUDA_MINOR})..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

if   [[ "$CUDA_MAJOR" == "12" && "$CUDA_MINOR" -ge "4" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [[ "$CUDA_MAJOR" == "12" && "$CUDA_MINOR" -ge "1" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "$CUDA_MAJOR" == "11" && "$CUDA_MINOR" -ge "8" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cu124"
  warn "Unrecognised CUDA version — defaulting to cu124"
fi

pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" --quiet
ok "PyTorch installed from $TORCH_INDEX"

python3 - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print("[FAIL] torch.cuda.is_available() is False"); sys.exit(1)
x = torch.zeros(1, device="cuda"); torch.cuda.synchronize()
print(f"[OK]   torch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}")
PYEOF

# STEP 7 — HuggingFace + quantization stack (pinned to uai env versions)
info "Installing HuggingFace + quantization stack..."
pip install --quiet \
  transformers==4.51.3 \
  accelerate==1.13.0 \
  huggingface-hub==0.30.2 \
  tokenizers==0.21.0 \
  safetensors==0.4.3 \
  datasets \
  optimum==1.24.0

ok "HuggingFace stack installed"

info "Installing bitsandbytes..."
pip install --quiet "bitsandbytes==0.49.2"
python3 -c "import bitsandbytes as bnb; print('[OK]  bitsandbytes:', bnb.__version__)" \
  || warn "bitsandbytes import failed"

info "Installing auto-gptq..."
pip install --quiet "auto-gptq==0.7.1"
python3 -c "import auto_gptq; print('[OK]  auto-gptq:', auto_gptq.__version__)" \
  || warn "auto-gptq import failed"

info "Installing autoawq..."
pip install --quiet "autoawq==0.2.9"
python3 -c "import awq; print('[OK]  autoawq installed')" 2>/dev/null \
  || warn "autoawq import failed"

# STEP 8 — Profiling helpers
info "Installing profiling helpers..."
pip install --quiet nvtx numpy scipy matplotlib netcal wandb
ok "Profiling helpers installed"

# STEP 9 — Kineto env vars (suppresses FabricManager IPC errors on single L4)
# KINETO_USE_DAEMON=0 disables the NVSwitch daemon that only exists on A100 SXM multi-GPU nodes
info "Configuring Kineto env vars..."
cat >> "$VENV_DIR/bin/activate" <<'ENVEOF'

export KINETO_USE_DAEMON=0
export KINETO_DAEMON_INIT_WAIT_USECS=50000
_TORCH_LIB=$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null)
if [[ -n "$_TORCH_LIB" ]]; then
  export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
unset _TORCH_LIB
ENVEOF
ok "Kineto vars written to venv activate"

export KINETO_USE_DAEMON=0
export KINETO_DAEMON_INIT_WAIT_USECS=50000

# STEP 10 — .bashrc
cat >> "$HOME/.bashrc" <<'RCEOF'

source "$HOME/uai_env/bin/activate" 2>/dev/null || true
export KINETO_USE_DAEMON=0
export KINETO_DAEMON_INIT_WAIT_USECS=50000
_TORCH_LIB=$(python3 -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'lib')" 2>/dev/null)
if [[ -n "$_TORCH_LIB" ]]; then
  export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi
unset _TORCH_LIB
RCEOF
ok ".bashrc updated"

# STEP 11 — Final verification
info "Running verification..."
python3 - <<'PYEOF'
import sys, os, torch

assert torch.cuda.is_available(), "torch CUDA not available"
x = torch.zeros(1, device="cuda"); torch.cuda.synchronize()
print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}")

from torch.profiler import ProfilerActivity, profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, profile_memory=False) as p:
    _ = torch.mm(torch.randn(64, 64, device="cuda"), torch.randn(64, 64, device="cuda"))
    torch.cuda.synchronize()
has_cuda = any(getattr(e, "cuda_time_total", 0) > 0 for e in p.key_averages())
print(f"Kineto CUDA backend: {'ACTIVE' if has_cuda else 'CPU-only fallback'}")

import torch.cuda.nvtx as nvtx
nvtx.range_push("test"); nvtx.range_pop()
print("NVTX: OK")

for lib, name in [("bitsandbytes", "bnb"), ("auto_gptq", "auto_gptq"), ("awq", "awq"), ("transformers", "transformers")]:
    try:
        m = __import__(lib)
        ver = getattr(m, "__version__", "ok")
        print(f"{lib}: {ver}")
    except Exception as e:
        print(f"{lib}: FAILED ({e})")

print(f"KINETO_USE_DAEMON={os.environ.get('KINETO_USE_DAEMON', '<not set>')}  (expected: 0)")
print("Setup complete — ready for ncu profiling")
PYEOF

echo ""
echo -e "${GREEN}Done.${NC}"
echo ""
echo "Next steps:"
echo "  source ~/uai_env/bin/activate"
echo "  export HF_TOKEN=hf_..."
echo "  cd /path/to/uncertainty-aware-inference/TeamB"
echo "  sudo -E /usr/local/cuda/bin/ncu ... ~/uai_env/bin/python ncu_fp16.py"
echo ""