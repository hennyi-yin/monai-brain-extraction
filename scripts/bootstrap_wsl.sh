#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p "$HOME/.local/bin" "$HOME/.venvs"
if ! command -v uv >/dev/null; then
  curl -fL --retry 3 https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz -o /tmp/monai-uv.tar.gz
  tar -xzf /tmp/monai-uv.tar.gz -C "$HOME/.local/bin" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv "$HOME/.venvs/monai-brain" --python python3 --allow-existing
uv pip install --python "$HOME/.venvs/monai-brain/bin/python" --index-strategy unsafe-best-match -r requirements.txt
if [[ ! -x "$HOME/.local/micromamba/bin/micromamba" ]]; then
  mkdir -p "$HOME/.local/micromamba"
  curl -fL --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/monai-micromamba.tar.bz2
  python3 -m tarfile -e /tmp/monai-micromamba.tar.bz2 "$HOME/.local/micromamba"
fi
if [[ ! -x "$HOME/.venvs/fsl-bet/bin/bet" ]]; then
  "$HOME/.local/micromamba/bin/micromamba" create -y -p "$HOME/.venvs/fsl-bet" --file environment-fsl-explicit.txt
fi
printf 'Setup complete. Activate the training environment and set FSLDIR as in docs/local_run.md.\n'
