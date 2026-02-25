#!/usr/bin/env bash
set -euo pipefail

# Run LightRAG server from repo root.
# This script:
# - ensures .env exists (created from env.example if missing)
# - patches ONLY LLM-related env vars to point to vLLM (embedding settings are preserved)
# - starts lightrag-server

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  cp env.example .env
fi

python - <<'PY'
import re
from pathlib import Path

env_path = Path('.env')
text = env_path.read_text(encoding='utf-8')

updates = {
    'LLM_BINDING': 'openai',
    'LLM_MODEL': '/mnt2/data3/nlp/ws/model/qwen3-30B-A3B',
    'LLM_BINDING_HOST': 'http://223.109.239.14:10009/v1',
    'LLM_BINDING_API_KEY': 'dummy',
}

for k, v in updates.items():
    if re.search(rf'(?m)^{re.escape(k)}=', text):
        text = re.sub(rf'(?m)^{re.escape(k)}=.*$', f'{k}={v}', text)
    else:
        if not text.endswith('\n'):
            text += '\n'
        text += f'{k}={v}\n'

env_path.write_text(text, encoding='utf-8')
print('Patched .env LLM settings for vLLM. Embedding settings left unchanged.')
PY

exec lightrag-server
