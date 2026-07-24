#!/usr/bin/env bash
# Self-contained dependency install for this externally-managed (PEP 668) host.
#
# DEFAULT: isolated .venv — the correct choice on a machine shared with other Python/ML
# projects, so this project's numpy/pandas/torch pins can NEVER disturb ~/.local.
# Pass --user to install into user-site instead (only if you know it won't clash).
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-venv}"

if [[ "$MODE" == "--user" || "$MODE" == "user" ]]; then
  echo "▶ Installing to USER site (--user --break-system-packages)"
  echo "  ⚠ This shares ~/.local with your other projects and may upgrade numpy/pandas globally."
  pip install --user --break-system-packages -r requirements.txt
else
  echo "▶ Creating isolated virtualenv at .venv (recommended)"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  echo ""
  echo "  Activate it in new shells with:  source .venv/bin/activate"
fi

echo "✔ Dependencies installed."
echo "Next:"
echo "  cp secrets/.env.example secrets/.env   # add your Kite keys"
echo "  python main.py kite-login              # then re-run with the real request_token"
