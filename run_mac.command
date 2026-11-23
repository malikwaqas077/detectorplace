#!/bin/bash
# DetectorPlace one-click launcher for macOS.
# First run: downloads the v0.2 branch into ~/DetectorPlace, makes a Python environment and installs
# packages (5-10 minutes). Later runs start in a few seconds. Opens the app in your browser.
set -e
DIR="$HOME/DetectorPlace"
BRANCH="v0.2-cad-vision"

if [ ! -d "$DIR/.git" ]; then
  echo "Downloading DetectorPlace into $DIR ..."
  git clone --branch "$BRANCH" https://github.com/malikwaqas077/detectorplace "$DIR"
else
  echo "Updating DetectorPlace ..."
  git -C "$DIR" fetch origin "$BRANCH" && git -C "$DIR" checkout "$BRANCH" && git -C "$DIR" pull --ff-only origin "$BRANCH" || true
fi
cd "$DIR"

PY=""
for c in python3.12 python3.11 python3.13 python3.10 /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  # No suitable Python and Homebrew may not be writable: get a private Python with uv (no admin rights needed).
  UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
  if [ ! -x "$UV" ]; then
    echo "No Python 3.10+ found. Installing uv (a Python manager, into ~/.local/bin, no admin needed) ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    UV="$HOME/.local/bin/uv"
  fi
  echo "Installing Python 3.12 for DetectorPlace only ..."
  "$UV" python install 3.12
  PY="$("$UV" python find 3.12)"
fi
if [ -z "$PY" ]; then
  echo "Could not get Python 3.10+. Install it from https://www.python.org/downloads/macos/ and run this again."
  exit 1
fi
echo "Using $($PY --version) at $(command -v $PY)"

if [ ! -x ".venv/bin/python" ]; then
  "$PY" -m venv .venv
fi
if [ ! -f ".venv/.installed" ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "Installing packages (first run only, 5-10 minutes) ..."
  .venv/bin/python -m pip install --upgrade pip -q
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.installed
fi

echo ""
echo "Checking everything works (20 tests) ..."
.venv/bin/python -m pytest -q || echo "Some tests failed: the app may still run, but tell Claude what failed."

echo ""
echo "Starting DetectorPlace at http://localhost:8501  (close this window or press Ctrl+C to stop)"
echo "The very first start runs the demo pipeline once (about 2 minutes); after that it opens instantly."
.venv/bin/python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
