#!/usr/bin/env bash
# Start TypeReel Studio:  ./run.sh   (first run: pip install -r requirements.txt)
cd "$(dirname "$0")"
python3 -c "import streamlit, PIL, numpy" 2>/dev/null || python3 -m pip install -r requirements.txt
exec python3 -m streamlit run ui.py
