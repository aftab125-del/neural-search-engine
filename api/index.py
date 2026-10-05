"""
Vercel Serverless Function entrypoint for NeuralSearch.
Exposes the ASGI FastAPI `app` directly to Vercel's Python runtime.
"""
import os
import sys
from pathlib import Path

# Add project root and src directory to Python sys.path so neuralsearch is cleanly discoverable
CURRENT_FILE = Path(__file__).resolve()
API_DIR = CURRENT_FILE.parent
ROOT_DIR = API_DIR.parent
SRC_DIR = ROOT_DIR / "src"

for p in (str(SRC_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Declare serverless environment flag
os.environ.setdefault("VERCEL", "1")

# Import the FastAPI application instance
from neuralsearch.api.app import app
