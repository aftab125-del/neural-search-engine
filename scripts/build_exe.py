"""
Build script to package NeuralSearch into a standalone Windows executable (.exe).
Uses PyInstaller to bundle the Python environment, ONNX models, and web UI.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
UI_DIR = BASE_DIR / "src" / "neuralsearch" / "ui"
ENTRY_POINT = BASE_DIR / "src" / "neuralsearch" / "desktop.py"

def build():
    print(f"[*] Building NeuralSearch Windows Executable from {ENTRY_POINT}...")
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=NeuralSearch",
        "--onedir",  # onedir is faster to launch and handles ONNX runtime DLLs reliably
        "--noconfirm",
        f"--add-data={UI_DIR};neuralsearch/ui",
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespans",
        "--hidden-import=uvicorn.lifespans.on",
        str(ENTRY_POINT),
    ]
    
    print("[*] Running command:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)
    print("\n[SUCCESS] Build complete! Executable is located in:")
    print(f"          {BASE_DIR / 'dist' / 'NeuralSearch' / 'NeuralSearch.exe'}")

if __name__ == "__main__":
    build()
