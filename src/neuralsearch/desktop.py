"""
Desktop entrypoint for NeuralSearch executable.
Launches the local search engine and automatically opens the user's default web browser.
"""

import sys
import time
import threading
import webbrowser
import uvicorn
from .api.app import app

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

def main():
    print("=" * 60)
    print("      NEURALSEARCH: PRIVATE & AD-FREE SEARCH ENGINE")
    print("=" * 60)
    print("  * Local Privacy Proxy & Neural Reranker Active")
    print("  * Serving at: http://127.0.0.1:8000")
    print("  * Opening your browser automatically...")
    print("  * Press Ctrl+C in this terminal window to stop.")
    print("=" * 60)
    
    # Launch browser after a brief delay so server has time to bind
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Uvicorn server
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    main()
