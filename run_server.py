"""Simple runner script to launch the Fraud Detection API and Operator Dashboard."""

import os
import sys
import subprocess

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Auto-detect and switch to virtual environment if run from global python
current_dir = os.path.abspath(os.path.dirname(__file__))
venv_python_win = os.path.join(current_dir, ".venv", "Scripts", "python.exe")
venv_python_nix = os.path.join(current_dir, ".venv", "bin", "python")
venv_python = venv_python_win if os.path.exists(venv_python_win) else venv_python_nix

if os.path.exists(venv_python):
    current_exe = os.path.abspath(sys.executable).lower()
    target_exe = os.path.abspath(venv_python).lower()
    if current_exe != target_exe:
        print("[Launcher] Switching from system Python to project virtual environment:")
        print(f"           -> {venv_python}")
        ret = subprocess.call([venv_python] + sys.argv)
        sys.exit(ret)

try:
    import uvicorn
    import lightgbm  # noqa: F401 - import probes that the model runtime is installed
except ImportError as e:
    print(f"\n[ERROR] Missing required dependencies ({e}).")
    print("Please run:")
    print("   .venv\\Scripts\\activate")
    print("   uv pip install -r requirements.txt\n")
    sys.exit(1)

if __name__ == "__main__":
    print("=" * 70)
    print(">>> Launching FRAUD-SHIELD RT Server & FinTech Operator Dashboard")
    print("   * API & Docs: http://127.0.0.1:8000/docs")
    print("   * Dashboard:  http://127.0.0.1:8000/")
    print("   * WebSocket:  ws://127.0.0.1:8000/ws/transactions")
    print("=" * 70)
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
