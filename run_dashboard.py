from __future__ import annotations

import subprocess
import sys
from pathlib import Path

"""
Simple launcher so the dashboard can be started with a normal Python run action.

Usage:
1. Run this file in VS Code or terminal.
2. It starts: streamlit run dashboard/app.py
"""


def main() -> int:
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "dashboard" / "app.py"

    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(command, cwd=str(project_root))


if __name__ == "__main__":
    raise SystemExit(main())
