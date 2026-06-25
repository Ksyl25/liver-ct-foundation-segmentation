"""Launch helper for the Streamlit viewer."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    """Run the Streamlit dashboard."""

    command = [sys.executable, "-m", "streamlit", "run", "app/dashboard.py"]
    print("Running:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
