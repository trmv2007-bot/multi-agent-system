#!/usr/bin/env python
"""Launch the multi-agent MCP server (stdio transport).

MCP clients start this file as a subprocess. It puts the project root on
sys.path first, so it works whether or not the package is installed:

    python C:\\Users\\rishi\\multi-agent-system\\run_mcp.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Settings load .env relative to the working directory, which a client
# launching us from elsewhere would otherwise break.
os.chdir(PROJECT_ROOT)

from src.mcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
