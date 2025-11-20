"""
Pytest configuration for test discovery and imports.

This file ensures that the scr module can be imported during testing.
"""

import sys
from pathlib import Path

# Add the project root to Python path so 'scr' module can be imported
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
