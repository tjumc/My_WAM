#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "tools" / "preprocess_action_dit_backbone.py"), run_name="__main__")
