#!/usr/bin/env python3
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "preprocess" / "precompute_text_embeds_direct.py"), run_name="__main__")
