#!/usr/bin/env python3
"""project_config.py — paths and model configuration.

Usage:
    from project_config import BASE_DIR, MODELS, OLLAMA_BASE
"""

import os
from pathlib import Path

# Project root (this file's directory)
BASE_DIR = Path(__file__).resolve().parent

# Model paths
MODELS = {
    "depth_vits": os.path.join(BASE_DIR, "models", "depth_anything_v2_vits.pth"),
    "depth_vitl": os.path.join(BASE_DIR, "models", "depth_anything_v2_vitl.pth"),
    "yolo_bdd":   os.path.join(BASE_DIR, "models", "best.pt"),
}

# Ollama
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Depth-Anything-V2 library
DEPTH_LIB = os.path.join(BASE_DIR, "Depth-Anything-V2")
