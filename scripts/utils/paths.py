"""
Utility to get relative paths from project root
Ensures scripts in scripts/ can access dataset/, models/, data_collection/
"""

import os
from pathlib import Path


def get_project_root():
    """
    Get project root directory
    This script can be run from scripts/*/ or root/
    """
    # Get current directory
    current_dir = Path(__file__).resolve()
    
    # Find root (contains README.md)
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "README.md").exists():
            return str(parent)
    
    # Fallback: assume we're at root or scripts/*
    current = current_dir
    while current.name == "scripts" or current.name in ["data_preprocessing", "training", "inference", "utils"]:
        current = current.parent
    
    return str(current)


PROJECT_ROOT = get_project_root()


def get_dataset_dir():
    """Get path to dataset/"""
    return os.path.join(PROJECT_ROOT, "dataset")


def get_models_dir():
    """Get path to models/"""
    return os.path.join(PROJECT_ROOT, "models")


def get_data_collection_dir():
    """Get path to data_collection/"""
    return os.path.join(PROJECT_ROOT, "data_collection")


def get_model_path(filename="best_model.h5"):
    """Get path to model file"""
    return os.path.join(get_models_dir(), filename)

