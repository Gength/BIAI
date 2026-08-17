"""Recoverable checkpoint rotation used before a fresh training run."""
import datetime
import os
import shutil


def backup_existing(path):
    """Copy an existing checkpoint file/directory to a timestamped sibling."""
    if not os.path.exists(path):
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = f"{path}.backup-{stamp}"
    suffix = 1
    while os.path.exists(destination):
        destination = f"{path}.backup-{stamp}-{suffix}"
        suffix += 1
    if os.path.isdir(path):
        shutil.copytree(path, destination)
    else:
        shutil.copy2(path, destination)
    print(f"Backed up existing checkpoint to {destination}")
    return destination


def clear_completion_marker(path):
    """A new run is incomplete until its trainer writes the marker again."""
    if os.path.exists(path):
        os.remove(path)
