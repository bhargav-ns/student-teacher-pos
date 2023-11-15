import torch
import os

def save_model(model, path):
    """Save a PyTorch model state."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load a state into a PyTorch model."""
    model.load_state_dict(torch.load(path))

def create_dir_if_not_exists(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)