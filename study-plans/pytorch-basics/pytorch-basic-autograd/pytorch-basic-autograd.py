import torch

def compute_gradient(values):
    """
    Returns: list of float gradient values dy/dx
    """
    return [float(3 * (num ** 2) + 2) for num in values]