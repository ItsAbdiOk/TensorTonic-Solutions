import torch

def activate(x, method="relu"):
    y = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    
    if method == "relu":
        return torch.relu(y).tolist()
        
    elif method == "sigmoid":
        return torch.sigmoid(y).tolist()
        
    elif method == "tanh":
        return torch.tanh(y).tolist()
        
    elif method == "leaky_relu":
        return torch.maximum(y, y * 0.01).tolist()