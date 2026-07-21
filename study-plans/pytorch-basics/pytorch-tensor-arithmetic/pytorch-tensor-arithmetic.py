import torch
def tensor_op(x, y, op):
    X = torch.tensor(x, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)

    if op == "add":
        return torch.add(X, Y).tolist()
    elif op == "matmul":
        return torch.matmul(X, Y).tolist()
    elif op == "power":
        return torch.pow(X, Y).tolist()
    elif op == "multiply":
        return torch.mul(X, Y).tolist()
    elif op == "max":
        return torch.max(X, Y).tolist()