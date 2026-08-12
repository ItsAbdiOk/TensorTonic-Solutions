import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    Pred =torch.tensor(pred, dtype=torch.float32)
    Target =torch.tensor(target, dtype=torch.float32)

    if method == "mse":
        x = torch.mean((Pred - Target)**2)
        return x.item()
    if method == "huber":
        a = (Pred - Target).abs()
        quad = 0.5 * a ** 2
        linear = delta * (a - 0.5 * delta)
        x = torch.where(a <= delta, quad, linear)
        return x.mean().item()
    if method == "cross_entropy":
        Target = torch.tensor(target, dtype=torch.long)
        max = Pred.max(dim=1, keepdim=True).values
        shift = Pred - max
        exp_sum = torch.exp(shift).sum(dim=1)
        lse = torch.log(exp_sum) + max.squeeze(1)
        num_rows = Pred.shape[0]                                     
        correct = torch.tensor([Pred[i, Target[i]].item() for i in range(num_rows)])
        loss = (lse - correct).mean().item()
        return loss        