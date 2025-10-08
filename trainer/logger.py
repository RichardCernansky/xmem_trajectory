import torch.nn as nn

def grad_norm(module: nn.Module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5

# g_ef = grad_norm(self.early_fusion)
# g_head = grad_norm(self.head)
# print(f"[grads] early_fusion={g_ef:.4e} head={g_head:.4e}")