import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# graph helpers
def induce_subgraph(edge_index: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Induziert Kanten nur zwischen Knoten, die node_mask=True haben.
    edge_index: [2, E] (long), node_mask: [N] bool
    returns: remap_edge_index [2, E_sub] mit kompakter Knoten-ID-Remap
    """
    assert edge_index.dtype == torch.long
    assert node_mask.dtype == torch.bool
    device = edge_index.device
    N = node_mask.numel()

    old2new = torch.full((N,), -1, dtype=torch.long, device=device)
    allowed = torch.nonzero(node_mask, as_tuple=False).flatten()
    old2new[allowed] = torch.arange(allowed.numel(), device=device)

    src, dst = edge_index
    keep = node_mask[src] & node_mask[dst]
    ei = edge_index[:, keep]

    return old2new[ei]

def _logits_to_posprob(logits: torch.Tensor) -> torch.Tensor:
    """
    Robust: akzeptiert [N] (binary), [N,1] (binary) oder [N,C] (multiclass).
    Liefert Positiv-Probability (bei multiclass: Klasse 1 = 'illicit').
    """
    if logits.dim() == 1:
        return torch.sigmoid(logits)
    if logits.size(1) == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, dim=1)[:, 1]

# train / eval

def train(model, data, optimizer, train_mask, criterion, max_grad_norm: float | None = None):
    """
    data: PyG Data (mit split-spezifischem edge_index)
    train_mask: bool [N]
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(data.x, data.edge_index)  # [N,C] oder [N]
    loss = criterion(logits[train_mask], data.y[train_mask].long())
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def evaluate(model, data, mask):
    """
    data: PyG Data (mit split-spezifischem edge_index)
    mask: bool [N]
    returns: (acc, f1_macro, pr_auc)
    """
    model.eval()
    logits = model(data.x, data.edge_index)

    y_true = data.y[mask].long().cpu()
    pos_prob = _logits_to_posprob(logits)[mask].cpu()

    if logits.dim() <= 2 and (logits.ndim == 1 or logits.size(-1) <= 1):
        y_pred = (pos_prob >= 0.5).long()
    else:
        y_pred = logits.argmax(dim=1)[mask].cpu()

    denom = int(mask.sum().item())
    if denom == 0:
        return 0.0, 0.0, 0.0

    acc = float((y_pred == y_true).sum().item() / denom)
    f1  = float(f1_score(y_true, y_pred, average="macro"))
    pr_auc = float(average_precision_score(y_true, pos_prob))
    return acc, f1, pr_auc
