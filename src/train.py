import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from torch_geometric.utils import train_test_split_edges

def train(model, data, optimizer, train_mask, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)

    prob = torch.softmax(out, dim=1)[:, 1]  # probability for illicit
    pred = out.argmax(dim=1)

    y_true = data.y[mask].cpu()
    y_pred = pred[mask].cpu()
    y_score = prob[mask].cpu()

    acc = (y_pred == y_true).sum().item() / mask.sum().item()
    f1 = f1_score(y_true, y_pred, average="macro")
    pr_auc = average_precision_score(y_true, y_score)

    return acc, f1, pr_auc