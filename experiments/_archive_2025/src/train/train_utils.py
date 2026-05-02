import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Train the model for one epoch using Binary CrossEntropyLoss and optional mixup support.

    Args:
        model: torch.nn.Module (forward(x, targets=None) may return (logits, loss) if mixup is enabled)
        loader: DataLoader yielding batches with keys 'mel' and 'label'
        optimizer: torch optimizer
        device: torch.device
        scheduler: optional torch.scheduler (e.g., OneCycleLR)

    Returns:
        avg_loss: float
        avg_auc: float
    """
    model.train()
    losses = []
    all_targets = []
    all_outputs = []

    for batch in tqdm(loader, desc="Training", leave=False):
        inputs = batch['mel'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()
        # Forward: always pass targets (model may ignore if mixup is disabled)
        result = model(inputs, targets)
        if isinstance(result, tuple):
            logits, loss = result
        else:
            logits = result
            loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()
        if isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()

        # Compute probabilities for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_outputs.append(probs)
        all_targets.append(targets.cpu().numpy())
        losses.append(loss.item())

    # Aggregate
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = float(np.mean(losses))
    avg_auc = calculate_auc(all_targets, all_outputs)
    return avg_loss, avg_auc


def validate(model, loader, criterion, device):
    """
    Evaluate the model on validation data.

    Returns:
        avg_loss: float
        avg_auc: float
    """
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            inputs = batch['mel'].to(device)
            targets = batch['label'].to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_outputs.append(probs)
            all_targets.append(targets.cpu().numpy())
            losses.append(loss.item())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = float(np.mean(losses))
    avg_auc = calculate_auc(all_targets, all_outputs)
    return avg_loss, avg_auc


def calculate_auc(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Compute macro-average ROC-AUC for multiclass predictions.
    """
    num_classes = outputs.shape[1]
    aucs = []
    for cls in range(num_classes):
        # Binary ground truth for class cls
        gt = (targets[:, cls] > 0).astype(int)
        if np.unique(gt).shape[0] == 2:
            aucs.append(roc_auc_score(gt, outputs[:, cls]))
    return float(np.mean(aucs)) if aucs else 0.0
