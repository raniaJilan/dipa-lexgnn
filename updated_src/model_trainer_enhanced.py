import os
import time, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Limit CPU usage to prevent high CPU utilization
torch.set_num_threads(20)
torch.set_num_interop_threads(20)


# =============================================================================
# Loss Functions for Comparative Study
# =============================================================================

class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy Loss (Baseline from LEX-GNN paper)
    Used in Equations 2 and 8 of the paper
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        return self.loss_fn(logits, labels.squeeze())


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for imbalanced datasets
    Assigns higher weight to minority class (fraud)
    """
    def __init__(self, pos_weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, logits, labels):
        if self.pos_weight is None:
            # Calculate weight dynamically based on class distribution
            pos_count = labels.sum().item()
            neg_count = len(labels) - pos_count
            if pos_count > 0:
                # clamp to 10x to prevent Adam instability on small fraud batches
                weight = torch.tensor([1.0, min(neg_count / pos_count, 10.0)], device=labels.device)
            else:
                weight = torch.tensor([1.0, 1.0], device=labels.device)
        else:
            weight = torch.tensor([1.0, self.pos_weight], device=labels.device)
        
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        return loss_fn(logits, labels.squeeze())


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses on hard-to-classify examples
    Paper: Lin et al. "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, labels):
        labels = labels.squeeze()
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        # per-class alpha: alpha for fraud (1), 1-alpha for benign (0)
        alpha_t = torch.where(labels == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MarginLoss(nn.Module):
    """
    Margin Loss for better separation between classes
    Encourages larger margins between fraud and benign predictions
    """
    def __init__(self, margin=1.0):
        super(MarginLoss, self).__init__()
        self.margin = margin
        
    def forward(self, logits, labels):
        labels = labels.squeeze()
        # {0,1} → {-1,+1} so hinge loss matches standard SVM formulation
        signed_labels = 2 * labels.float() - 1
        # decision score on raw logits (no softmax — margin losses need logit space)
        scores = logits[:, 1] - logits[:, 0]
        loss = F.relu(self.margin - signed_labels * scores)
        return loss.mean()


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss) for robust training
    Less sensitive to outliers than MSE
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, logits, labels):
        labels = labels.squeeze()
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of positive class
        pred_probs = probs[:, 1]
        
        # Convert labels to float
        target = labels.float()
        
        # Huber loss
        diff = torch.abs(pred_probs - target)
        loss = torch.where(diff < self.delta,
                          0.5 * diff ** 2,
                          self.delta * (diff - 0.5 * self.delta))
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for metric learning
    Pulls same-class samples together, pushes different-class samples apart
    """
    def __init__(self, margin=2.0, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        """
        embeddings: node embeddings from the model
        labels: ground truth labels
        """
        labels = labels.squeeze()
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        # Create label mask
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Positive pairs (same label)
        pos_loss = labels_equal.float() * distances ** 2
        
        # Negative pairs (different label)
        neg_loss = (~labels_equal).float() * F.relu(self.margin - distances) ** 2
        
        # Combine losses
        loss = (pos_loss + neg_loss).sum() / (labels_equal.numel())
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for imbalanced classification
    Originally from image segmentation, works well for fraud detection
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, labels):
        labels = labels.squeeze()
        probs = F.softmax(logits, dim=1)
        
        # Get probabilities for positive class
        pred_probs = probs[:, 1]
        
        # One-hot encode labels
        targets = labels.float()
        
        # Dice coefficient
        intersection = (pred_probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (pred_probs.sum() + targets.sum() + self.smooth)
        
        # Dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss for main classification and auxiliary pre-label prediction
    This follows the LEX-GNN paper approach (Equation 9)
    """
    def __init__(self, main_loss, aux_loss=None, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.main_loss = main_loss
        self.aux_loss = aux_loss if aux_loss is not None else CrossEntropyLoss()
        self.beta = beta
    
    def forward(self, main_logits, aux_logits, labels):
        """
        main_logits: final classification logits
        aux_logits: pre-label prediction logits (from label exploration)
        labels: ground truth labels
        """
        loss_main = self.main_loss(main_logits, labels)
        loss_aux = self.aux_loss(aux_logits, labels)
        return loss_main + self.beta * loss_aux


# =============================================================================
# Loss Function Factory
# =============================================================================

def get_loss_function(loss_name, **kwargs):
    """
    Factory function to create loss functions
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional parameters for the loss function
    
    Returns:
        Loss function instance
    """
    loss_functions = {
        'ce': CrossEntropyLoss,
        'cross_entropy': CrossEntropyLoss,
        'weighted_ce': WeightedCrossEntropyLoss,
        'focal': FocalLoss,
        'margin': MarginLoss,
        'huber': HuberLoss,
        'contrastive': ContrastiveLoss,
        'dice': DiceLoss,
    }
    
    loss_name = loss_name.lower()
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name](**kwargs)


# =============================================================================
# Utility Functions
# =============================================================================

def reset_model_parameters(model):
    """Reset model parameters to random initialization"""
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def test(model, loader, device):
    """
    Test the model on validation/test set
    Returns multiple evaluation metrics
    """
    labels = []
    output_list = [[], [], [], []]
    model.eval()
    
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in loader:
            blocks = [b.to(device) for b in blocks]
            output_labels = blocks[-1].dstdata['y'].data.cpu().numpy()
            output, output1 = model(blocks)
            output = torch.softmax(output, dim=1).data.cpu().numpy()
            
            prediction = output.argmax(axis=1)
            confidence = output.max(axis=1)
            anomaly_confidence = output[:, 1]

            output_list[0].extend(prediction.tolist())
            output_list[1].extend(confidence.tolist())
            output_list[2].extend(anomaly_confidence.tolist())
            labels.extend(output_labels.tolist())
            
            output1 = torch.softmax(output1[-1], dim=1).data.cpu().numpy()
            output_list[3].extend(output1[:, 1].tolist()[:len(output_labels)])
    
    output_list = np.array(output_list)
    labels = np.array(labels)

    # Calculate metrics
    f1_macro = f1_score(labels, output_list[0], average='macro')
    auc = roc_auc_score(labels, output_list[2])
    ap = average_precision_score(labels, output_list[2])
    fpr, tpr, thresholds = roc_curve(labels, output_list[0])
    gmean = (tpr[1] * (1 - fpr[1])) ** (1/2)
    prec_macro = precision_score(labels, output_list[0], average='macro', zero_division=0)
    rec_macro = recall_score(labels, output_list[0], average='macro', zero_division=0)
    
    auc1 = roc_auc_score(labels, output_list[3])
    
    return auc, f1_macro, gmean, ap, auc1, prec_macro, rec_macro


def train(model, train_loader, valid_loader, epochs, valid_epochs, 
          beta, lr, weight_decay, early_stop, seed, device, 
          loss_name='ce', loss_params=None):
    """
    Train the model with specified loss function
    
    Args:
        model: The LEX-GNN model
        train_loader: Training data loader
        valid_loader: Validation data loader
        epochs: Maximum number of epochs
        valid_epochs: Validate every N epochs
        beta: Weight for auxiliary loss (label exploration)
        lr: Learning rate
        weight_decay: Weight decay for regularization
        early_stop: Early stopping patience
        seed: Random seed
        device: Device to train on
        loss_name: Name of the loss function to use
        loss_params: Additional parameters for the loss function
    
    Returns:
        Tuple of (best_model, best_epoch, total_training_time)
    """
    if loss_params is None:
        loss_params = {}
    
    # Initialize model
    model.apply(reset_model_parameters)
    model.to(device)
    _is_ddp = dist.is_initialized()
    _rank = dist.get_rank() if _is_ddp else 0
    if _is_ddp:
        model = DDP(model, device_ids=[device.index])
    if _rank == 0:
        print(f"\nTraining with {loss_name.upper()} loss", flush=True)

    # Create loss functions
    if loss_name == 'contrastive':
        main_loss_fn = get_loss_function(loss_name, **loss_params)
        aux_loss_fn = CrossEntropyLoss()
        use_contrastive = True
    else:
        main_loss_fn = get_loss_function(loss_name, **loss_params)
        aux_loss_fn = CrossEntropyLoss()
        use_contrastive = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    auc_best, f1_best, epoch_best = 1e-10, 1e-10, 0
    epoch = 1
    total_time = 0.0
    model_best = copy.deepcopy(model.module if _is_ddp else model)
    
    while epoch <= epochs:
        model.train()
        avg_loss, avg_loss_main, avg_loss_aux = [], [], []
        start_time = time.time()

        for batch in train_loader:
            _, output_nodes, blocks = batch
            blocks = [b.to(device) for b in blocks]
            output_labels = blocks[-1].dstdata['y'].type(torch.LongTensor).to(device)
            idx_pre = blocks[-1].srcdata['y_mask'].type(torch.LongTensor).to(device) != 2
            output_labels1 = blocks[-1].srcdata['y'].type(torch.LongTensor).to(device)[idx_pre]

            logit, q_list = model(blocks)
            
            # Main classification loss
            if use_contrastive:
                # For contrastive loss, we need the embeddings (hidden representation)
                # We'll use the logit as a proxy for embeddings
                loss_main = main_loss_fn(logit, output_labels.squeeze())
            else:
                loss_main = main_loss_fn(logit, output_labels.squeeze())
            
            # Auxiliary loss for label exploration (Equation 2 in paper)
            loss_aux = aux_loss_fn(q_list[-1][idx_pre], output_labels1.squeeze())
            
            # Combined loss (Equation 9 in paper)
            total_loss = loss_main + loss_aux * beta

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            avg_loss.append(total_loss.item())
            avg_loss_main.append(loss_main.item())
            avg_loss_aux.append(loss_aux.item())
            del blocks

        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        torch.cuda.empty_cache()

        # Validation
        if epoch % valid_epochs == 0:
            auc_val, f1_val, gmn_val, ap_val, auc1, prec_val, rec_val = test(model, valid_loader, device)

            # Re-initialize if initial performance is too poor
            if auc_val <= 0.51:
                model.apply(reset_model_parameters)
                auc_val = 0
                epoch = 0
                auc_best, f1_best, epoch_best = 1e-10, 1e-10, 0
                model_best = copy.deepcopy(model.module if _is_ddp else model)
                if _rank == 0:
                    print(f'Epoch: {epoch} | Poor init (AUC<=0.51), reinitializing weights', flush=True)

            gain_auc = (auc_val - auc_best) / auc_best if auc_best > 0 else 0

            is_best = gain_auc > 0
            if is_best:
                auc_best, f1_best, epoch_best = auc_val, f1_val, epoch
                model_best = copy.deepcopy(model.module if _is_ddp else model)

            if _rank == 0:
                marker = ' *' if is_best else ''
                line = (f'Epoch: {str(epoch).rjust(3, " ")} | '
                       f'Loss: {np.mean(avg_loss):.4f} '
                       f'(Main: {np.mean(avg_loss_main):.4f}, Aux: {np.mean(avg_loss_aux):.4f}) | '
                       f'AUC: {auc_val:.4f} | F1: {f1_val:.4f} | '
                       f'P: {prec_val:.4f} | R: {rec_val:.4f} | '
                       f'GM: {gmn_val:.4f} | AP: {ap_val:.4f}{marker}')
                print(line, flush=True)
        
        # Early stopping
        if (epoch - epoch_best) > early_stop:
            if _rank == 0:
                print(f"Early stopping at epoch {epoch}", flush=True)
            break

        epoch += 1

    if _rank == 0:
        print(f"Best epoch: {epoch_best}, Best AUC: {auc_best:.4f}", flush=True)
    return model_best, epoch_best, total_time


def train_with_multiple_losses(model_class, model_params, train_loader, valid_loader, 
                                test_loader, epochs, valid_epochs, beta, lr, weight_decay,
                                early_stop, seed, device, loss_configs):
    """
    Train the model with multiple loss functions and compare results
    
    Args:
        model_class: Model class (e.g., LEXGNN)
        model_params: Parameters to initialize the model
        train_loader: Training data loader
        valid_loader: Validation data loader  
        test_loader: Test data loader
        epochs: Maximum number of epochs
        valid_epochs: Validate every N epochs
        beta: Weight for auxiliary loss
        lr: Learning rate
        weight_decay: Weight decay
        early_stop: Early stopping patience
        seed: Random seed
        device: Device to train on
        loss_configs: List of dictionaries with 'name' and 'params' keys
                     Example: [{'name': 'ce', 'params': {}}, 
                              {'name': 'focal', 'params': {'alpha': 0.25, 'gamma': 2.0}}]
    
    Returns:
        Dictionary with results for each loss function
    """
    results = {}
    
    for config in loss_configs:
        loss_name = config['name']
        loss_params = config.get('params', {})
        
        print("\n" + "="*80)
        print(f"Training with {loss_name.upper()} Loss")
        print("="*80)
        
        # Initialize new model for each loss function
        model = model_class(**model_params).to(device)
        
        # Train
        model_best, epoch_best, train_time = train(
            model, train_loader, valid_loader, epochs, valid_epochs,
            beta, lr, weight_decay, early_stop, seed, device,
            loss_name, loss_params
        )
        
        # Test
        auc, f1, gm, ap, auc1, prec, rec = test(model_best, test_loader, device)
        
        results[loss_name] = {
            'auc': auc,
            'f1': f1,
            'gmean': gm,
            'ap': ap,
            'auc_pre': auc1,
            'precision': prec,
            'recall': rec,
            'best_epoch': epoch_best,
            'train_time': train_time
        }
        
        print(f'\n{loss_name.upper()} Results:')
        print(f'AUC: {auc:.4f} | F1-macro: {f1:.4f} | Precision: {prec:.4f} | '
              f'Recall: {rec:.4f} | G-mean: {gm:.4f} | AP: {ap:.4f}')
        print(f'Best Epoch: {epoch_best} | Training Time: {train_time:.2f}s')
    
    return results


def save_results_to_csv(results, filename='loss_comparison_results.csv'):
    """
    Save comparison results to CSV file
    
    Args:
        results: Dictionary with results from train_with_multiple_losses
        filename: Output CSV filename
    """
    import pandas as pd
    
    df = pd.DataFrame(results).T
    df.index.name = 'loss_function'
    df.to_csv(filename)
    print(f"\nResults saved to {filename}")
    return df