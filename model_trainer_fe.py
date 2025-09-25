import os
import time, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, average_precision_score, precision_score, recall_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        # alpha: None | float | list/tuple/tensor per-class weight
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([1 - float(alpha), float(alpha)], dtype=torch.float32)  # binary helper

    def forward(self, logits, targets):
        # logits: [N, C], targets: [N] (Long)
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)         # [N, C]
        probs     = log_probs.exp()                      # [N, C]

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # [N]

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device, dtype=logits.dtype)
            at = alpha.gather(0, targets)                               # per-sample alpha
            loss = -at * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def reset_model_parameters(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def test(model, loader, device):
    labels = []
    output_list = [[], [], [], []]
    model.eval()
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
        output_list[3].extend(output1[: ,1].tolist()[:len(output_labels)])
    output_list = np.array(output_list)
    labels = np.array(labels)

    f1_macro = f1_score(labels, output_list[0], average='macro')
    auc = roc_auc_score(labels, output_list[2])
    ap = average_precision_score(labels, output_list[2])
    fpr, tpr, thresholds = roc_curve(labels, output_list[0])
    gmean = (tpr[1] *(1- fpr[1]))**(1/2)
    prec_macro = precision_score(labels, output_list[0], average='macro', zero_division=0)
    rec_macro  = recall_score(labels,  output_list[0], average='macro', zero_division=0)
    
    auc1 = roc_auc_score(labels, output_list[3])
    return auc, f1_macro, gmean, ap, auc1, prec_macro, rec_macro


def train(model, train_loader, valid_loader, epochs, valid_epochs, 
                beta, lr, weight_decay, early_stop, seed, device):
    
    model.apply(reset_model_parameters)
    model.to(device)
    print()
    
    loss_fn_cls = FocalLoss(alpha=[0.25, 0.75], gamma=2.0)   # contoh: lebih beratkan kelas 1 (fraud)
    loss_fn_aux = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    auc_best, f1_best, epoch_best = 1e-10, 1e-10, 0
    epoch = 1
    
    while epoch <= epochs:
        model.train()
        avg_loss, avg_loss1 = [], []
        epoch_time = 0.0
        torch.cuda.empty_cache()
        start_time = time.time()

        for batch in train_loader:
            _, output_nodes, blocks = batch
            blocks = [b.to(device) for b in blocks]
            output_labels = blocks[-1].dstdata['y'].type(torch.LongTensor).cuda()
            idx_pre = blocks[-1].srcdata['y_mask'].type(torch.LongTensor).cuda() != 2
            output_labels1 = blocks[-1].srcdata['y'].type(torch.LongTensor).cuda()[idx_pre]

            logit, q_list = model(blocks)
            loss = loss_fn_cls(logit, output_labels.squeeze()) #pakai focal
            loss1 = loss_fn_aux(q_list[-1][idx_pre], output_labels1.squeeze()) #pakai CE
            Loss = loss + loss1 * beta 

            Loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()
            avg_loss.append(loss.item())#/ len(output_labels)) 
            avg_loss1.append(loss1.item())# / len(output_labels1)) 

        end_time = time.time()
        epoch_time += end_time - start_time

        if epoch % valid_epochs == 0:
            auc_val, f1_val, gmn_val, ap_val, auc1, prec_val, rec_val = test(model, valid_loader, device)

            if auc_val <= 0.51:
                #print(f"Epoch: {epoch}, Suboptimal initial values. Re-initializing model weights.")
                model.apply(reset_model_parameters)
                auc_val = 0
                epoch = 0

            gain_auc = (auc_val - auc_best) / auc_best
            gain_f1 = (f1_val - f1_best) / f1_best
            if gain_auc > 0:
                auc_best, f1_best, epoch_best = auc_val, f1_val, epoch
                model_best = copy.deepcopy(model)

                line = f'Epoch: {str(epoch).rjust(3, " ")} | loss: {np.mean(avg_loss):.4f} | AUC_val: {auc_best:.4f}| F1: {f1_val:.4f} | P: {prec_val:.4f} | R: {rec_val:.4f} | '
            f'GM: {gmn_val:.4f} | AP: {ap_val:.4f} | '
            print(line)
        
        if (epoch - epoch_best) > early_stop:
            break
        
        epoch += 1
        
    return model_best, epoch_best, epoch_time
