import torch.nn as nn
import torch.nn.functional as F
import torch 

class BCEWithLogitsLossWithIgnoreIndex(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, weight=None):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).to(inputs.device)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, :inputs.shape[1], :, :]  # remove 255 from 1hot
        # targets is B x C x H x W so shape[1] is C

        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == 'mean':
            #return loss.mean()
            # if targets have only zeros, we skip them
            return torch.masked_select(loss, targets.sum(dim=1) != 0).mean()
        elif self.reduction == 'sum':
            #return loss.sum()
            return torch.masked_select(loss, targets.sum(dim=1) != 0).sum()
        else:
            #return loss
            return loss * targets.sum(dim=1)

class LabelGuidedOutputDistillation(nn.Module):
      
    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, masks=None):
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        # labels.shape[1] expand tagets.shape[1]
         
        new_labels = torch.zeros_like(outputs)
        new_labels[:, :labels.shape[1]] = labels
        
        unique_labels = torch.unique(masks)
        for i in range(len(unique_labels)):
            if unique_labels[i] != 0 and unique_labels[i] != 255:
                mask = torch.where(masks == unique_labels[i], 1, 0)
                new_labels[:, unique_labels[i]] = mask * labels[:, 0]
                new_labels[:, 0] = (1 - mask) * new_labels[:, 0]
        loss = (outputs * new_labels).mean(dim=1)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        # outputs.shape is B
        return outputs
    
class AdaptiveFeatureDistillation(nn.Module):
    def __init__(self, reduction='mean', alpha=1):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs, targets, weights=1):
        
        loss = (inputs - targets) ** 2
        loss = loss * weights * self.alpha
        
        if self.reduction == 'mean':
            if torch.is_tensor(weights):
                mask = torch.where(weights > 0, 1, 0)
                count = torch.sum(mask.expand_as(loss))
                return torch.sum(loss) / count
            elif weights == 1:
                return torch.mean(loss)
            
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class OtrthogonalLoss(nn.Module):
    def __init__(self, reduction='mean', classes=None):
        super().__init__()
        self.reduction = reduction
        self.classes = classes
    
    def forward(self, class_token, weight=1.):
        class_token = class_token / class_token.norm(dim=-1, keepdim=True)
        class_token_sim = torch.matmul(class_token, class_token.permute(0, 2, 1).detach())
            # class_token_sim.shape is B x C x C
        for i in range(len(class_token_sim)):
            class_token_sim[i].fill_diagonal_(0)

        
        class_token_sim[:, :self.classes[0]] = 0
    
        non_zero_mask = class_token_sim != 0
        loss_orth = class_token_sim[non_zero_mask].abs().mean()
        
        return loss_orth * weight
    
class KnowledgeDistillationLoss(nn.Module):
      
    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, masks=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        
        loss = (outputs * labels).mean(dim=1)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs