import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import distances, losses, miners

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)

def euclidean_dist(x, y):
  '''
  Compute euclidean distance between two tensors
  '''
  # x: N x D
  # y: M x D
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  if d != y.size(1):
    raise Exception

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)

class DCELoss(nn.Module):
  def __init__(self, device, gamma=0.05):
    super().__init__()
    self.gamma = gamma
    self.device = device

  def forward(self, features, labels, prototypes, n_query, n_classes):
    unique_labels = torch.unique(labels)
    features = torch.cat(
      [features[(labels == l).nonzero(as_tuple=True)[0]] for l in unique_labels]
    )

    dists = euclidean_dist(features, prototypes)
    # dists = (-self.gamma * dists).exp() 

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = (
      torch.arange(0, n_classes, device=self.device, dtype=torch.long)
      .view(n_classes, 1, 1)
      .expand(n_classes, n_query, 1)
    )

    loss_val = -log_p_y.gather(2, target_inds).mean()
    return loss_val

class PtLoss(nn.Module):
  def __init__(self, device, args):
    super().__init__()
    self.args = args
    self.lambda_1 = args.lambda_1
    self.lambda_2 = args.lambda_2
    
    self.dce = DCELoss(device, gamma=args.temp_scale)
    self.ce = torch.nn.CrossEntropyLoss()

  def forward(self, features, outputs, labels, prototypes, n_query, n_classes):
    dce_loss = self.dce(features, labels, prototypes, n_query, n_classes)
    cls_loss = self.ce(outputs, labels.long())

    return self.lambda_1 * dce_loss +\
           self.lambda_2 * cls_loss

class MetricLoss(torch.nn.Module):
  def __init__(self, device, args):
    super().__init__()
    self.args = args
    self.lambda_1 = args.lambda_1 # Metric loss coef
    self.lambda_2 = args.lambda_2 # CE coef

    self.ce = torch.nn.CrossEntropyLoss()
    self.miner = miners.BatchEasyHardMiner() # for ContrastiveLoss 
    # self.metric = losses.NTXentLoss(temperature=0.07)
    # self.metric = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    self.metric = losses.TripletMarginLoss(margin=0.05)
    # self.metric = losses.CosFaceLoss(
    #   args.n_classes,
    #   args.hidden_dims,
    #   margin=0.35, scale=64
    # ).to(device)

  def forward(self, logits, labels):
    cls_loss = self.ce(logits, labels.long())
    
    ## loss with miner
    # miner_output = self.miner(logits, labels.long())
    # metric_loss = self.metric(logits, labels.long(), miner_output)
    ## loss without minier
    metric_loss = self.metric(logits, labels.long())

    return self.lambda_1 * metric_loss +\
           self.lambda_2 * cls_loss