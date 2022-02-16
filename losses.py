import torch
from pytorch_metric_learning import distances, losses, miners

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