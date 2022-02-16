import torch

def compute_prototypes(
  support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
  """
  Compute class prototypes from support features and labels
  Args:
    support_features: for each instance in the support set, its feature vector
    support_labels: for each instance in the support set, its label
  Returns:
    for each label of the support set, the average feature vector of instances with this label
  """
  seen_labels = torch.unique(support_labels)

  # Prototype i is the mean of all instances of features corresponding to labels == i
  return torch.cat(
    [
      support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
      for l in seen_labels
    ]
  )


class BatchLearner:
  def __init__(self, criterion, device, args):
    self.criterion = criterion
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.hidden_dims, device=device)
      for l in range(args.n_classes)
    }

  def train(self, model, batch, optimizer, args):
    model.train()  
    optimizer.zero_grad()

    images, labels = batch
    images, labels = images.to(self.device), labels.to(self.device)

    ## == Forward ===========================
    outputs, features = model.forward(images)
    loss = self.criterion(outputs, labels)
    
    ## == Backward ==========================
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    ## == Calculate Prototypes ==============
    # unique_label = torch.unique(labels)

    # batch_prototypes = compute_prototypes(features, labels)
    # old_prototypes = torch.cat(
    #   [self.prototypes[l.item()] for l in unique_label]
    # )
    # new_prototypes = args.beta * old_prototypes + (1 - args.beta) * batch_prototypes
    
    # for idx, l in enumerate(unique_label):
    #   self.prototypes[l.item()] = new_prototypes[idx].reshape(1, -1).detach()
    ## =======================================

    return loss.detach().item()
  
  def evaluate(self, model, dataloader, known_labels):
    model.eval()
    
    total_loss = 0.0
    total_dist_acc = 0.0
    correct_cls_acc = 0.0
    total_cls_acc = 0

    # = For Distance-based Acc. ==
    known_labels = torch.tensor(list(known_labels), device=self.device)
    pts = torch.cat(
      [self.prototypes[l.item()] for l in known_labels]
    )

    with torch.no_grad():
      for j, data in enumerate(dataloader):
        sample, labels = data
        sample, labels = sample.to(self.device), labels.to(self.device)
        logits, features = model.forward(sample)

        ## == Cls-based Acc. ===================
        _, predicted = torch.max(logits, 1)
        total_cls_acc += labels.size(0)
        correct_cls_acc += (predicted == labels).sum().item()
        
        ## == Distance-based Acc. ============== 
        dists = torch.cdist(features, pts)  #[]
        argmin_dists = torch.min(dists, dim=1).indices
        pred_labels = known_labels[argmin_dists]
        
        acc = (labels==pred_labels).sum().item() / labels.size(0)
        total_dist_acc += acc

        ## == loss =============================
        loss = self.criterion(logits, labels)
        loss = loss.mean()
        total_loss += loss.item()
      
      total_loss /= len(dataloader)
      total_dist_acc /= len(dataloader)
      total_cls_acc = correct_cls_acc / total_cls_acc  

    return total_loss, total_dist_acc, total_cls_acc

  
  def calculate_prototypes(self, model, dataloader):
    model.eval()
    
    all_features = []
    all_labels = []
    with torch.no_grad():
      for j, data in enumerate(dataloader):
        sample, labels = data
        sample, labels = sample.to(self.device), labels.to(self.device)
        _, features = model.forward(sample)
        all_features.append(features)
        all_labels.append(labels)
      
      all_features = torch.cat(all_features, dim=0)
      all_labels = torch.cat(all_labels, dim=0)
      
      unique_labels = torch.unique(all_labels)
      pts = compute_prototypes(all_features, all_labels)
      for idx, l in enumerate(unique_labels):
        self.prototypes[l.item()] = pts[idx].reshape(1, -1).detach()
        

  def load(self, pkl_path):
    self.__dict__.update(torch.load(pkl_path))

  def save(self, pkl_path):
    torch.save(self.__dict__, pkl_path)
