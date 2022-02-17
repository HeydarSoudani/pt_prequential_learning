import torch
from learners.pt_learner import compute_prototypes


## According to the article, this learner does not use query set
class ReptileLearner:
  def __init__(self, criterion, device, args):
    self.criterion = criterion
    self.device = device

    self.prototypes = {
      l: torch.zeros(1, args.hidden_dims, device=device)
      for l in range(args.n_classes)
    }

  def train(self, model, queue, optimizer, iteration, args):
    model.train()

    old_vars = [param.data.clone() for param in model.parameters()]
    queue_length = len(queue)
    losses = 0

    for k in range(args.update_step):
      for i in range(queue_length):
          optimizer.zero_grad()

          batch = queue[i]
          support_images = batch['data']
          support_labels = batch['label']
          support_images = support_images.reshape(-1, *support_images.shape[2:])
          support_labels = support_labels.flatten() 
          support_images = support_images.to(self.device)
          support_labels = support_labels.to(self.device)
         
          logits, support_features = model.forward(support_images)

          loss = self.criterion(logits, support_labels)
          loss.backward()
          losses += loss.detach().item()

          torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
          optimizer.step()

          # Update pts ========
          # unique_label = torch.unique(support_labels)
          # episode_prototypes = compute_prototypes(
          #   support_features, support_labels
          # )
          # old_prototypes = torch.cat(
          #   [self.prototypes[l.item()] for l in unique_label]
          # )
          # if args.beta_type == 'evolving':
          #   beta = args.beta * iteration / args.meta_iteration
          # elif args.beta_type == 'fixed':
          #   beta = args.beta

          # new_prototypes = beta * old_prototypes + (1 - beta) * episode_prototypes
          # for idx, l in enumerate(unique_label):
          #   self.prototypes[l.item()] = new_prototypes[idx].reshape(1, -1).detach()


    beta = args.beta * (1 - iteration / args.meta_iteration)
    for idx, param in enumerate(model.parameters()):
      param.data = (1 - beta) * old_vars[idx].data + beta * param.data

    return losses / (queue_length * args.update_step)


  def evaluate(self, model, dataloader, known_labels, args):
    model.eval()
    
    total_loss = 0.0
    total_dist_acc = 0.0
    correct_cls_acc = 0.0
    total_cls_acc = 0

    known_labels = torch.tensor(list(known_labels), device=self.device)
    pts = torch.cat(
      [self.prototypes[l.item()] for l in known_labels]
    )

    with torch.no_grad():
      for i, batch in enumerate(dataloader):
        samples, labels = batch
        labels = labels.flatten()
        samples, labels = samples.to(self.device), labels.to(self.device)
        logits, features = model.forward(samples)

        ## == Distance-based Acc. ============== 
        dists = torch.cdist(features, pts)  #[]
        argmin_dists = torch.min(dists, dim=1).indices
        pred_labels = known_labels[argmin_dists]
        
        acc = (labels==pred_labels).sum().item() / labels.size(0)
        total_dist_acc += acc

        ## == Cls-based Acc. ===================
        _, predicted = torch.max(logits, 1)
        total_cls_acc += labels.size(0)
        correct_cls_acc += (predicted == labels).sum().item()

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
