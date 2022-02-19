import torch
import torch.nn as nn
import math
import torchvision.models as models

def Xavier(m):
  if m.__class__.__name__ == 'Linear':
    fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
    std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    m.weight.data.uniform_(-a, a)
    if m.bias is not None:
      m.bias.data.fill_(0.0)


class MyPretrainedResnet18(nn.Module):
  def __init__(self, args, bias=True):
    super(MyPretrainedResnet18, self).__init__()

    # == Pretrain with torch ===============
    self.pretrained = models.resnet18(pretrained=True)
    self.pretrained.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # == freeze all layers but the last fc =
    for name, param in self.pretrained.named_parameters():
      if name not in ['fc.weight', 'fc.bias']:
      # if not name.startswith(('layer4', 'fc')):
        param.requires_grad = False

    # == Hidden layers =====================
    self.hidden = nn.Sequential(nn.Linear(1000, args.hidden_dims),
                                nn.ReLU(True),
                                nn.Dropout(args.dropout))
    self.hidden.apply(Xavier)

    # == Classifier ========================
    self.linear = nn.Linear(args.hidden_dims, args.n_classes, bias=bias)
    
  def forward(self, samples):
    # x = samples.view(samples.size(0), -1)
    x = self.pretrained(samples)
    features = self.hidden(x)
    outputs = self.linear(features)
    return outputs, features

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0] # store device
    return self

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)


class MLP(nn.Module):
  def __init__(self, n_input, args, bias=True):
    super(MLP, self).__init__()
    self.device = None

    self.hidden = nn.Sequential(nn.Linear(n_input, 256),
                                nn.ReLU(True),
                                nn.Dropout(args.dropout),
                                nn.Linear(256, args.hidden_dims),
                                nn.ReLU(True),
                                nn.Dropout(args.dropout))
    # self.hidden = nn.Sequential(nn.Linear(n_input, args.hidden_dims),
    #                             nn.ReLU(True),
    #                             nn.Dropout(args.dropout))

    self.linear = nn.Linear(args.hidden_dims, args.n_classes, bias=bias)
    self.hidden.apply(Xavier)
  
  def forward(self, samples):
    x = samples.view(samples.size(0), -1)
    features = self.hidden(x)
    outputs = self.linear(features)
    return outputs, features

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0] # store device
    return self

  def save(self, path):
    torch.save(self.state_dict(), path)

  def load(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)
