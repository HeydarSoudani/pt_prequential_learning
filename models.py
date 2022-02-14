import torch
import torch.nn as nn
import math

def Xavier(m):
  if m.__class__.__name__ == 'Linear':
    fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
    std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    m.weight.data.uniform_(-a, a)
    if m.bias is not None:
      m.bias.data.fill_(0.0)


class MLP(nn.Module):
  def __init__(self, n_input, n_feature, n_output, args, bias=True):
    super(MLP, self).__init__()
    self.device = None

    self.hidden = nn.Sequential(nn.Linear(n_input, 100),
                                nn.ReLU(True),
                                nn.Dropout(args.dropout),
                                nn.Linear(100, n_feature),
                                nn.ReLU(True),
                                nn.Dropout(args.dropout))
    self.linear = nn.Linear(n_feature, n_output, bias=bias)
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
