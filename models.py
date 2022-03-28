import torch
import torch.nn as nn
import torchvision.models as models
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



class MyPretrainedResnet18(nn.Module):
  def __init__(self, args, bias=True):
    super(MyPretrainedResnet18, self).__init__()

    # == Pretrain with torch ===============
    self.pretrained = models.resnet18(pretrained=True)
    
    # == 1-channel ===
    arch = list(self.pretrained.children())
    w = arch[0].weight
    arch[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    arch[0].weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
    
    self.pretrained = nn.Sequential(*arch)
    
    # self.pretrained.fc = nn.Sequential(nn.Linear(512, args.hidden_dims),
    #                                     nn.ReLU(True),
    #                                     nn.Dropout(args.dropout))


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
    print(samples.shape)
    x = self.pretrained(samples)
    print(x.shape)
    features = self.hidden(x)
    print(features.shape)
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


class Conv_4(nn.Module):
	def __init__(self, args):
		super(Conv_4, self).__init__()
		
		if args.dataset in ['mnist', 'permutedmnist', 'rotatedmnist', 'pfmnist', 'rfmnist']:
			img_channels = 1	  	# 1
			self.last_layer = 1 	# 3 for 3-layers - 1 for 4-layers
		elif args.dataset in ['cifar10', 'cifar100']:
			img_channels = 3	  	# 3 
			self.last_layer = 2 	# 4 for 3-layers - 2 for 4-layers

		self.filters_length = 256    # 128 for 3-layers - 256 for 4-layers

		self.layer1 = nn.Sequential(
			nn.Conv2d(img_channels, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
			# nn.ReLU(),
			nn.PReLU(),
			nn.Conv2d(32, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
			nn.BatchNorm2d(32),
			nn.PReLU(),
			# nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),   #input: 28 * 28 * 32, output: 14 * 14 * 32
			nn.Dropout(args.dropout)
		)
		
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=5, padding=2), #input: 14 * 14 * 32, output: 14 * 14 * 64
			nn.PReLU(),
			# nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=5, padding=2), #input: 14 * 14 * 64, output: 14 * 14 * 64
			nn.BatchNorm2d(64),
			nn.PReLU(),
			# nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),   #input: 14 * 14 * 64, output: 7* 7 * 64
			nn.Dropout(args.dropout)
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1), #input: 7 * 7 * 64, output: 7 * 7 * 128
			nn.PReLU(),
			# nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1), #input: 7 * 7 * 128, output: 7 * 7 * 128
			nn.BatchNorm2d(128),
			nn.PReLU(),
			# nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),   #input: 7 * 7 * 128, output: 3* 3 * 128
			nn.Dropout(args.dropout)
		)
		
		self.layer4 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1), #input: 3 * 3 * 128, output: 3 * 3 * 256
			nn.PReLU(),
			# nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1), #input: 3*3*256, output: 3*3*256
			nn.BatchNorm2d(256),
			nn.PReLU(),
			# nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),   #input: 3*3*256, output: 1*1*256
			nn.Dropout(args.dropout)
		)

		self.ip1 = nn.Linear(self.filters_length*self.last_layer*self.last_layer, args.hidden_dims)
		self.preluip1 = nn.PReLU()
		self.dropoutip1 = nn.Dropout(args.dropout)
		self.classifier = nn.Linear(args.hidden_dims, 10)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = x.view(-1, self.filters_length*self.last_layer*self.last_layer)

		features = self.preluip1(self.ip1(x))
		x = self.dropoutip1(features)
		logits = self.classifier(x)
		
		return logits, features

	def to(self, *args, **kwargs):
		self = super().to(*args, **kwargs)
		self.device = args[0] # store device
		
		self.layer1 = self.layer1.to(*args, **kwargs)
		self.layer2 = self.layer2.to(*args, **kwargs)
		self.layer3 = self.layer3.to(*args, **kwargs)
		self.layer4 = self.layer4.to(*args, **kwargs)

		self.ip1 = self.ip1.to(*args, **kwargs)
		self.preluip1 = self.preluip1.to(*args, **kwargs)
		self.dropoutip1 = self.dropoutip1.to(*args, **kwargs)
		self.classifier = self.classifier.to(*args, **kwargs)
		return self

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		state_dict = torch.load(path)
		self.load_state_dict(state_dict)
  
