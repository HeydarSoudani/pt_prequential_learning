import os
import torch
import argparse
import numpy as np

from models import MLP, MyPretrainedResnet18, Conv_4
from losses import MetricLoss
from trainer import prequential_learn
from learners.pt_learner import PtLearner
from learners.reptile_learner import ReptileLearner
from learners.batch_learner import BatchLearner
from losses import PtLoss, MetricLoss

## == Params =========================
parser = argparse.ArgumentParser()
parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'permuted_mnist',
    'rotated_mnist',
    'fmnist',
    'permuted_fmnist',
    'rotated_fmnist'
  ],
  default='permuted_mnist',
  help=''
)
parser.add_argument(
  '--algorithm',
  type=str,
  choices=['batch', 'prototype', 'reptile'],
  default='batch',
  help=''
)

# train phase
parser.add_argument('--start_epoch', type=int, default=0, help='')
parser.add_argument('--epochs', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--meta_iteration', type=int, default=3000, help='')
parser.add_argument('--log_interval', type=int, default=100, help='must be less then meta_iteration parameter')
parser.add_argument('--ways', type=int, default=5, help='')
parser.add_argument('--shot', type=int, default=5, help='')
parser.add_argument('--query_num', type=int, default=5, help='')

# Network
parser.add_argument('--dropout', type=float, default=0.2, help='')
parser.add_argument('--hidden_dims', type=int, default=128, help='')

# Prototype
parser.add_argument('--beta', type=float, default=1.0, help='Update Prototype in Prototypical algorithm')

# Loss function
parser.add_argument("--lambda_1", type=float, default=1.0, help="Metric Coefficien in loss function")
parser.add_argument("--lambda_2", type=float, default=1.0, help="CE Coefficient in loss function")
parser.add_argument("--temp_scale", type=float, default=0.2, help="Temperature scale for DCE in loss function",)

# Optimizer
parser.add_argument('--lr', type=float, default=0.1, help='')
parser.add_argument('--momentum', type=float, default=0.9, help='')
parser.add_argument('--wd', type=float, default=1e-4, help='')  #l2 regularization
parser.add_argument('--grad_clip', type=float, default=0.1)   # before was 5.0

# Scheduler
parser.add_argument("--scheduler", action="store_true", help="use scheduler")
parser.add_argument("--step_size", default=8, type=int)
parser.add_argument('--gamma', type=float, default=0.5, help='for lr step')

# Device and Randomness
parser.add_argument('--cuda', action='store_true',help='use CUDA')
parser.add_argument('--seed', type=int, default=2, help='')

# Save and load model
parser.add_argument('--save', type=str, default='saved/', help='')

args = parser.parse_args()   

## == additional params ================
args.n_classes = 10
args.n_chunk = 70
# if args.dataset in ['mnist', 'permuted_mnist', 'rotated_mnist']:
#   args.n_chunk = 70

## == Device ===========================
if torch.cuda.is_available():
  if not args.cuda:
    args.cuda = True
  torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Device: {}'.format(device))

## == Apply seed =======================
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if args.cuda:
  torch.cuda.manual_seed_all(args.seed)

## == Save dir =========================
if not os.path.exists(args.save):
  os.makedirs(args.save)

## == Model ============================
model = MLP(784, args)
# model = MyPretrainedResnet18(args)
# model = Conv_4(args)
model.to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params: {}'.format(total_params))
print('Total trainable params: {}'.format(total_params_trainable))

## == Loss & Learner Definition =========
if args.algorithm == 'prototype':
  criterion = PtLoss(device, args)
  learner = PtLearner(criterion, device, args)
elif args.algorithm == 'reptile':
  criterion = torch.nn.CrossEntropyLoss()
  learner = ReptileLearner(criterion, device, args)
elif args.algorithm == 'batch':
  criterion = MetricLoss(device, args)
  learner = BatchLearner(criterion, device, args)


if __name__ == '__main__':
  prequential_learn(model, learner, args, device)
