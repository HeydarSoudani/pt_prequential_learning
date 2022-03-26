import torch
import torchvision
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

sys.path.insert(1, 'D:/uni/MS/_MS_thesis/codes/our_prequential_learning')
from dataset import ChunkDataset
from samplers.pt_sampler import PtSampler

## == Params ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--n_tasks', type=int, default=5, help='')
parser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'mnist',
    'permutedmnist',
    'rotatedmnist',
    'fmnist'
  ],
  default='permutedmnist',
  help='')
parser.add_argument('--seed', type=int, default=5, help='')
args = parser.parse_args()

## == additional params ===============
args.dataset_path = 'dataset/{}.csv'.format(args.dataset)
args.n_classes = 10

if args.dataset in ['mnist', 'permutedmnist', 'fmnist']:
  args.chunk_num = 70
elif args.dataset in ['rotatedmnist']:
  args.chunk_num = 65

## == Apply seed ======================
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def imshow(imgs):
  # imgs *= 255.0
  grid_imgs = torchvision.utils.make_grid(torch.tensor(imgs), nrow=10)
  plt.imshow(grid_imgs.permute(1, 2, 0))
  plt.show()


def show_samples():
  
  fig, axs = plt.subplots(3, 10)
  ## == Chunk-based version ================
  for index, chunk_idx in enumerate(range(30)):
    print('== Chunk {} =='.format(chunk_idx+1))
    
    # == Define Dataset & test Dataloder ========
    data = read_csv(args.dataset_path, sep=',', header=None).values 
    chunk_data = data[chunk_idx*1000:(chunk_idx+1)*1000]
    dataset = ChunkDataset(chunk_data, args)
    print('Chunk labels: {}'.format(dataset.label_set))

    sampler = PtSampler(
      dataset,
      n_way=10,
      n_shot=1,
      n_query=0,
      n_tasks=1)
    dataloader = DataLoader(
      dataset,
      batch_sampler=sampler,
      num_workers=1,
      pin_memory=True,
      collate_fn=sampler.episodic_collate_fn)

    batch = next(iter(dataloader))
    support_images, support_labels, _, _ = batch
    support_images = torch.squeeze(support_images, 1)
    
    # imshow(support_images)
    grid_imgs = torchvision.utils.make_grid(torch.tensor(support_images), nrow=10)
    
    axs[int(index/10)][index%10].set_title('chunk {}'.format(chunk_idx+1))
    axs[int(index/10)][index%10].set_xticks([])
    axs[int(index/10)][index%10].set_yticks([])
    axs[int(index/10)][index%10].imshow(grid_imgs.permute(1, 2, 0))
    print('Chunk {} done!'.format(chunk_idx+1))
  
  plt.show()

if __name__ == '__main__':
  show_samples()
