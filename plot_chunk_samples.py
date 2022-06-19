import torch
import torchvision
from torch.utils.data import DataLoader
import os
import sys
import argparse
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

# sys.path.insert(1, 'D:/uni/MS/_MS_thesis/codes/our_prequential_learning')
from dataset import ChunkDataset
from samplers.pt_sampler import PtSampler

## == Params ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='permuted_mnist', help='') #[permuted_mnist, permuted_fmnist, rotated_mnist, rotated_fmnist]
parser.add_argument('--n_drift', type=int, default=3, help='')
parser.add_argument('--saved', type=str, default='./dataset/', help='')
parser.add_argument('--change_points', type=str, default='1,2,3', help='')
parser.add_argument('--seed', type=int, default=1, help='')
args = parser.parse_args()

## == additional params ===============
args.dataset_path = 'dataset/{}.csv'.format(args.dataset)
args.n_chunk = 70
args.chunk_size = 1000
args.n_classes = 10

## == Apply seed ======================
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def imshow(imgs):
  # imgs *= 255.0
  grid_imgs = torchvision.utils.make_grid(torch.tensor(imgs), nrow=10)
  plt.imshow(grid_imgs.permute(1, 2, 0))
  plt.show()


if __name__ == '__main__':
  # change_drift_points = np.random.choice(np.arange(5, args.n_chunk-5), args.n_drift, replace=False)
  # change_drift_points = list(np.sort(change_drift_points))
  change_drift_points = [int(item) for item in args.change_points.split(',')]
  print('Change drift points: {}'.format(change_drift_points))
  
  fig, axs = plt.subplots(args.n_drift+1,)

  for idx, current_point in enumerate(change_drift_points + [args.n_chunk]):
    if idx == 0: pervious_point = 0
    else: pervious_point = change_drift_points[idx-1]
    
    # == Define Dataset & test Dataloder ========
    data = read_csv(args.dataset_path, sep=',', header=None).values 
    chunk_data = data[pervious_point*args.chunk_size:current_point*args.chunk_size]
    dataset = ChunkDataset(chunk_data, args)

    sampler = PtSampler(
      dataset,
      n_way=args.n_classes,
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
    
    # For rotation
    # angles = [0, 20, 40, 60, 80, 100, 120]
    # axs[idx].set_title('rotation {}$^\circ$'.format(angles[idx]), fontsize=9)
    # For permutation
    axs[idx].set_title('permutation {}'.format(idx+1), fontsize=11)

    axs[idx].set_xticks([])
    axs[idx].set_yticks([])
    axs[idx].imshow(grid_imgs.permute(1, 2, 0))
    print('Change {} done!'.format(idx+1))
  
  fig.savefig('samples.png', format='png', dpi=800)
  plt.show()
