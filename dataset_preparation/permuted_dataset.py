import pandas as pd 
import numpy as np
import argparse
import torch
import os


## == Params ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', help='') #[mnist, fmnist, cifar10, cifar100, mini_imagenet]
parser.add_argument('--n_drift', type=int, default=3, help='')
parser.add_argument('--seed', type=int, default=2, help='')
parser.add_argument('--saved', type=str, default='./dataset/', help='')
args = parser.parse_args()

# = Add some variables to args =========
args.data_path = 'data/{}'.format(args.dataset)
args.dataset_file = 'permuted_{}.csv'.format(args.dataset)

if args.dataset in ['mnist']:
  args.n_chunk = 70
  args.chunk_size = 1000

## == Apply seed =======================
np.random.seed(args.seed)

## == Save dir =========================
if not os.path.exists(args.saved):
  os.makedirs(args.saved)


if __name__ == '__main__':
  ## ========================================
  # == Get MNIST dataset ====================
  if args.dataset in ['mnist', 'permuted_mnist']:
    train_data = pd.read_csv(os.path.join(args.data_path, "mnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "mnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  ## ========================================
  # == Get Fashion-MNIST dataset ============
  if args.dataset == 'fmnist':
    train_data = pd.read_csv(os.path.join(args.data_path, "fmnist_train.csv"), sep=',').values
    test_data = pd.read_csv(os.path.join(args.data_path, "fmnist_test.csv"), sep=',').values
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
  ## ========================================
  ## ========================================

  train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
  test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
  all_data = np.concatenate((train_data, test_data), axis=0)
  np.random.shuffle(all_data)

  all_images = all_data[:, :-1]
  all_labels = all_data[:, -1]
  n_feature = all_images.shape[1]

  ### === Permuted dataset (Vector) =========
  change_drift_points = np.random.choice(np.arange(5, args.n_chunk-5), args.n_drift, replace=False)
  change_drift_points = list(np.sort(change_drift_points))
  print('Change drift points: {}'.format(change_drift_points))
  p_data_list = []
  
  for idx, current_point in enumerate(change_drift_points + [args.n_chunk]):
    if idx == 0: pervious_point = 0
    else: pervious_point = change_drift_points[idx-1]  
    perm = torch.randperm(n_feature)

    p_data_list.append(
      np.concatenate((
        all_images[pervious_point*args.chunk_size:current_point*args.chunk_size, perm],
        all_labels[pervious_point*args.chunk_size:current_point*args.chunk_size].reshape(-1, 1)
      ), axis=1)
    )

  p_all_data = np.concatenate(p_data_list, axis=0)
  pd.DataFrame(p_all_data).to_csv(os.path.join(args.saved, args.dataset_file),
    header=None,
    index=None
  )
  print('All data saved in {}'.format(os.path.join(args.saved, args.dataset_file)))


  

