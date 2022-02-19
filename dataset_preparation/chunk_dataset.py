import pandas as pd 
import numpy as np
import argparse
import pickle
import os

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

## == Params ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', help='') #[mnist, fmnist, cifar10, cifar100, mini_imagenet]
parser.add_argument('--seed', type=int, default=2, help='')
parser.add_argument('--saved', type=str, default='./dataset/', help='')
args = parser.parse_args()

# = Add some variables to args =========
args.data_path = 'data/{}'.format(args.dataset)
args.dataset_file = '{}.csv'.format(args.dataset)

## == Apply seed =======================
np.random.seed(args.seed)

## == Save dir =========================
if not os.path.exists(args.saved):
  os.makedirs(args.saved)


if __name__ == '__main__':
  ## ========================================
  # == Get MNIST dataset ====================
  if args.dataset == 'mnist':
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
  print('all data: {}'.format(all_data.shape))
  pd.DataFrame(all_data).to_csv(os.path.join(args.saved, args.dataset_file),
    header=None,
    index=None
  )
  print('All data saved in {}'.format(os.path.join(args.saved, args.dataset_file)))
