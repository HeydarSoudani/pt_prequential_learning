import torch
import torchvision.transforms as transforms
import pandas as pd 
import numpy as np
import argparse
import os


## == Params ===========================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='fmnist', help='') #[mnist, fmnist, cifar10, cifar100, mini_imagenet]
parser.add_argument('--n_drift', type=int, default=3, help='')
parser.add_argument('--saved', type=str, default='./dataset/', help='')
parser.add_argument('--seed', type=int, default=1, help='')
args = parser.parse_args()

# = Add some variables to args =========
args.dataset_file = 'rotated_{}.csv'.format(args.dataset)
args.data_path = 'data/{}'.format(args.dataset)
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

  all_images = all_data[:, :-1]
  all_labels = all_data[:, -1]
  n_feature = all_images.shape[1]

  ### === Permuted dataset (Vector) =========
  change_drift_points = np.random.choice(np.arange(5, args.n_chunk-5), args.n_drift, replace=False)
  change_drift_points = list(np.sort(change_drift_points))
  print('Change drift points: {}'.format(change_drift_points))
  r_data_list = []
  
  angles = [0, 20, 40, 60, 80, 100, 120]
  # angles = [0, 10, 20, 30, 40, 50, 60]
  for idx, current_point in enumerate(change_drift_points + [args.n_chunk]):
    if idx == 0: pervious_point = 0
    else: pervious_point = change_drift_points[idx-1]  
    
    # Select images for current task
    temp_images = all_images[pervious_point*args.chunk_size:current_point*args.chunk_size]
    
    # rotate each image
    tensor_view = (1, 28, 28)
    rotated_xtrain_list = []
    for img in temp_images:
      x_tensor = (torch.tensor(img, dtype=torch.float) / 255).view(tensor_view)
      pil_img = transforms.ToPILImage()(x_tensor)
      rotated_pil_img = transforms.functional.rotate(pil_img, angles[idx])
      rotated_img = transforms.ToTensor()(rotated_pil_img)
      rotated_img = rotated_img*255.0
      rotated_xtrain_list.append(rotated_img)
    
    rotated_xtrain = torch.stack(rotated_xtrain_list)
    rotated_xtrain = rotated_xtrain.clone().detach().numpy()
    rotated_xtrain = rotated_xtrain.reshape(rotated_xtrain.shape[0], -1)

    r_data_list.append(
      np.concatenate((
        rotated_xtrain,
        all_labels[pervious_point*args.chunk_size:current_point*args.chunk_size].reshape(-1, 1)
      ), axis=1)
    )



  r_all_data = np.concatenate(r_data_list, axis=0)
  pd.DataFrame(r_all_data).to_csv(os.path.join(args.saved, args.dataset_file),
    header=None,
    index=None
  )
  print('All data saved in {}'.format(os.path.join(args.saved, args.dataset_file)))


  

