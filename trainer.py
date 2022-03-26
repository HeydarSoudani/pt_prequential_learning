from torch.utils.data import DataLoader
from pandas import read_csv
import numpy as np
import time

from dataset import ChunkDataset
from trainers.batch_train import train as batch_train
from trainers.episodic_train import train as episodic_train


def prequential_learn(model, learner, args, device):

  cls_accs = []
  dist_accs = []
  test_time = []
  train_time = []

  for chunk_idx in range(args.n_chunk):
    print('=== Chunk {} ============'.format(chunk_idx+1))
    
    # == Define Dataset & test Dataloder ========
    data = read_csv('dataset/{}.csv'.format(args.dataset), sep=',', header=None).values 
    chunk_data = data[chunk_idx*1000:(chunk_idx+1)*1000]
    dataset = ChunkDataset(chunk_data, args)
    test_dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=False)
    # print('Chunk labels: {}'.format(dataset.label_set))

    # == testing ========================   
    known_labels = dataset.label_set
    test_start = time.time()
    _, acc_dis, acc_cls = learner.evaluate(model,
                                            test_dataloader,
                                            known_labels)
    test_time.append(time.time() - test_start)
    print('Dist: {:.4f}, Cls: {}'.format(acc_dis, acc_cls))
    cls_accs.append(acc_cls)
    dist_accs.append(acc_dis)
    
    # == training =======================
    if chunk_idx != args.n_chunk-1 :
      train_start = time.time()
      ### == Train Model (Batch) ===========
      if args.algorithm == 'batch':
        batch_train(
          model,
          learner,
          dataset,
          args, device)

      ### == Train Model (Episodic) ========
      else:
        episodic_train(
          model,
          learner,
          dataset,
          args, device)
    
      # Claculate Pts.
      print('Prototypes are calculating ...')
      learner.calculate_prototypes(model, test_dataloader, args)
      train_time.append(time.time() - train_start)

  ## === Overal evaluation ==========
  test_time = np.array(test_time)
  print('test time(s): ({:.3f}, {:.3f}, {:.3f})'.format(np.sum(test_time), np.mean(test_time), np.std(test_time)))

  train_time = np.array(train_time)
  print('train time(s): ({:.3f}, {:.3f}, {:.3f})'.format(np.sum(train_time), np.mean(train_time), np.std(train_time)))

  print(dist_accs)
  dist_accs = np.array(dist_accs)
  print('Classification rate (dist): ({:.3f}, {:.3f})'.format(np.mean(dist_accs), np.std(dist_accs)))

  print(cls_accs)
  cls_accs = np.array(cls_accs)
  print('Classification rate (cls): ({:.3f}, {:.3f})'.format(np.mean(cls_accs), np.std(cls_accs)))

