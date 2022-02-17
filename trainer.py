from torch.optim import SGD
from torch.utils.data import DataLoader
from pandas import read_csv
import numpy as np

from dataset import ChunkDataset
from trainers.batch_train import train as batch_train
from trainers.episodic_train import train as episodic_train


def prequential_learn(model, learner, args, device):

  optim = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

  dist_accs = []  
  for chunk_idx in range(args.chunk_num):
    print('=== Chunk {} ==='.format(chunk_idx+1))
    
    # == Define Dataset & test Dataloder ========
    data = read_csv('dataset/{}.csv'.format(args.dataset), sep=',', header=None).values 
    chunk_data = data[chunk_idx*1000:(chunk_idx+1)*1000]
    dataset = ChunkDataset(chunk_data, args)
    test_dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=False)


    # == testing ========================
    if chunk_idx != 0:
      
      known_labels = dataset.label_set
      _, acc_dis, acc_cls = learner.evaluate(model,
                                              test_dataloader,
                                              known_labels)
      print('Dist: {:.4f}, Cls: {}'.format(acc_dis, acc_cls))
      dist_accs.append(acc_dis)
    # == training =======================
    if chunk_idx != args.chunk_num-1 :
      
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
      learner.calculate_prototypes(model, train_dataloader, args)
  
  ## Overal evaluation
  dist_accs = np.array(dist_accs)
  print('Classification rate: ({:.3f}, {:.3f})'.format(np.mean(dist_accs), np.std(dist_accs)))


