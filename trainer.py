from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from pandas import read_csv
import time 
from dataset import ChunkDataset



def prequential_learn(model, learner, args, device):
  model.to(device)
  optim = SGD(model.parameters(), lr=0.001, momentum=0.95)

  chunk_num = 70
  for chunk_idx in range(chunk_num):
      
      # == Define Dataset & Dataloder =====
      data = read_csv('dataset/permutedmnist.csv', sep=',', header=None).values 
      chunk_data = data[chunk_idx*1000:(chunk_idx+1)*1000]
      dataset = ChunkDataset(chunk_data, args)
      test_dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=False)
      train_dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

      # == testing ========================
      if chunk_idx != 0:
        
        known_labels = dataset.label_set
        _, acc_dis, acc_cls = learner.evaluate(model,
                                               test_dataloader,
                                               known_labels)
        print('Dist: {:.4f}, Cls: {}'.format(acc_dis, acc_cls))


      # == training =======================
      if chunk_idx != chunk_num-1 :
        
        global_time = time.time()
        min_loss = float('inf')
        
        for epoch_item in range(args.start_epoch, args.epochs):
          print('=== Epoch {} ==='.format(epoch_item+1))
          train_loss = 0.
          for i, batch in enumerate(train_dataloader):
            
            loss = learner.train(model, batch, optim, args)
            train_loss += loss
            print('Loss: {:.4f}'.format(loss))
      
        # Claculate Pts.
        print('Prototypes are calculating ...')
        learner.calculate_prototypes(model, train_dataloader)

