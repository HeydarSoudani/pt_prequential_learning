from torch.optim import Adam
from torch.utils.data import DataLoader
import time

def train(model,
          learner,
          dataset,
          args, device):
  
  ## == Batch loader =============
  train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

  ## == Learn model ==============
  optim = Adam(model.parameters(), lr=args.lr)

  min_loss = float('inf')
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      train_loss = 0.
      for i, batch in enumerate(train_dataloader):
        loss = learner.train(model, batch, optim, args)
        train_loss += loss
      print('Epoch: {}, train_loss: {:.4f}'.format(epoch_item+1, train_loss))

  except KeyboardInterrupt:
    print('skipping training')  