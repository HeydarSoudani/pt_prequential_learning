from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time

from samplers.pt_sampler import PtSampler
from samplers.reptile_sampler import ReptileSampler

def train(model,
          learner,
          dataset,
          args, device):
  model.to(device)
  
  ## == Episodic loader ==========
  sampler = PtSampler(
    dataset,
    n_way=args.ways,
    n_shot=args.shot,
    n_query=args.query_num,
    n_tasks=args.meta_iteration
  )
  train_dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=1,
    pin_memory=True,
    collate_fn=sampler.episodic_collate_fn,
  )

  ## == Learn model ==============
  optim = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
  scheduler = StepLR(
    optim,
    step_size=args.step_size,
    gamma=args.gamma,
  )

  min_loss = float('inf')
  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      # print('=== Epoch %d ===' % epoch_item)
      train_loss = 0.0
      trainloader = iter(train_dataloader)

      for miteration_item in range(args.meta_iteration):
        batch = next(trainloader)
        loss = learner.train(model, batch, optim, miteration_item, args)
        train_loss += loss

        ## == validation ==============
        if (miteration_item + 1) % args.log_interval == 0:
          train_loss_total = train_loss / args.log_interval
          train_loss = 0.0
          # evalute on val_dataset
          # ...
          print('=== Step: %d, Train Loss: %f' % (miteration_item+1, train_loss_total))
        
        if args.scheduler:
          scheduler.step()
  
  except KeyboardInterrupt:
    print('skipping training')  