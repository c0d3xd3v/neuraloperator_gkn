import sys
import time
import torch
from ngsolve import *
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from hlp.hdf5 import load_pde_dataset
from gkn.KernelNN import KernelNN
from gkn.utilities import LpLoss
from hlp.nn import save_check_point
from hlp.nn import load_check_point
from gkn.gaussian_batch_normalizer import GaussianBatchNormalizer


# if __name__== "__main__":

dataset_path = 'data/train_data.h5'
checkpoint_path = 'data/checkpoint.pt'

model, optimizer, scheduler, epoch, learning_rate, scheduler_step, scheduler_gamma, normalizer, target_normalizer = load_check_point(checkpoint_path)
#myloss = LpLoss(size_average=False)
train_data = load_pde_dataset(dataset_path)

time_restrict=True
max_time_in_hours = 5.75
start = time.time()
epochs = 50
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

model.train()

for epochn in range(epochs):
    train_mse = 0.0
    for batch in train_loader:

        local_batch = batch.clone()

        data_tensor = local_batch.x
        target_tensor = local_batch.y

        normalizer.update(data_tensor)
        target_normalizer.update(target_tensor)

        local_batch.x = normalizer.encode(data_tensor)
        local_batch.y = target_normalizer.encode(target_tensor)

        optimizer.zero_grad()
        out = model(local_batch)
        out_np = out.view(-1, 1).detach().cpu().numpy()

        mse = F.mse_loss(out.view(-1, 1), local_batch.y.view(-1,1))
        mse.backward()
        optimizer.step()
        train_mse += mse.item()

        end = time.time()
        diff = end - start
        diff_h = diff/3600.

        if(diff_h >= max_time_in_hours and time_restrict==True):
            model.eval()
            save_check_point(
                model,
                optimizer,
                epoch + epochn,
                learning_rate,
                scheduler_step,
                scheduler_gamma,
                checkpoint_path,
                normalizer,
                target_normalizer)
            sys.exit(0)

    print(f'epoch : {epoch + epochn}, mse : {train_mse/len(train_loader)}')
    scheduler.step()
    model.eval()

save_check_point(
    model,
    optimizer,
    epoch + epochn,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    checkpoint_path,
    normalizer,
    target_normalizer)
