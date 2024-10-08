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


# if __name__== "__main__":

dataset_path = 'data/train_data.h5'
checkpoint_path = 'data/checkpoint.pt'
train_mesh_path = "data/train_mesh.vol"

mesh = Mesh(train_mesh_path)

#fes_order = 1
#fes = H1(mesh, order=fes_order, dirichlet="rectangle", complex=False)
#gfu = GridFunction(fes)

model, optimizer, scheduler, epoch, learning_rate, scheduler_step, scheduler_gamma = load_check_point(checkpoint_path)
#myloss = LpLoss(size_average=False)
train_data = load_pde_dataset(dataset_path)


time_restrict=True
max_time_in_hours = 2.0
start = time.time()
epochs = 5
batch_size = 8
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

model.train()

for epochn in range(epochs):
    train_mse = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        out_np = out.view(-1, 1).detach().cpu().numpy()

        #for k in range(len(gfu.vec)):
        #    gfu.vec.data[k] = out_np[k][0]
        #Draw(gfu, mesh, "gfu_train")

        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
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
                model.width,
                model.ker_width,
                model.depth,
                model.edge_features,
                model.node_features,
                optimizer,
                epochn,
                learning_rate,
                scheduler_step,
                scheduler_gamma,
                checkpoint_path)
            sys.exit(0)

    print(f'epoch : {epochn}, mse : {train_mse/len(train_loader)}')
    scheduler.step()
    model.eval()

save_check_point(
    model,
    model.width,
    model.ker_width,
    model.depth,
    model.edge_features,
    model.node_features,
    optimizer,
    epochn,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    checkpoint_path)
