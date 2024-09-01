import sys
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from hlp.hdf5 import load_pde_dataset
from gkn.KernelNN import KernelNN
from gkn.utilities import LpLoss
from hlp.nn import save_check_point
from hlp.nn import load_check_point


if __name__== "__main__":

    dataset_path = 'data/train_data.h5'
    checkpoint_path = 'data/checkpoint.pt'

    width = 32
    ker_width = 32
    depth = 6
    edge_features = 8
    node_features = 7
    batch_size = 2

    learning_rate = 0.005
    scheduler_step = 50
    scheduler_gamma = 0.5

    model = KernelNN(width, ker_width, depth, edge_features, in_width=node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    #myloss = LpLoss(size_average=False)
    train_data = load_pde_dataset(dataset_path)


    time_restrict=True
    max_time_in_hours = 0.05
    start = time.time()
    epochs = 100
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.train()

    for epochn in range(epochs):
        train_mse = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            out_np = out.view(-1, 1).detach().cpu().numpy()

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
                    width,
                    ker_width,
                    depth,
                    edge_features,
                    node_features,
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
                width,
                ker_width,
                depth,
                edge_features,
                node_features,
                optimizer,
                epochn,
                learning_rate,
                scheduler_step,
                scheduler_gamma,
                checkpoint_path)
