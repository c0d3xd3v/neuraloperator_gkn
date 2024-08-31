import sys
import time
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from hlp.hdf5 import load_pde_dataset

from gkn.utilities import *
from gkn.nn_conv import NNConv_old, NNConv, NNConv_Gaussian


class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth - 1):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x


if __name__== "__main__":
    time_restrict=True
    max_time_in_hours = 5.5
    start = time.time()

    dataset_path = 'data/train_data.h5'

    width = 32
    ker_width = 32
    depth = 6
    edge_features = 8
    node_features = 7
    batch_size = 2
    epochs = 100
    learning_rate = 0.005
    scheduler_step = 50
    scheduler_gamma = 0.5

    model = KernelNN(width, ker_width, depth, edge_features, in_width=node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=False)

    train_data = load_pde_dataset(dataset_path)
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
            print(f'remaining time : {diff_h}')
            if(diff_h >= max_time_in_hours and time_restrict==True):
                sys.exit(0)

        print(train_mse/len(train_loader))
        scheduler.step()
        model.eval()

    torch.save(model.state_dict(), "data/current_model.pt")

