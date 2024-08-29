from ngsolve import *

import torch
import numpy as np

from netgen_utilities import generate_unit_rectangle

from pde.poission import solvePoission
from gkn.utilities import ball_connectivity, GaussianNormalizer

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


mesh = generate_unit_rectangle()

vertices = [[p[0], p[1], p[2]] for p in mesh.ngmesh.Points()]
meshpoints = [mesh(v[0], v[1], v[2]) for v in vertices]
vertices = np.transpose(np.array(vertices))
r = 0.05
edge_index, _ = ball_connectivity(vertices.T, r)

fes_order = 1
fes = H1(mesh, order=fes_order, dirichlet="rectangle", complex=False)
gfu = GridFunction(fes)
gfut = GridFunction(gfu.space,multidim=0)

train_data = []

for i in range(10):
    for j in range(10):
        for k in range(10):
            o0 = k/10. + 0.05
            x0 = cos(i / 10. * pi * 2)
            y0 = sin(j / 10. * pi * 2)

            o1 = 1.0
            x1 = 0.0
            y1 = 0.0

            source0 = CF(exp(-0.5*(((x - x0)/o0)**2 + ((y - y0)/(o0))**2)))
            coeff0 = CF(1.) # CF(exp(-0.5*(((x - x1)/o1)**2 + ((y - y1)/(o1))**2)))
            gfu = solvePoission(fes, gfu, g=source0, c=coeff0)

            coeffg = GridFunction(fes)
            coeffg.Set(coeff0)
            coeffg = grad(coeffg)
            coeffx = coeffg[0]
            coeffy = coeffg[1]

            U = torch.Tensor([gfu(x) for x in meshpoints])
            A = torch.Tensor([coeff0(x) for x in meshpoints])
            Ax = torch.Tensor([coeffx(x) for x in meshpoints])
            Ay = torch.Tensor([coeffy(x) for x in meshpoints])
            Rhs = torch.Tensor([source0(x) for x in meshpoints])
            vertices = torch.Tensor(vertices)

            gn = GaussianNormalizer(U)
            U = gn.encode(U)
            gn = GaussianNormalizer(A)
            A = gn.encode(A)
            gn = GaussianNormalizer(Ax)
            Ax = gn.encode(Ax)
            gn = GaussianNormalizer(Ay)
            Ay = gn.encode(Ay)
            gn = GaussianNormalizer(Rhs)
            Rhs = gn.encode(Rhs)

            X = torch.cat([
                            vertices.T,
                            A.reshape(-1, 1),
                            Ax.reshape(-1, 1),
                            Ay.reshape(-1, 1),
                            Rhs.reshape(-1, 1)
                        ], dim=1)

            edge_attr = []
            for edge in edge_index.T:
                v0 = vertices.T[int(edge[0].item())]
                mp0 = mesh(v0[0], v0[1], v0[2])

                v1 = vertices.T[int(edge[1].item())]
                mp1 = mesh(v1[0], v1[1], v1[2])

                a0 = coeff0(mp0)
                a1 = coeff0(mp1)

                edge_attr.append([v0[0].item(), v0[1].item(), v0[2].item(),
                                  v1[0].item(), v1[1].item(), v1[2].item(),
                                  a0, a1])

            edge_attr = torch.Tensor(edge_attr)
            data_test = Data(edge_index=torch.Tensor(edge_index).type(torch.int64),
                             edge_attr=edge_attr,
                             x=X, y=U, coeff=A)
            train_data.append(data_test)

            #Draw(source0, mesh, "source")
            #Draw(coeff0, mesh, "coeff0")
            Draw(gfu, mesh, "gfu")

            print(f"{o0} : {len(train_data)} : {data_test}")


import torch.nn.functional as F
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


width = 128
ker_width = 128
depth = 6
edge_features = 8
node_features = 7
batch_size = 8
epochs = 100
learning_rate = 0.005
scheduler_step = 50
scheduler_gamma = 0.5

model = KernelNN(width, ker_width, depth, edge_features, in_width=node_features)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
myloss = LpLoss(size_average=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
model.train()

gfu_train = GridFunction(fes)

for epochn in range(epochs):
    train_mse = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        out_np = out.view(-1, 1).detach().cpu().numpy()
        #print(len(gfu.vec.data), len(out_np))
        #v0 = [0] * len(gfu.vec)
        for k in range(len(gfu.vec)):
            gfu_train.vec.data[k] = out_np[k][0]
        Draw(gfu_train, mesh, "gfu_train")
        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
        mse.backward()
        optimizer.step()
        train_mse += mse.item()
    print(train_mse/len(train_loader))
    scheduler.step()
    model.eval()

torch.save(model.state_dict(), "current_model.pt")
