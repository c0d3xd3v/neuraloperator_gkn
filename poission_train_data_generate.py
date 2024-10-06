import math
import torch
import ngsolve

import numpy as np

from ngsolve import H1, GridFunction, CF, x, y, grad, Draw

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from pde.poission import solvePoission
from hlp.hdf5 import write_pde_dataset_to_hdf5
from hlp.netgen_utilities import generate_unit_rectangle
from gkn.utilities import ball_connectivity, GaussianNormalizer


# output parameters
filename = 'data/train_data.h5'
unit_rect_sampling = 0.0175
r = 1.0*unit_rect_sampling
fes_order = 1

mesh = generate_unit_rectangle(maxh=unit_rect_sampling)
mesh.ngmesh.Save("data/train_mesh.vol")

# static training data, mesh sampling
vertices = [[p[0], p[1], p[2]] for p in mesh.ngmesh.Points()]
meshpoints = [mesh(v[0], v[1], v[2]) for v in vertices]
vertices = np.transpose(np.array(vertices))
edge_index, _ = ball_connectivity(vertices.T, r)

# finite element space creation
fes = H1(mesh, order=fes_order, dirichlet="rectangle", complex=False)
gfu = GridFunction(fes)

train_data = []

for i in range(1,10):
    for j in range(1,10):
        for k in range(1,10):
            o0 = (k/9.)
            x0 = math.cos((i / 10.) * math.pi * 2)
            y0 = math.sin((j / 10.) * math.pi * 2)

            o1 = 1.0
            x1 = 0.0
            y1 = 0.0

            source0 = CF(ngsolve.exp(-0.5*(((x - x0)/o0)**2 + ((y - y0)/(o0))**2)))
            coeff0 = CF(1.)
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
            print(f"{i} : {j} : {k} \n {data_test}")

write_pde_dataset_to_hdf5(filename, train_data)
