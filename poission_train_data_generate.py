import math
import torch
import ngsolve
import numpy as np
from ngsolve import H1, GridFunction, CF, x, y, grad, Draw
from torch_geometric.data import Data
from pde.poission import solvePoission
from hlp.hdf5 import write_pde_dataset_to_hdf5
from hlp.netgen_utilities import generate_unit_rectangle, generate_unit_rectangle_with_hole
from gkn.utilities import ball_connectivity, GaussianNormalizer

# Output-Parameter
filename = 'data/train_data_2.h5'
unit_rect_sampling = 0.1
r = 0.5 * unit_rect_sampling
fes_order = 2

# Erstelle das Mesh
rectange_mesh = generate_unit_rectangle(maxh=unit_rect_sampling)
circle_mesh = generate_unit_rectangle_with_hole(maxh=unit_rect_sampling)
#mesh.ngmesh.Save("data/train_mesh.vol")
meshes = [rectange_mesh, circle_mesh]

train_data = []

# Sammle alle Werte für die Normalisierung
all_U, all_A, all_Ax, all_Ay, all_Rhs = [], [], [], [], []

for mesh in meshes:
    # Statistische Trainingsdaten, Mesh-Sampling
    vertices = [[p[0], p[1], p[2]] for p in mesh.ngmesh.Points()]
    meshpoints = [mesh(v[0], v[1], v[2]) for v in vertices]
    vertices = np.transpose(np.array(vertices))
    edge_index, _ = ball_connectivity(vertices.T, r)

    # Erstelle den Finite Element Space
    fes = H1(mesh, order=fes_order, dirichlet="rectangle", complex=False)
    gfu = GridFunction(fes)

    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                # Parameter bestimmen
                o0 = (k / 9.)
                x0 = math.cos((i / 10.) * math.pi * 2)
                y0 = math.sin((j / 10.) * math.pi * 2)

                source0 = CF(ngsolve.exp(-0.5 * (((x - x0) / o0) ** 2 + ((y - y0) / (o0)) ** 2)))
                coeff0 = CF(1.)
                gfu = solvePoission(fes, gfu, g=source0, c=coeff0)

                Draw(gfu)

                coeffg = GridFunction(fes)
                coeffg.Set(coeff0)
                coeffg = grad(coeffg)
                coeffx = coeffg[0]
                coeffy = coeffg[1]

                # Werte sammeln
                U = torch.Tensor([gfu(x) for x in meshpoints])
                A = torch.Tensor([coeff0(x) for x in meshpoints])
                Ax = torch.Tensor([coeffx(x) for x in meshpoints])
                Ay = torch.Tensor([coeffy(x) for x in meshpoints])
                Rhs = torch.Tensor([source0(x) for x in meshpoints])

                all_U.append(U)
                all_A.append(A)
                all_Ax.append(Ax)
                all_Ay.append(Ay)
                all_Rhs.append(Rhs)

                vertices = torch.Tensor(vertices)

                # Erstelle die Graph-Datenstruktur
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
                                x=X, 
                                y=U, 
                                coeff=A)            
                train_data.append(data_test)


# Normalisierung der gesammelten Daten
all_U = torch.cat(all_U)
all_A = torch.cat(all_A)
all_Ax = torch.cat(all_Ax)
all_Ay = torch.cat(all_Ay)
all_Rhs = torch.cat(all_Rhs)

# Erstelle den Normalizer für U
gn_U = GaussianNormalizer(all_U)
normalized_U = gn_U.encode(all_U)

# Erstelle den Normalizer für A
gn_A = GaussianNormalizer(all_A)
normalized_A = gn_A.encode(all_A)

# Erstelle den Normalizer für Ax
gn_Ax = GaussianNormalizer(all_Ax)
normalized_Ax = gn_Ax.encode(all_Ax)

# Erstelle den Normalizer für Ay
gn_Ay = GaussianNormalizer(all_Ay)
normalized_Ay = gn_Ay.encode(all_Ay)

# Erstelle den Normalizer für Rhs
gn_Rhs = GaussianNormalizer(all_Rhs)
normalized_Rhs = gn_Rhs.encode(all_Rhs)

# Wende die Normalisierung auf den gesamten train_data an
for i in range(len(train_data)):
    train_data[i].y = gn_U.encode(train_data[i].y)
    train_data[i].coeff = gn_A.encode(train_data[i].coeff)

    # Zugriff auf die X-Daten
    Ax = train_data[i].x[:, -3]  # Angenommen, Ax ist die drittletzte Spalte
    Ay = train_data[i].x[:, -2]  # Angenommen, Ay ist die vorletzte Spalte
    Rhs = train_data[i].x[:, -1]  # Angenommen, Rhs ist die letzte Spalte

    # Normalisierung
    train_data[i].x[:, -3] = gn_Ax.encode(Ax)
    train_data[i].x[:, -2] = gn_Ay.encode(Ay)
    train_data[i].x[:, -1] = gn_Rhs.encode(Rhs)

# Schreibe die normalisierten Daten in eine HDF5-Datei
write_pde_dataset_to_hdf5(filename, train_data)
