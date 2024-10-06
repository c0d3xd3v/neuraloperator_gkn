import torch

from gkn.utilities import ball_connectivity, GaussianNormalizer


def construct_poission_dirichlet_data(meshpoints, vertices, edge_index, gfu, coeff0, coeffx, coeffy, source0):
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
    return data_test
