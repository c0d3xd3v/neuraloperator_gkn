import math

from ngsolve import *

from ngsolve import H1
from ngsolve import GridFunction
from ngsolve import CF
from ngsolve import x
from ngsolve import y
from ngsolve import grad
from ngsolve import exp
from ngsolve import Mesh
from ngsolve import sin
from ngsolve import Draw

from netgen.occ import *
from netgen.geom2d import SplineGeometry

import torch
import numpy as np

from torch_geometric.data import Data

from gkn.utilities import ball_connectivity
from hlp.hdf5 import write_pde_dataset_to_hdf5

def trianglesToEdgeList(mesh):
    tris = [(t[0][0:3] - 1).tolist() for t in np.array(mesh.ngmesh.Elements2D())]
    edges = []*0
    for tri in tris:
        edges.append([int(tri[0]), int(tri[1])])
        edges.append([int(tri[1]), int(tri[2])])
        edges.append([int(tri[2]), int(tri[0])])
    edges = np.transpose(np.array(edges))
    return edges


def generate_unit_rectangle(maxh=0.1):
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-0.5,-0.5),
                     p2=( 0.5, 0.5),
                     bc="rectangle",
                     leftdomain=1,
                     rightdomain=0)
    return Mesh(geo.GenerateMesh(maxh=maxh))


def generate_unit_rectangle_with_hole(maxh=0.1):
    air = Circle((0.5, 0.5), 0.8).Face()
    air.edges.name = 'rectangle'
    scatterer = MoveTo(0.7, 0.3).Rectangle(0.05, 0.4).Face()
    scatterer.edges.name = 'rectangle'
    geo = OCCGeometry(air - scatterer, dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    return mesh


def generate_unit_circle(maxh=0.1):
    air = Circle((0.0, 0.0), 0.5).Face()
    air.edges.name = 'rectangle'
    geo = OCCGeometry(air, dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=maxh))
    return mesh


# Function to get boundary points
def get_boundary_node_ids(mesh):
    # Initialize a list to store boundary node IDs
    boundary_nodeIds = set()  # Using a set to avoid duplicates
        
    for boundary_name in mesh.GetBoundaries():
        # Access the boundary region
        boundary_region = mesh.Boundaries(boundary_name)
        # Iterate over elements in the boundary region
        for element in boundary_region.Elements():
            # Access vertex IDs of the element
            for v in element.vertices:
                # Add node IDs to the set
                boundary_nodeIds.add(v.nr)

    return list(boundary_nodeIds)  # Return as a list


def sample_from_ngsolve_mesh(fes, mesh, source0, coeff0,  r = 0.2):
    vertices = [[p[0], p[1], p[2]] for p in mesh.ngmesh.Points()]
    meshpoints = [mesh(v[0], v[1], v[2]) for v in vertices]

    boundary_nodes = get_boundary_node_ids(mesh)
    node_boundary_feature = [0]*len(vertices)

    for i in boundary_nodes:
        node_boundary_feature[i] = 1.0
    node_boundary_feature = torch.Tensor(np.array(node_boundary_feature).T)

    vertices = np.transpose(np.array(vertices))
    edge_index, _ = ball_connectivity(vertices.T, r)

    coeffg = GridFunction(fes)
    coeffg.Set(coeff0)
    coeffg = grad(coeffg)
    coeffx = coeffg[0]
    coeffy = coeffg[1]

    #U = torch.Tensor([gfu(x) for x in meshpoints])
    A = torch.Tensor([coeff0(x) for x in meshpoints])
    Ax = torch.Tensor([coeffx(x) for x in meshpoints])
    Ay = torch.Tensor([coeffy(x) for x in meshpoints])
    Rhs = torch.Tensor([source0(x) for x in meshpoints])
    vertices = torch.Tensor(vertices)

    X = torch.cat([
        vertices.T,
        node_boundary_feature.reshape(-1, 1),
        A.reshape(-1, 1),
        Ax.reshape(-1, 1),
        Ay.reshape(-1, 1),
        Rhs.reshape(-1, 1)
    ], dim=1)

    print(edge_index.shape)
    edge_attr = []
    for edge in edge_index.T:
        v0 = vertices.T[int(edge[0].item())]
        mp0 = mesh(v0[0], v0[1], v0[2])

        v1 = vertices.T[int(edge[1].item())]
        mp1 = mesh(v1[0], v1[1], v1[2])

        a0 = coeff0(mp0)
        a1 = coeff0(mp1)

        attr = [v0[0].item(), v0[1].item(), v0[2].item(),
                          v1[0].item(), v1[1].item(), v1[2].item(),
                          a0, a1]
        edge_attr.append(attr)
    edge_attr = torch.Tensor(edge_attr)

    data_test = Data(edge_index=torch.Tensor(edge_index).type(torch.int64),
                     edge_attr=edge_attr,
                     x=X, y=None, coeff=A)
    return data_test


def generate_rectangle_geometry(fes_order, unit_rect_sampling):
    s = 1.0
    scatterer = MoveTo(-s*0.5, -s*0.5).Rectangle(s, s).Face()
    scatterer.edges.name = 'rectangle'
    #air = Circle((0.0, 0.0), 0.5*s).Face()
    #air.edges.name = 'rectangle'
    geo = OCCGeometry(scatterer, dim=2)

    mesh = Mesh(geo.GenerateMesh(maxh=unit_rect_sampling))
    fes = H1(mesh, order=fes_order, dirichlet="rectangle", complex=False)

    #k = 10
    #j = 0
    #i = 0
    #o0 = k / 10.0 + 0.05
    #x0 = -2*math.cos(i / 10. * math.pi * 2)
    #y0 = math.sin(j / 10. * math.pi * 2)

    source = CF(1.0) #CF(10.0*exp(-0.5 * (((x - x0) / o0) ** 2 + ((y - y0) / o0) ** 2)))
    coeff = CF(1.0)

    return fes, mesh, source, coeff


def save_as_hdf5(filename, mesh, train_data):
    vertices = [[p[0], p[1], p[2]] for p in mesh.ngmesh.Points()]
    triangles = [(t[0][0:3] - 1).tolist() for t in np.array(mesh.ngmesh.Elements2D())]

    # In NumPy-Arrays umwandeln
    vertices_np = np.array(vertices)       # Shape: (n_vertices, 3)
    triangles_np = np.array(triangles)     # Shape: (n_triangles, 3)

    # Transponieren:
    vertices_T = vertices_np.T             # Shape: (3, n_vertices)
    triangles_T = triangles_np.T           # Shape: (3, n_triangles)

    # Optional: wieder in Listen umwandeln (falls nötig)
    vertices_T_list = vertices_T.tolist()
    triangles_T_list = triangles_T.tolist()

    print(f'vertices       : {len(vertices_T_list)}')
    print(f'triangles      : {len(triangles_T_list)}')

    write_pde_dataset_to_hdf5(filename, train_data, vertices_T_list, triangles_T_list)
