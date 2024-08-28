from ngsolve import *
from netgen.geom2d import SplineGeometry


def trianglesToEdgeList(mesh):
    tris = [(t[0][0:3] - 1).tolist() for t in np.array(mesh.ngmesh.Elements2D())]
    edges = []*0
    for tri in tris:
        edges.append([int(tri[0]), int(tri[1])])
        edges.append([int(tri[1]), int(tri[2])])
        edges.append([int(tri[2]), int(tri[0])])
    edges = np.transpose(np.array(edges))
    return edges


def generate_unit_rectangle():
    geo = SplineGeometry()
    geo.AddRectangle(p1=(-1,-1),
                     p2=(1,1),
                     bc="rectangle",
                     leftdomain=1,
                     rightdomain=0)
    return Mesh(geo.GenerateMesh(maxh=0.1))