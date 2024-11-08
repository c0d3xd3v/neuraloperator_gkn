from ngsolve import *
from netgen.occ import *
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
