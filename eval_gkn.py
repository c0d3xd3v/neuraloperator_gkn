from gkn.gkn import GKN
from hlp.hdf5 import write_pde_dataset_to_hdf5
from hlp.netgen_utilities import generate_rectangle_geometry


checkpoint_path = 'data/checkpoint.pt'
unit_rect_sampling = 0.025
r = 1.25*unit_rect_sampling
fes_order = 1

fes, mesh, source, coeff = generate_rectangle_geometry(fes_order, unit_rect_sampling)

gkn = GKN(checkpoint_path)
gkn.solve(fes, mesh, source, coeff, r)
gkn.result_as_ngsolve_grid_function()
gkn.save_result_as_hdf5("out_data.h5")
