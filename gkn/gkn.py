import numpy as np
from ngsolve import GridFunction

from hlp.nn import load_check_point
from hlp.netgen_utilities import save_as_hdf5
from hlp.netgen_utilities import sample_from_ngsolve_mesh


class GKN():
    def __init__(self, model_path):
        print("Graph Kernel Network")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = None
        self.learning_rate = None
        self.scheduler_step = None
        self.scheduler_gamma = None
        self.normalizer = None
        self.target_normalizer = None

        self.out_np = None
        self.fes = None
        self.data = None

        (self.model, 
        self.optimizer, 
        self.scheduler, 
        self.epoch, 
        self.learning_rate, 
        self.scheduler_step, 
        self.scheduler_gamma, 
        self.normalizer, 
        self.target_normalizer) = load_check_point(model_path)


    def solve(self, fes, mesh, source, coeff, r=0.2):
        data_test = sample_from_ngsolve_mesh(fes, mesh, source, coeff, r)
        
        self.local_batch = data_test.clone()
        self.data = data_test.clone()
        data_tensor = self.local_batch.x

        self.local_batch.x = self.normalizer.encode(data_tensor)
        out = self.model(self.local_batch)
        out = out.view(-1, 1).detach().cpu()
        
        self.fes = fes
        self.out_np = self.normalizer.decode(out).numpy()

        self.data.y = out.reshape(-1)


    def result_as_ngsolve_grid_function(self):
        if self.fes != None:
            gfu = GridFunction(self.fes)
            for k in range(len(gfu.vec)):
                gfu.vec.data[k] = self.out_np[k][0]
            return gfu


    def save_result_as_hdf5(self, filename):
        if self.fes != None:
            save_as_hdf5(filename, self.fes.mesh, [self.data])
