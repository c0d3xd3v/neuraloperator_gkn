"""Create an HDF5 file in memory and retrieve the raw bytes

This could be used, for instance, in a server producing small HDF5
files on demand.
"""
import h5py
import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_pde_dataset(pde_file_name):

    loaded_train_data = []

    with h5py.File(pde_file_name, 'r') as h5file:
        for i, key in enumerate(h5file.keys()):
            group = h5file[key]
            x = torch.tensor(group['x'][:])
            edge_index = torch.tensor(group['edge_index'][:])
            edge_attr = torch.tensor(group['edge_attr'][:])
            y = torch.tensor(group['y'][:])
            coeff = torch.tensor(group['coeff'][:])

            data = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y=y, coeff=coeff)
            loaded_train_data.append(data)
            print(f'{i} {data}')

    return loaded_train_data


if __name__=="__main__":

    batch_size = 8
    train_data = load_pde_dataset('train_data.h5')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
