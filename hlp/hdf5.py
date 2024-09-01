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

    return loaded_train_data


def write_pde_dataset_to_hdf5(pde_file_name, train_data):

    with h5py.File(pde_file_name, 'w') as h5file:
        for i, data in enumerate(train_data):
            group = h5file.create_group(f'data_{i}')
            group.create_dataset('x', data=data.x.numpy())
            group.create_dataset('edge_index', data=data.edge_index.numpy())
            group.create_dataset('edge_attr', data=data.edge_attr.numpy())
            group.create_dataset('y', data=data.y.numpy())
            group.create_dataset('coeff', data=data.coeff.numpy())
    print("Die Data-Objekte wurden erfolgreich in der HDF5-Datei gespeichert.")


if __name__=="__main__":

    filename = 'data/train_data.h5'
    batch_size = 8
    train_data = load_pde_dataset(filename)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    write_pde_dataset_to_hdf5(filename,train_data)
