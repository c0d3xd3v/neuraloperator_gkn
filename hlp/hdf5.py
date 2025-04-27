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

def write_pde_dataset_to_hdf5(pde_file_name, train_data, vertices, triangles):

    with h5py.File(pde_file_name, 'w') as h5file:
        for i, data in enumerate(train_data):
            group = h5file.create_group(f'data_{i}')
            group.create_dataset('x', data=data.x.numpy())
            group.create_dataset('edge_index', data=data.edge_index.numpy())
            group.create_dataset('edge_attr', data=data.edge_attr.numpy())
            group.create_dataset('y', data=data.y.numpy())
            group.create_dataset('coeff', data=data.coeff.numpy())
        
                # Speichern des Dreiecksnetzes
        mesh_group = h5file.create_group('triangle_mesh')
        mesh_group.create_dataset('vertices', data=vertices)
        mesh_group.create_dataset('triangles', data=triangles)

    print("Die Data-Objekte wurden erfolgreich in der HDF5-Datei gespeichert.")


def read_pde_dataset_from_hdf5(pde_file_name):
    train_data = []
    vertices, triangles = None, None
    
    with h5py.File(pde_file_name, 'r') as h5file:
        # Trainingsdaten lesen
        for key in h5file.keys():
            if key.startswith("data_"):
                group = h5file[key]
                data = {
                    'x': np.array(group['x']),
                    'edge_index': np.array(group['edge_index']),
                    'edge_attr': np.array(group['edge_attr']),
                    'y': np.array(group['y']),
                    'coeff': np.array(group['coeff'])
                }
                train_data.append(data)
        
        # Dreiecksnetz lesen
        if 'triangle_mesh' in h5file:
            mesh_group = h5file['triangle_mesh']
            vertices = np.array(mesh_group['vertices'])
            triangles = np.array(mesh_group['triangles'])
    
    print("Die Data-Objekte und das Dreiecksnetz wurden erfolgreich aus der HDF5-Datei gelesen.")
    return train_data, vertices, triangles

def read_pde_dataset_from_hdf5_torch(pde_file_name):
    train_data = []
    vertices, triangles = None, None
    
    with h5py.File(pde_file_name, 'r') as h5file:
        # Trainingsdaten lesen
        for key in h5file.keys():
            if key.startswith("data_"):
                group = h5file[key]

                x = torch.tensor(group['x'][:])
                edge_index = torch.tensor(group['edge_index'][:])
                edge_attr = torch.tensor(group['edge_attr'][:])
                y = torch.tensor(group['y'][:])
                coeff = torch.tensor(group['coeff'][:])

                data = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y=y, coeff=coeff)
                train_data.append(data)
        
        # Dreiecksnetz lesen
        if 'triangle_mesh' in h5file:
            mesh_group = h5file['triangle_mesh']
            vertices = np.array(mesh_group['vertices'])
            triangles = np.array(mesh_group['triangles'])
    
    print("Die Data-Objekte und das Dreiecksnetz wurden erfolgreich aus der HDF5-Datei gelesen.")
    return train_data, vertices, triangles

if __name__=="__main__":

    filename = 'out_data.h5'
    batch_size = 8
    train_data, vertices, triangles = read_pde_dataset_from_hdf5(filename)

    print(f'nodes     : {vertices.shape}')
    print(f'triangles : {triangles.shape}')
    print(f'datasets  : {len(train_data)}')
    print(f'keys      : {train_data[0].keys()}')
