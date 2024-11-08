import os
import torch
from gkn.KernelNN import KernelNN

from gkn.gaussian_batch_normalizer import GaussianBatchNormalizer


def save_check_point(model, 
                     optimizer, 
                     epochn, 
                     learning_rate,
                     scheduler_step, 
                     scheduler_gamma, 
                     checkpoint_dir, 
                     normalizer, 
                     target_normalizer):
    f_path = checkpoint_dir
    checkpoint = {
        'state_dict': model.state_dict(),
        'width' : model.width,
        'ker_width' : model.ker_width,
        'depth' : model.depth,
        'edge_features' : model.edge_features,
        'node_features' : model.node_features,
        'optimizer': optimizer.state_dict(),
        'epoch' : epochn,
        'learning_rate' : learning_rate,
        'scheduler_step' : scheduler_step,
        'scheduler_gamma' : scheduler_gamma,
        'normalizer_mean': normalizer.mean,   # Speichere den Mittelwert des Normalizers
        'normalizer_std': normalizer.var,
        'target_normalizer_mean' : target_normalizer.mean,
        'target_normalizer_std' : target_normalizer.var
    }
    torch.save(checkpoint, f_path)


def load_check_point(checkpoint_fpath):
    epoch = -1
    has_statedict = False
    if os.path.isfile(checkpoint_fpath):

        checkpoint = torch.load(checkpoint_fpath, weights_only=True)

        width = checkpoint['width']
        ker_width = checkpoint['ker_width']
        depth = checkpoint['depth']
        edge_features = checkpoint['edge_features']
        node_features = checkpoint['node_features']
        learning_rate = checkpoint['learning_rate']
        scheduler_step = checkpoint['scheduler_step']
        scheduler_gamma = checkpoint['scheduler_gamma']
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        has_statedict = True
        
        normalizer = GaussianBatchNormalizer()
        target_normalizer = GaussianBatchNormalizer()

        normalizer.mean = checkpoint['normalizer_mean']
        normalizer.var = checkpoint['normalizer_std']
        normalizer.n = epoch

        target_normalizer.mean = checkpoint['target_normalizer_mean']
        target_normalizer.var = checkpoint['target_normalizer_std']
        target_normalizer.n = epoch

        print(f'load from file : {checkpoint_fpath}')
    else:
        width = 64
        ker_width = 64
        depth = 6
        edge_features = 8
        node_features = 8

        learning_rate = 0.00005
        scheduler_step = 50
        scheduler_gamma = 0.5

        normalizer = GaussianBatchNormalizer()
        target_normalizer = GaussianBatchNormalizer()

    model = KernelNN(width, ker_width, depth, edge_features, in_width=node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    if has_statedict:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, scheduler, epoch + 1, learning_rate, scheduler_step, scheduler_gamma, normalizer, target_normalizer

