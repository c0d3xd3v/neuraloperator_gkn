import os
import torch
from gkn.KernelNN import KernelNN


def save_check_point(model, width, ker_width, depth, edge_features, node_features,
                     optimizer, epochn, learning_rate,
                     scheduler_step, scheduler_gamma, checkpoint_dir):
    f_path = checkpoint_dir
    checkpoint = {
        'state_dict': model.state_dict(),
        'width' : width,
        'ker_width' : ker_width,
        'depth' : depth,
        'edge_features' : edge_features,
        'node_features' : node_features,
        'optimizer': optimizer.state_dict(),
        'epoch' : epochn,
        'learning_rate' : learning_rate,
        'scheduler_step' : scheduler_step,
        'scheduler_gamma' : scheduler_gamma
    }
    torch.save(checkpoint, f_path)


def load_check_point(checkpoint_fpath):
    epoch = 0
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
        print(f'load from file : {checkpoint_fpath}')
    else:
        width = 64
        ker_width = 64
        depth = 2
        edge_features = 8
        node_features = 7

        learning_rate = 0.00005
        scheduler_step = 50
        scheduler_gamma = 0.5

    model = KernelNN(width, ker_width, depth, edge_features, in_width=node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    if has_statedict:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, scheduler, epoch, learning_rate, scheduler_step, scheduler_gamma

