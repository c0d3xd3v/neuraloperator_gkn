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

        model = KernelNN(width, ker_width, depth, edge_features, in_width=node_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1

    return model, optimizer, scheduler, epoch

