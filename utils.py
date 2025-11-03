import os
import numpy as np
import torch
from torch.utils.data import Dataset


def load_sparse(path):
    """Load sparse matrix from .npz file"""
    data = np.load(path)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat


class EHRDataset(Dataset):
    """Dataset for EHR data with disease codes"""
    def __init__(self, path, label='m', batch_size=32, shuffle=True, device='cpu'):
        """
        Args:
            path: path to dataset directory
            label: 'm' for multi-label code prediction, 'h' for heart failure
            batch_size: batch size
            shuffle: whether to shuffle data
            device: torch device
        """
        self.path = path
        self.label_type = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        # Load data
        self.code_x = load_sparse(os.path.join(path, 'code_x.npz'))
        self.visit_lens = np.load(os.path.join(path, 'visit_lens.npz'))['lens']

        if label == 'm':
            self.labels = load_sparse(os.path.join(path, 'code_y.npz'))
        elif label == 'h':
            self.labels = np.load(os.path.join(path, 'hf_y.npz'))['hf_y']
        else:
            raise ValueError(f'Invalid label type: {label}')

        # Check for divided and neighbors (for models that need graph structure)
        self.has_divided = os.path.exists(os.path.join(path, 'divided.npz'))
        self.has_neighbors = os.path.exists(os.path.join(path, 'neighbors.npz'))

        if self.has_divided:
            self.divided = load_sparse(os.path.join(path, 'divided.npz'))
        if self.has_neighbors:
            self.neighbors = load_sparse(os.path.join(path, 'neighbors.npz'))

        self.num_samples = len(self.code_x)
        self.indices = np.arange(self.num_samples)

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Return number of batches"""
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        """Get batch by index"""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]

        # Get batch data
        code_x = self.code_x[batch_indices]
        visit_lens = self.visit_lens[batch_indices]
        labels = self.labels[batch_indices]

        # Convert to tensors
        code_x = torch.FloatTensor(code_x).to(self.device)
        visit_lens = torch.LongTensor(visit_lens).to(self.device)

        if self.label_type == 'm':
            labels = torch.FloatTensor(labels).to(self.device)
        else:
            labels = torch.FloatTensor(labels).to(self.device)

        # Return with divided and neighbors if available
        if self.has_divided and self.has_neighbors:
            divided = self.divided[batch_indices]
            neighbors = self.neighbors[batch_indices]
            divided = torch.FloatTensor(divided).to(self.device)
            neighbors = torch.LongTensor(neighbors).to(self.device)
            return code_x, visit_lens, divided, labels, neighbors
        else:
            # Return dummy values for compatibility
            divided = torch.zeros_like(code_x).to(self.device)
            neighbors = torch.zeros((code_x.shape[0], code_x.shape[1], 1), dtype=torch.long).to(self.device)
            return code_x, visit_lens, divided, labels, neighbors

    def on_epoch_end(self):
        """Shuffle data at end of epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def size(self):
        """Return total number of samples"""
        return self.num_samples

    def label(self):
        """Return all labels"""
        return self.labels


def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    elif minutes > 0:
        return f'{minutes}m {secs}s'
    else:
        return f'{secs}s'


class MultiStepLRScheduler:
    """Multi-step learning rate scheduler"""
    def __init__(self, optimizer, epochs, init_lr, milestones, lrs):
        """
        Args:
            optimizer: torch optimizer
            epochs: total epochs
            init_lr: initial learning rate
            milestones: list of epoch milestones
            lrs: list of learning rates for each milestone
        """
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_lr = init_lr
        self.milestones = milestones
        self.lrs = lrs
        self.current_epoch = 0

    def step(self):
        """Update learning rate"""
        if self.current_epoch in self.milestones:
            idx = self.milestones.index(self.current_epoch)
            lr = self.lrs[idx]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1


def historical_hot(code_x, code_num, lens):
    """Get historical hot encoding from last visit"""
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        if l > 0:
            result[i] = x[l - 1]
    return result
