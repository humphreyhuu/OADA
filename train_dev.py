"""
Domain Adaptation Training Script for OADA Model

This script trains the OADA (Domain-Adaptive EHR Prediction) model using
a 3-stage curriculum learning approach with unlabeled target domain data.

Key features:
- Samples 500 patients from validation set as unlabeled target domain
- Uses 3-stage training: invariant learning → sparse reconstruction → domain supervision
- Combines source (labeled) and target (unlabeled) data in each batch
- Evaluates on remaining validation set and full test set

Reference: OADA.md, plan.md
"""

import os
import random
import time
import _pickle as pickle

import torch
import numpy as np

from models.Transformer import Transformer
from models.models import OADAModel, OADAWrapper
from utils import EHRDataset, format_time, MultiStepLRScheduler, historical_hot
from metrics import evaluate_codes, evaluate_hf


class TargetDomainDataset(EHRDataset):
    """
    Custom dataset for unlabeled target domain data.

    Inherits from EHRDataset but samples specific indices and ignores labels.
    """

    def __init__(self, path, indices, label, batch_size, shuffle, device):
        """
        Args:
            path: Path to data directory
            indices: List of patient indices to include
            label: Task type ('m' or 'h')
            batch_size: Batch size
            shuffle: Whether to shuffle data
            device: Torch device
        """
        super().__init__(path, label, batch_size, shuffle, device)

        # Filter to selected indices
        self.code_x = self.code_x[indices]
        self.visit_lens = self.visit_lens[indices]

        # Handle divided and neighbors (only if they exist)
        if hasattr(self, 'divided'):
            self.divided = self.divided[indices]
        if hasattr(self, 'neighbors'):
            self.neighbors = self.neighbors[indices]

        # Remove labels (convert to unlabeled data)
        # Keep labels shape but fill with zeros for compatibility
        self.labels = self.labels[indices]
        if self.label_type == 'm':
            self.labels = np.zeros_like(self.labels)
        else:  # 'h'
            self.labels = np.zeros(len(indices), dtype=np.float32)

        # Update size
        self.num_samples = len(indices)
        self.indices = np.arange(self.num_samples)

        # Regenerate shuffled indices
        if self.shuffle:
            np.random.shuffle(self.indices)


def sample_target_domain(valid_path, num_samples, seed):
    """
    Sample patients from validation set to use as unlabeled target domain.

    Args:
        valid_path: Path to validation data directory
        num_samples: Number of patients to sample
        seed: Random seed for reproducibility

    Returns:
        List of patient indices
    """
    # Load validation data to get number of patients
    code_x_path = os.path.join(valid_path, 'code_x.npz')
    with np.load(code_x_path) as data:
        num_patients = data['shape'][0]

    # Set seed and sample
    random.seed(seed)
    np.random.seed(seed)

    indices = list(range(num_patients))
    random.shuffle(indices)
    target_indices = indices[:num_samples]

    print(f'Sampled {len(target_indices)} patients from {num_patients} validation patients')
    print(f'First 10 target indices: {target_indices[:10]}')

    return target_indices


def get_stage(epoch, stage1_epochs=30, stage2_epochs=60):
    """
    Determine training stage based on current epoch.

    Args:
        epoch: Current epoch (0-indexed)
        stage1_epochs: Number of epochs for stage 1
        stage2_epochs: Number of epochs before stage 3

    Returns:
        Stage number (1, 2, or 3)
    """
    if epoch < stage1_epochs:
        return 1
    elif epoch < stage2_epochs:
        return 2
    else:
        return 3


if __name__ == '__main__':
    # Configuration
    seed = 6669
    dataset = 'eicu'  # 'mimic3', 'mimic4', or 'eicu'
    task = 'h'  # 'm' for multi-label code prediction, 'h' for heart failure
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # OADA hyperparameters
    embedding_dim = 128
    latent_dim = 256  # Overcomplete: k = 2d
    heads = 4
    dropout = 0.1
    num_layers = 3
    batch_size = 32
    epochs = 100

    # Loss weights
    lambda_mmd = 0.1
    lambda_rec = 1.0
    lambda_sp = 0.01
    lambda_dom = 0.5

    # Stage transitions
    stage1_epochs = 30
    stage2_epochs = 60

    # Target domain sampling
    num_target_samples = 500

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print('=' * 80)
    print('OADA Domain-Adaptive EHR Prediction Training')
    print('=' * 80)
    print(f'Dataset: {dataset}')
    print(f'Task: {task}')
    print(f'Device: {device}')
    print(f'Seed: {seed}')
    print('-' * 80)
    print('Hyperparameters:')
    print(f'  Embedding dim: {embedding_dim}, Latent dim: {latent_dim}')
    print(f'  Batch size: {batch_size}, Epochs: {epochs}')
    print(f'  Lambda MMD: {lambda_mmd}, Lambda Rec: {lambda_rec}')
    print(f'  Lambda Sparsity: {lambda_sp}, Lambda Domain: {lambda_dom}')
    print(f'  Stage transitions: 1→2 at epoch {stage1_epochs}, 2→3 at epoch {stage2_epochs}')
    print('=' * 80)

    # Load encoded data to get code_num
    dataset_path = os.path.join('data', dataset)
    encoded_path = os.path.join(dataset_path, 'encoded')
    code_map = pickle.load(open(os.path.join(encoded_path, 'code_map.pkl'), 'rb'))
    code_num = len(code_map)

    print(f'\nTotal number of codes: {code_num}')

    # Load data paths
    standard_path = os.path.join(dataset_path, 'standard')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')

    # Sample target domain from validation set
    print('\nSampling target domain from validation set...')
    target_indices = sample_target_domain(valid_path, num_target_samples, seed)

    # Load source domain (training data - labeled)
    print('\nLoading source domain (training data)...')
    source_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print(f'  Source domain: {len(source_data.code_x)} patients, {len(source_data)} batches')

    # Load target domain (sampled validation data - unlabeled)
    print('\nLoading target domain (sampled validation data)...')
    target_data = TargetDomainDataset(
        path=valid_path,
        indices=target_indices,
        label=task,
        batch_size=batch_size,
        shuffle=True,
        device=device
    )
    print(f'  Target domain: {len(target_data.code_x)} patients, {len(target_data)} batches')

    # Load validation data (remaining validation patients for model selection)
    print('\nLoading validation data (for model selection)...')
    # Create indices for remaining validation patients
    code_x_path = os.path.join(valid_path, 'code_x.npz')
    with np.load(code_x_path) as data:
        num_valid_patients = data['shape'][0]
    all_valid_indices = set(range(num_valid_patients))
    target_indices_set = set(target_indices)
    remaining_valid_indices = list(all_valid_indices - target_indices_set)

    # Load validation data with labels (for model selection)
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    # Filter to remaining indices
    valid_data.code_x = valid_data.code_x[remaining_valid_indices]
    valid_data.visit_lens = valid_data.visit_lens[remaining_valid_indices]
    valid_data.labels = valid_data.labels[remaining_valid_indices]

    # Handle divided and neighbors (only if they exist)
    if hasattr(valid_data, 'divided'):
        valid_data.divided = valid_data.divided[remaining_valid_indices]
    if hasattr(valid_data, 'neighbors'):
        valid_data.neighbors = valid_data.neighbors[remaining_valid_indices]

    valid_data.num_samples = len(remaining_valid_indices)
    valid_data.indices = np.arange(valid_data.num_samples)
    print(f'  Validation: {len(valid_data.code_x)} patients, {len(valid_data)} batches')

    # Load test data
    print('\nLoading test data...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print(f'  Test: {len(test_data.code_x)} patients, {len(test_data)} batches')

    # Get historical hot encoding for evaluation (if needed)
    if task == 'm':
        valid_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)
        test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)
    else:
        valid_historical = None
        test_historical = None

    # Task configuration
    task_conf = {
        'm': {
            'dropout': 0.1,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.001,
                'milestones': [20, 40, 60],
                'lrs': [0.0001, 0.00001, 0.000001]
            }
        },
        'h': {
            'dropout': 0.1,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.001,
                'milestones': [10, 20, 40],
                'lrs': [0.0001, 0.00001, 0.000001]
            }
        }
    }

    output_size = task_conf[task]['output_size']
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']

    # Initialize Transformer backbone
    feature_keys = ['diagnosis']
    code_nums = {'diagnosis': code_num}

    print('\nInitializing Transformer backbone...')
    backbone = Transformer(
        feature_keys=feature_keys,
        code_nums=code_nums,
        embedding_dim=embedding_dim,
        output_size=embedding_dim,  # Output embeddings, not task predictions
        activation=False,  # No activation for embeddings
        heads=heads,
        dropout=dropout_rate,
        num_layers=num_layers
    ).to(device)

    # Initialize OADA model
    print('Initializing OADA model...')
    oada_model = OADAModel(
        backbone=backbone,
        embedding_dim=len(feature_keys) * embedding_dim,
        latent_dim=latent_dim,
        output_size=output_size,
        lambda_mmd=lambda_mmd,
        lambda_rec=lambda_rec,
        lambda_sp=lambda_sp,
        lambda_dom=lambda_dom
    ).to(device)

    # Wrap model for compatibility with EHRDataset interface
    model = OADAWrapper(oada_model, device).to(device)

    # Count parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {pytorch_total_params:,}')

    # Loss function (for supervised task loss)
    loss_fn = torch.nn.BCEWithLogitsLoss() if task == 'h' else torch.nn.BCELoss()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=task_conf[task]['lr']['init_lr'])
    scheduler = MultiStepLRScheduler(
        optimizer,
        epochs,
        task_conf[task]['lr']['init_lr'],
        task_conf[task]['lr']['milestones'],
        task_conf[task]['lr']['lrs']
    )

    # Create directory for saving parameters
    param_path = os.path.join('data', 'params', dataset, 'oada', task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    print(f'\nModel parameters will be saved to: {param_path}')
    print('=' * 80)

    # Training loop
    best_valid_f1 = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        stage = get_stage(epoch, stage1_epochs, stage2_epochs)
        print(f'\nEpoch {epoch + 1} / {epochs} (Stage {stage}):')

        model.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_mmd_loss = 0.0
        total_sae_loss = 0.0
        total_dom_loss = 0.0
        total_num = 0

        # Determine number of steps (max of source and target batches)
        max_steps = max(len(source_data), len(target_data))

        st = time.time()
        scheduler.step()

        for step in range(max_steps):
            optimizer.zero_grad()

            # Get source batch (labeled)
            source_idx = step % len(source_data)
            code_x_s, visit_lens_s, divided_s, y_s, neighbors_s = source_data[source_idx]
            batch_size_s = len(code_x_s)
            domain_labels_s = torch.zeros(batch_size_s, device=device)  # 0 = source

            # Get target batch (unlabeled)
            target_idx = step % len(target_data)
            code_x_t, visit_lens_t, divided_t, y_t, neighbors_t = target_data[target_idx]
            batch_size_t = len(code_x_t)
            domain_labels_t = torch.ones(batch_size_t, device=device)  # 1 = target

            # Forward pass - source (with supervision)
            task_output_s, losses_s = model(code_x_s, domain_labels_s, stage,
                                            divided_s, neighbors_s, visit_lens_s)

            # Task loss (supervised, source only)
            if task == 'h':
                task_output_s = task_output_s.squeeze()
            task_loss = loss_fn(task_output_s, y_s)

            # Forward pass - target (no supervision)
            _, losses_t = model(code_x_t, domain_labels_t, stage,
                               divided_t, neighbors_t, visit_lens_t)

            # Total loss
            loss = (
                task_loss +  # Supervised task loss (source only)
                losses_s['total'] +  # Source domain losses (MMD + SAE + domain)
                losses_t['total']  # Target domain losses (MMD + SAE + domain)
            )

            loss.backward()
            optimizer.step()

            # Accumulate losses
            batch_total = batch_size_s + batch_size_t
            total_loss += loss.item() * batch_total
            total_task_loss += task_loss.item() * batch_size_s
            total_mmd_loss += (losses_s.get('mmd', torch.tensor(0.0)).item() +
                              losses_t.get('mmd', torch.tensor(0.0)).item()) * batch_total
            total_sae_loss += (losses_s.get('sae', torch.tensor(0.0)).item() +
                              losses_t.get('sae', torch.tensor(0.0)).item()) * batch_total
            total_dom_loss += (losses_s.get('domain', torch.tensor(0.0)).item() +
                              losses_t.get('domain', torch.tensor(0.0)).item()) * batch_total
            total_num += batch_total

            # Progress display
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (max_steps - step - 1))
            print(f'\r    Step {step + 1} / {max_steps}, remaining: {remaining_time}, '
                  f'loss: {total_loss / total_num:.4f} '
                  f'(task: {total_task_loss / total_num:.4f}, '
                  f'mmd: {total_mmd_loss / total_num:.4f}, '
                  f'sae: {total_sae_loss / total_num:.4f}, '
                  f'dom: {total_dom_loss / total_num:.4f})',
                  end='')

        source_data.on_epoch_end()
        target_data.on_epoch_end()

        et = time.time()
        time_cost = format_time(et - st)
        print(f'\r    Step {max_steps} / {max_steps}, time: {time_cost}, '
              f'loss: {total_loss / total_num:.4f} '
              f'(task: {total_task_loss / total_num:.4f}, '
              f'mmd: {total_mmd_loss / total_num:.4f}, '
              f'sae: {total_sae_loss / total_num:.4f}, '
              f'dom: {total_dom_loss / total_num:.4f})')

        # Validation
        print('  Validating...')
        # For validation, we need to adapt the model wrapper to handle single domain
        model.eval()

        # Simple evaluation wrapper
        class EvalWrapper(torch.nn.Module):
            def __init__(self, oada_wrapper, current_stage):
                super().__init__()
                self.wrapper = oada_wrapper
                self.current_stage = current_stage

            def forward(self, code_x, divided, neighbors, visit_lens):
                # For evaluation, use current training stage
                batch_size = len(code_x)
                domain_labels = torch.zeros(batch_size, device=code_x.device)
                task_output, _ = self.wrapper(code_x, domain_labels, self.current_stage,
                                             divided=divided, neighbors=neighbors,
                                             visit_lens=visit_lens)
                return task_output

        eval_model = EvalWrapper(model, stage)

        if task == 'm':
            valid_loss, valid_f1 = evaluate_fn(eval_model, valid_data, loss_fn, output_size, valid_historical)
        else:
            valid_loss, valid_f1 = evaluate_fn(eval_model, valid_data, loss_fn, output_size, None)

        print(f'  Validation - Loss: {valid_loss:.4f}, F1: {valid_f1:.4f}')

        # Save model
        torch.save(model.state_dict(), os.path.join(param_path, f'{epoch}.pt'))

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(param_path, 'best.pt'))
            print(f'  *** New best model saved! (F1: {valid_f1:.4f}) ***')

    print('\n' + '=' * 80)
    print(f'Training completed!')
    print(f'Best validation F1: {best_valid_f1:.4f} at epoch {best_epoch + 1}')
    print('=' * 80)

    # Test with best model
    print('\nTesting with best model...')
    model.load_state_dict(torch.load(os.path.join(param_path, 'best.pt')))
    model.eval()

    # Use stage 3 for final testing (all components trained)
    class EvalWrapper(torch.nn.Module):
        def __init__(self, oada_wrapper, current_stage):
            super().__init__()
            self.wrapper = oada_wrapper
            self.current_stage = current_stage

        def forward(self, code_x, divided, neighbors, visit_lens):
            # For evaluation, use current training stage
            batch_size = len(code_x)
            domain_labels = torch.zeros(batch_size, device=code_x.device)
            task_output, _ = self.wrapper(code_x, domain_labels, self.current_stage,
                                         divided=divided, neighbors=neighbors,
                                         visit_lens=visit_lens)
            return task_output

    eval_model = EvalWrapper(model, current_stage=3)

    if task == 'm':
        test_loss, test_f1 = evaluate_fn(eval_model, test_data, loss_fn, output_size, test_historical)
    else:
        test_loss, test_f1 = evaluate_fn(eval_model, test_data, loss_fn, output_size, None)

    print(f'\nTest Results:')
    print(f'  Loss: {test_loss:.4f}')
    print(f'  F1 Score: {test_f1:.4f}')
    print('=' * 80)

    print('\nTraining complete! Model saved to:', param_path)
