import os
import random
import time
import _pickle as pickle

import torch
import numpy as np

from models.Transformer import Transformer
from utils import EHRDataset, format_time, MultiStepLRScheduler, historical_hot
from metrics import evaluate_codes, evaluate_hf


class TransformerAdapter(torch.nn.Module):
    """Adapter to convert input format for Transformer model"""
    def __init__(self, transformer_model, device):
        super(TransformerAdapter, self).__init__()
        self.transformer = transformer_model
        self.device = device

    def prepare_input(self, code_x):
        """Convert binary code matrix to transformer input format"""
        batch_size_actual = code_x.shape[0]
        max_visits = code_x.shape[1]

        # Convert binary matrix to indices format
        code_x_indices = []
        for b in range(batch_size_actual):
            visit_codes = []
            for v in range(max_visits):
                codes_in_visit = torch.nonzero(code_x[b, v], as_tuple=True)[0]
                visit_codes.append(codes_in_visit)
            code_x_indices.append(visit_codes)

        # Pad to same length
        max_codes_per_visit = max([max([len(v) for v in patient]) if len(patient) > 0 else 1
                                   for patient in code_x_indices])
        max_codes_per_visit = max(max_codes_per_visit, 1)

        code_x_padded = torch.zeros((batch_size_actual, max_visits, max_codes_per_visit), dtype=torch.long).to(self.device)
        for b in range(batch_size_actual):
            for v in range(max_visits):
                codes = code_x_indices[b][v]
                if len(codes) > 0:
                    code_x_padded[b, v, :len(codes)] = codes + 1  # +1 for padding_idx

        return {'diagnosis': code_x_padded}

    def forward(self, code_x, divided, neighbors, visit_lens):
        """
        Adapt the interface to match train_demo.py model
        Args:
            code_x: binary code matrix [batch, max_visits, code_num]
            divided: not used for Transformer
            neighbors: not used for Transformer
            visit_lens: not used for Transformer
        Returns:
            output tensor
        """
        # Convert code_x to Transformer input format
        code_x_dict = self.prepare_input(code_x)

        # Forward pass (activation is inside Transformer now)
        output = self.transformer(code_x_dict)

        return output


if __name__ == '__main__':
    seed = 6669
    dataset = 'mimic4'  # 'mimic3', 'mimic4', or 'eicu'
    task = 'h'  # 'm' for multi-label code prediction, 'h' for heart failure
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    # Transformer hyperparameters
    embedding_dim = 128
    heads = 4
    dropout = 0.1
    num_layers = 3
    batch_size = 64
    epochs = 100

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load encoded data to get code_num
    dataset_path = os.path.join('data', dataset)
    encoded_path = os.path.join(dataset_path, 'encoded')
    code_map = pickle.load(open(os.path.join(encoded_path, 'code_map.pkl'), 'rb'))
    code_num = len(code_map)

    print(f'Dataset: {dataset}')
    print(f'Total number of codes: {code_num}')

    # Load data
    standard_path = os.path.join(dataset_path, 'standard')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')

    print('Loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('Loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('Loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    # Get historical hot encoding for evaluation
    valid_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

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

    # Initialize Transformer model
    # Note: Transformer expects feature_keys and code_nums
    # We'll use 'diagnosis' as the single feature key
    feature_keys = ['diagnosis']
    code_nums = {'diagnosis': code_num}

    print('Initializing Transformer model ...')
    transformer = Transformer(
        feature_keys=feature_keys,
        code_nums=code_nums,
        embedding_dim=embedding_dim,
        output_size=output_size,
        activation=True,  # Use activation inside Transformer
        heads=heads,
        dropout=dropout_rate,
        num_layers=num_layers
    ).to(device)

    # Wrap the Transformer to match the metrics interface
    model = TransformerAdapter(transformer, device).to(device)

    # Loss function
    loss_fn = torch.nn.BCELoss()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=task_conf[task]['lr']['init_lr'])
    scheduler = MultiStepLRScheduler(
        optimizer,
        epochs,
        task_conf[task]['lr']['init_lr'],
        task_conf[task]['lr']['milestones'],
        task_conf[task]['lr']['lrs']
    )

    # Count parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {pytorch_total_params}')

    # Create directory for saving parameters
    param_path = os.path.join('data', 'params', dataset, 'transformer', task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    # Training loop
    best_valid_f1 = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1} / {epochs}:')
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        scheduler.step()

        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, visit_lens, divided, y, neighbors = train_data[step]

            # Forward pass through wrapper model
            output = model(code_x, divided, neighbors, visit_lens)

            if task == 'h':
                output = output.squeeze()

            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)

            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print(f'\r    Step {step + 1} / {steps}, remaining time: {remaining_time}, loss: {total_loss / total_num:.4f}', end='')

        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print(f'\r    Step {steps} / {steps}, time cost: {time_cost}, loss: {total_loss / total_num:.4f}')

        # Evaluation
        print('Validating ...')
        if task == 'm':
            valid_loss, valid_f1 = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical)
        else:
            valid_loss, valid_f1 = evaluate_fn(model, valid_data, loss_fn, output_size, None)

        # Save model
        torch.save(model.state_dict(), os.path.join(param_path, f'{epoch}.pt'))

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(param_path, 'best.pt'))

    print(f'\nBest validation F1: {best_valid_f1:.4f} at epoch {best_epoch + 1}')

    # Test with best model
    print('\nTesting with best model ...')
    model.load_state_dict(torch.load(os.path.join(param_path, 'best.pt')))
    if task == 'm':
        test_loss, test_f1 = evaluate_fn(model, test_data, loss_fn, output_size, test_historical)
    else:
        test_loss, test_f1 = evaluate_fn(model, test_data, loss_fn, output_size, None)
    print(f'Test F1: {test_f1:.4f}')
