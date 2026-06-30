"""
Active learning training loop for cholec80 EndoNet.

Implements an iterative pool-based active learning cycle:

  Round 0 … N-1:
    1. Train EndoNet on the current labeled set.
    2. Evaluate on the held-out test set (phase accuracy & F1, tool accuracy).
    3. Query the most informative *n_query* frames using the chosen strategy.
    4. (Simulated annotation) mark queried frames as labeled.
    5. Repeat.

Results are written to JSON for later comparison across strategies.

Usage
-----
# from the active_learning/ directory
python active_train.py \
    --train_csv  /path/to/EndoNet/train_set_info.csv \
    --test_csv   /path/to/EndoNet/test_set_info.csv  \
    --strategy   entropy                              \
    --n_rounds   10                                   \
    --query_budget 200                                \
    --epochs_per_round 20                             \
    --output_dir ./results

Supported strategies (--strategy):
    random  least_confidence  margin  entropy
    mc_dropout  coreset  badge
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ── EndoNet models / config ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'EndoNet'))
import config as endonet_config
from model import AlexNet, EasyFCNet

# ── Active learning components ────────────────────────────────────────────
from al_dataset import CholecPoolDataset, CholecBaseDataset
from strategies import get_strategy
import al_config


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

_THRESHOLD = al_config.training_params['threshold']


def _build_models(device):
    """Instantiate and move fresh AlexNet + EasyFCNet to *device*."""
    model1 = AlexNet(init_weights=True, freezing=False, **endonet_config.net_params)
    model2 = EasyFCNet(init_weights=True, freezing=False, **endonet_config.net_params)
    model1.to(device)
    model2.to(device)
    return model1, model2


def _build_optimisers(model1, model2):
    lr_feat = al_config.training_params['learning_rate_feature']
    lr_cls  = al_config.training_params['learning_rate_classifier']
    mom     = al_config.training_params['momentum']
    opt1 = torch.optim.SGD(model1.parameters(), lr=lr_feat, momentum=mom)
    opt2 = torch.optim.SGD(model2.fc_phase.parameters(), lr=lr_cls)
    return opt1, opt2


# ──────────────────────────────────────────────────────────────────────────
# Train / eval
# ──────────────────────────────────────────────────────────────────────────

def train_one_epoch(model1, model2, loader, opt1, opt2, device):
    """One SGD epoch over *loader*.  Returns (loss, tool_acc, phase_acc)."""
    model1.train()
    model2.train()

    crit_tool  = nn.MultiLabelSoftMarginLoss()
    crit_phase = nn.CrossEntropyLoss()

    total_loss = 0.0
    tool_preds,  tool_trues  = [], []
    phase_preds, phase_trues = [], []

    for X, y_tool, y_phase in tqdm(loader, desc='  Train', leave=False):
        X           = X.to(device)
        y_tool      = y_tool.to(device)
        y_phase_idx = y_phase.argmax(dim=1).long().to(device)

        opt1.zero_grad()
        opt2.zero_grad()

        feats, tool_out = model1(X)
        combined        = torch.cat([feats, tool_out], dim=1)
        phase_out       = model2(combined)

        loss = crit_tool(tool_out, y_tool) + crit_phase(phase_out, y_phase_idx)
        loss.backward()
        opt1.step()
        opt2.step()

        total_loss += loss.item()

        tool_bin = (tool_out.detach().cpu().numpy() > _THRESHOLD).astype(float)
        tool_preds.extend(tool_bin.tolist())
        tool_trues.extend(y_tool.cpu().numpy().tolist())
        phase_preds.extend(phase_out.argmax(dim=1).cpu().numpy().tolist())
        phase_trues.extend(y_phase_idx.cpu().numpy().tolist())

    tool_acc  = accuracy_score(
        np.array(tool_trues).ravel(), np.array(tool_preds).ravel())
    phase_acc = accuracy_score(phase_trues, phase_preds)
    return total_loss / max(len(loader), 1), tool_acc, phase_acc


@torch.no_grad()
def evaluate(model1, model2, loader, device):
    """
    Evaluate on *loader*.

    Returns
    -------
    (loss, phase_acc, phase_f1_macro, tool_acc)
    """
    model1.eval()
    model2.eval()

    crit_tool  = nn.MultiLabelSoftMarginLoss()
    crit_phase = nn.CrossEntropyLoss()

    total_loss = 0.0
    tool_preds,  tool_trues  = [], []
    phase_preds, phase_trues = [], []

    for X, y_tool, y_phase in tqdm(loader, desc='  Eval', leave=False):
        X           = X.to(device)
        y_tool      = y_tool.to(device)
        y_phase_idx = y_phase.argmax(dim=1).long().to(device)

        feats, tool_out = model1(X)
        combined        = torch.cat([feats, tool_out], dim=1)
        phase_out       = model2(combined)

        loss = crit_tool(tool_out, y_tool) + crit_phase(phase_out, y_phase_idx)
        total_loss += loss.item()

        tool_bin = (tool_out.cpu().numpy() > _THRESHOLD).astype(float)
        tool_preds.extend(tool_bin.tolist())
        tool_trues.extend(y_tool.cpu().numpy().tolist())
        phase_preds.extend(phase_out.argmax(dim=1).cpu().numpy().tolist())
        phase_trues.extend(y_phase_idx.cpu().numpy().tolist())

    phase_acc = accuracy_score(phase_trues, phase_preds)
    phase_f1  = f1_score(phase_trues, phase_preds,
                         average='macro', zero_division=0)
    tool_acc  = accuracy_score(
        np.array(tool_trues).ravel(), np.array(tool_preds).ravel())
    return total_loss / max(len(loader), 1), phase_acc, phase_f1, tool_acc


# ──────────────────────────────────────────────────────────────────────────
# Main AL loop
# ──────────────────────────────────────────────────────────────────────────

def run_al(args) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Strategy: {args.strategy}')

    # ── Pool and test sets ────────────────────────────────────────────
    pool = CholecPoolDataset(
        csv_path=args.train_csv,
        initial_labeled_ratio=args.initial_ratio,
        seed=args.seed,
    )
    test_loader = DataLoader(
        CholecBaseDataset(args.test_csv),
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results = {'strategy': args.strategy, 'args': vars(args), 'rounds': []}

    for al_round in range(args.n_rounds):
        print(f'\n{"="*62}')
        print(f'  AL Round {al_round + 1}/{args.n_rounds}  '
              f'| strategy={args.strategy}  '
              f'| labeled={pool.n_labeled}/{pool.n_total}')
        print('='*62)

        # ── (Re)train model on current labeled set ────────────────────
        model1, model2 = _build_models(device)
        opt1, opt2     = _build_optimisers(model1, model2)
        labeled_loader = pool.get_labeled_loader(
            batch_size=args.batch_size, shuffle=True)

        # keep track of the best in-round checkpoint (by train phase acc)
        best_train_acc = -1.0
        best_state     = None

        for epoch in range(args.epochs_per_round):
            tr_loss, tr_tool, tr_phase = train_one_epoch(
                model1, model2, labeled_loader, opt1, opt2, device)

            log_every = max(1, args.epochs_per_round // 4)
            if (epoch + 1) % log_every == 0:
                print(f'  Epoch {epoch + 1:3d}/{args.epochs_per_round}  '
                      f'loss={tr_loss:.4f}  '
                      f'tool_acc={tr_tool:.3f}  '
                      f'phase_acc={tr_phase:.3f}')

            if tr_phase > best_train_acc:
                best_train_acc = tr_phase
                best_state = {
                    'm1': {k: v.clone() for k, v in model1.state_dict().items()},
                    'm2': {k: v.clone() for k, v in model2.state_dict().items()},
                }

        # restore best weights before evaluation and querying
        if best_state is not None:
            model1.load_state_dict(best_state['m1'])
            model2.load_state_dict(best_state['m2'])

        # ── Evaluate on held-out test set ─────────────────────────────
        te_loss, te_phase_acc, te_phase_f1, te_tool_acc = evaluate(
            model1, model2, test_loader, device)
        print(f'  Test  loss={te_loss:.4f}  '
              f'phase_acc={te_phase_acc:.4f}  '
              f'phase_f1={te_phase_f1:.4f}  '
              f'tool_acc={te_tool_acc:.4f}')

        round_info = {
            'round':          al_round + 1,
            'n_labeled':      pool.n_labeled,
            'n_unlabeled':    pool.n_unlabeled,
            'test_loss':      round(te_loss, 6),
            'test_phase_acc': round(te_phase_acc, 6),
            'test_phase_f1':  round(te_phase_f1, 6),
            'test_tool_acc':  round(te_tool_acc, 6),
        }
        results['rounds'].append(round_info)

        # ── Save intermediate results ─────────────────────────────────
        out_json = os.path.join(args.output_dir, f'al_{args.strategy}.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)

        # ── Query new samples (skip on last round) ────────────────────
        if al_round < args.n_rounds - 1 and pool.n_unlabeled > 0:
            n_query = min(args.query_budget, pool.n_unlabeled)

            strategy_kwargs: dict = {}
            if args.strategy == 'mc_dropout':
                strategy_kwargs['n_forward_passes'] = al_config.mc_dropout_passes
            if args.strategy == 'badge':
                strategy_kwargs['pca_dim'] = al_config.badge_pca_dim

            strategy = get_strategy(args.strategy, model1, model2, device,
                                    **strategy_kwargs)
            unlabeled_loader = pool.get_unlabeled_loader(
                batch_size=args.batch_size)

            if args.strategy == 'coreset':
                selected = strategy.query(
                    unlabeled_loader, n_query,
                    labeled_loader=labeled_loader)
            else:
                selected = strategy.query(unlabeled_loader, n_query)

            pool.label_samples(selected)
            print(f'  Queried {len(selected)} frames  '
                  f'→ labeled={pool.n_labeled}/{pool.n_total}')

    # ── Save final model ──────────────────────────────────────────────
    ckpt_path = os.path.join(args.output_dir,
                             f'model_{args.strategy}_final.pth')
    torch.save({
        'model1_state_dict': model1.state_dict(),
        'model2_state_dict': model2.state_dict(),
        'strategy':          args.strategy,
        'n_labeled':         pool.n_labeled,
        'n_total':           pool.n_total,
    }, ckpt_path)
    print(f'\nFinal checkpoint → {ckpt_path}')
    print(f'Results           → {out_json}')
    return results


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Active learning for cholec80 EndoNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--train_csv', required=True,
                   help='Path to train_set_info.csv')
    p.add_argument('--test_csv', required=True,
                   help='Path to test_set_info.csv')
    p.add_argument(
        '--strategy', default='entropy',
        choices=['random', 'least_confidence', 'margin', 'entropy',
                 'mc_dropout', 'coreset', 'badge'],
        help='Active learning query strategy',
    )
    p.add_argument('--n_rounds',        type=int,   default=10,
                   help='Number of AL cycles')
    p.add_argument('--query_budget',    type=int,   default=200,
                   help='Frames to annotate per round')
    p.add_argument('--initial_ratio',   type=float, default=0.05,
                   help='Fraction of training data labeled at round 0')
    p.add_argument('--epochs_per_round', type=int,  default=20,
                   help='SGD epochs per AL round')
    p.add_argument('--batch_size',      type=int,   default=16)
    p.add_argument('--seed',            type=int,   default=42,
                   help='Random seed for pool initialisation')
    p.add_argument('--output_dir',      default='./results',
                   help='Directory for JSON results and .pth checkpoints')
    return p.parse_args()


if __name__ == '__main__':
    run_al(parse_args())
