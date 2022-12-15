"""
Source : https://github.com/Bjarten/early-stopping-pytorch
"""
import os
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0.01,
        path="checkpoint.pt",
        trace_func=print,
        checkpoint_fn=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.checkpoint_fn = None

    def __call__(self, val_loss, save_dict):

        score = -val_loss
        save_checkpoint = False
        if self.best_score is None:
            self.best_score = score
            save_checkpoint = True
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, save_dict)
            save_checkpoint = True
            self.counter = 0
        return save_checkpoint

    def save_checkpoint(self, val_loss, save_dict):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # intermediate
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        torch.save(save_dict, f"checkpoints/step-{save_dict['step']}_counter-{self.counter}.pt")
        self.val_loss_min = val_loss
