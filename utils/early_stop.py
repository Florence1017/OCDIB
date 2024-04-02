import datetime
import numpy as np
import torch
import os
# from utils.helper import save_config


class EarlyStopping(object):
    def __init__(self, checkpoint_path, config=None, patience=10, is_ours=False):
        dt = datetime.datetime.now()
        if is_ours:
            self.filepath = os.path.join(checkpoint_path, 'early_stop_{}_{:02d}-{:02d}'.format(
                dt.date(), dt.hour, dt.minute))
        else:
            self.filepath = os.path.join(checkpoint_path, 'early_stop')

        self.filename = os.path.join(self.filepath, "model.pth")
        # if config:
        #     save_config(config, self.filepath)
        self.patience = patience
        self.counter = 0
        self.ck_path = checkpoint_path

        self.best_loss = None
        self.early_stop = False

    def step(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss >= self.best_loss):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss < self.best_loss):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        if not os.path.exists(self.ck_path):
            os.makedirs(self.ck_path)
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model, filename=None, filepath=None):
        """Load the latest checkpoint."""
        if filename is None and not filepath is None:
            path = os.path.join(self.ck_path, filepath, "model.pth")
        elif filepath is None and not filename is None:
            path = os.path.join(self.filepath, filename)
        elif filepath is None and filename is None:
            path = self.filename
        else:
            path = os.path.join(self.ck_path, filepath, filename)
        model.load_state_dict(torch.load(path))
        # if filename is None:
        #     model.load_state_dict(torch.load(self.filename))
        # else:
        #     model.load_state_dict(torch.load(os.path.join(self.filepath, filename)))
        return model
