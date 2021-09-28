import logging
import time
import os
import torch

import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, config_file, mode="train"):
        parent_path = f"work/{config_file}"
        model_path = f"work/models/{config_file}"
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        filename = self.get_time()
        log_file = os.path.join(parent_path, f"{filename}.{mode}.log")
        self.loss_file = os.path.join(parent_path, f"{filename}.{mode}.jpg")
        self.model_file = os.path.join(model_path, f"{filename}.pkl")
        logging.basicConfig(level=logging.INFO, filename=log_file)
        self.logger = logging.getLogger(mode)

    def get_time(self):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    def info(self, message):
        print(message)
        self.logger.info(f"{self.get_time()}:{message}")

    def log_finish(self):
        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        self.logger.info(f"Training process finished at {current_time}")

    def plot_loss(self, epoch_list, loss_list):
        plt.plot(epoch_list, loss_list)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(self.loss_file)

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_file, _use_new_zipfile_serialization=False)
