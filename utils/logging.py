import logging
import time
import os
import torch

import matplotlib.pyplot as plt


class Logger(object):
    def __init__(self, config_file):
        parent_path = f"./work/{config_file}"
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
        filename = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        log_file = os.path.join(parent_path, f"{filename}.log")
        self.loss_file = os.path.join(parent_path, f"{filename}.jpg")
        self.model_file = os.path.join(parent_path, f"{filename}.pkl")
        logging.basicConfig(level=logging.INFO, filename=log_file)
        self.logger = logging.getLogger('Train')

    def info(self, message):
        print(message)
        self.logger.info(message)
    
    def log_finish(self):
        current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
        self.logger.info(f'Training process finished at {current_time}')

    def plot_loss(self, epoch_list, loss_list):
        plt.plot(epoch_list, loss_list)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(self.loss_file)
        
    def save_model(self, model):
        torch.save(model.state_dict(), self.model_file)
