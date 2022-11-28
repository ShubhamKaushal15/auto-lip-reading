import os
from functions import *
import pandas as pd
from data.dataset import LipReadSet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import sys
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(config_name):
    """
    Runs the train scripts based on config file
    """

    with open(os.path.join("configs", config_name), "r") as f:
        config = json.load(f)

    model = ... # Needs to be hard-coded
    # transform = ... TODO: albumentations feature

    ctc_loss = nn.CTCLoss()
    adam_optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0., amsgrad=True)

    train_dataset = LipReadSet(config['video_path'],
                         config['anno_path'],
                         config['train_list'],
                         config['vid_padding'],
                         config['txt_padding'])

    validation_dataset = LipReadSet(config['video_path'],
                         config['anno_path'],
                         config['validation_list'],
                         config['vid_padding'],
                         config['txt_padding'])

    test_dataset = LipReadSet(config['video_path'],
                         config['anno_path'],
                         config['test_list'],
                         config['vid_padding'],
                         config['txt_padding'], 'test')
     
    train_loader = DataLoader(train_dataset, 
                              batch_size = config['batch_size'], 
                              num_workers = config['num_workers'],
                              shuffle = True)

    validation_loader = DataLoader(validation_dataset, 
                                   batch_size = config['batch_size'],
                                   num_workers = config['num_workers'], 
                                   shuffle = True)

    test_loader = DataLoader(test_dataset, 
                             batch_size = config['batch_size'],
                             num_workers = config['num_workers'], 
                             shuffle = True)

    
    train(model, config['epochs'], ctc_loss, adam_optimizer, config['alias'],
        train_loader, validation_loader, test_loader, device, os.path.join("models"))

    return 1   



if __name__ == '__main__':
    """
    Takes name of config file as argument
    """
    
    config_name = sys.argv[1]
    main(config_name)