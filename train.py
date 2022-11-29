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
from lipnet import ConvGRU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(config_name):
    """
    Runs the train scripts based on config file
    """

    with open(os.path.join("configs", f"{config_name}.json"), "r") as f:
        config = json.load(f)

    model = ConvGRU() # Needs to be hard-coded
    model.to(device)

    pretrained_dict = torch.load(config['pretrained_path'])
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:{}'.format(missed_params))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

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
    Takes name (not path) of config file as argument
    """
    config_name = sys.argv[1]

    main(config_name)