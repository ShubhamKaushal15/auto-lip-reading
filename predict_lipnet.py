import os
import torch
from tqdm import tqdm
from data.dataset import LipReadSet
import numpy as np
import sys
import json
from lipnet import ConvGRU
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(config_name):
    """
    Runs the train scripts based on config file
    """

    with open(os.path.join("configs", f"{config_name}.json"), "r") as f:
        config = json.load(f)

    model = ConvGRU() # Needs to be hard-coded
    model.to(device)

    test_dataset = LipReadSet(config['video_path'],
                         config['anno_path'],
                         config['test_list'],
                         config['vid_padding'],
                         config['txt_padding'], 'test')

    test_loader = DataLoader(test_dataset, 
                             batch_size = config['batch_size'],
                             num_workers = config['num_workers'], 
                             shuffle = True)
    
    loaded_checkpoint = torch.load(os.path.join("models", "lipnet_unseen_mark2", "lipnet_unseen_mark2.pt"))
    model.load_state_dict(loaded_checkpoint['model_state_dict'])

    
    predict(model, test_loader, device)

    return 1   

def predict(model, data_loader, device):
    """
    Calculates the loss and error of the model on the data
    """

    model.eval() # setting model to eval mode
    
    print("Predicting")
    
    for _, data in enumerate(tqdm(data_loader), 0):
        
        # get the inputs; data is a dictionary
        inputs = data.get('vid').to(device)
        targets = data.get('txt').to(device)
        vid_len = data.get('vid_len').to(device)
        txt_len = data.get('txt_len').to(device)

        # getting predictions
        outputs = model(inputs)
        pred_txt = LipReadSet.ctc_decode(outputs)

        # getting target text
        target_txt = [LipReadSet.arr2txt(targets[_]) for _ in range(targets.size(0))]
        
        # print(pred_txt)
        # print(target_txt)
        
        break
        
    print(list(zip(pred_txt, target_txt)))
    
    return 1
          


if __name__ == '__main__':
    """
    Takes name (not path) of config file as argument
    """
    config_name = sys.argv[1]

    main(config_name)