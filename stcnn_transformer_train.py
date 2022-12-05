import torch
import numpy as np
from tqdm import tqdm
import os
from functions import *
from data.dataset import LipReadSet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import sys
import json

from constants import START_IDX, STOP_IDX
from stcnn_transformer import STCNNTransformer
from data.dataset import LipReadSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model, loss_function, data_loader, device):
    """
    Calculates the loss and error of the model on the data
    """

    model.eval() # setting model to eval mode

    running_loss = 0.0
    running_wer = np.array([])
    running_cer = np.array([])
    
    print("Evaluating")

    for _, data in enumerate(tqdm(data_loader), 0):
        
        # get the inputs; data is a dictionary
        inputs = data.get('vid').to(device)
        targets = data.get('txt').to(device)

        tgt_mask = create_mask(targets.size(1) - 1)

        # getting predictions
        outputs = model(inputs, targets[:, :-1], tgt_mask = tgt_mask)
        pred_txt = LipReadSet.ctc_decode(outputs)

        # calculate CE loss
        loss = loss_function(outputs.permute(0, 2, 1), targets[:, 1:]) # CE wants (B x C x T)
        running_loss += loss.item()

        # getting target text
        target_txt = [LipReadSet.arr2txt(targets[_]) for _ in range(targets.size(0))]

        running_wer = np.append(running_wer, LipReadSet.wer(pred_txt, target_txt))
        running_cer = np.append(running_cer, LipReadSet.cer(pred_txt, target_txt))

    final_loss = running_loss / len(data_loader)
    final_wer = np.mean(running_wer)
    final_cer = np.mean(running_cer)

    return final_loss, final_wer, final_cer

def train(model, num_epochs, loss_function, optimizer, model_alias, 
          train_loader, val_loader, test_loader, device,
          model_save_dir = os.path.join("models")):
    """
    Trains the model using given hyperparameters. Saves the best model based
    on validation loss.
    Saves the list of validation losses of the model during training.
    """
    model_save_path = os.path.join(model_save_dir, model_alias)

    best_loss = 10000
    
    if not os.path.exists(model_save_path):
      print(f"{model_save_path} path does not exist. Creating...")
      os.makedirs(f"{model_save_path}")

      with open(os.path.join(model_save_path, "model.info"), 'w') as description_file:
          description_file.write(f"Model name: {type(model)}\n")
          description_file.write(f"Optimizer: {type(optimizer)}\n")
          description_file.write("Best Loss:\n")
          description_file.write("WER:\n")
          description_file.write("CER:")
    else:
        print("Resuming training from previous best checkpoint...")

        model.to(device)
        # model.double()

        loaded_checkpoint = torch.load(os.path.join(model_save_path, f"{model_alias}.pt"))
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0., amsgrad=True)
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

        with open(os.path.join(model_save_path, "model.info"), 'r') as description_file:
            lines = description_file.readlines()
            best_loss = float(lines[-3].split(":")[-1].strip())

    validation_loss_file_path = os.path.join(model_save_path, "validation_losses.info")
    validation_wer_file_path = os.path.join(model_save_path, "validation_wer.info")
    validation_cer_file_path = os.path.join(model_save_path, "validation_cer.info")

    if not os.path.exists(validation_loss_file_path):
        with open(validation_loss_file_path, 'w') as a, open(validation_wer_file_path, 'w') as b, open(validation_cer_file_path, 'w') as c:
            pass

    model.to(device) # get model to current device
    # model.double()

    for i in range(num_epochs):  # loop over the dataset multiple times
        
        print(f"Training epoch: {i}")

        model.train() # set model to train mode

        for _, data in enumerate(tqdm(train_loader), 0):

            # get the inputs; data is a dictionary
            inputs = data.get('vid').to(device)
            targets = data.get('txt').to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            tgt_mask = create_mask(targets.size(1) - 1)

            # getting predictions
            outputs = model(inputs, targets[:, :-1], tgt_mask=tgt_mask) # B x T x C

            # calculate CE loss
            loss = loss_function(outputs.permute(0, 2, 1), targets[:, 1:]) # CE wants (B x C x T)

            # backprop
            loss.backward()

            # update gradients
            optimizer.step()

        val_loss, val_wer, val_cer = evaluate(model, loss_function, val_loader, device)

        with open(validation_loss_file_path, 'a') as loss_file:
            loss_file.write(f"{val_loss}\n")

        with open(validation_wer_file_path, 'a') as loss_file:
            loss_file.write(f"{val_wer}\n")

        with open(validation_cer_file_path, 'a') as loss_file:
            loss_file.write(f"{val_cer}\n")

        # Saving best model based on validation loss
        if val_loss <= best_loss:

            best_loss = val_loss

            save_path = os.path.join(model_save_path, f"{model_alias}.pt")
            if os.path.exists(save_path):
                os.remove(save_path)

            checkpoint = {'model': type(model),
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict()}
            torch.save(checkpoint, save_path)

            # Save best loss in the model info file
            lines = open(os.path.join(model_save_path, "model.info"), 'r').readlines() # !may cause memory issues!
            lines[-3] = f"Best Loss: {best_loss}\n"
            lines[-2] = f"WER: {val_wer}\n"
            lines[-1] = f"CER: {val_cer}"
            description_file = open(os.path.join(model_save_path, "model.info"), 'w')
            description_file.writelines(lines)
            description_file.close()

    loaded_checkpoint = torch.load(os.path.join(model_save_path, f"{model_alias}.pt"))
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    model.to(device)

    with open(os.path.join(model_save_path, "training.success"), 'w') as success_file:
        test_loss, test_wer, test_cer = evaluate(model, loss_function, test_loader, device)
        success_file.write(f"Test Loss: {test_loss:.6f}\n")
        success_file.write(f"Test WER: {test_wer:.6f}\n")
        success_file.write(f"Test CER: {test_cer:.6f}") 

    return 1

def create_mask(len_text):
    """
    Input: len_text = T
    Output: T x T mask
    """
    mask = torch.tril(torch.ones(len_text, len_text) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    return mask 

def predict(model, input_video, dim_text):

    model.eval()
    model = model.to(device)

    y_seq = torch.tensor([START_IDX], dtype=torch.long, device=device)
    y_seq = y_seq.unsqueeze(dim=1)

    for i in range(dim_text):
        # Predict a character
        pred = model(input_video, y_seq)
        idx = torch.argmax(pred[pred.shape[0]-1])

        # Append character
        idx = torch.tensor([idx], dtype=torch.long, device=device)
        idx = idx.unsqueeze(dim=1)
        y_seq = torch.cat([y_seq, idx], dim=0)
        
        # If we hit the stop token, finish
        if idx == STOP_IDX:
            break

    return y_seq


def main(config_name):
    """
    Runs the train scripts based on config file
    """

    with open(os.path.join("configs", f"{config_name}.json"), "r") as f:
        config = json.load(f)

    model = STCNNTransformer() # Needs to be hard-coded
    model.to(device)

    # NO NEED TO COMMENT THIS IN #############################
    # pretrained_dict = torch.load(config['pretrained_path'])
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    # missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    # print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    # print('miss matched params:{}'.format(missed_params))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    ce_loss = nn.CrossEntropyLoss()
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
                         config['txt_padding'], 'validation')

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

    
    train(model, config['epochs'], ce_loss, adam_optimizer, config['alias'],
        train_loader, validation_loader, test_loader, device, os.path.join("models"))

    return 1   



if __name__ == '__main__':
    """
    Takes name (not path) of config file as argument
    """
    config_name = sys.argv[1]

    main(config_name)