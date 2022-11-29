import os
import torch
from tqdm import tqdm
from data.dataset import LipReadSet

def evaluate(model, loss_function, data_loader, device):
    """
    Calculates the loss and error of the model on the data
    """

    model.eval() # setting model to eval mode

    running_loss = 0.0
    running_wer = 0
    running_cer = 0

    for _, data in enumerate(data_loader, 0):
        
        # get the inputs; data is a dictionary
        inputs = data.get('vid').to(device)
        targets = data.get('txt').to(device)
        vid_len = data.get('vid_len').to(device)
        txt_len = data.get('txt_len').to(device)

        # getting predictions
        outputs = model(inputs)
        pred_txt = LipReadSet.ctc_decode(outputs)

        # calculate CTC loss
        loss = loss_function(outputs.transpose(0, 1).log_softmax(-1), targets, vid_len.view(-1), txt_len.view(-1))
        running_loss += loss.item()

        # getting target text
        target_txt = [LipReadSet.arr2txt(targets[_]) for _ in range(targets.size(0))]

        running_wer += LipReadSet.wer(pred_txt, target_txt)
        running_cer += LipReadSet.cer(pred_txt, target_txt)

    final_loss = running_loss / len(data_loader)
    final_wer = running_wer / len(data_loader)
    final_cer = running_cer / len(data_loader)

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
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

        with open(os.path.join(model_save_path, "model.info"), 'r') as description_file:
            lines = description_file.readlines()
            best_loss = float(lines[-1].split(":")[-1].strip())

    validation_loss_file_path = os.path.join(model_save_path, "validation_losses.info")
    validation_wer_file_path = os.path.join(model_save_path, "validation_wer.info")
    validation_cer_file_path = os.path.join(model_save_path, "validation_cer.info")

    if not os.path.exists(validation_loss_file_path):
        with open(validation_loss_file_path, 'w') as a, open(validation_wer_file_path, 'w') as b, open(validation_cer_file_path, 'w') as c:
            pass

    model.to(device) # get model to current device
    # model.double()

    for _ in range(num_epochs):  # loop over the dataset multiple times

        model.train() # set model to train mode

        for _, data in enumerate(tqdm(train_loader), 0):

            # get the inputs; data is a dictionary
            inputs = data.get('vid').to(device)
            targets = data.get('txt').to(device)
            vid_len = data.get('vid_len').to(device)
            txt_len = data.get('txt_len').to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # getting predictions
            outputs = model(inputs)

            # calculate CTC loss
            loss = loss_function(outputs.transpose(0, 1).log_softmax(-1), targets, vid_len.view(-1), txt_len.view(-1))

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