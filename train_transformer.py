import torch
import numpy as np
from tqdm import tqdm
import os

from constants import START_IDX, STOP_IDX
from transformer import LipTransformer
from data.dataset import LipReadSet, EncodingDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def demo_transformer():
    x = np.random.rand(75, 1, 3072)
    y = np.zeros((34, 1))
    y[0][0] = 28
    y[5][0] = 29
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    model = LipTransformer(dim=128, nhead=4, nlayers=3)
    model.to(device)
    model.train()
    
    tgt_mask = model.create_mask(y.shape[0])
    tgt_mask = tgt_mask.to(device)
    logits = model(x, y, tgt_mask)
    
    logits = torch.squeeze(logits)
    preds = torch.argmax(logits, dim=1)
    print(preds)


def demo_prediction():
    model = LipTransformer(nhead=4, nlayers=3)
    x = torch.tensor(np.random.rand(75, 1, 3072), dtype=torch.float, device=device)
    print(predict(model, x, 30))


def pred_to_text(y_seq):
    ten = y_seq.squeeze()
    seq = ten.to_lis


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


def load(model, optimizer, model_alias, model_save_dir = os.path.join("models")):
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

        loaded_checkpoint = torch.load(os.path.join(model_save_path, f"{model_alias}.pt"))
        model.load_state_dict(loaded_checkpoint['model_state_dict'])
        optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

        with open(os.path.join(model_save_path, "model.info"), 'r') as description_file:
            lines = description_file.readlines()
            best_loss = float(lines[-3].split(":")[-1].strip())

    train_loss_path = os.path.join(model_save_path, "train_losses.info")
    val_loss_path = os.path.join(model_save_path, "validation_losses.info")
    val_wer_path = os.path.join(model_save_path, "validation_wer.info")
    val_cer_path = os.path.join(model_save_path, "validation_cer.info")

    if not os.path.exists(val_loss_path):
        with open(train_loss_path, 'w') as z, open(val_loss_path, 'w') as a, open(val_wer_path, 'w') as b, open(val_cer_path, 'w') as c:
            pass

    return best_loss, model_save_path, train_loss_path, val_loss_path, val_wer_path, val_cer_path


def update_checkpoint(model, loss_fn, model_save_path, model_alias):
    loaded_checkpoint = torch.load(os.path.join(model_save_path, f"{model_alias}.pt"))
    model.load_state_dict(loaded_checkpoint['model_state_dict'])
    model.to(device)

    with open(os.path.join(model_save_path, "training.success"), 'w') as success_file:
        test_loss, test_wer, test_cer = eval(model, loss_fn, test_loader, device)
        success_file.write(f"Test Loss: {test_loss:.6f}\n")
        success_file.write(f"Test WER: {test_wer:.6f}\n")
        success_file.write(f"Test CER: {test_cer:.6f}") 


def update_loss(model, optimizer, loss, wer, cer, model_save_path, model_alias):
    save_path = os.path.join(model_save_path, f"{model_alias}.pt")
    if os.path.exists(save_path):
        os.remove(save_path)

    checkpoint = {'model': type(model),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict()}
    torch.save(checkpoint, save_path)

    # Save best loss in the model info file
    lines = open(os.path.join(model_save_path, "model.info"), 'r').readlines() # !may cause memory issues!
    lines[-3] = f"Best Loss: {loss}\n"
    lines[-2] = f"WER: {wer}\n"
    lines[-1] = f"CER: {cer}"
    description_file = open(os.path.join(model_save_path, "model.info"), 'w')
    description_file.writelines(lines)
    description_file.close()

"""
preds: (N x T)
"""
ALPHABET = "_ abcdefghijklmnopqrstuvwxyz<>"
def indices_to_chars(preds):
    # TODO: speed this up!!!!
    # Is there a pytorch way to do this?
    words = list()
    for n in range(preds.shape[0]):
        indices = preds[n]
        word = ""
        for i in indices:
            word += ALPHABET[i.item()]
        words.append(word)

    return words
    
"""
logits: (N X T X C)
"""
def decode_logits(logits):
    preds = torch.argmax(logits, 2)
    return indices_to_chars(preds)


def eval(model, loss_fn, val_data):
    # VALIDATION
    val_loss = 0.0
    running_wer = np.array([])
    running_cer = np.array([])

    model.eval()
    for vid, label in tqdm(val_data):
        vid = vid.to(device)
        label = label.to(device)

        # Reorder vid to (S x N x V)
        vid = torch.permute(vid, (1, 0, 2))
        # Reorder label to (T x N)
        label_in = torch.permute(label, (1, 0))

        # Forward pass
        tgt_mask = model.create_mask(label_in.shape[0]).to(device)
        logits = model(vid, label_in, tgt_mask)

        logits_loss = torch.permute(logits, (1, 2, 0))
        """
        logits: (N x C x T)
        label: (N x T)
        """
        loss = loss_fn(logits_loss, label)
        val_loss += loss.item()

        pred_txt = decode_logits(logits)
        target_txt = indices_to_chars(label)

        running_wer = np.append(running_wer,LipReadSet.wer(pred_txt, target_txt))
        running_cer = np.append(running_cer, LipReadSet.cer(pred_txt, target_txt))

    val_loss = val_loss / len(val_data)
    final_wer = np.mean(running_wer)
    final_cer = np.mean(running_cer)

    return val_loss, final_wer, final_cer


def test():
    pass

def train(model, optimizer, train_data, val_data, model_alias, epochs=1, model_save_dir = os.path.join("models")):

    best_loss, model_save_path, train_loss_path, val_loss_path, val_wer_path, val_cer_path = load(model, optimizer, model_alias, model_save_dir)

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("Epoch ", str(epoch))
        print("Train")
        # TRAIN
        model.train()
        train_loss = 0.0
        for vid, label in tqdm(train_data):
            """
            vid: (N x S x V)
            label: (N x T)
            """
            vid = vid.to(device)
            label = label.to(device)

            # Reorder vid to (S x N x V)
            vid = torch.permute(vid, (1, 0, 2))
            # Reorder label to (T x N)
            label_in = torch.permute(label, (1, 0))

            # Forward pass
            tgt_mask = model.create_mask(label_in.shape[0]).to(device)
            logits = model(vid, label_in, tgt_mask)

            logits = torch.permute(logits, (1, 2, 0))
            """
            logits: (N x C x T)
            label: (N x T)
            """
            loss = loss_fn(logits, label)
            train_loss += loss.item()

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_data)
        with open(train_loss_path, 'a') as train_file:
            train_file.write(f"{train_loss}\n")

        # Write loss, wer, cer to file
        print("Validation")
        val_loss, val_wer, val_cer = eval(model, loss_fn, val_data)

        with open(val_loss_path, 'a') as loss_file:
            loss_file.write(f"{val_loss}\n")
        with open(val_wer_path, 'a') as wer_file:
            wer_file.write(f"{val_wer}\n")
        with open(val_cer_path, 'a') as cer_file:
            cer_file.write(f"{val_cer}\n")

        # Saving best model based on validation loss
        if val_loss <= best_loss:
            best_loss = val_loss
            update_loss(model, optimizer, best_loss, val_wer, val_cer, model_save_path, model_alias)
            
    # TODO: Implement testing with predictions
    #update_checkpoint(model, loss_fn, model_save_path, model_alias)

    return model


def get_data(path):
    train = EncodingDataset(path, "train")
    val = EncodingDataset(path, "val")

    train_data = torch.utils.data.DataLoader(train, batch_size=128, num_workers=4)
    val_data = torch.utils.data.DataLoader(val, batch_size=128, num_workers=4)
    return train_data, val_data

if __name__ == "__main__":
    print("Getting data")
    train_data, val_data = get_data("../encoding_data")
    print("Creating model")
    model = LipTransformer(dim=512, nhead=8, nlayers=6)
    optimizer = torch.optim.Adam(model.parameters())
    print("Training model")
    model = train(model, optimizer, train_data, val_data, model_alias="transformer", epochs=100)

