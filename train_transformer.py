import torch
import numpy as np

from constants import START_TOKEN, STOP_TOKEN
from transformer import LipTransformer
from data.dataset import LipReadSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_transformer():
    x = np.random.rand(75, 1, 3072)
    y = np.zeros((34, 1))
    y[0][0] = 28
    y[5][0] = 29
    x = torch.tensor(x, dtype=torch.float, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    model = LipTransformer(dim=512, nhead=4, nlayers=3)
    model.to(device)
    model.train()
    
    logits = model(x, y)
    
    logits = torch.squeeze(logits)
    print(logits.shape)
    preds = torch.argmax(logits, dim=1)
    
    print(preds)


def pred_to_text(y_seq):
    ten = y_seq.squeeze()
    seq = ten.to_lis


def predict(model, input_video, dim_text):

    model.eval()
    model = model.to(device)

    y_seq = torch.tensor([START_TOKEN], dtype=torch.long, device=device)
    y_seq = y_seq.unsqueeze(dim=1)

    for i in range(dim_text):
        pred = model(input_video, y_seq)
        idx = torch.argmax(pred[pred.shape[0]-1])

        # If we hit the stop token, finish
        if idx == STOP_TOKEN:
            break

        idx = torch.tensor([idx], dtype=torch.long, device=device)
        idx = idx.unsqueeze(dim=1)
        y_seq = torch.cat([y_seq, idx], dim=0)
        

    return y_seq

    
if __name__ == "__main__":
    train_transformer()
    # model = LipTransformer(nhead=4, nlayers=3)
    # x = torch.tensor(np.random.rand(75, 1, 3072), dtype=torch.float, device=device)
    # print(predict(model, x, 30))
