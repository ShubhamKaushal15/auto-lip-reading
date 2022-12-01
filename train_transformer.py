import torch
import numpy as np

from transformer import LipTransformer


def train_transformer():
    x = np.random.rand(75, 1, 3072)
    y = np.zeros((34, 1, 29))
    y[0][0][27] = 1
    y[30][0][28] = 1
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    model = LipTransformer()
    
    logits = model(x, y)
    
    logits = torch.squeeze(logits)
    
    preds = torch.argmax(logits, dim=1)
    
    print(preds)
    
    
if __name__ == "__main__":
    train_transformer()