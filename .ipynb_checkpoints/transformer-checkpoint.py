import torch
import math

class LipTransformer(torch.nn.Module):
    
    
    class PositionalEncoding(torch.nn.Module):
        
        def __init__(self, d_model: int=512, dropout: float = 0.1, max_len: int=3072):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Tensor, shape [seq_len, batch_size, embedding_dim]
            """
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)
        
    
    def __init__(self, dim: int=512, dropout: float=0.1, dim_video: int=3072, dim_text : int=29, nhead: int=8, nlayers: int=6):
        super().__init__()
        
        self.video_encoder = self.PositionalEncoding(d_model=dim_video,
                                          dropout=dropout,
                                          max_len=dim_video)
        
        self.text_encoder = torch.nn.Sequential(torch.nn.Linear(dim_text, dim_video),
                                                  torch.nn.Dropout(p=dropout),
                                                  torch.nn.ReLU())
        
        self.model = torch.nn.Transformer(d_model=dim_video,
                                          nhead=nhead,
                                          num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers)
        
        self.FC = torch.nn.Sequential(torch.nn.Linear(dim_video, 128),
                                      torch.nn.Linear(128, 64),
                                      torch.nn.Linear(64, dim_text))
        
        
    def forward(self, X, y):
        """
        Input X shape: (S x E) -> (75 x 3072)
        Input y shape: (T x C) -> (32 + 2 x 29)
        Encoded y shape: (T x E) -> (34 x 3072)
        Output shape: (T x E) -> (34 x 3072)
        Final output shape: (T x C) -> (34 x 29)
        """
        vid_enc = self.video_encoder(X)
        text_enc = self.text_encoder(y)
        out = self.model(vid_enc, text_enc)
        return self.FC(out)
    
    
def predict(model, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()

    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

        pred = model(input_sequence, y_input, tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()
