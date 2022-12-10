import torch
import math

class LipTransformer(torch.nn.Module):
    
    class PositionalEncoding(torch.nn.Module):
        
        # TODO: is max_len correct?
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
        
    
    def __init__(self, dim: int=512, dropout: float=0.1, dim_video: int=3072, dim_text : int=30, nhead: int=8, nlayers: int=6):
        super().__init__()

        """
        Input: (S x N x V)
        Output: (S x N x E)
        """
        self.video_encoder = torch.nn.Sequential(torch.nn.Linear(dim_video, dim),
                                                 torch.nn.ReLU(), 
                                                 self.PositionalEncoding(d_model=dim, 
                                                                         dropout=dropout,
                                                                         max_len=dim_video))
        
        """
        Input: (T x N)
        Output: (T x N x E)
        """
        self.text_encoder = torch.nn.Embedding(dim_text, dim)
        
        """
        Inputs: (S x N x E), (T x N x E)
        Output: (T x N x E)
        """
        self.model = torch.nn.Transformer(d_model=dim,
                                          nhead=nhead,
                                          num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers)

        """
        Input: (T x N x E)
        Output: (T x N x C)
        """
        self.FC = torch.nn.Sequential(torch.nn.Linear(dim, 128),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(128, 64),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(64, dim_text))

    """
    Input: len_text = T
    Output: T x T mask
    """
    def create_mask(self, len_text):
        mask = torch.tril(torch.ones(len_text, len_text) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask 
        

    def forward(self, X, y, tgt_mask=None):
        """
        Mask: Whether or not to perform masking, this should be true during training

        Input X shape: (S x N x V)
        Encoded X shape: (S x N x E)
        Input y shape: (T x N)
        Encoded y shape: (T x N x E)
        Output shape: (T x N x E)
        Final output shape: (T x N x C)
        """
        vid_enc = self.video_encoder(X)
        text_enc = self.text_encoder(y)
        out = self.model(vid_enc, text_enc, tgt_mask=tgt_mask)
        logits = self.FC(out)
        return logits
    
    
