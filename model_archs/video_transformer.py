import torch 
import torch.nn as nn
import torch.nn.init as init
import math

class PositionalEncoding(torch.nn.Module):
        
        # TODO: is max_len correct?
        def __init__(self, d_model: int=512, dropout: float = 0.1, max_len: int=1200):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=dropout)

            position = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: Tensor, shape [batch_size, seq_len, embedding_dim]
            """
            x = x + self.pe[:x.size(1)]
            return self.dropout(x)

class VideoTransformer(torch.nn.Module):
    def __init__(self, nhead: int=8, num_encoder_layers: int=6, num_decoder_layers: int=6, 
                    transformer_dim: int=512, output_dim: int=30, dropout_p=0.5):
        super(VideoTransformer, self).__init__()
        self.patch_linear_embedding = nn.Conv3d(1, 512, (8, 8, 8), (8, 8, 8)) # 128 x 64 --> 64 x 32
          

        """
        Input: (S x E)
        Output: (S x E)
        """
        self.pos_embedding = PositionalEncoding(d_model=transformer_dim, dropout=0.1)
        
        """
        Input: (T)
        Output: (T x E)
        """
        self.text_encoder = torch.nn.Sequential(torch.nn.Embedding(30, transformer_dim),
                                                PositionalEncoding(d_model=transformer_dim, 
                                                                    dropout=0.1))
        
        """
        Inputs: (S x E), (T x E)
        Output: (T x E)
        """
        self.model = torch.nn.Transformer(d_model=transformer_dim,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)

        """
        Input: (T x E)
        Output: (T x C)
        """
        self.FC = torch.nn.Sequential(torch.nn.Linear(transformer_dim, 128),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(128, 64),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(64, output_dim))

        
           
    

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

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int=0):
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)
        
    def forward(self, x, y, tgt_mask, tgt_key_padding_mask):
        x = torch.mean(x, axis=1)
        x = x.unsqueeze(1)
        #print(x.shape)
        x = self.patch_linear_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1,2)
        
        
        # (B, T, E)
        
        # video_encoder takes B x T ...
        vid_enc = self.pos_embedding(x)

        # y shape: B x T x 1. text_enc shape: B x T x dim_model
        text_enc = self.text_encoder(y)

        # permute to (T, B, ...)
        out = self.model(vid_enc.permute(1, 0, 2).contiguous(),
                        text_enc.permute(1, 0, 2).contiguous(), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        # (T x B x C)
        logits = self.FC(out)
        logits = logits.permute(1, 0, 2).contiguous() # (B, T, ...)

        return logits

        
