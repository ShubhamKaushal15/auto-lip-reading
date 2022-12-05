import torch 
import torch.nn as nn
import torch.nn.init as init
import math

class PositionalEncoding(torch.nn.Module):
        
        # TODO: is max_len correct?
        def __init__(self, d_model: int=512, dropout: float = 0.1, max_len: int=75):
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

class STCNNTransformer(torch.nn.Module):
    def __init__(self, transformer_dim: int=512, output_dim: int=30, dropout_p=0.5):
        super(STCNNTransformer, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        """
        Input: (S x V)
        Output: (S x E)
        """
        self.video_encoder = torch.nn.Sequential(torch.nn.Linear(96*4*8, transformer_dim),
                                                 PositionalEncoding(d_model=transformer_dim, 
                                                                    dropout=0.1))
        
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
                                          nhead=8,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6)

        """
        Input: (T x E)
        Output: (T x C)
        """
        self.FC = torch.nn.Sequential(torch.nn.Linear(transformer_dim, 128),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(128, 64),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(64, output_dim))

        
        self.dropout_p  = dropout_p

        self.relu = nn.ReLU(inplace=True)
        self.dropout3d = nn.Dropout3d(self.dropout_p)  
        self._init()
    
    def _init(self):
        
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)
        
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)        
        
        # init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
        # init.constant_(self.FC.bias, 0)

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
        
    def forward(self, x, y, tgt_mask):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)

        # (B, C, T, H, W)->(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # (B, T, C, H, W)->(B, T, C*H*W)
        x = x.view(x.size(0), x.size(1), -1)

        # video_encoder takes B x T ...
        vid_enc = self.video_encoder(x)

        # y shape: B x T x 1. text_enc shape: B x T x dim_model
        text_enc = self.text_encoder(y)

        # permute to (T, B, ...)
        out = self.model(vid_enc.permute(1, 0, 2).contiguous(),
                        text_enc.permute(1, 0, 2).contiguous(), tgt_mask=tgt_mask)
        
        # (T x B x C)
        logits = self.FC(out)
        logits = logits.permute(1, 0, 2).contiguous() # (B, T, ...)

        return logits
        
        
