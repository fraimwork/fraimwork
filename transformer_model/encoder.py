import torch
from torch import nn

from mha import MultiHeadAttention
from pe import PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer Encoder Layer as a torch.nn.Module, using the MultiHeadAttention mechanism implemented earlier and other existing modules in torch.nn
    """
    
    ## The __init__ method has been given to you. Do NOT modify ANY of the code in this method!
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout_prob,
        ):
        '''
        Initialize the TransformerEncoderLayer. Defines some model parameters and hyperparameters

        Input:
            embedding_dim (int) - dimension of the Transformer encoder layer (aka d_model)

            ffn_embedding_dim (int) - inner dimension of the position-wise feedforward network (size of W_1 and b_1 in the paper)

            num_attention_heads (int) - number of attention heads in the encoder self attention

            dropout_prob (float) - dropout probability between 0 and 1 for the dropout module
        '''

        super().__init__()

        # This is the model dimension, d_model
        self.embedding_dim = embedding_dim

        # Build the self attention mechanism using MultiHeadAttention
        self.self_attn = MultiHeadAttention(
            self.embedding_dim,
            num_attention_heads
        )

        # layer norm for the self attention layer's output
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        # activation_fn, fc1 and fc2 for implementing the position-wise feed-forward network
        self.activation_fn = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim, bias = True)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim, bias = True)

        # layer norm for the position-wise feed-forward network's out
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        # dropout layer for regularization
        self.dropout = nn.Dropout(dropout_prob)

    ## You need to fill in the missing code in the forward method below
    def forward(
        self,
        x,
        self_attn_padding_mask = None,
    ):
        """
        Applies the self attention module + Dropout + Add & Norm operation, and the position-wise feedforward network + Dropout + Add & Norm operation. Note that LayerNorm is applied after the self-attention, and another time after the ffn modules, similar to the original Transformer implementation.

        Input:
            x (torch.Tensor) - input tensor of size B x T x embedding_dim from the encoder input or the previous encoder layer; serves as input to the TransformerEncoderLayer's self attention mechanism.

            self_attn_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T, where for each self_attn_padding_mask[b] for the b-th source in the batch, the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

        Output:
            x (torch.Tensor) - the encoder layer's output, of size B x T x embedding_dim, after the self attention module + Dropout + Add & Norm operation, and the position-wise feedforward network + Dropout + Add & Norm operation.

        """
        residual = x

        ### Implement encoder self-attention, dropout, and then the Add & Norm opertion
        ##### YOUR CODE STARTS HERE #####

        x = self.self_attn(x, x, x, key_padding_mask=self_attn_padding_mask)
        x = self.dropout(x)
        x = self.self_attn_layer_norm(x + residual)
        #### YOUR CODE ENDS HERE #####

        ## Implement encoder position-wise feedforward, dropout, and then the Add & Norm opertion
        #### YOUR CODE STARTS HERE #####

        residual = x

        x = self.fc2(self.activation_fn(self.fc1(x)))

        x = self.dropout(x)

        x = self.final_layer_norm(x + residual)


        ##### YOUR CODE ENDS HERE #####

        return x
    
## The TransformerEncoder class has been implemented for you. Do NOT modify ANY of the code in this class!
class TransformerEncoder(nn.Module):
    """
    Stacks the Transformer Encoder Layer implemented earlier together to form a Transformer Encoder.

    """
    def __init__(
            self,
            num_layers, 
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout_prob = 0.1,
            output_layer_size = None
        ):
        """
        Initialize the TransformerEncoder. Defines an nn.ModuleList of TransformerEncoderLayer, and an optional output layer
        Input:
            num_layers (int) - number of encoder layers in the TransformerEncoder

            embedding_dim (int) - dimension of the Transformer encoder and the Transformer encoder layer(aka d_model)

            ffn_embedding_dim (int) - inner dimension of the position-wise feedforward network in the TransformerEncoderLayer (size of W_1 and b_1 in the paper)

            num_attention_heads (int) - number of attention heads in the encoder self attention
            dropout_prob (float) - dropout probability between 0 and 1 for the dropout module in the TransformerEncoderLayer

            output_layer_size (None/int): if it is not None, then it is the size of the output layer
        """
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(embedding_dim=embedding_dim,
                                                             ffn_embedding_dim=ffn_embedding_dim,num_attention_heads=num_attention_heads,
                                                             dropout_prob=dropout_prob,
                                                             ) for _ in range(num_layers)])
        self.output_layer = None
        if output_layer_size is not None:
            self.output_layer = nn.Linear(embedding_dim, output_layer_size)

    def forward(self, x, encoder_padding_mask=None):
        """
        Applies the encoder layers in self.layers one by one, followed by an optional output layer if it exists

        Input:
            x (torch.Tensor) - input tensor of size B x T x embedding_dim; input to the TransformerEncoderLayer's self attention mechanism

            encoder_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T, where for each encoder_padding_mask[b] for the b-th source in the batch, the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

        Output:
            x (torch.Tensor) - the Transformer encoder's output, of size B x T x embedding_dim, if output layer is None, or of size B x T x output_layer_size, if there is an output layer.

        """
        ## Iterate through the modules in nn.ModuleList and apply them one by one
        for l in self.layers:
            x = l(x, self_attn_padding_mask=encoder_padding_mask)
            
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x
    


