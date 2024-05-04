import torch
from torch import nn

from mha import MultiHeadAttention
from pe import PositionalEncoding

class TransformerDecoderLayer(nn.Module):
    """
    Implements a single Transformer Decoder Layer as a torch.nn.Module, using the MultiHeadAttention mechanism implemented earlier and other existing modules in torch.nn
    """
    ## The __init__ method has been given to you. Do NOT modify ANY of the code in this method!
    def __init__(
        self, 
        embedding_dim, 
        ffn_embedding_dim,
        num_attention_heads,
        dropout_prob=0.1,
        no_encoder_attn=False,
    ):
        '''
        Initialize the TransformerDecoderLayer. Defines some model parameters and hyperparameters

        Input:
            embedding_dim (int) - dimension of the Transformer decoder layer (aka d_model). Note that for convenience, it is set to the same as the Transformer encoder layers.

            ffn_embedding_dim (int) - inner dimension of the position-wise feedforward network (size of W_1 and b_1 in the paper)

            num_attention_heads (int) - number of attention heads in the encoder self attention

            dropout_prob (float) - dropout probability between 0 and 1 for the dropout module

            no_encoder_attn (bool) - whether the decoder layer is standalone (no_encoder_attn = True; auto-regressive modeling only), or there is encoder output for it to calculate attention (no_encoder_attn = False; encoder-decoder modeling)
        '''
        super().__init__()
        self.embedding_dim = embedding_dim

        ## dropout layer for regularization
        self.dropout = nn.Dropout(dropout_prob)

        ## decoder self attention mechanism
        self.self_attn =  MultiHeadAttention(
            self.embedding_dim,
            num_attention_heads,
        )

        ## layer normalization for decoder self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            ## if an encoder-decoder architecture is built, we also need encoder-decoder attention
            ## for simplicity, we assume encoder and decoder has the same embedding dimension (aka d_model)
            self.encoder_attn = MultiHeadAttention(
            self.embedding_dim,
            num_attention_heads,
            )
            ## layer normalization for encoder-decoder attention layer
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embedding_dim)

        ## activation_fn, fc1 and fc2 are for implementing position-wise feed-forward network
        self.activation_fn = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim, bias = True)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim, bias = True)

        ## layer normalization for position-wise feed-forward network
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    ## You need to fill in the missing code in the forward method below
    def forward(
        self,
        x,
        encoder_out = None,
        encoder_padding_mask = None,
        self_attn_padding_mask = None,
        self_attn_mask = None,
    ):
        """
        Applies the self attention module + Dropout + Add & Norm operation, the encoder-decoder attention + Dropout + Add & Norm operation (if self.encoder_attn is not None), and the position-wise feedforward network + Dropout + Add & Norm operation. Note that LayerNorm is applied after the self-attention operation, after the encoder-decoder attention operation and another time after the ffn modules, similar to the original Transformer implementation.

        Input:
            x (torch.Tensor) - input tensor of size B x T_d x embedding_dim from the decoder input or the previous encoder layer, where T_d is the decoder's temporal dimension; serves as input to the TransformerDecoderLayer's self attention mechanism.

            encoder_out (None/torch.Tensor) - If it is not None, then it is the output from the TransformerEncoder as a tensor of size B x T_e x embedding_dim, where T_e is the encoder's temporal dimension; serves as part of the input to the TransformerDecoderLayer's self attention mechanism (hint: which part?).
            
            encoder_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_e, where for each encoder_padding_mask[b] for the b-th source in the batched tensor encoder_out[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

            self_attn_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_d, where for each self_attn_padding_mask[b] for the b-th source in the batched tensor x[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

            self_attn_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size 1 x T_d x T_d or B x T_d x T_d. It is used for decoder self-attention to enforce auto-regressive property during parallel training; suppose the maximum length of a batch is 5, then the attention_mask for any input in the batch will look like this for each input of the batch.
            0 1 1 1 1
            0 0 1 1 1
            0 0 0 1 1
            0 0 0 0 1
            0 0 0 0 0
            The non-zero positions will be ignored and disallowed for attention while the zero positions will be allowed for attention.

        Output:
            x (torch.Tensor) - the decoder layer's output, of size B x T_d x embedding_dim, after the self attention module + Dropout + Add & Norm operation, the encoder-decoder attention + Dropout + Add & Norm operation (if self.encoder_attn is not None), and the position-wise feedforward network + Dropout + Add & Norm operation.

        """
        ## Implement decoder self-attention; dropout comes after self_attn
        ##### YOUR CODE STARTS HERE #####
        xs = self.self_attn_layer_norm(x + self.dropout(self.self_attn(x,x,x, key_padding_mask = self_attn_padding_mask,attention_mask = self_attn_mask)))


        ##### YOUR CODE ENDS HERE #####

        ## Implement encoder-decoder attention; dropout comes after encoder_attn
        if self.encoder_attn is not None:
            ### REMOVE THIS AFTER YOU FINISHED THE CODE IN THIS IF STATEMENT!
            ##### YOUR CODE STARTS HERE #####
            xd = self.encoder_attn_layer_norm(xs + self.dropout(self.encoder_attn(xs,encoder_out, encoder_out, key_padding_mask  = encoder_padding_mask)  ))


             ##### YOUR CODE ENDS HERE #####
            
        ## Implement position-wise feed-forward network (hint: it should be the same as in the encoder layer)
        ##### YOUR CODE STARTS HERE #####
        xf = self.final_layer_norm(xd + self.dropout(self.fc2(self.activation_fn(self.fc1(xd))) )  )




        ##### YOUR CODE ENDS HERE #####
        return xf
    
    ## You need to fill in the missing code in the forward_one_step_ec method below (for extra credit)
    def forward_one_step_ec(
        self,
        x,
        encoder_out = None,
        encoder_padding_mask = None,
        self_attn_padding_mask = None,
        self_attn_mask = None,
        cache = None,
    ):
        """
        Applies the self attention module + Dropout + Add & Norm operation, the encoder-decoder attention + Dropout + Add & Norm operation (if self.encoder_attn is not None), and the position-wise feedforward network + Dropout + Add & Norm operation, but for just a single time step at the last time step. Note that LayerNorm is applied after the self-attention operation, after the encoder-decoder attention operation and another time after the ffn modules, similar to the original Transformer implementation.

        Input:
            x (torch.Tensor) - input tensor of size B x T_d x embedding_dim from the decoder input or the previous encoder layer, where T_d is the decoder's temporal dimension; serves as input to the TransformerDecoderLayer's self attention mechanism. You need to correctly slice x in the function below so that it is only calculating a one-step (one frame in length in the temporal dimension) decoder output of the last time step.

            encoder_out (None/torch.Tensor) - If it is not None, then it is the output from the TransformerEncoder as a tensor of size B x T_e x embedding_dim, where T_e is the encoder's temporal dimension; serves as part of the input to the TransformerDecoderLayer's self attention mechanism (hint: which part?).
            
            encoder_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_e, where for each encoder_padding_mask[b] for the b-th source in the batched tensor encoder_out[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

            self_attn_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_d, where for each self_attn_padding_mask[b] for the b-th source in the batched tensor x[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence. If it is not None, then you need to correctly slice it in the function below so that it is corresponds to the self_attn_padding_mask for calculating a one-step (one frame in length in the temporal dimension) decoder output of the last time step.

            self_attn_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size 1 x T_d x T_d or B x T_d x T_d. It is used for decoder self-attention to enforce auto-regressive property during parallel training; suppose the maximum length of a batch is 5, then the attention_mask for any input in the batch will look like this for each input of the batch.
            0 1 1 1 1
            0 0 1 1 1
            0 0 0 1 1
            0 0 0 0 1
            0 0 0 0 0
            The non-zero positions will be ignored and disallowed for attention while the zero positions will be allowed for attention. If it is not None, then you need to correctly slice it in the function below so that it is corresponds to the self_attn_padding_mask for calculating a one-step (one frame in length in the temporal dimension) decoder output of the last time step.

            cache (torch.Tensor) - the output from this decoder layer previously computed up until the previous time step before the last; hence it is of size B x (T_d-1) x embedding_dim. It is to be concatenated with the single time-step output calculated in this function before being returned


        Returns:
            x (torch.Tensor) - Output tensor B x T_d x embedding_dim, which is a concatenation of cache (previously computed up until the previous time step before the last) and the newly computed one-step decoder output for the last time step.
        """
        residual = x


        if cache is None:
            ag_x = x
            ag_self_attn_mask = self_attn_mask
            ag_self_attn_padding_mask = self_attn_padding_mask

        else:
            assert cache.shape == (
                x.shape[0],
                x.shape[1] - 1,
                self.embedding_dim,
            ), f"{cache.shape} == {(x.shape[0], x.shape[1] - 1, self.embedding_dim)}"
            ##### YOUR CODE STARTS HERE #####
            ## Hint: For the , you need to slice it so that it only has the last time step in the time dimension
            ## You also need to adjust the masks accordingly
 








            ##### YOUR CODE ENDS HERE #####
        
        ## Implement decoder self-attention, but the output should only contain one time step
        ##### YOUR CODE STARTS HERE #####


        ##### YOUR CODE ENDS HERE #####

        if self.encoder_attn is not None:
            pass ### REMOVE THIS AFTER YOU FINISHED THE CODE IN THIS IF STATEMENT!
            ## Implement encoder-decoder attention, but the output should only contain one time step
            ##### YOUR CODE STARTS HERE #####
        


            ##### YOUR CODE ENDS HERE #####

        ## Implement position-wise feed-forward network (hint: it should be the same as in the encoder layer)
        ##### YOUR CODE STARTS HERE #####






        ##### YOUR CODE ENDS HERE #####

        ## concatenate the new time step layer output with those from previous time steps
        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x

    
## The TransformerEncoder class has been implemented for you, except for the extra-credit part
## Do not modify anything in __init__ or in forward
class TransformerDecoder(nn.Module):
    """
    Stacks the Transformer Decoder Layer implemented earlier together to form a Transformer Decoder.

    """
    ## The __init__ method has been given to you. Do NOT modify ANY of the code in this method!
    def __init__(
        self,
        num_layers, 
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout_prob,
        no_encoder_attn = False,
        output_layer_size = None
    ):
        """
        Initialize the TransformerDecoder. Defines an nn.ModuleList of TransformerDecoderLayer, and an optional output layer
        Input:
            num_layers (int) - number of decoder layers in the TransformerDecoder

            embedding_dim (int) - dimension of the Transformer decoder and the Transformer decoder layer(aka d_model); for simplicity, it is assumed to be the same as that of the TransformerEncoder.

            ffn_embedding_dim (int) - inner dimension of the position-wise feedforward network in the TransformerDecoderLayer (size of W_1 and b_1 in the paper)

            num_attention_heads (int) - number of attention heads in the decoder self attention, as well as the encoder-decoder attention

            dropout_prob (float) - dropout probability between 0 and 1 for the dropout module in the TransformerDecoderLayer

            no_encoder_attn (bool) - whether the decoder layer is standalone (no_encoder_attn = True; auto-regressive modeling only), or there is encoder output for it to calculate attention (no_encoder_attn = False; encoder-decoder modeling)            

            output_layer_size (None/int): if it is not None, then it is the size of the output layer of the decoder
        """
        
        super().__init__()

        self.layers = nn.ModuleList([TransformerDecoderLayer(embedding_dim=embedding_dim,
                                                             ffn_embedding_dim=ffn_embedding_dim,num_attention_heads=num_attention_heads,
                                                             dropout_prob=dropout_prob,
                                                             no_encoder_attn=no_encoder_attn,
                                                             ) for _ in range(num_layers)])
        
        self.output_layer = None
        if output_layer_size is not None:
            self.output_layer = nn.Linear(embedding_dim, output_layer_size)
            
    ## The forward method has been given to you. Do NOT modify ANY of the code in this method!
    def forward(self, 
                x, 
                decoder_padding_mask = None, 
                decoder_attention_mask = None, 
                encoder_out = None, 
                encoder_padding_mask = None):
        """
        Applies the encoder layers in self.layers one by one, followed by an optional output layer if it exists

        Input:
            x (torch.Tensor) - input tensor of size B x T_d x embedding_dim; input to the TransformerDecoderLayer's self attention mechanism

            decoder_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_d, where for each decoder_padding_mask[b] for the b-th source in the batched tensor x[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

            decoder_attention_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size 1 x T_d x T_d or B x T_d x T_d. It is used for decoder self-attention to enforce auto-regressive property during parallel training; suppose the maximum length of a batch is 5, then the attention_mask for any input in the batch will look like this for each input of the batch.
            0 1 1 1 1
            0 0 1 1 1
            0 0 0 1 1
            0 0 0 0 1
            0 0 0 0 0
            The non-zero positions will be ignored and disallowed for attention while the zero positions will be allowed for attention.

            encoder_out (None/torch.Tensor) - If it is not None, then it is the output from the TransformerEncoder as a tensor of size B x T_e x embedding_dim, where T_e is the encoder's temporal dimension; serves as part of the input to the TransformerDecoderLayer's self attention mechanism (hint: which part?).

            encoder_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_e, where for each encoder_padding_mask[b] for the b-th source in the batch, the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

        Output:
            x (torch.Tensor) - the Transformer decoder's output, of size B x T_d x embedding_dim, if output layer is None, or of size B x T_d x output_layer_size, if there is an output layer.

        """
        ## Iterate through the modules in nn.ModuleList and apply them one by one
        for l in self.layers:
            x = l(x, 
                  encoder_out = encoder_out,
                  encoder_padding_mask = encoder_padding_mask,
                  self_attn_padding_mask = decoder_padding_mask,
                  self_attn_mask = decoder_attention_mask)
            
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x
    
    ## You need to fill in the missing code in the forward_one_step_ec method below (for extra credit)
    def forward_one_step_ec(self, 
                            x, 
                            decoder_padding_mask = None, 
                            decoder_attention_mask = None, 
                            encoder_out = None, 
                            encoder_padding_mask = None,
                            cache = None):
        """Forward one step.

        Input:
            x (torch.Tensor) - input tensor of size B x T_d x embedding_dim; input to the TransformerDecoderLayer's self attention mechanism

            decoder_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_d, where for each decoder_padding_mask[b] for the b-th source in the batched tensor x[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

            decoder_attention_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size 1 x T_d x T_d or B x T_d x T_d. It is used for decoder self-attention to enforce auto-regressive property during parallel training; suppose the maximum length of a batch is 5, then the attention_mask for any input in the batch will look like this for each input of the batch.
            0 1 1 1 1
            0 0 1 1 1
            0 0 0 1 1
            0 0 0 0 1
            0 0 0 0 0
            The non-zero positions will be ignored and disallowed for attention while the zero positions will be allowed for attention.

            encoder_out (None/torch.Tensor) - If it is not None, then it is the output from the TransformerEncoder as a tensor of size B x T_e x embedding_dim, where T_e is the encoder's temporal dimension; serves as part of the input to the TransformerDecoderLayer's self attention mechanism (hint: which part?).

            encoder_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_e, where for each encoder_padding_mask[b] for the b-th source in the batch, the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

            cache (None/List[torch.Tensor]) -  If it is not None, then it is a list of cache tensors of each decoder layer calculated until and including the previous time step; hence, if it is not None, then each tensor in the list is of size B x (T_d-1) x embedding_dim; the list length is equal to len(self.layers), or the number of decoder layers.

        Output:
            y (torch.Tensor) -  Output tensor from the Transformer decoder consisting of a single time step, of size B x 1 x embedding_dim, if output layer is None, or of size B x 1 x output_layer_size, if there is an output layer.

            new_cache (List[torch.Tensor]) -  List of cache tensors of each decoder layer for use by the auto-regressive decoding of the next time step; each tensor is of size B x T_d x embedding_dim; the list length is equal to len(self.layers), or the number of decoder layers.

        """

        ## Initialize an empty cache if the input cache is None (i.e., first time step)
        if cache is None:
            cache = [None] * len(self.layers)


        ##### YOUR CODE STARTS HERE #####
        ## Invoke forward_one_step_ec method for each of the decoder layers
        ## Hint: Remember to save a new version of cache for use by the next time step, in a list (it shall be returned later together with the one-frame output)










        ## Obtain the step-wise decoder output as the item in the last time step, at the last layer



        ##### YOUR CODE ENDS HERE #####

        if self.output_layer is not None:
            y = self.output_layer(y)

        return y, new_cache


    
        