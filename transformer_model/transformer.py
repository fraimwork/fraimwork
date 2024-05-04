from encoder import TransformerEncoder
from decoder import TransformerDecoder
from pe import PositionalEncoding

import torch
from torch import nn

## The below helper function has been given to you for convenience
def subsequent_mask(size, device="cpu", dtype=torch.long):
    """
    Create mask for subsequent steps size x size; this may be useful for creating decoder attention masks for parallel auto-regressive training.
    subsequent_mask(3) will return a torch.tensor of dtype, on the device, as:
    [[0, 1, 1],
    [0, 0, 1],
    [0, 0, 0]]
    Input:
        size (int) - size of mask
        device (str/torch.Tensor.device): where the return tensor will be located, say, "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.dtype) - result dtype
    Output:
        torch.Tensor - mask for subsequent steps with shape as size x size, of "dtype", on the "device"
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return 1 - torch.tril(ret, out=ret)

## The below helper function has been given to you for convenience
def length_to_padding_mask(lengths, device="cpu", dtype=torch.long):
    """
    Convert a list/1D tensor/1D array of length in to padding masks used by the encoder and the decoder's attention mechanism

    For example, length_to_padding_mask([3, 4, 5]) will return a torch.tensor of dtype, on the device, as:
    [[0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0]]

    Input:
        lengths (List/torch.Tensor/np.array) - a 1D iterable List/torch.Tensor/np.array 
        device (str/torch.Tensor.device): where the return tensor will be located, say, "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.dtype) - result dtype

    Output:
        ret (torch.Tensor) - a padding mask of size len(lengths) x max(lengths), with non-zero positions indicating locations out of bounds , of "dtype", on the "device"
    """

    ret = torch.zeros(len(lengths), max(lengths), device=device, dtype=dtype)
    for idx in range(len(lengths)):
        ret[idx, lengths[idx]:] = 1
    return ret

class Transformer(nn.Module):
    """
    Implements the Transformer architecture for a discrete sequence-to-sequence task using the previously implemented PositionalEncoding, TransformerEncoder, and TransformerDecoder, as well as other existing torch.nn modules.
    """
    ## The __init__ method has been given to you. Do NOT modify ANY of the code in this method!
    def __init__(self, src_vocab_size, tgt_vocab_size, sos_idx, eos_idx, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout_prob):
        '''
        Initialize the Transformer. Defines some model parameters and hyperparameters
        Input:
            src_vocab_size (int) - the size of the vocabulary for the source discrete sequences (i.e., encoder-side)
            tgt_vocab_size (int) - the size of the vocabulary for the target discrete sequences (i.e., decoder-side)
            sos_idx (int) - the start of sentence discrete index at the start of every source and target sequence
            eos_idx (int) - the end of sentence discrete index at the end of every source and target sequence
            d_model (int) - model dimension used by both the Transformer encoder and decoder (aka embedding_dim)
            num_heads (int) - number of attention heads used by both the Transformer encoder and decoder
            num_layers (int) - number of encoder/decoder layers used by both the Transformer encoder and decoder
            d_ff (int) - inner dimension of the position-wise feedforward network used by both the Transformer encoder and decoder (size of W_1 and b_1 in the paper; aka ffn_embedding_dim)
            max_seq_length (int) - maximum input/output sequence length for PositionalEncoding, and supported by the Transformer encoder/decoder
            dropout_prob (float) - dropout probability between 0 and 1 for the dropout modules used throughout the Transformer model implementation
        '''

        super(Transformer, self).__init__()

        # Encoder input embedding
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)

        # Decoder input embedding
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding used by both the Transformer encoder and the decoder
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_length)

        # Transformer Encoder
        self.encoder = TransformerEncoder(
            num_layers = num_layers, 
            embedding_dim = d_model,
            ffn_embedding_dim = d_ff,
            num_attention_heads = num_heads,
            dropout_prob = dropout_prob,
            output_layer_size = None
        )

        # Transformer Decoder
        self.decoder = TransformerDecoder(
            num_layers = num_layers, 
            embedding_dim = d_model,
            ffn_embedding_dim = d_ff,
            num_attention_heads = num_heads,
            dropout_prob = dropout_prob,
            no_encoder_attn = False,
            output_layer_size = tgt_vocab_size
        )

        # Need to save sos_idx and eos_idx for auto-regressive decoding
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        # additional dropout after positional encoding
        self.dropout = nn.Dropout(dropout_prob)

    ## You need to fill in the missing code in the forward_encoder method below
    def forward_encoder(self, src, src_lengths):
        """
        Applies the Transformer encoder to src, where each sequence in src has been padded to the max(src_lengths)

        Input:
            src (torch.Tensor) - Encoder's input tensor of size B x T_e x d_model

            src_lengths (torch.Tensor) - A 1D iterable of Long/Int of length B, where the b-th length in src_lengths corresponds to the actual length of src[b] (beyond that is the pre-padded region); T_e = max(src_lengths)

        Output:
            enc_output (torch.Tensor) - the Transformer encoder's output, of size B x T_e x d_model

            src_padding_mask (torch.Tensor) - the encoder_padding_mask/key_padding_mask used by the Transformer encoder's self-attention; this should be created from src_lengths

        """


        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        ##### YOUR CODE STARTS HERE #####
        ## Use the given mask creator functions to create encoder padding mask
        src_padding_mask = length_to_padding_mask(src_lengths)
        
        ## Pass the encoder input into the encoder, with the correct mask(s)
        enc_output = self.encoder(src_embedded, encoder_padding_mask = src_padding_mask)

        ##### YOUR CODE ENDS HERE #####


        return enc_output, src_padding_mask

    ## You need to fill in the missing code in the forward_decoder method below
    def forward_decoder(self, enc_output, src_padding_mask, tgt, tgt_lengths):
        """
        Applies the Transformer decoder to tgt and enc_output (possibly as used during training to obtain the next token prediction under teacher-forcing), where sequences in enc_output are associated with src_padding_mask, and each sequence in tgt has been padded to the max(tgt_lengths)
        Input:
            enc_output (torch.Tensor) - the Transformer encoder's output, of size B x T_e x d_model
            src_padding_mask (torch.Tensor) - the encoder_padding_mask/key_padding_mask associated with enc_output. It is a torch.IntTensor/torch.LongTensor of size B x T_e, where for each src_padding_mask[b] for the b-th source in the batched tensor enc_output[b], the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence
            tgt (torch.Tensor) - Decoder's input tensor of size B x T_d x d_model
            tgt_lengths (torch.Tensor) - A 1D iterable of Long/Int of length B, where the b-th length in tgt_lengths corresponds to the actual length of tgt[b] (beyond that is the pre-padded region); T_d = max(tgt_lengths)
        Output:
            dec_output (torch.Tensor) - the Transformer's final output from the decoder of size B x T_d x tgt_vocab_size, as there is an output layer.
        """
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        decoder_padding_mask = length_to_padding_mask(tgt_lengths)
        decoder_attention_mask = subsequent_mask(max(tgt_lengths)).unsqueeze(0)
        return self.decoder(tgt_embedded, decoder_padding_mask = decoder_padding_mask,decoder_attention_mask = decoder_attention_mask, encoder_out = enc_output, encoder_padding_mask = src_padding_mask)

    ## The forward method has been given to you. Do NOT modify ANY of the code in this method!
    def forward(self, src, tgt, src_lengths, tgt_lengths):

        """
        Applies the entire Transformer encoder-decoder to src and target, possibly as used during training to obtain the next token prediction under teacher-forcing; each sequence in src has been padded to the max(src_lengths); each sequence in tgt has been padded to the max(tgt_lengths)

        Input:
            src (torch.Tensor) - Encoder's input tensor of size B x T_e x d_model

            src_lengths (torch.Tensor) - A 1D iterable of Long/Int of length B, where the b-th length in src_lengths corresponds to the actual length of src[b] (beyond that is the pre-padded region); T_e = max(src_lengths)

            tgt (torch.Tensor) - Decoder's input tensor of size B x T_d x d_model

            tgt_lengths (torch.Tensor) - A 1D iterable of Long/Int of length B, where the b-th length in tgt_lengths corresponds to the actual length of tgt[b] (beyond that is the pre-padded region); T_d = max(tgt_lengths)

        Output:
            dec_output (torch.Tensor) - the Transformer's final output from the decoder of size B x T_d x tgt_vocab_size, as there is an output layer.
        """

        ## Get encoder output and encoder padding mask
        enc_output, src_padding_mask = self.forward_encoder(src, src_lengths)

        ## Get decoder output
        dec_output = self.forward_decoder(enc_output, src_padding_mask, tgt, tgt_lengths)

        return dec_output
    
    ## You need to fill out (probably about 3 lines of code) in the inference method below; do not modify anything else
    def inference(self, src, src_lengths, max_output_length):
        """
        Applies the entire Transformer encoder-decoder to src and target, possibly as used during inference to auto-regressively obtain the next token; each sequence in src has been padded to the max(src_lengths)
        Input:
            src (torch.Tensor) - Encoder's input tensor of size B x T_e x d_model
            src_lengths (torch.Tensor) - A 1D iterable of Long/Int of length B, where the b-th length in src_lengths corresponds to the actual length of src[b] (beyond that is the pre-padded region); T_e = max(src_lengths)
            
        Output:
            decoded_list (List(torch.Tensor) - a list of auto-regressively obtained decoder output token predictions; the b-th item of the decoded_list should be the output from src[b], and each of the sequence predictions in decoded_list is of a possibly different length. 
            decoder_layer_cache_list (List(List(torch.Tensor))) - a list of decoder_layer_cache; the b-th item of the decoded_layer_cache_list should be the decoder_layer_cache for the src[b], which itself is a list of torch.Tensor, as returned by self.decoder.forward_one_step_ec (see the function definition there for more details) when the auto-regressive inference ends for src[b].
        """
        
        ## Get encoder output
        enc_output, _ = self.forward_encoder(src, src_lengths)

        # Prepare a list for storing the auto-regressively obtained output sequence for each input sequence; as well as a list of (a list of decoder layer caches) for grading
        decoded_list = []
        decoded_layer_cache_list = []

        # Note: as the auto-regressively obtained output sequence length can be different, we separately obtain them as opposed to a batch operation
        for item_idx in range(len(src_lengths)):

            ## Obtain the encoded input sequence
            hs = enc_output[item_idx:item_idx+1, :src_lengths[item_idx], :]

            ##### YOUR CODE STARTS HERE #####
            ## You need to initialize the tgt for the first time step for the current sequence as a torch.LongTensor of size (1, 1); note that all sequences start with the self.sos_idx


            ##### YOUR CODE ENDS HERE #####

            decoder_layer_cache = None

            for idx in range(max_output_length):
                tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

                decoder_attn_mask = subsequent_mask(idx + 1, device = hs.device).unsqueeze(0)
                decoder_output, decoder_layer_cache = self.decoder.forward_one_step_ec(
                    x = tgt_embedded,
                    decoder_attention_mask = decoder_attn_mask, 
                    encoder_out = hs, 
                    cache = decoder_layer_cache
                )

                ## Obtain the discrete token for the single frame that have been returned
                decoder_next_pred = decoder_output.argmax(dim=-1)[0]
                ## Concatenate the current discrete token with previously obtained output token sequence from the previous time step

                ##### YOUR CODE STARTS HERE #####





                ##### YOUR CODE ENDS HERE #####
            
            ## append the finished output tokens and states for the current input to return lists
            decoded_list.append(tgt.squeeze(0))
            decoded_layer_cache_list.append(decoder_layer_cache)

        return decoded_list, decoded_layer_cache_list