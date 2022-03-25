
from os import sep, set_blocking
import pdb
from pickle import NONE
from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
# from torch._C import _set_mkldnn_enabled, per_channel_affine
from torch.nn import functional as F
from torch.nn import LSTM, GRU
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_packed_sequence


def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print("Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "classification") or (task == "regression"):
        num_labels = len(data.class_names) if task == "classification" else data.labels_df.shape[1]  # dimensionality of labels
        if config['model'] == 'transformer':
            return TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                        config['num_heads'],
                                                        config['num_layers'], config['dim_feedforward'],
                                                        num_classes=num_labels,
                                                        dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                        activation=config['activation'],
                                                        norm=config['normalization_layer'], freeze=config['freeze'], task_type = task)
       
        elif (config['model'] == 'LSTM') or (config['model'] == 'GRU'):
            return TSLstmEncoderClassiregressor(config['model'], feat_dim, max_seq_len, config['d_model'],
                                                        config['num_heads'],
                                                        config['num_layers'], config['dim_feedforward'],
                                                        num_classes=num_labels, hidden_size=config['hidden_size'], batch_size=config['batch_size'],
                                                        dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                        activation=config['activation'],
                                                        norm=config['normalization_layer'], freeze=config['freeze'], gpu=config['gpu'] )
        
        elif config['model'] == 'SimpleConv':
            return TSSimpleConvEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                        config['num_layers'], num_classes=num_labels, hidden_size=config['hidden_size'],
                                                        kernel_size=config['kernel_size'], stride=config['stride'], dilation=config['dilation'],
                                                        dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                        activation=config['activation'],
                                                        norm=config['normalization_layer'], freeze=config['freeze']) 
        elif config['model'] == 'CausalConv':
            return TSCausalConvEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                        config['num_layers'], num_classes=num_labels, hidden_size=config['hidden_size'], 
                                                        kernel_size=config['kernel_size'], stride=config['stride'], isdilated=config['isdilated'],
                                                        dropout=config['dropout'], activation=config['activation'], out_channel=config['out_channel'],
                                                        norm=config['normalization_layer'], freeze=config['freeze'] )       
        elif config['model'] == 'EACausalConv':
            return TSEACausalConvEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],config['num_layers'], 
                                                        dim_feedforward=config['dim_feedforward'], num_classes=num_labels, hidden_size=config['hidden_size'], 
                                                        kernel_size=config['kernel_size'], stride=config['stride'], isdilated=config['isdilated'],
                                                        dropout=config['dropout'], activation=config['activation'], out_channel=config['out_channel'],
                                                        norm=config['normalization_layer'], freeze=config['freeze'],pos_encoding=config['pos_encoding'], alpha = config['alpha'],
                                                        beta = config['beta'], k = config['k'], v = config['v'], task_type = task)                                      
                                                       
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)

        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False,task_type = None):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.task_type = task_type
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output_reg = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output_rep = output.reshape(output_reg.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output_rep)  # (batch_size, num_classes)
   
        return output, output_rep

class TSLstmEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """
    #feat_dim dimensionality of data features
    def __init__(self, config_model, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes, hidden_size, batch_size, gpu,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSLstmEncoderClassiregressor, self).__init__()
        self.config_model = config_model
        self.max_len = max_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gpu = gpu
        self.project_inp = nn.Linear(feat_dim, d_model)

        if(config_model == "LSTM"):
            self.TSLstm = LSTM(feat_dim, self.hidden_size, self.num_layers)
        elif(config_model == "GRU"):
            self.TSGru = GRU(feat_dim, self.hidden_size, self.num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(hidden_size, max_len, num_classes)

    def build_output_module(self, hidden_size, max_len, num_classes):
        output_layer = nn.Linear(hidden_size * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        X[~padding_masks]=0

        inp = X.permute(1, 0, 2)
        
        bs= inp.size(1)
       
        h0 = torch.randn(self.num_layers, bs, self.hidden_size).cuda()
       
        if self.config_model == "LSTM":
            hc = torch.randn(self.num_layers, bs, self.hidden_size).cuda()
            output, (hn,cn) = self.TSLstm(inp, (h0, hc)) #(seq_len, batch_size, feat_dim)
        elif self.config_model == "GRU":
            output, hn = self.TSGru(inp, h0) #(seq_len, batch_size, feat_dim)
        output = output.permute(1, 0, 2)

        output = self.act(output)  
        
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output_rep = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * hidden_size)
        output = self.output_layer(output_rep)  # (batch_size, num_classes)

        return output, output_rep


class TSSimpleConvEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """
    #feat_dim dimensionality of data features
    def __init__(self, feat_dim, max_len, d_model, num_layers, num_classes, hidden_size, kernel_size, stride, dilation,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSSimpleConvEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.dilation = dilation
        self.TSConv1 = nn.Conv1d(feat_dim, d_model, kernel_size, stride, dilation=self.dilation, padding= 'same') 
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, feat_dim_out, max_len, num_classes):
        output_layer = nn.Linear(feat_dim_out * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        X[~padding_masks]=0
        inp = X.permute(0, 2, 1)
        output = self.TSConv1(inp)  #(batch_size, feat_dim_out, seq_len_out)
        output = output.permute(0, 2, 1)  #(batch_size, seq_len_out, feat_dim_out)

        output = self.act(output) 
        output = self.dropout1(output) 
        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * hidden_size)
        output = self.output_layer(output)  # (batch_size, num_classes)
        return output, None


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class Chomp2d(torch.nn.Module):
    """
    Removes the last elements of a time series.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.
    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

class EAAugmentedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding,
                 alpha,beta, k, v, Nh, att_downsample, with_att_conv,
                #  alpha=1.0, beta=0.5, relative=True):
                   relative=True):
        super(EAAugmentedConv1d, self).__init__()
        self.dk = int(out_channels * k)
        self.dv = int(out_channels * v)
        self.K = k
        self.Nh = Nh  #num_head
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.att_downsample = att_downsample  
        self.with_att_conv = with_att_conv
        self.alpha = alpha
        self.beta = beta
        self.relative = relative

        assert self.dk //self.Nh  or self.dk == 0
       
        self.conv = torch.nn.utils.weight_norm(torch.nn.Conv1d(self.in_channels,
                                  self.out_channels - self.dv,
                                  self.kernel_size,
                                  dilation = dilation,
                                  padding = padding
                                 )) if k<1 else None

    
        self.pool_input = torch.nn.Conv1d(in_channels , in_channels, 
                                kernel_size, dilation = dilation, padding=padding,bias=False) if k > 0 else None
        self.chomp = Chomp1d(padding) if k < 1 else None

        self.pool_input_att = torch.nn.Conv2d(Nh , Nh , 
                                kernel_size, dilation = dilation, padding=padding,bias=False) if k > 0 else None

        #att_downsample always false
        if att_downsample and k > 0:
            self.pool_x = torch.nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
            self.upsample_x = torch.nn.Upsample(scale_factor=2, mode= 'linear')
            self.pool_att = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.upsample_att = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        
        self.qkv_conv = torch.nn.Conv1d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1, bias=True)if k > 0 else None

        
        if with_att_conv and k > 0 :
            self.att_conv = nn.Sequential(
                    nn.Conv2d(Nh, Nh, 3, padding=1, bias=True),
                    nn.LeakyReLU())
            self.att_conv2 = nn.Sequential(
                    nn.Conv2d(2*Nh, Nh, 3, padding=1, bias=True),
                    nn.LeakyReLU())

    def forward(self, x, prev_att):
        # conv_out
        # (batch_size, out_channels, seq_len)   
        
        # assert self.dk //self.Nh  or self.dk == 0
        conv_out = None if self.conv is None else self.conv(x) 
        # chomp first  
        if self.K < 1 and self.padding == 0:
            conv_out = conv_out
        else:
          conv_out = None if self.chomp is None  else self.chomp(conv_out)

        if(self.dk > 0):

            if self.att_downsample:
                x = self.pool_x(x)
                if prev_att is not None:
                    prev_att = self.pool_att(prev_att)
            batch, _, seq_len = x.size()
        
            # dvh = dv / Nh, dkh = dk / Nh
            # q, k, v
            # (batch_size, Nh, dk // Nh, seq_len) or (batch num_head dimention_for_q/k/v seq_len)
        
            q, k, v = self.compute_qkv(x, self.dk, self.dv, self.Nh)        
            # logits (batch  num_head  seq_len  seq_len)
            logits = torch.matmul(q.transpose(2, 3), k)
       
            if self.relative:
                rel_logits = self.relative_logits(q)
                logits += rel_logits

            #prev_att -> batch  num_head  seq_len  seq_len
            if self.with_att_conv:
                att_matrix = (1 - self.beta) * logits + self.beta * prev_att
                logits = self.att_conv(att_matrix)
                logits = self.alpha * logits + (1 - self.alpha) * att_matrix

            weights = F.softmax(logits, dim=-1)

            attn_out = torch.matmul(weights, v.transpose(2, 3))
            # (batch, Nh, dvh-> dimention of v, seq_len)
            attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, seq_len))
    
            # combine_heads_2d
            # (batch, out_channels, seq_len)
            attn_out = self.combine_heads_2d(attn_out)
            if self.att_downsample:
                attn_out = self.upsample_x(attn_out)
                logits = self.upsample_att(logits)

            if conv_out is not None:
                output = torch.cat((conv_out, attn_out), dim=1)
            else:
                output = attn_out

            return output, logits

        else:
            
            logits = None
            
            return conv_out, logits

    def compute_qkv(self, x, dk, dv, Nh):
        N, _, Seq_len = x.size()

        qkv = self.qkv_conv(x)

        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
 
        q = self.split_heads(q, Nh)
        k = self.split_heads(k, Nh)
        v = self.split_heads(v, Nh)

        dkh = dk // Nh

        q = q * (dkh ** -0.5)
        
        return q, k, v

    def split_heads(self, x, Nh):
        batch, channels, seq_len = x.size()
        ret_shape = (batch, Nh, channels // Nh, seq_len)
        split = torch.reshape(x, ret_shape)

        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, seq_len = x.size()
        ret_shape = (batch, Nh * dv, seq_len)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        #  q     batch  num_head  q_dimension seq_len
        import pdb; 
        B, Nh, dk, seq_len = q.size()
        q = torch.transpose(q, 2, 3)
 
        key_rel = nn.Parameter(
                torch.randn((2 * seq_len - 1, self.dk // Nh), requires_grad=True).cuda())

        rel_logits = self.relative_logits_1d(q, key_rel, seq_len, Nh)
 
        return rel_logits

    def relative_logits_1d(self, q, rel_k, seq_len, Nh):
        rel_logits = torch.einsum('bhld,md->bhlm', q, rel_k)

        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, seq_len, seq_len))

        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)
 
        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]

        return final_x

        
class EACausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels,dim_feedforward, kernel_size, dilation, layer_idx,  alpha, beta, k , v,dropout,
                 final=False):
        super(EACausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        self.layer_idx = layer_idx
        padding = (kernel_size - 1) * dilation
        with_att_conv = True
        att_downsample = False
        if layer_idx == 0 :
            with_att_conv = False
       
        self.conv1 = EAAugmentedConv1d(
            in_channels= in_channels, out_channels= out_channels,
                    # kernel_size= kernel_size,  k = 0.25,  v = 0.25, dilation  = dilation, padding=padding,
                    kernel_size= kernel_size, alpha = alpha, beta = beta,k = k,  v = v, dilation  = dilation, padding=padding,
                    Nh = 8,  att_downsample= att_downsample,  with_att_conv= with_att_conv
        )
        # The truncation makes the convolution causal
        self.chomp1 = Chomp1d(padding)
        self.chomp3 = Chomp2d(padding) if k > 0 else None

        self.linear1 = Linear(out_channels, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, out_channels)
        self.linear3 = Linear(out_channels, dim_feedforward)
        self.linear4 = Linear(dim_feedforward, out_channels)

        self.relu1 = torch.nn.LeakyReLU()
        self.relu2 = torch.nn.LeakyReLU()
        self.relu3 = torch.nn.LeakyReLU() if k > 0 else None
        self.relu4 = torch.nn.LeakyReLU() if k > 0 else None

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout) if k > 0 else None
        self.dropout4 = Dropout(dropout) if k > 0 else None
        self.norm1 = BatchNorm1d(out_channels, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(out_channels, eps=1e-5) if k<1 else None
        self.norm3 = BatchNorm1d(out_channels, eps=1e-5)  if k<1 else None# normalizes each feature across batch samples and time steps


        # Second causal convolution
        self.conv2 = EAAugmentedConv1d(
            in_channels= out_channels, out_channels= out_channels,
            # kernel_size= kernel_size,  k = 0.25,  v = 0.25, dilation  = dilation, padding=padding,
            kernel_size= kernel_size, alpha = alpha, beta = beta,k = k,  v = v,  dilation  = dilation, padding=padding,
            Nh = 8,  att_downsample= att_downsample,  with_att_conv= with_att_conv
        )
        self.chomp2 = Chomp1d(padding)
        self.chomp4 = Chomp2d(padding) if k > 0 else None
        
        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None


        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None
    def forward(self, x, prev_att):

    
        out , att = self.conv1(x, prev_att)
        out = out if self.norm2 is None else self.norm2(out)
        
        out = out.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        out = self.linear2(self.dropout1(self.relu1(self.linear1(out))))
        out = out.permute(1, 2, 0)
    
        att = att if self.relu3 is None else self.relu3(att)
        att = att if self.dropout3 is None else self.dropout3(att)

        out , att = self.conv2(out, att)     
        out = out if self.norm3 is None else self.norm3(out)

        out = out.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        out = self.linear4(self.dropout2(self.relu2(self.linear3(out))))
        out = out.permute(1, 2, 0)
       
        att = att if self.relu4 is None else self.relu4(att)
        att = att if self.dropout4 is None else self.dropout4(att)

        res = x if self.upordownsample is None else self.upordownsample(x)

        if self.relu is None:
            return self.norm1(out + res),att
        else:
            return self.norm1(self.relu(out + res)), att



class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class MultiInputsSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class EACausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels,dim_feedforward, depth, out_channels,
                 kernel_size, isdilated, alpha, beta, k , v,dropout):
        super(EACausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [EACausalConvolutionBlock(
                in_channels_block, channels,dim_feedforward, kernel_size, dilation_size, i,  alpha, beta, k , v, dropout
            )]
            if isdilated == "TRUE":
                dilation_size *= 2  # Doubles the dilation size at each step
            elif isdilated == "FALSE":
                dilation_size = 1

        # Last layer
        layers += [EACausalConvolutionBlock(
            channels, out_channels,dim_feedforward, kernel_size, dilation_size, i+1 ,  alpha, beta, k , v, dropout
        )]

        self.network = MultiInputsSequential(*layers)
  

    def forward(self, x, prev_att):

        return self.network(x, prev_att)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).
    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size, isdilated):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            if isdilated == "TRUE":
                dilation_size *= 2  # Doubles the dilation size at each step
            elif isdilated == "FALSE":
                dilation_size = 1

        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
   
        return self.network(x)

class EACausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels,dim_feedforward, depth, reduced_size,
                 out_channels, kernel_size, isdilated, alpha, beta, k , v,dropout):
        super(EACausalCNNEncoder, self).__init__()
        self.ea_causal_cnn = EACausalCNN(
            in_channels, channels,dim_feedforward, depth, reduced_size, kernel_size, isdilated, alpha, beta, k , v, dropout
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        
    def forward(self, x, prev_att):
        
        return self.ea_causal_cnn(x, prev_att)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).
    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size, isdilated):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size, isdilated
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
       
        self.network = causal_cnn

    def forward(self, x):
       
        return self.network(x)
class TSEACausalConvEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """
    #feat_dim dimensionality of data features
    def __init__(self, feat_dim, max_len, d_model, num_layers, num_classes, hidden_size, kernel_size, stride, isdilated, out_channel,dim_feedforward,
                alpha, beta, k , v,pos_encoding = 'learnable',
                 dropout=0.1, activation='gelu', norm='BatchNorm', freeze=False,  task_type = None):
        super(TSEACausalConvEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.isdilated = isdilated
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)
        self.task_type = task_type


        """
        Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
        batch size, `C` is the number of input channels, and `L` is the length of
        the input. Outputs a three-dimensional tensor (`B`, `C`).
        @param in_channels Number of input channels.
        @param channels Number of channels manipulated in the causal CNN.
        @param depth Depth of the causal CNN.
        @param reduced_size Fixed length to which the output time series of the
            causal CNN is reduced.
        @param out_channels Number of output channels.
        @param kernel_size Kernel size of the applied non-residual convolutions.
        """

        self.TSEACausalCnn = EACausalCNNEncoder(d_model, hidden_size,dim_feedforward, num_layers, reduced_size=d_model, out_channels=out_channel, kernel_size=kernel_size, isdilated=isdilated,
           alpha = alpha, beta = beta, k = k, v = v ,dropout=dropout)


    def build_output_module(self, feat_dim_out, max_len, num_classes):
        output_layer = nn.Linear(feat_dim_out * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        X[~padding_masks]=0
        inp = self.project_inp(X) * math.sqrt(
            self.d_model) 
        inp = self.pos_enc(inp)
        inp = inp.permute(0, 2, 1)
        output, att = self.TSEACausalCnn(inp, None)  #(batch_size, feat_dim_out, seq_len_out)
       
        output = output.permute(0, 2, 1)  #(batch_size, seq_len_out, feat_dim_out)
        output = self.act(output) 

        output = self.dropout1(output) 

  
        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output_rep = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * hidden_size)
        
        output = self.output_layer(output_rep)  # (batch_size, num_classes)


        return output, output_rep

class TSCausalConvEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """
    #feat_dim dimensionality of data features
    def __init__(self, feat_dim, max_len, d_model, num_layers, num_classes, hidden_size, kernel_size, stride, isdilated, out_channel,
                 dropout=0.1, activation='gelu', norm='BatchNorm', freeze=False):
        super(TSCausalConvEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.isdilated = isdilated

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)


        """
        Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
        batch size, `C` is the number of input channels, and `L` is the length of
        the input. Outputs a three-dimensional tensor (`B`, `C`).
        @param in_channels Number of input channels.
        @param channels Number of channels manipulated in the causal CNN.
        @param depth Depth of the causal CNN.
        @param reduced_size Fixed length to which the output time series of the
            causal CNN is reduced.
        @param out_channels Number of output channels.
        @param kernel_size Kernel size of the applied non-residual convolutions.
        """
        self.TSCausalCnn =  CausalCNNEncoder(self.feat_dim, hidden_size, num_layers, reduced_size=d_model, out_channels=out_channel, kernel_size=kernel_size, isdilated=isdilated)
    def build_output_module(self, feat_dim_out, max_len, num_classes):
        output_layer = nn.Linear(feat_dim_out * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        X[~padding_masks]=0
        inp = X.permute(0, 2, 1)
        output = self.TSCausalCnn(inp)  #(batch_size, feat_dim_out, seq_len_out)
       
        output = output.permute(0, 2, 1)  #(batch_size, seq_len_out, feat_dim_out)
        output = self.act(output) 

        output = self.dropout1(output) 
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output_rep = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * hidden_size)
       
        output = self.output_layer(output_rep)  # (batch_size, num_classes)
        return output, output_rep
