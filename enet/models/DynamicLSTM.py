import numpy
import numpy as np
import torch
import torch.nn as nn


class DynamicLSTM(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Dynamic LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False
        """
        super(DynamicLSTM, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs)

    def forward(self, x, seq_lens):
        """
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort

        :param x: FloatTensor, pre-padded input sequence (batch_size, seq_len, feature_dim)
        :param x_len: numpy list, indicating corresponding actual sequence length
        :return: output, (h_n, c_n)
        - **output**: FloatTensor, packed output sequence (batch_size, seq_len, feature_dim * num_directions)
            containing the output features `(h_t)` from the last layer of the LSTM, for each t.
        - **h_n**: FloatTensor, (num_layers * num_directions, batch, hidden_size)
            containing the hidden state for `t = seq_len`
        - **c_n**: FloatTensor, (num_layers * num_directions, batch, hidden_size)
            containing the cell state for `t = seq_len`
        """
        # 1. sort
        x_sort_idx = torch.sort(seq_lens, descending=True)[1]
        x_unsort_idx = torch.sort(x_sort_idx)[1]
        seq_lens = seq_lens[x_sort_idx]
        x = x[x_sort_idx]
        # 2. pack
        x_p = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True)
        # 3. process using RNN
        out_pack, (ht, ct) = self.lstm.forward(x_p, None)
        # 4. unsort h and c
        ht = ht.detach()
        ht.transpose_(0, 1)
        ct = ht.detach()
        ct.transpose_(0, 1)
        # 5. unpack output
        out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)  # (sequence, lengths)
        out = out[0]  #
        # 6. unsort out
        out = out[x_unsort_idx]
        return out, (ht, ct)