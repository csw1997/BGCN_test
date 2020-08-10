import torch
import torch.nn as nn
from torch.nn import functional as F

class ArgumentClassify(nn.Module):
    def __init__(self, in_features, out_features):
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
        super(ArgumentClassify, self).__init__()
        self.argument_classify_step1 = nn.Linear(in_features, 1000)
        self.argument_classify_step2 = nn.Linear(1000, 500)
        self.argument_classify_step3 = nn.Linear(500, out_features)

    def forward(self, word_rep_seqs):
        x = self.argument_classify_step1(word_rep_seqs)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.argument_classify_step2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.argument_classify_step3(x)
        return x
