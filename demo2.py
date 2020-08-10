from torch import nn
import torch
import time
import decimal

class MyLSTM(nn.Module):
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
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs)

    def forward(self, x, state):
        return self.lstm.forward(x, state)

model = MyLSTM(2, 64, 1, batch_first=True, bidirectional=False)

epochs = 1000

x = torch.randn(5, 3, 2)

h0 = torch.zeros(1, x.size(0), 64)
c0 = torch.zeros(1, x.size(0), 64)

start = time.time()
for i in range(epochs):
    model.forward(x, (h0, c0))
end = time.time()
print('Running time: {} Seconds'.format(end-start))

x_len = torch.LongTensor([3, 3, 3, 3, 3])
x_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
start = time.time()
for i in range(epochs):
    model.forward(x_p, (h0, c0))
end = time.time()
print('Running time: {} Seconds'.format(end-start))


y = torch.randn(1, 15, 2)
h0 = torch.zeros(1, y.size(0), 64)
c0 = torch.zeros(1, y.size(0), 64)

start = time.time()
for i in range(epochs):
    model.forward(y, (h0, c0))
end = time.time()
print('Running time: {} Seconds'.format(end-start))
