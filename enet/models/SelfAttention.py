import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, multi_head_num, input_features, output_features):
        '''
        A single convolutional unit
        :param D: int, input feature dim
        :param H: int, hidden feature dim
        :param return_sequences: boolean, whether return sequence
        '''
        super(AttentionLayer, self).__init__()

        self.multi_head_num = multi_head_num

        self.heads = nn.ModuleList()
        for i in range(multi_head_num):
            subModuleList = nn.ModuleList([nn.Linear(input_features, output_features) for i in range(3)])
            self.heads.append(subModuleList)
            

    def calc_attention_weights(self, Q, K, seq_lens):
        '''
        Softmax with mask
        :param x: torch.FloatTensor, logits, [batch_size, seq_len]
        :param mask: torch.ByteTensor, masks for sentences, [batch_size, seq_len]
        :return: torch.FloatTensor, probabilities, [batch_size, seq_len]
        '''
        max_seq_len = Q.size(1)
        weights_masks = []
        for i in range(seq_lens.size(0)):
            weights_mask = torch.zeros(max_seq_len, 1)
            weights_mask[0:seq_lens[i], :] = 1
            weights_mask = torch.matmul(weights_mask, weights_mask.transpose(0, 1))
            weights_masks.append(weights_mask)
        weights_masks = torch.stack(weights_masks).cuda()
        attention_weights = torch.exp(torch.bmm(Q, K.transpose(-1, -2))/torch.sqrt(torch.FloatTensor([Q.size(-1)])).cuda()) * weights_masks
        attention_weights /= torch.sum(attention_weights, dim=-1, keepdim=True) + 1e-8
        assert not torch.isnan(attention_weights).any()
        return attention_weights

    def forward(self, x_text, seq_lens):
        '''
        Forward this module
        :param x_text: torch.FloatTensor, input features, [batch_size, seq_len, D]
        :param mask: torch.ByteTensor, masks for features, [batch_size, seq_len]
        :param x_attention: torch.FloatTensor, input features No. 2 to attent with x_text, [batch_size, seq_len, D]
        :return: torch.FloatTensor, output features, if return sequences, output shape is [batch, SEQ_LEN, D];
                    otherwise output shape is [batch, D]
        '''
        x_text_attentions = []
        for i in range(self.multi_head_num):
            subModuleList = self.heads[i]
            Q = subModuleList[0].forward(x_text)
            Q = F.sigmoid(Q)
            K = subModuleList[1].forward(x_text)
            K = F.sigmoid(K)
            V = subModuleList[2].forward(x_text)
            attention_weights = self.calc_attention_weights(Q, K, seq_lens)
            assert not torch.isnan(attention_weights).any()
            x_text_attention = torch.bmm(attention_weights, V)
            x_text_attentions.append(x_text_attention)

        return torch.cat(x_text_attentions, dim=-1)


if __name__ == "__main__":
    al = AttentionLayer(4, 50, 250)
    x = torch.randn(5, 10, 50)
    print(x.size())
    seq_lens = torch.tensor([10, 7, 6, 5, 9])
    y = al.forward(x, seq_lens)
