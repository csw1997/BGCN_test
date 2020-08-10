import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from ipdb import set_trace
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from enet.models.GCN import GraphConvolution
from enet.configuration import consts
from enet.models.DynamicLSTM import DynamicLSTM
from enet.models.SelfAttention import AttentionLayer
from enet.models.ArgumentClassify import ArgumentClassify

class Model(nn.Module):
    def __init__(self, hps, word_embedding_weight_matrix=None):
        super(Model, self).__init__()
        self.word_dict_size = hps["dict_size"]
        self.word_embedding_size = hps["word_embedding_size"]
        self.word_embedding_weight_matrix = word_embedding_weight_matrix
        self.pos_class_num = hps["pos_class_num"]
        self.pos_embedding_size = hps["pos_embedding_size"]
        self.trigger_class_num = hps["event_class_num"]
        self.argument_class_num = hps["argument_class_num"]
        self.lstm_hidden_size = hps["lstm_hidden_size"]
        self.lstm_num_layers = hps["lstm_num_layers"]
        self.gcn_layers = hps["gcn_layers"]
        self.gcn_out_dim = hps["gcn_out_dim"]
        self.multi_head_num = hps["multi_head_num"]
        self.single_attention_out_dim = hps["single_attention_out_dim"]
        self.word_rep_size = self.word_embedding_size+self.pos_embedding_size

        # Word Embedding Layer
        # word_dict_size * word_embedding_size
        self.word_embedding = nn.Embedding(self.word_dict_size, self.word_embedding_size)
        if word_embedding_weight_matrix is not None:
            self.word_embedding.weight.data.copy_(word_embedding_weight_matrix)
            self.word_embedding.weight.requires_grad = False

        # Pos Embedding Layer
        # Pos_class_num * pos_embedding_size
        self.pos_embedding = nn.Embedding(self.pos_class_num, self.pos_embedding_size)
        
        # LSTM Encoder
        self.lstm = DynamicLSTM(self.word_rep_size, self.lstm_hidden_size, self.lstm_num_layers, batch_first=True, bidirectional=True)

        # Dropout Layer
        self.drop_out = nn.Dropout(p=0.5)
        
        # GCN Encoder
        self.gcns = nn.ModuleList()
        gcn = GraphConvolution(in_features=2*self.lstm_hidden_size,
                                   out_features=self.gcn_out_dim)
        self.gcns.append(gcn)
        for i in range(hps["gcn_layers"]-1):
            gcn = GraphConvolution(in_features=self.gcn_out_dim,
                                   out_features=self.gcn_out_dim)
            self.gcns.append(gcn)
        
        # SelfAttention Encoder
        self.sa = AttentionLayer(self.multi_head_num, self.gcn_out_dim, self.single_attention_out_dim)
        # Trigger classify layer
        self.trigger_classify = nn.Linear(self.multi_head_num*self.single_attention_out_dim, self.trigger_class_num)
        # Argument classify layer
        self.argument_classify = ArgumentClassify(self.multi_head_num*self.single_attention_out_dim+self.gcn_out_dim, self.argument_class_num)

    @staticmethod
    def get_triggers(batch_trigger_classes, word_seq_lens, id2trigger):
        batch_size = batch_trigger_classes.size(0)
        batch_triggers = []
        for i in range(batch_size):
            triggers = {}
            sent_trigger_classes = batch_trigger_classes[i]
            for j in range(word_seq_lens[i]):
                label = id2trigger[sent_trigger_classes[j]]
                if label.startswith("B-"):
                    tg_end = j+1
                    for k in range(j+1, word_seq_lens[i]):
                        if id2trigger[sent_trigger_classes[k]] == "I-"+label[2:]:
                            continue
                        else:
                            tg_end = k
                            break
                    triggers[(j, tg_end)] = sent_trigger_classes[j]
                else:
                    continue
            batch_triggers.append(triggers)
        return batch_triggers

    def forward(self, word_seqs, word_seq_lens, pos_tags, adj, entities=None, id2trigger=None):
        mask = np.zeros(shape=word_seqs.size(), dtype=np.uint8)
        for i in range(word_seqs.size(0)):
            seq_len = word_seq_lens[i]
            mask[i, 0:seq_len] = np.ones(shape=(seq_len, ), dtype=np.uint8)
        mask = torch.ByteTensor(mask)
        
        x = torch.cat([self.word_embedding(word_seqs), self.pos_embedding(pos_tags)], dim=-1)
        x, _ = self.lstm(x, word_seq_lens)
        x = self.drop_out(x)
        for gcn in self.gcns:
            x = gcn(x, adj)
            x = F.relu(x)
        x = self.drop_out(x)
        x_ = self.sa(x, word_seq_lens)

        batch_trigger_logits = self.trigger_classify(x_)
        assert not torch.isnan(batch_trigger_logits).any()
        batch_size = word_seqs.size(0)
        batch_trigger_classes = torch.max(batch_trigger_logits, dim=-1)[1]
        batch_triggers = self.get_triggers(batch_trigger_classes, word_seq_lens, id2trigger)
        batch_trigger_arguments_logits = []
        for i in range(batch_size):
            if len(entities[i]) == 0:
                batch_trigger_arguments_logits.append({})
                continue
            trigger_arguments_logits = {}
            for trigger in batch_triggers[i].keys():
                arguments_rep = []
                trigger_rep = x_[i][trigger[0]:trigger[1]].sum(dim=0) / (trigger[1]-trigger[0])
                for et_start, et_end, et_type in entities[i]:
                    entity_rep = x[i][et_start:et_end].sum(dim=0) / (et_end-et_start)
                    arguments_rep.append(torch.cat([trigger_rep, entity_rep]))
                trigger_arguments_logits[trigger] = self.argument_classify(torch.stack(arguments_rep))
                assert not torch.isnan(trigger_arguments_logits[trigger]).any()
            batch_trigger_arguments_logits.append(trigger_arguments_logits)
        
        return mask, batch_trigger_logits, batch_trigger_arguments_logits

def is_intersect(x1, x2, y1, y2):
    if max(x2, y2) - min(x1, y1) < (x2-x1) + (y2-y1):
        return True
    else:
        return False

class CustomLoss(nn.Module):
    def __init__(self, hps): 
        super(CustomLoss, self).__init__()
        self.loss_alpha = hps["loss_alpha"]
        self.loss_beta = hps["loss_beta"]

    def forward(self, mask, trigger_logits, trigger_labels, trigger_arguments_logits, trigger_arguments_labels, batch_entities, use_argument_loss=False):
        batch_size, seq_len = trigger_logits.size(0), trigger_logits.size(1)
        mask = mask.view(batch_size*seq_len, )
        masked_index = torch.tensor([x for x in range(batch_size*seq_len) if mask[x] == 1]).cuda()
        trigger_logits = trigger_logits.view(batch_size*seq_len, -1).index_select(0, masked_index)
        # index_select()索引查找
        trigger_labels = trigger_labels.view(batch_size*seq_len).index_select(0, masked_index)

        weight = torch.FloatTensor([1]+[self.loss_alpha]*(trigger_logits.size(-1)-1)).cuda()
        trigger_loss = F.nll_loss(F.log_softmax(trigger_logits, dim=1), trigger_labels, weight=weight)/trigger_labels.size(0)

        batch_arguments_logits = []
        batch_arguments_labels = []
        for i in range(batch_size):
            for trigger_pred, arguments_logits in trigger_arguments_logits[i].items():
                batch_arguments_logits.append(arguments_logits)
                find_trigger = False
                for trigger_true in trigger_arguments_labels[i].keys():
                    if is_intersect(trigger_true[0], trigger_true[1], trigger_pred[0], trigger_pred[1]):
                        batch_arguments_labels.append(torch.tensor(trigger_arguments_labels[i][trigger_true]).cuda())
                        find_trigger = True
                        break
                if not find_trigger:
                    batch_arguments_labels.append(torch.tensor([0]*arguments_logits.size(0)).cuda())
                
        argument_loss = 0
        if use_argument_loss and batch_arguments_logits != []:
            argument_loss = F.nll_loss(F.log_softmax(torch.cat(batch_arguments_logits), dim=1), torch.cat(batch_arguments_labels))/len(batch_arguments_labels)
        loss = trigger_loss + self.loss_beta * argument_loss
        assert loss > 0
        return loss

def get_scores(labels_true, labels_pred):
    sent_num = len(labels_true)
    I_TP, I_FP, I_FN = 0, 0, 0
    C_TP, C_FP, C_FN = 0, 0, 0
    for i in range(sent_num):
        for pos, tp in labels_pred[i].items():
            if pos in labels_true[i]:
                I_TP += 1
                if tp == labels_true[i][pos]:
                    C_TP += 1
                else:
                    C_FP += 1
            else:
                I_FP += 1
                C_FP += 1

        for pos, tp in labels_true[i].items():
            if pos not in labels_pred[i]:
                I_FN += 1
                C_FN += 1
    
    I_P, I_R, I_F1 = 0, 0, 0
    if I_TP != 0:
        I_P, I_R = I_TP/(I_TP+I_FP), I_TP/(I_TP+I_FN)
        I_F1 = 2*I_P*I_R/(I_P+I_R)
    C_P, C_R, C_F1 = 0, 0, 0
    if C_TP != 0:
        C_P, C_R = C_TP/(C_TP+C_FP), C_TP/(C_TP+C_FN)
        C_F1 = 2*C_P*C_R/(C_P+C_R)
        
    return I_P, I_R, I_F1, C_P, C_R, C_F1

def trigger_identification_and_classification_score(batch_trigger_logits_list, batch_trigger_labels_list, word_seq_lens_list, id2trigger):
    all_trigger_labels_pred = []
    all_trigger_labels_true = []
    for i, batch_trigger_logits in enumerate(batch_trigger_logits_list):
        batch_trigger_labels_pred = Model.get_triggers(torch.max(batch_trigger_logits, dim=-1)[1], word_seq_lens_list[i], id2trigger)
        batch_trigger_labels_true = Model.get_triggers(batch_trigger_labels_list[i], word_seq_lens_list[i], id2trigger)
        all_trigger_labels_pred.extend(batch_trigger_labels_pred)
        all_trigger_labels_true.extend(batch_trigger_labels_true)
    
    I_P, I_R, I_F1, C_P, C_R, C_F1 = get_scores(all_trigger_labels_true, all_trigger_labels_pred)
    return I_P, I_R, I_F1, C_P, C_R, C_F1

def argument_identification_and_classification_score(batch_trigger_arguments_logits_list, batch_entities_list, batch_trigger_arguments_labels_list):
    all_argument_labels_pred = []
    all_argument_labels_true = []
    for i, batch_trigger_arguments_logits in enumerate(batch_trigger_arguments_logits_list): 
        for j, sent_trigger_arguments_labels in enumerate(batch_trigger_arguments_labels_list[i]):
            sent_arguments_labels_true = {}
            for trigger, arguments_labels in sent_trigger_arguments_labels.items():
                for k, argument_label in enumerate(arguments_labels):
                    if argument_label != 0:
                        sent_arguments_labels_true[trigger+batch_entities_list[i][j][k]] = argument_label
            all_argument_labels_true.append(sent_arguments_labels_true)

            sent_arguments_labels_pred = {}
            for trigger, arguments_logits in batch_trigger_arguments_logits_list[i][j].items():
                for k, argument_label in enumerate(torch.max(arguments_logits, dim=-1)[1]):
                    argument_label = int(argument_label)
                    if argument_label != 0:
                        sent_arguments_labels_pred[trigger+batch_entities_list[i][j][k]] = argument_label
            all_argument_labels_pred.append(sent_arguments_labels_pred)

    I_P, I_R, I_F1, C_P, C_R, C_F1 = get_scores(all_argument_labels_true, all_argument_labels_pred)
    return I_P, I_R, I_F1, C_P, C_R, C_F1
                    
            
