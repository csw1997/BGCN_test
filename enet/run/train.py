from enet.models.model import Model, CustomLoss
from enet.models.model import trigger_identification_and_classification_score, argument_identification_and_classification_score
from enet.corpus.data import ACE2005Dataset
from torchtext.vocab import Vectors
from enet.corpus.data import TEXT, TRIGGER_LABEL, ARGUMENT_LABEL, ENTITY, POSTAG
from torchtext.data import BucketIterator
import torch
from torch import nn
from ipdb import set_trace
from torchtext.vocab import GloVe
from torch import autograd
import os

train_set = ACE2005Dataset(path="ace-05-splits/JMEE_train.json")
dev_set = ACE2005Dataset(path="ace-05-splits/JMEE_dev.json")
test_set = ACE2005Dataset(path="ace-05-splits/JMEE_test.json")

TEXT.build_vocab(train_set, dev_set, test_set, vectors=GloVe(name='6B', dim=300))
# vectors = Vectors(name="D:/dogtime/mission/论文相关/资源/JMEE_refactoring/.vector_cache/sgns.wiki.txt",cache="D:/dogtime/mission/论文相关/资源/JMEE_refactoring/.vector_cache")
# TEXT.build_vocab(train_set, dev_set, test_set, vectors=vectors)
TRIGGER_LABEL.build_vocab(train_set, dev_set, test_set)
ARGUMENT_LABEL.build_vocab(train_set, dev_set, test_set)
ENTITY.build_vocab(train_set, dev_set, test_set)
POSTAG.build_vocab(train_set, dev_set, test_set)

weight_matrix = TEXT.vocab.vectors

train_iter, dev_iter, test_iter = BucketIterator.splits(
    (train_set, dev_set, test_set),
    batch_sizes = (128,128,128),
    shuffle = True,
    device = torch.device("cuda"),
    sort_key = lambda x:len(x.WORDS)
)

hps = {}
hps["dict_size"] = len(TEXT.vocab.itos)
hps["word_embedding_size"] = 300
hps["pos_class_num"] = len(POSTAG.vocab.itos)
hps["pos_embedding_size"] = 50
hps["event_class_num"] = len(TRIGGER_LABEL.vocab.itos)
hps["argument_class_num"] = len(ARGUMENT_LABEL.vocab.itos)
hps["lstm_hidden_size"] = 200
hps["lstm_num_layers"] = 1
hps["gcn_layers"] = 3
hps["gcn_out_dim"] = 400
hps["multi_head_num"] = 4
hps["single_attention_out_dim"] = 250
hps["loss_alpha"] = 2
hps["loss_beta"] = 0.05


num_epochs = 20
fine_tune = True
if os.path.exists("out/model.pt") and fine_tune:
    model = torch.load("out/model.pt")
else:
    model = Model(hps, word_embedding_weight_matrix=weight_matrix)

cuda_gpu = torch.cuda.is_available()
if cuda_gpu:
    model.cuda()
criterion = CustomLoss(hps)
use_argument_loss = False
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=1e-7)
id2trigger = TRIGGER_LABEL.vocab.itos
history = {
    "train_tg": [],
    "train_ag": [],
    "dev_tg": [],
    "dev_ag": []
}

for epoch in range(num_epochs):
    if num_epochs == 10:
        model.word_embedding.weight.requires_grad = True
    train_mask_list = []
    train_batch_trigger_logits_list = []
    train_batch_trigger_labels_list = []
    train_batch_trigger_arguments_logits_list = []
    train_batch_trigger_arguments_labels_list = []
    train_batch_entities_list = []
    train_word_seq_lens_list = []
    
    if epoch == 10:
        use_argument_loss = True
    
    for batch in train_iter:
        word_seqs, word_seq_lens = batch.WORDS
        batch_trigger_labels = batch.TRIGGER_LABELS
        batch_trigger_arguments_labels = batch.ARGUMENTS
        entities = batch.ENTITIES
        pos_tags = batch.POSTAGS
        adj = batch.ADJ
        
        mask, batch_trigger_logits, batch_trigger_arguments_logits = model.forward(word_seqs, word_seq_lens, pos_tags, adj, entities, id2trigger)
        loss = criterion(mask, batch_trigger_logits, batch_trigger_labels, batch_trigger_arguments_logits, batch_trigger_arguments_labels, entities, use_argument_loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(filter(lambda p: p.requires_grad, model.parameters())), 2)
        optimizer.step()

        train_mask_list.append(mask)
        train_batch_trigger_logits_list.append(batch_trigger_logits)
        train_batch_trigger_labels_list.append(batch_trigger_labels)
        train_batch_trigger_arguments_logits_list.append(batch_trigger_arguments_logits)
        train_batch_trigger_arguments_labels_list.append(batch_trigger_arguments_labels)
        train_batch_entities_list.append(entities)
        train_word_seq_lens_list.append(word_seq_lens)
        
        print("Epoch %d," % epoch, "batch average loss:", loss.data.item())

    train_trigger_identification_precision, train_trigger_identification_recall, train_trigger_identification_f1, \
        train_trigger_classification_precision, train_trigger_classification_recall, train_trigger_classification_f1 \
         = trigger_identification_and_classification_score(train_batch_trigger_logits_list, \
             train_batch_trigger_labels_list, train_word_seq_lens_list, id2trigger)

    print("Epoch %d, train: tg_ident_P: %.2f, tg_ident_R: %.2f, tg_ident_F1: %.2f, tg_clas_P: %.2f, tg_clas_R: %.2f, tg_clas_F1: %.2f" % \
        (epoch, train_trigger_identification_precision, train_trigger_identification_recall, train_trigger_identification_f1, \
        train_trigger_classification_precision, train_trigger_classification_recall, train_trigger_classification_f1))
    
    history["train_tg"].append((train_trigger_identification_precision, train_trigger_identification_recall, train_trigger_identification_f1, \
        train_trigger_classification_precision, train_trigger_classification_recall, train_trigger_classification_f1))

    train_argument_identification_precision, train_argument_identification_recall, train_argument_identification_f1, \
        train_argument_classification_precision, train_argument_classification_recall, train_argument_classification_f1 \
        = argument_identification_and_classification_score(train_batch_trigger_arguments_logits_list, \
            train_batch_entities_list, train_batch_trigger_arguments_labels_list)

    print("Epoch %d, train: ag_ident_P: %.2f, ag_ident_R: %.2f, ag_ident_F1: %.2f, ag_clas_P: %.2f, ag_clas_R: %.2f, ag_clas_F1: %.2f" % \
        (epoch, train_argument_identification_precision, train_argument_identification_recall, train_argument_identification_f1, \
        train_argument_classification_precision, train_argument_classification_recall, train_argument_classification_f1))
    
    history["train_ag"].append((train_argument_identification_precision, train_argument_identification_recall, train_argument_identification_f1, \
        train_argument_classification_precision, train_argument_classification_recall, train_argument_classification_f1))

    dev_mask_list = []
    dev_batch_trigger_logits_list = []
    dev_batch_trigger_labels_list = []
    dev_batch_trigger_arguments_logits_list = []
    dev_batch_trigger_arguments_labels_list = []
    dev_batch_entities_list = []
    dev_word_seq_lens_list = []
    for batch in dev_iter:
        word_seqs, word_seq_lens = batch.WORDS
        batch_trigger_labels = batch.TRIGGER_LABELS
        batch_trigger_arguments_labels = batch.ARGUMENTS
        entities = batch.ENTITIES
        pos_tags = batch.POSTAGS
        adj = batch.ADJ
        mask, batch_trigger_logits, batch_trigger_arguments_logits = model.forward(word_seqs, word_seq_lens, pos_tags, adj, entities, id2trigger)
        dev_mask_list.append(mask)
        dev_batch_trigger_logits_list.append(batch_trigger_logits)
        dev_batch_trigger_labels_list.append(batch_trigger_labels)
        dev_batch_trigger_arguments_logits_list.append(batch_trigger_arguments_logits)
        dev_batch_trigger_arguments_labels_list.append(batch_trigger_arguments_labels)
        dev_batch_entities_list.append(entities)
        dev_word_seq_lens_list.append(word_seq_lens)

    dev_identification_precision, dev_identification_recall, dev_identification_f1, \
        dev_classification_precision, dev_classification_recall, dev_classification_f1 \
         = trigger_identification_and_classification_score(dev_batch_trigger_logits_list, \
             dev_batch_trigger_labels_list, dev_word_seq_lens_list, id2trigger)

    print("Epoch %d, dev:   tg_ident_P: %.2f, tg_ident_R: %.2f, tg_ident_F1: %.2f, tg_clas_P: %.2f, tg_clas_R: %.2f, tg_clas_F1: %.2f" % \
        (epoch, dev_identification_precision, dev_identification_recall, dev_identification_f1, \
        dev_classification_precision, dev_classification_recall, dev_classification_f1))
    
    history["dev_tg"].append((dev_identification_precision, dev_identification_recall, dev_identification_f1, \
        dev_classification_precision, dev_classification_recall, dev_classification_f1))

    dev_argument_identification_precision, dev_argument_identification_recall, dev_argument_identification_f1, \
        dev_argument_classification_precision, dev_argument_classification_recall, dev_argument_classification_f1 \
        = argument_identification_and_classification_score(dev_batch_trigger_arguments_logits_list, \
            dev_batch_entities_list, dev_batch_trigger_arguments_labels_list)

    print("Epoch %d, dev:   ag_ident_P: %.2f, ag_ident_R: %.2f, ag_ident_F1: %.2f, ag_clas_P: %.2f, ag_clas_R: %.2f, ag_clas_F1: %.2f" % \
        (epoch, dev_argument_identification_precision, dev_argument_identification_recall, dev_argument_identification_f1, \
        dev_argument_classification_precision, dev_argument_classification_recall, dev_argument_classification_f1))
    
    history["dev_ag"].append((dev_argument_identification_precision, dev_argument_identification_recall, dev_argument_identification_f1, \
        dev_argument_classification_precision, dev_argument_classification_recall, dev_argument_classification_f1))
    
torch.save(model, "out/model.pt")
