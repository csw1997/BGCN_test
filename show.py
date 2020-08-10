import torch
import json
from enet.configuration import consts
import numpy as np
import re
from stanfordcorenlp import StanfordCoreNLP
from ipdb import set_trace

nlp = StanfordCoreNLP("D:/dogtime/mission/论文相关/资源/JMEE_Chinese_preprocess/stanford-corenlp-full-2018-10-05")

model = torch.load("out/model.pt")

sent = "Obama beats McCain."

sent_info = {
    "words": None,
    "pos-tags": None,
    "stanford-colcc": None
}

sent_info["words"] = nlp.word_tokenize(sent)
sent_info["pos-tags"] = [x for _, x in nlp.pos_tag(sent)]
sent_info["stanford-colcc"] = [relation + "/dep=" + str(ed - 1) + "/gov=" + str(st - 1) for relation, st ,ed in nlp.dependency_parse(sent)]

with open("out/word_dict.json", "r") as f:
    word_dict = json.load(f)

with open("out/pos_dict.json", "r") as f:
    pos_dict = json.load(f)

with open("out/trigger_label_itos.json", "r") as f:
    trigger_label_itos = json.load(f)

word_seq = [word_dict.get(word.lower(), word_dict["<unk>"]) for word in sent_info["words"]]
word_seq_len = len(word_seq)
pos_tags = [pos_dict[pos] for pos in sent_info["pos-tags"]]

adj = np.zeros((consts.edge_types, len(word_seq), len(word_seq)))
for i in range(len(word_seq)):
    adj[0, i, i] = 1
for edge in sent_info["stanford-colcc"][1:]:
    if re.search(r"dep=(\d*)", edge) is not None and re.search(r"gov=(\d*)", edge) is not None and \
        re.search(r"dep=(\d*)", edge).group(1) != "" and re.search(r"gov=(\d*)", edge).group(1) != "":
        dep = int(re.search(r"dep=(\d*)", edge).group(1))
        gov = int(re.search(r"gov=(\d*)", edge).group(1))
        adj[1, gov, dep] = 1
        adj[2, dep, gov] = 1

word_seq = torch.tensor(word_seq).unsqueeze(0).cuda()
word_seq_len = torch.tensor(word_seq_len).unsqueeze(0).cuda()
pos_tags = torch.tensor(pos_tags).unsqueeze(0).cuda()

adj = torch.tensor(adj).unsqueeze(0).cuda()

set_trace()
mask, batch_trigger_logits = model.forward(word_seq, word_seq_len, pos_tags, torch.tensor(adj),id2trigger=trigger_label_itos)
predicted_batch_trigger_classes = torch.max(batch_trigger_logits, dim=2)[1].cpu().numpy()

for i in range(len(predicted_batch_trigger_classes)):
    labels = [trigger_label_itos[label] for label in predicted_batch_trigger_classes[i]]
    print(labels)
