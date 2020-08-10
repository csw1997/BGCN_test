import torch
from torchtext.data import Field, Example, Pipeline, TabularDataset, Dataset
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
import json
from collections import Counter, OrderedDict
from ipdb import set_trace
from torchtext.vocab import GloVe
import os
import numpy as np
from enet.configuration import consts
import re

# 论元field
# field对象指定要如何处理某个字段
# Field class models common text processing datatypes that can be represented by tensors. 
# It holds a Vocab object that defines the set of possible values for elements of the field and their corresponding numerical representations. 
# The Field object also holds other parameters relating to how a datatype should be numericalized, 
# such as a tokenization method and the kind of Tensor that should be produced.
class ArgumentField(Field):
    def __init__(self, *args, **kwargs):
        super(ArgumentField, self).__init__(*args, **kwargs)

    # Construct the Vocab object for this field from one or more datasets.
    # arguments:Dataset objects or other iterable data sources from which
    # to construct the Vocab object that represents the set of possible values for this field.
    # keywords argument:  Passed to the constructor of Vocab.
    # 所谓构建词表，即需要给每个单词编码，也就是用数字来表示每个单词，这样才能够传入模型中。
    def build_vocab(self, *args, **kwargs):
        # Counter()计数器，追踪值的出现次数
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                # getattr(object, name[, default]) 返回对象属性值
                # is 比较的是两个实例对象是不是完全相同，它们是不是同一个对象，占用的内存地址是否相同。
                # if field is self 也就是获得ArgumentField对应的字段ARGUMENTS（name）
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for ex in data:
                for trigger, arguments_type in ex.items():
                    # update()增加元素
                    counter.update(arguments_type)
        # class torchtext.vocab.Vocab :Defines a vocabulary object that will be used to numericalize a field.
        # vocab_cls: class torchtext.vocab.Vocab
        # 从collections.Counter创建Vocab对象。__init__()
        # counter – collections.Counter object holding the frequencies of each value found in the data.
        self.vocab = self.vocab_cls(counter, specials=[], **kwargs)

    # Turn a batch of examples that use this field into a Variable.
    # arr (List[List[str]], or tuple of (List[List[str]], List[int])) if self.include_lengths is True.
    def numericalize(self, arr, device=None):
        # stoi: A collections.defaultdict instance mapping token strings to numerical identifiers.
        arr = [{trigger: [self.vocab.stoi[argument_type] for argument_type in arguments_type] for trigger, arguments_type in ex.items()} for ex in arr]
        return arr

class EntityField(Field):
    def __init__(self, *args, **kwargs):
        super(EntityField, self).__init__(*args, **kwargs)

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [getattr(arg, name) for name, field in
                            arg.fields.items() if field is self]
            else:
                sources.append(arg)
        for data in sources:
            for ex in data:
                for _, _, et_type in ex:
                    counter.update([et_type])
        self.vocab = self.vocab_cls(counter, specials=[], **kwargs)

    def numericalize(self, arr, device=None):
        arr = [[(et_pos_st, et_pos_ed, self.vocab.stoi[et_type]) for et_pos_st, et_pos_ed, et_type in ex] for ex in arr]
        return arr

class ADJField(Field):
    def __init__(self, *args, **kwargs):
        super(ADJField, self).__init__(*args, **kwargs)

    def pad(self, minibatch):
        minibatch = list(minibatch)
        max_length = 0
        for x in minibatch:
            max_length = max(max_length, x.shape[1])
        padded = []
        for x in minibatch:
            origen_len = x.shape[1]
            padded_x = np.zeros((consts.edge_types, max_length, max_length))
            padded_x[:, 0:origen_len, 0:origen_len] = x
            padded.append(padded_x)
        return padded



TEXT = Field(lower=True, use_vocab=True ,include_lengths=True, batch_first=True)
TRIGGER_LABEL = Field(lower=False, use_vocab=True, pad_token="O", unk_token=None, batch_first=True)
ARGUMENT_LABEL = ArgumentField(sequential=False, lower=False, use_vocab=True, batch_first=True)
ENTITY = EntityField(sequential=False, lower=False, use_vocab=True, batch_first=True)
POSTAG = Field(lower=False, use_vocab=True, batch_first=True)
ADJ = ADJField(use_vocab=False, batch_first=True)

default_fields = [("WORDS", TEXT), ("POSTAGS", POSTAG), ("ADJ", ADJ), ("TRIGGER_LABELS", TRIGGER_LABEL), ("ARGUMENTS", ARGUMENT_LABEL), ("ENTITIES", ENTITY)]

# 自定义Dataset，Dataset's subclass
# class torchtext.data.Dataset(examples, fields, filter_pred=None)
# torchtext预置的Dataset类的API如下，我们必须至少传入examples和fields这两个参数。
# examples为由torchtext中的Example对象构造的列表，Example为对数据集中一条数据的抽象。
# fields可简单理解为每一列数据和Field对象的绑定关系
class ACE2005Dataset(Dataset):
    def __init__(self, path, fields=default_fields):
        self.examples = self.get_examples(path, fields)
        # 传入examples和fields给父类
        # 首先找到ACE2005Dataset的父类（比如是类A），然后把类ACE2005Dataset的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数
        super(ACE2005Dataset, self).__init__(self.examples, fields)

    def get_examples(self, path, fields):
        examples = []
        with open(path, "r", encoding='utf-8') as f:
            raw_dataset = json.load(f)
        for data in raw_dataset:
            word_seq = data["words"]
            pos_tag = data["pos-tags"]
            if len(word_seq) > consts.max_seq_length:
                continue
            adj = np.zeros((consts.edge_types, len(word_seq), len(word_seq)))
            for i in range(len(word_seq)):
                adj[0, i, i] = 1
            for edge in data["stanford-colcc"][1:]:
                if re.search(r"dep=(\d*)", edge) is not None and re.search(r"gov=(\d*)", edge) is not None and \
                    re.search(r"dep=(\d*)", edge).group(1) != "" and re.search(r"gov=(\d*)", edge).group(1) != "":
                    dep = int(re.search(r"dep=(\d*)", edge).group(1))
                    gov = int(re.search(r"gov=(\d*)", edge).group(1))
                    adj[1, gov, dep] = 1
                    adj[2, dep, gov] = 1
            
            trigger_labels = ["O"] * len(word_seq)
            # trigger_labels = ["O"] * len(sentence) 
            trigger_arguments_labels = {}
            entities = {(entity["start"], entity["end"]): entity["entity-type"] for entity in data["golden-entity-mentions"]}
            entities = [(pos[0], pos[1], et_type) for pos, et_type in entities.items()]
            for event in data["golden-event-mentions"]:
                event_type = event["event_type"]
                trigger_start = event["trigger"]["start"]
                trigger_end = event["trigger"]["end"]
                # print("trigger_start",trigger_start,"len(trigger_labels)",len(trigger_labels))
                trigger_labels[trigger_start] = "B-"+event_type
                trigger_labels[trigger_start+1:trigger_end] = ["I-"+event_type]*(trigger_end-trigger_start-1)

                argument_labels = OrderedDict()
                for entity in entities:
                    argument_labels[(entity[0], entity[1])] = "O" 
                for argument in event["arguments"]:
                    assert (argument["start"], argument["end"]) in argument_labels, (argument["text"])
                    argument_labels[(argument["start"], argument["end"])] = argument["role"]

                trigger_arguments_labels[(trigger_start, trigger_end)] = [label for label in argument_labels.values()]

            examples.append(Example.fromlist([word_seq, pos_tag, adj, trigger_labels, trigger_arguments_labels, entities], fields))
        return examples

if __name__ == "__main__":
    train_set = ACE2005Dataset(path="ace-05-splits/JMEE_train.json")
    dev_set = ACE2005Dataset(path="ace-05-splits/JMEE_dev.json")
    test_set = ACE2005Dataset(path="ace-05-splits/JMEE_test.json")

    TEXT.build_vocab(train_set, dev_set, test_set, vectors=GloVe(name='6B', dim=300))
    # use Chinese wiki word embedding
    # vectors = Vectors(name="D:/dogtime/mission/论文相关/资源/JMEE_refactoring/.vector_cache/sgns.wiki.txt",cache="D:/dogtime/mission/论文相关/资源/JMEE_refactoring/.vector_cache")
    # TEXT.build_vocab(train_set, dev_set, test_set, vectors=vectors)
    TRIGGER_LABEL.build_vocab(train_set, dev_set, test_set)
    ARGUMENT_LABEL.build_vocab(train_set, dev_set, test_set)
    ENTITY.build_vocab(train_set, dev_set, test_set)
    POSTAG.build_vocab(train_set, dev_set, test_set)

    train_iter = BucketIterator(
        train_set,
        batch_size = 16,
        shuffle = True,
        device = torch.device("cuda"),
        sort_key = lambda x:len(x.WORDS)
    )
    for batch in train_iter:
        arguments = batch.ARGUMENTS
        adj = batch.ADJ
        break