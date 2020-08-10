#%%
from bert import tokenization,modeling
import tensorflow as tf
import numpy as np  
import os
import collections
import json

# %%
#配置运行变量
flags = tf.flags
FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string('f', '', 'kernel')
bert_path = "D:/dogtime/mission/论文相关/资源/JMEE_refactoring/uncased_L-12_H-768_A-12/"
flags.DEFINE_string(
    'bert_config_file', os.path.join(bert_path, 'bert_config.json'),
    'config json file corresponding to the pre-trained BERT model.'
)
flags.DEFINE_string(
    'bert_vocab_file', os.path.join(bert_path,'vocab.txt'),
    'the config vocab file',
)
flags.DEFINE_string(
    'init_checkpoint', os.path.join(bert_path,'bert_model.ckpt'),
    'from a pre-trained BERT get an initial checkpoint',
)
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

#%%
def convert2Uni(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8','ignore')
    else:
        print(type(text))
        print('####################wrong################')

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    vocab.setdefault('blank', 2)
    index = 0
    with open(vocab_file, 'rb') as reader:
    # with tf.gfile.GFile(vocab_file, 'r') as reader:
        while True:
            tmp = reader.readline()
            if not tmp:
                break
            token = convert2Uni(tmp)
            token = token.strip()
            vocab[token] = index 
            index+=1
    return vocab

def inputs(vectors, maxlen = 50):
    length = len(vectors)
    if length > maxlen:
        return vectors[0:maxlen], [1]*maxlen, [0]*maxlen
    else:
        input = vectors+[0]*(maxlen-length)
        mask = [1]*length + [0]*(maxlen-length)
        segment = [0]*maxlen
        return input, mask, segment


def response_request(text):
    vectors = [dictionary.get('[CLS]')] + [dictionary.get(i) if i in dictionary else dictionary.get('[UNK]') for i in list(text)] + [dictionary.get('[SEP]')]
    input, mask, segment = inputs(vectors)

    input_ids = np.reshape(np.array(input), [1, -1])
    input_mask = np.reshape(np.array(mask), [1, -1])
    segment_ids = np.reshape(np.array(segment), [1, -1])

    embedding = tf.squeeze(model.get_sequence_output())
    rst = sess.run(embedding, feed_dict={'input_ids_p:0':input_ids, 'input_mask_p:0':input_mask, 'segment_ids_p:0':segment_ids})

    return rst ,json.dumps(rst.tolist(), ensure_ascii=False)
#%%
dictionary = load_vocab(FLAGS.bert_vocab_file)
init_checkpoint = FLAGS.init_checkpoint

sess = tf.Session()
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

input_ids_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='input_ids_p')
input_mask_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='input_mask_p')
segment_ids_p = tf.placeholder(shape=[None, None], dtype = tf.int32, name='segment_ids_p')

model = modeling.BertModel(
    config = bert_config,
    is_training = FLAGS.use_tpu,
    input_ids = input_ids_p,
    input_mask = input_mask_p,
    token_type_ids = segment_ids_p,
    use_one_hot_embeddings = FLAGS.use_tpu,
)
print('####################################')
restore_saver = tf.train.Saver()
restore_saver.restore(sess, init_checkpoint)

print(response_request('我叫叶露。'))
s,t = response_request("我叫叶露。")
# %%
