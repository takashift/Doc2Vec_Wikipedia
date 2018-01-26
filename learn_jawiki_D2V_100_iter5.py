
# coding: utf-8

# In[1]:

import gensim
import smart_open
import argparse


# In[2]:

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--save_model', '-s', default='model', type=str)
class args:
    input = ["~/corpus_jawiki/jawiki_00_wakati.txt",
    "~/corpus_jawiki/jawiki_01_wakati.txt",
    "~/corpus_jawiki/jawiki_02_wakati.txt",
    "~/corpus_jawiki/jawiki_03_wakati.txt",
    "~/corpus_jawiki/jawiki_04_wakati.txt",
    "~/corpus_jawiki/jawiki_05_wakati.txt"]
    save_model = "./wikiD2V_100dEN_iter5.model"


# In[ ]:

def read_corpus(fname):
    with smart_open.smart_open(fname, encoding="utf-8") as f:
        for i, line in enumerate(f):
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line, min_len=1), [i])


# In[ ]:

# train_corpus = list(read_corpus(args.input))

'''
size: ベクトル化した際の次元数
alpha: 学習率
sample: 単語を無視する際の頻度の閾値
min_count: 学習に使う単語の最低出現回数
workers: 学習時のスレッド数
'''

model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=10)

# 膨大なデータを読み込むために逐次処理
update=False
for file in args.input:
    # 適当な処理"load"をして学習用のsentencesをfiles[i]から抜き出す
    # sentences = load(file)
    train_corpus = list(read_corpus(file))
    model.build_vocab(train_corpus, update=update)
    update=True
for file in args.input:
    train_corpus = list(read_corpus(file))
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

# model.build_vocab(train_corpus)
# model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

model.save(args.save_model)


# In[ ]:



