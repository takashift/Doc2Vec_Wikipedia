
# coding: utf-8

# In[2]:

import gensim
import MeCab
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse


# In[3]:

class args:
    purpose = ""
    model = ""
    dictionary = ""


# In[4]:

mecab = MeCab.Tagger("-Owakati" + ("" if not args.dictionary else " -d " + args.dictionary))

model = gensim.models.Doc2Vec.load(args.model)


# In[5]:

purpose = []
# answers = []
for line in open(args.purpose, "r", encoding="utf-8"):
    cols = line.strip().split('\n')
    purpose.append(gensim.utils.simple_preprocess(mecab.parse(cols[0]).strip(), min_len=1))
#     answers.append(cols[1])


# In[6]:

doc_vecs = []
for pp in purpose:
    doc_vecs.append(model.infer_vector(pp))


# In[1]:

while True:
    line = input("> ")
    if not line:
        break

#     vec = model.infer_vector(gensim.utils.simple_preprocess(mecab.parse(line), min_len=1))
#     sims = cosine_similarity([vec], doc_vecs)
    sims = model.n_similarity([mecab.parse(line)], purpose)
    index = np.argsort(sims[0])

    print(purpose[index[-1]], sims[index[-1]])
#     print()
#     print(answers[index[-1]])
#     print()

    print(purpose[index[-2]], sims[index[-2]])
    print(purpose[index[-3]], sims[index[-3]])
    print(purpose[index[-4]], sims[index[-4]])
    print()


# In[ ]:



