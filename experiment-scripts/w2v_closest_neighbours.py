#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os, sys, gensim

data_dir = sys.argv[1]
#data_dir = os.path.abspath(os.path.join(sys.argv[1], os.pardir))
embeddings_fpath = os.path.join(data_dir, 'embeddings/english.bin')
print(embeddings_fpath)

embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_fpath, binary=True)

#axes = plt.gca()
#axes.set_ylim([-0.5,0.5])

for word in ['king', 'queen', 'Stockholm']:
  print(word)
  s = embeddings.most_similar(word, topn=10)

  for w2 in s:
    print(w2)
    #print('  '+w2)

