#!/usr/bin/python3

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers

if len(sys.argv)%2 != 1:
  print('Usage: command.py datafile1 legend1 [datafile2 legend2 ...]')
  exit()
SCORE_COLUMN = 3


list_of_files = []
for i in range(1,len(sys.argv), 2):
  list_of_files.append((sys.argv[i], sys.argv[i+1]))
#  data_dir = os.path.join(sys.argv[1], 'saved_embeddings/validation/')
#data_dir = '/home/mogren/experiments/2017-char-rnn-wordrelations/nov-monolingual-resplit/tie_rel-F0.0-keep0.6/english/saved_embeddings/validation'


import pylab

datalist = [ ( pylab.loadtxt(filename), label ) for filename, label in list_of_files ]
for data, label in datalist:
    pylab.plot( data[:,0], data[:,SCORE_COLUMN], label=label )

pylab.legend()
pylab.title("Training trajectory")
pylab.xlabel("Training iterations")
pylab.ylabel("Accuracy")

pylab.show()

exit()

