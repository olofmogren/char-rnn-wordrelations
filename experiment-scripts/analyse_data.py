#!/usr/bin/python3

import sys,os

path = sys.argv[1]

data = {}

for partition in ['train', 'validation', 'test']:
  for language in sorted(os.listdir(os.path.join(path, partition))):
    data.setdefault(language, {})
    data[language].setdefault(partition, {})
    data[language][partition].setdefault('singles', 0)
    data[language][partition].setdefault('labels', {})
    l = os.listdir(os.path.join(path, os.path.join(partition, language)))
    print('{},{}: {} relations'.format(partition, language, len(l)))
    for f in l:
      fpath = os.path.join(path, os.path.join(partition, os.path.join(language, f)))
      with open(fpath) as fp:
        lines = len(fp.readlines())
        data[language][partition]['labels'][f] = lines
        if lines == 1:
          data[language][partition]['singles'] += 1
          
for l in data:
  partition = 'testplus'
  data[l].setdefault(partition, {})
  data[l][partition].setdefault('singles', 0)
  data[l][partition].setdefault('labels', {})
  for label in data[l]['test']['labels']:
    data[l]['testplus']['labels'][label] = data[l]['test']['labels'][label] + \
                                           data[l]['train']['labels'].get(label, 0) + \
                                           data[l]['validation']['labels'].get(label, 0)
    if data[l]['testplus']['labels'][label] == 1:
      data[l]['testplus']['singles'] += 1

  print('{}: testset {}/{} (singles/total). borrowing from train/validation: {}/{}. (train set: {}/{})'.format(l, data[l]['test']['singles'], len(data[l]['test']['labels']), data[l]['testplus']['singles'], len(data[l]['testplus']['labels']), data[l]['train']['singles'], len(data[l]['train']['labels'])))

