#!/usr/bin/python3

# -*- coding: utf-8-*-

import os, zipfile
from urllib.request import urlopen

data_url_sigmorphon2016 = 'https://github.com/ryancotterell/sigmorphon2016/archive/master.zip'
data_url_sigmorphon2017 = 'https://github.com/sigmorphon/conll2017/archive/master.zip'

translate_partition = {'train':'train', 'validation':'dev', 'test':'test'}

def read_sigmorphon2016(data_dir):
  zipfname = os.path.join(data_dir, 'sigmorphon2016.zip')
  if not os.path.exists(zipfname):
    response = urlopen(data_url_sigmorphon2016)
    CHUNK = 16 * 1024
    with open(zipfname , 'wb') as f:
      while True:
        chunk = response.read(CHUNK)
        if not chunk:
          break
        f.write(chunk)
  else:
    print('found {} on disk.'.format(data_url_sigmorphon2016))
  with zipfile.ZipFile(zipfname) as zf:
    zf.extractall(data_dir)
  tempsubdir = os.path.join(data_dir, 'sigmorphon2016-master/data')
  d = {}
  vocab = set()
  tags = {}
  lookup = {}
  for language in ['arabic', 'finnish', 'georgian', 'german', 'hungarian', 'maltese', 'navajo', 'russian', 'spanish', 'turkish']:
  #for language in ['german']:
    #print(language)
    # skip task3. unknown source tag.
    #for task in ['task1', 'task2','task3']:
    merge_count = 0
    merge_distinct_forms_count = 0
    for task in ['task1', 'task2']:
      for partition in ['train', 'validation', 'test']:
        if partition in ['validation', 'test'] and task == 'task1':
          print('Not using {} data from {}. (only using {} data from task2).'.format(partition, task, partition))
          #partition = 'train'
          continue
        d.setdefault(partition, {})
        lookup.setdefault(partition, {})
        fn = os.path.join(tempsubdir,'{}-{}-{}'.format(language, task, translate_partition[partition]))
        with open(fn, 'r') as f:
          for l in f:
            ws = l.split()
            if task == 'task1':
              tag1 = 'lemma=1'
              word1 = ws[0]
              tag2 = ws[1]
              word2 = ws[2]
              for t2 in tag2.split(','):
                if t2[:3] == 'pos':
                  tag1 += ','+t2
                  break
            elif task == 'task2':
              tag1 = ws[0]
              word1 = ws[1]
              tag2 = ws[2]
              word2 = ws[3]
            #print('word1: {}, tags1: {}, word2: {}, tags2: {}'.format(word1, tag1, word2, tag2))
            for c in word1+word2:
              vocab.add(c)
            if not language in d[partition]:
              d[partition][language] = {}
              lookup[partition][language] = {}
            retained_tags1 = []
            retained_tags2 = []
            for t in sorted(tag2.split(',')):
              if t == 'pos=ADJ': pos = 'adjective'
              elif t == 'pos=V': pos = 'verb'
              elif t == 'pos=N': pos = 'noun'
              # blacklisting, effectively 'collapsing' classes:
              #if not (t.startswith('arg=') or t.startswith('polar=') or t.startswith('val=') or t.startswith('poss=') or t.startswith('mood=') or t.startswith('per=') or t.startswith('aspect=') or t.startswith('alt=') or t.startswith('polite=') or t.startswith('evid=')):
              if True:
                ts = t.split('=')
                if len(ts)>1:
                  if ts[1][0] == '{':
                    content = ts[1][1:-1]
                    content = content.split('/')[0]
                    t = ts[0]+'='+content
                retained_tags2.append(t)
            pos1=None
            for t in sorted(tag1.split(',')):
              if t == 'pos=ADJ': pos1 = 'adjective'
              elif t == 'pos=V': pos1 = 'verb'
              elif t == 'pos=N': pos1 = 'noun'
              #if not (t.startswith('arg=') or t.startswith('polar=') or t.startswith('val=') or t.startswith('poss=') or t.startswith('mood=') or t.startswith('per=') or t.startswith('aspect=') or t.startswith('alt=') or t.startswith('polite=') or t.startswith('evid=')):
              if True:
                ts = t.split('=')
                if len(ts)>1:
                  if ts[1][0] == '{':
                    content = ts[1][1:-1]
                    content = content.split('/')[0]
                    t = ts[0]+'='+content
                retained_tags1.append(t)
            for t in retained_tags1+retained_tags2:
              tp = t.split('=')
              #print(tp)
              add_to_dictset(tags, [(tp[0],tp[1])])
            retained_tags1 = ','.join(retained_tags1)
            retained_tags2 = ','.join(retained_tags2)
            #print('retained 1: {}, 2: {}'.format(retained_tags1, retained_tags2))
              #  # We are not considering these in the classification head.
              #  retained_tags2.append(t)
              #elif t.startswith('case=') or t.startswith('num=') or t.startswith('comp=') or t.startswith('tense=') or t.startswith('per=') or t.startswith('gen='):
              #  # We are not considering gender in the classification head.
              #  retained_tags2.append(t)
            if not pos in d[partition][language]:
              d[partition][language][pos] = []
              lookup[partition][language][pos] = {}
            if partition == 'test':
              # a hack to make the test set pass through the rest of the code.
              # will not combine paradigms, only use pairs from existing test set.
              # the testrelation key will be used in datapreparer.paradigm_to_relations()
              location = len(d[partition][language][pos])
              d[partition][language][pos].append({})
              paradigm = d[partition][language][pos][location]
              paradigm['testrelation'] = [retained_tags1,retained_tags2]
              paradigm[retained_tags1] = word1
              paradigm[retained_tags2] = word2
              lookup[partition][language][pos][word1] = location
              lookup[partition][language][pos][word2] = location
            else:
              location = lookup[partition][language][pos].get(word1, None)
              if location is not None:
                # ONLY EXACT MATCH WITH THE SAME TAGS:
                if retained_tags1 in d[partition][language][pos][location].keys() and d[partition][language][pos][location][retained_tags1] == word1:
                  merge_count += 1
                  print('Found a form of {} ({}) in existing paradigm: {}'.format(word1, retained_tags1, d[partition][language][pos][location]))
                else:
                  location = None
              else:
                location = lookup[partition][language][pos].get(word2, None)
                if location is not None:
                  # ONLY EXACT MATCH WITH THE SAME TAGS:
                  if retained_tags2 in d[partition][language][pos][location].keys() and d[partition][language][pos][location][retained_tags2] == word2:
                    merge_count += 1
                    print('Found a form of {} ({}) in existing paradigm: {}'.format(word2, retained_tags2, d[partition][language][pos][location]))
                  else:
                    location = None
              if location is None:
                location = len(d[partition][language][pos])
                d[partition][language][pos].append({})
              paradigm = d[partition][language][pos][location]
              paradigm[retained_tags1] = word1
              #if pos != pos1:
              #  print('different pos in pair: {} {} {} {}'.format(retained_tags1, word1, retained_tags2, word2))
              if retained_tags1 != retained_tags2:
                paradigm[retained_tags2] = word2
              lookup[partition][language][pos][word1] = location
              lookup[partition][language][pos][word2] = location
    print('Language: {}, merge_count: {}'.format(language, merge_count))
  total_max = 0
  for partition in d:
    for l in d[partition]:
      for pos in d[partition][l]:
        mi = 1000
        ma = 0
        for par in d[partition][l][pos]:
          mi = min(mi, len(par))
          ma = max(ma, len(par))
          total_max = max(ma, total_max)
          #if len(par) == ma:
          #  for val in par.values():
          #    print(val)
        print('{}, {}, {}: {} paradigms, min {} forms, max {} forms (total max: {}), vocab size: {}'.format(l, partition, pos, len(d[partition][l][pos]), mi, ma, total_max, len(vocab)))
  return d, vocab, tags

def add_to_dictset(d, list_of_pairs):
  for key,val in list_of_pairs:
    d.setdefault(key, set())
    d[key].add(val)


