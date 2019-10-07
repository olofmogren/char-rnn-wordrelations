#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Copyright (C) 2017 Olof Mogren

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, random, time, math, random, os, argparse, sys, pickle
import datareader_saldo, datareader_sigmorphon, datareader_scowl
from urllib.request import urlopen
import zipfile, gzip
import gensim
import requests


num_examples               = {}
num_examples['validation'] = 1000
num_examples['test']       = 1000

supported_languages  = ['english', 'swedish', 'arabic', 'finnish', 'georgian', 'german', 'hungarian', 'maltese', 'navajo', 'russian', 'spanish', 'turkish']

ONE_RELATION_PER_TAGFLIP = False

def prepare_data(data_dir, languages, id_prob, test_words, dataset_partitions_without_singles=[], filter_tags=[]):
  '''
    This function reads data from disk. If not found, it calls download_data() to download it.
    download_data() downloads the data, and then calls prepare_data() once more to
    read the downloaded data.

  '''
  all_relation_labels  = []
  relations            = {}
  flattened_train_set  = {}
  all_tags             = {}
  vocab                = []
  vocab_size           = -1
  reverse_vocab        = {}
  num_relation_classes = None
  # all letters will be further populated from dataset in prepare_data().
  all_characters          = string.ascii_letters + " .,;'-ÅåÄäÖöÜüß"
  native_characters       = {}
  all_relations = {}
  for language in supported_languages:
    all_relations[language] = {}
  
  partitions = ['train','validation','test']
  for partition in partitions:
    relations[partition] = {}
    for language in supported_languages:
      relations[partition][language] = []

  all_characters_set = set(all_characters)
  native_characters_sets = {}
  exists = True
  all_relation_labels_set = set()
  fname = os.path.join(data_dir, 'tags.pkl')
  if os.path.exists(fname):
    with open(fname, 'rb') as f:
      all_tags = pickle.load(f)
  else:
    print('{} not found.'.format(fname))
    exists = False

  if exists:
    for language in languages:
      lang_relation_labels_set = set()
      native_characters_sets[language] = set()
      relation_labels_fn = os.path.join(data_dir, '{}-relations.txt'.format(language))
      if not os.path.exists(relation_labels_fn):
        print('{} not found.'.format(relation_labels_fn))
        exists = False
        break
      with open(relation_labels_fn, 'r') as f_rel_names:
        for line in f_rel_names:
          #print('{}: {}'.format(language, line.strip()))
          all_relation_labels_set.add(line.strip())
          lang_relation_labels_set.add(line.strip())
      for relation in lang_relation_labels_set:
        if not exists:
          break
        all_relations[language].setdefault(relation, [])
        for partition in relations:
          if not os.path.exists(os.path.join(data_dir, '{}/{}'.format(partition,language))):
            print('{} does not exist! Will download.'.format(os.path.join(data_dir, '{}/{}'.format(partition,language))))
            exists = False
            break
          if os.path.exists(os.path.join(data_dir, '{}/{}/{}.txt'.format(partition,language,relation))):
            r = {'label': relation, 'wordpairs': []}
            with open(os.path.join(data_dir, '{}/{}/{}.txt'.format(partition,language,relation)), 'r') as f:
              for line in f:
                words = line.split()
                #print(words)
                r['wordpairs'].append(((words[0], tagstring_to_dict(words[1])), (words[2], tagstring_to_dict(words[3]))))
                all_characters_set.update(set(words[0]))
                all_characters_set.update(set(words[2]))
                native_characters_sets[language].update(set(words[0]))
                native_characters_sets[language].update(set(words[2]))
            relations[partition][language].append(r)
            all_relations[language][relation] += r['wordpairs']
          #else:
          #  print('Could not find: {}'.format(os.path.join(data_dir, '{}/{}/{}.txt'.format(partition,language,relation))))
          #  Files does not need to exist. Not all relations have examples in all partitions (should have training examples though).
          #  exists = False
          #  break
      if not exists:
        print('All data does not yet exist. Download.')
        break
  if exists:
    all_characters = sorted(list(all_characters_set))
    for l in languages:
      native_characters[l] = sorted(list(native_characters_sets[l]))
    if id_prob > 0.0:
      print('--id_prob: {}'.format(id_prob))
      all_relation_labels_set.add('id')
    all_relation_labels = sorted(list(all_relation_labels_set))
    # each relation is counted twice, as they can be reversed.
    num_relation_classes = len(all_relation_labels)*2 #*len(languages.split(','))
    
    # DONE!

    if test_words:
      test_words = test_words.split(',')
      last_test_word_associations = set([])
      test_word_associations = set(test_words)
      print('test_words: {}'.format(' '.join(test_words)))
      while test_word_associations != last_test_word_associations:
        sys.stdout.write('.')
        #print('test_word_associations changed.')
        last_test_word_associations = test_word_associations.copy()
        for p in ['train', 'validation']:
          for l in languages:
            for r in relations[p][l]:
              newlist = []
              for ((word1, tags1_d), (word2, tags2_d)) in r['wordpairs']:
                remove_wordpair = False
                for test_word in last_test_word_associations:
                  if test_word in word1 or test_word in word2:
                    #print('found test_word {} in {} and/or {}'.format(test_word, word1, word2))
                    test_word_associations.add(word1)
                    test_word_associations.add(word2)
                    remove_wordpair = True
                if not remove_wordpair:
                  newlist.append(((word1, tags1_d), (word2, tags2_d)))
              r['wordpairs'] = newlist
      print('\nFound and removed {} associated test_words: {}'.format(len(test_word_associations), ' '.join(sorted(list(test_word_associations)))))

    if len(filter_tags):
      # update relations, all_tags, num_relation_classes,all_relation_labels
      # flattened_train_set is created later.

      print('Filtering tags: {}'.format(filter_tags))

      new_all_tags = {}
      new_all_relation_labels = set()
      new_relations = {}
      for p in relations:
        for l in relations[p]:
          for r in relations[p][l]:
            tags = r['label'].split('-')
            source_tags = tags[0]
            source_tags = [tuple(x.split('=')) for x in source_tags.split(',')]
            source_tags = [(k,v) for (k,v) in source_tags if k not in filter_tags]
            target_tags = tags[1]
            target_tags = [tuple(x.split('=')) for x in target_tags.split(',')]
            target_tags = [(k,v) for (k,v) in target_tags if k not in filter_tags]
            newlabel = ','.join(['{}={}'.format(k,v) for k,v in source_tags])+'-'+\
                       ','.join(['{}={}'.format(k,v) for k,v in target_tags])
            found = False
            new_relations.setdefault(p, {})
            new_relations[p].setdefault(l, [])
            if newlabel in new_all_relation_labels:
              for new_r in new_relations[p][l]:
                if new_r['label'] == newlabel:
                  found = True
                  new_r['wordpairs'] += r['wordpairs']
            if not found:
              new_r = {'label': newlabel, 'wordpairs': r['wordpairs']}
              new_relations[p][l].append(new_r)
            new_all_relation_labels.add(newlabel)
            for (k,v) in source_tags:
              new_all_tags.setdefault(k, set())
              new_all_tags[k].add(v)
            for (k,v) in target_tags:
              new_all_tags.setdefault(k, set())
              new_all_tags[k].add(v)
      all_tags = new_all_tags
      all_relation_labels = sorted(list(new_all_relation_labels))
      num_relation_classes = len(all_relation_labels)*2 #*len(languages.split(','))
      relations = new_relations
      for t in all_tags:
        all_tags[t] = ['nil']+sorted(list(all_tags[t]))
        print('{}: {}'.format(t, all_tags[t]))
      print('Filtering tags done.')

    for t in all_tags:
      print('{}: {}'.format(t, all_tags[t]))

    for partition in dataset_partitions_without_singles:
      print('Filtering single-sized {} relations.'.format(partition))
      
      for l in relations[partition]:
        to_remove = []
        for i in range(len(relations[partition][l])-1, -1, -1):
          if len(relations[partition][l][i]['wordpairs']) <= 1:
            to_remove.append(i)
        for i in to_remove:
          # sorted backwards, so it's cool:
          del relations[partition][l][i]
      print('Filtering singles done.')

    print('Sorting relation labels.')
    for p in relations:
      for l in relations[p]:
        relations[p][l].sort(key=lambda r: r['label'])
    print('Sorting done.')

    print('Creating flattened training set list.')
    #if uniform_sampling:
    for language in relations['train']:
      #print('creating flattened_train_set[{}]'.format(language))
      flattened_train_set[language] = []
      for i,r in enumerate(relations['train'][language]):
        for (demo_word1, demo_tag1), (demo_word2, demo_tag2) in r['wordpairs']:
          flattened_train_set[language].append((r['label'], i, (demo_word1, demo_tag1), (demo_word2, demo_tag2)))
    print('Flattening done.')

    count = {}
    for p in relations:
      count[p] = 0
      for l in relations[p]:
        for r in relations[p][l]:
          count[p] += len(r['wordpairs'])
    print('Total words-pairs: train: {}, validation: {}, test: {}'.format(count['train'], count['validation'], count['test']))

    print('| Language  | Tr rels  | Tr s-rels | Tr w-pairs  | Val rels | Val s-rels | Val w-pairs | Tst rels | Tst s-rels | Tst w-pairs | Tot rels |')
    print('|:--------- | --------:| -----------:| --------:| -----------:| --------:| -----------:| --------:| -----------:| --------:| -----------:|')
    ambiguities_string = ''
    for l in languages:
      toprint = '| {} | '.format(l.ljust(9))
      ambiguities_string += '| {} | '.format(l.ljust(9))
      total_rels = set()
      for p in ['train', 'validation', 'test']:
        num_rels = 0
        num_single_rels = 0
        num_words = 0
        num_bi_ambiguities = 0
        num_poly_ambiguities = 0
        for r in relations[p][l]:
          num_rels += 1
          if len(r['wordpairs']) == 1:
            num_single_rels += 1
          total_rels.add(r['label'])
          num_words += len(r['wordpairs'])

          word_counts_s = {}
          word_counts_t = {}
          for (w_s,w_t) in r['wordpairs']:
            w_s = w_s[0]
            w_t = w_t[0]
            word_counts_s[w_s] = word_counts_s.get(w_s, 0)+1
            word_counts_t[w_t] = word_counts_t.get(w_t, 0)+1
          num_bi_ambiguities += len([word_counts_s[w] for w in word_counts_s if word_counts_s[w] == 2])
          num_bi_ambiguities += len([word_counts_t[w] for w in word_counts_t if word_counts_t[w] == 2])
          num_poly_ambiguities += len([word_counts_s[w] for w in word_counts_s if word_counts_s[w] > 2])
          num_poly_ambiguities += len([word_counts_t[w] for w in word_counts_t if word_counts_t[w] > 2])
        toprint += '{} | {} | {} | '.format(str(num_rels).rjust(8), str(num_single_rels).rjust(10), str(num_words).rjust(11))
        ambiguities_string += '{} | {} | '.format(str(num_bi_ambiguities).rjust(11), str(num_poly_ambiguities).rjust(11))
      ambiguities_string += '\n'
      toprint += '{} |'.format(str(len(total_rels)).rjust(8))
      print(toprint)
    print('| Language  | Tr bi-amb | Tr poly-amb  | Val bi-amb | Val poly-amb  | Tst bi-amb | Tst poly-amb  |')
    print('|:--------- | --------:| -----------:| --------:| -----------:| --------:| -----------:| --------:|')
    print(ambiguities_string)
    return relations,num_relation_classes,all_relation_labels,all_characters,native_characters,all_tags,test_words,flattened_train_set

  else:
    # FIRST TIME WE ARE RUNNING THIS SCRIPT, THINGS NEED TO BE DOWNLOADED HERE.
    # THEN RUN THE FIRST PART OF THE FUNCTION AGAIN (WILL BE CALLED FROM download_data()).
    return download_data(data_dir, languages, id_prob, test_words)


def download_data(data_dir, languages, id_prob, test_words):
  '''
    This function downloads the data and saves to disk. It populates only local variables,
    and then data is read in prepare_data(), populating global variables. This way,
    we are using the same code the first time we load the data, as when we load it the
    following times from disk.

    This may look like recursion, but only one call should ever be made back to prepare_data().

    Will download data from all supported_languages. The argument 'languages' is only passed on to
    prepare_data, which will read data only for languages in this list.
  '''
  all_relation_labels_set = set()
  if id_prob > 0.0:
    print('--id_prob: {}'.format(id_prob))
    all_relation_labels_set.add('id')
  all_relations           = {}
  sigmorphon_relations    = {}
  all_tags                = {}
  relations               = {}

  for language in supported_languages:
    all_relations[language] = {}
  
  partitions = ['train','validation','test']
  for partition in partitions:
    relations[partition] = {}
    for language in supported_languages:
      relations[partition][language] = []
  # need to download and parse different formats!
  data_en,vocab_en,tags_en = datareader_scowl.get_english_data(data_dir)
  for language in data_en:
    all_relations.setdefault(language, {})
    for pos in data_en[language]:
      for paradigm in data_en[language][pos]:
        if ONE_RELATION_PER_TAGFLIP:
          for relation, word1, complete_tags1, word2, complete_tags2 in paradigm_to_relations(paradigm):
            all_relations[language].setdefault(relation, [])
            #print('relation: {}, word1: {} tags1: {}, word2, {}, tags2: {}'.format(relation, word1, complete_tags1, word2, complete_tags2))
            all_relations[language][relation].append(((word1, complete_tags1),(word2, complete_tags2)))
            all_relation_labels_set.add(relation)
        else:
          tags = sorted(paradigm.keys())
          for t1 in range(len(tags)-1):
            tag1 = tags[t1]
            for t2 in range(t1+1, len(tags)):
              tag2 = tags[t2]
              if tag1 < tag2:
                relation = tag1+'-'+tag2 
                relation = relation.replace('/', '.').replace('{', '').replace('}', '')
                all_relations.setdefault(language, {})
                all_relations[language].setdefault(relation, [])
                all_relations[language][relation].append(((paradigm[tag1], tagstring_to_dict(tag1)), (paradigm[tag2], tagstring_to_dict(tag2))))
                all_relation_labels_set.add(relation)

  for t in tags_en:
    all_tags.setdefault(t, set())
    all_tags[t].update(tags_en[t])

  data_sw,vocab_sw,tags_sw = datareader_saldo.get_saldo_data(data_dir)
  for language in data_sw:
    all_relations.setdefault(language, {})
    for pos in data_sw[language]:
      for paradigm in data_sw[language][pos]:
        if ONE_RELATION_PER_TAGFLIP:
          for relation, word1, complete_tags1, word2, complete_tags2 in paradigm_to_relations(paradigm):
            all_relations[language].setdefault(relation, [])
            #print('relation: {}, word1: {} tags1: {}, word2, {}, tags2: {}'.format(relation, word1, complete_tags1, word2, complete_tags2))
            all_relations[language][relation].append(((word1, complete_tags1),(word2, complete_tags2)))
            all_relation_labels_set.add(relation)
        else:
          tags = sorted(paradigm.keys())
          for t1 in range(len(tags)-1):
            tag1 = tags[t1]
            for t2 in range(t1+1, len(tags)):
              tag2 = tags[t2]
              if tag1 < tag2:
                relation = tag1+'-'+tag2 
                relation = relation.replace('/', '.').replace('{', '').replace('}', '')
                all_relations[language].setdefault(relation, [])
                all_relations[language][relation].append(((paradigm[tag1], tagstring_to_dict(tag1)), (paradigm[tag2], tagstring_to_dict(tag2))))
                all_relation_labels_set.add(relation)

  for t in tags_sw:
    all_tags.setdefault(t, set())
    all_tags[t].update(tags_sw[t])

  data_s,vocab_s,tags_s = datareader_sigmorphon.read_sigmorphon2016(data_dir)
  for partition in data_s:
    sigmorphon_relations[partition] = {}
    for language in data_s[partition]:
      sigmorphon_relations[partition][language] = {}
      for pos in data_s[partition][language]:
        for paradigm in data_s[partition][language][pos]:
          if ONE_RELATION_PER_TAGFLIP:
            for relation, word1, complete_tags1, word2, complete_tags2 in paradigm_to_relations(paradigm):
              sigmorphon_relations[partition][language].setdefault(relation, [])
              #print('relation: {}, word1: {} tags1: {}, word2, {}, tags2: {}'.format(relation, word1, complete_tags1, word2, complete_tags2))
              sigmorphon_relations[partition][language][relation].append(((word1, complete_tags1),(word2, complete_tags2)))
              all_relation_labels_set.add(relation)
          else:
            if 'testrelation' in paradigm:
              tags = paradigm['testrelation']
            else:
              tags = sorted(paradigm.keys())
            for t1 in range(len(tags)-1):
              tag1 = tags[t1]
              for t2 in range(t1+1, len(tags)):
                tag2 = tags[t2]
                if tag1 < tag2:
                  relation = tag1+'-'+tag2 
                  relation = relation.replace('/', '.').replace('{', '').replace('}', '')
                  sigmorphon_relations[partition][language].setdefault(relation, [])
                  sigmorphon_relations[partition][language][relation].append(((paradigm[tag1], tagstring_to_dict(tag1)), (paradigm[tag2], tagstring_to_dict(tag2))))
                  all_relation_labels_set.add(relation)

  for t in tags_s:
    all_tags.setdefault(t, set())
    all_tags[t].update(tags_s[t])
  for t in all_tags:
    all_tags[t] = ['nil']+sorted(list(all_tags[t]))
    print('{}: {}'.format(t, all_tags[t]))


  all_relation_labels = sorted(list(all_relation_labels_set))
  # each relation is counted twice, as they can be reversed.
  num_relation_classes = len(all_relation_labels)*2 #*len(languages.split(','))

  for language in all_relations:
    relations['train'][language]      = []
    relations['validation'][language] = []
    relations['test'][language]       = []

  with open(os.path.join(data_dir, 'tags.pkl'), 'wb') as f:
    pickle.dump(all_tags, f)
  # not workshop selection:
  for language in ['english', 'swedish']:
    relation_labels_fn = os.path.join(data_dir, '{}-relations.txt'.format(language))
    total_count_language = 0
    with open(relation_labels_fn, 'w') as f_rel_names:
      for relation in all_relations[language]:
        f_rel_names.write(relation+'\n')
        if len(all_relations[language][relation]) > 10:
          random.shuffle(all_relations[language][relation])
        total_count_language += len(all_relations[language][relation])
        r = {'label': relation, 'wordpairs': all_relations[language][relation]}
        relations['train'][language].append(r)
    relation_labels = sorted(list(all_relations[language].keys()))
    
    for partition in ['validation', 'test']:
      num_examples_local = min(num_examples[partition], total_count_language*.1)
      print('{}, num_examples_local: {}'.format(language, num_examples_local))
      total_count = 0
      while total_count < num_examples_local:
        r = randomChoice(relations['train'][language])
        # need at least two datapoints in each non-empty category.
        found = False
        for i,relation_label in enumerate([r_part['label'] for r_part in relations[partition][language]]):
          if relation_label == r['label']:
            count = len(relations[partition][language][i])
            count_to_select = 1 if count > 0 else 2
            found = True
            relation_index = i
            break
        if not found:
          count_to_select = 2
        if len(r['wordpairs'])-count_to_select < 2:
          print('continuing. len({},{})={}'.format(language, relation, len(r['wordpairs'])))
          continue
        if not found:
          relation_index = len(relations[partition][language])
          relations[partition][language].append({'label': r['label'], 'wordpairs': []})
        relations[partition][language][relation_index]['wordpairs'] += r['wordpairs'][:count_to_select]
        r['wordpairs'] = r['wordpairs'][count_to_select:]
        total_count += count_to_select
  for language in supported_languages:
    printcounts = {}
    for partition in relations:
      try: os.makedirs(os.path.join(os.path.join(data_dir, partition), language))
      except: pass
      printcounts[partition] = {}
      for r in relations[partition][language]:
        printcounts[partition][r['label']] = len(r['wordpairs'])
        with open(os.path.join(data_dir, '{}/{}/{}.txt'.format(partition,language,r['label'])), 'w') as f:
          for ((word1, tags1_d), (word2, tags2_d)) in r['wordpairs']:
            f.write('{} {} {} {}\n'.format(word1, tagdict_to_string(tags1_d), word2, tagdict_to_string(tags2_d)))
    for relation in sorted(list(all_relations[language].keys())):
      for partition in relations:
        printcounts[partition].setdefault(relation, 0)
      print('{}: {}: total: {}, num_train: {}, num_valid: {}, num_test: {}'.format(language, relation, len(all_relations[language][relation]), printcounts['train'][relation], printcounts['validation'][relation], printcounts['test'][relation]))

  for language in sigmorphon_relations['train']:
    labels = set([])
    for partition in sigmorphon_relations:
      labels.update(sigmorphon_relations[partition][language])
    relation_labels_fn = os.path.join(data_dir, '{}-relations.txt'.format(language))
    total_count_language = 0
    with open(relation_labels_fn, 'w') as f_rel_names:
      for relation in sorted(list(labels)):
        f_rel_names.write(relation+'\n')
    # Official dataset split:
    for partition in sigmorphon_relations:
      for relation in sigmorphon_relations[partition][language]:
        total_count_language += len(sigmorphon_relations[partition][language][relation])
        r = {'label': relation, 'wordpairs': sigmorphon_relations[partition][language][relation]}
        relations[partition][language].append(r)
        random.shuffle(r['wordpairs'])
  for language in supported_languages:
    printcounts = {}
    for partition in relations:
      try: os.makedirs(os.path.join(os.path.join(data_dir, partition), language))
      except: pass
      printcounts[partition] = {}
      for r in relations[partition][language]:
        printcounts[partition][r['label']] = len(r['wordpairs'])
        with open(os.path.join(data_dir, '{}/{}/{}.txt'.format(partition,language,r['label'])), 'w') as f:
          #print('Saving {}'.format(os.path.join(data_dir, '{}/{}/{}.txt'.format(partition,language,relation))))
          #lines = 0
          for ((word1, tags1_d), (word2, tags2_d)) in r['wordpairs']:
            f.write('{} {} {} {}\n'.format(word1, tagdict_to_string(tags1_d), word2, tagdict_to_string(tags2_d)))
            #lines += 1
          #print('{} lines.'.format(lines))
    for relation in sorted(list(all_relations[language].keys())):
      for partition in relations:
        printcounts[partition].setdefault(relation, 0)
      print('{}: {}: total: {}, num_train: {}, num_valid: {}, num_test: {}'.format(language, relation, len(all_relations[language][relation]), printcounts['train'][relation], printcounts['validation'][relation], printcounts['test'][relation]))

  # NOW PREPARE DATA (READ IT FROM DISK)
  return prepare_data(data_dir, languages, id_prob, test_words)

def tagstring_to_dict(tagstring):
  #print(tagstring)
  return dict([x.split('=') for x in tagstring.split(',')])

def tagdict_to_string(tags_d):
  return ','.join(['='.join([k,v]) for k,v in tags_d.items()])

def paradigm_to_relations(paradigm):
  tags = sorted(paradigm.keys())
  if 'testrelation' in tags:
    tags = paradigm['testrelation']
  for t1 in range(len(tags)-1):
    tag1 = tags[t1]
    for t2 in range(t1+1, len(tags)):
      tag2 = tags[t2]
      tag1 = tag1.replace('/', '.').replace('{', '').replace('}', '')
      tag2 = tag2.replace('/', '.').replace('{', '').replace('}', '')
      tags1_d = tagstring_to_dict(tag1)
      tags2_d = tagstring_to_dict(tag2)
      for tag in tags1_d:
        if tag in tags2_d and tags1_d[tag] != tags2_d[tag]:
          if tags1_d[tag] < tags2_d[tag]:
            tagval1 = tags1_d[tag]
            tagval2 = tags2_d[tag]
            pos = tags1_d['pos']
            if 'pos' in tags2_d and tags1_d['pos'] != tags2_d['pos']:
              pos += '.'+tags2_d['pos']
            complete_tags1 = tags1_d
            complete_tags2 = tags2_d
            word1 = paradigm[tag1]
            word2 = paradigm[tag2]
          else:
            tagval1 = tags2_d[tag]
            tagval2 = tags1_d[tag]
            pos = tags2_d['pos']
            if 'pos' in tags1_d and tags1_d['pos'] != tags2_d['pos']:
              pos += '.'+tags1_d['pos']
            complete_tags1 = tags2_d
            complete_tags2 = tags1_d
            word1 = paradigm[tag2]
            word2 = paradigm[tag1]
          relation = pos+':'+tag+':'+tagval1+'-'+tagval2
          yield relation, word1, complete_tags1, word2, complete_tags2

def get_embeddingfile(data_dir, filename, url):
  filepath = os.path.join(data_dir, filename)
  if not os.path.exists(filepath):
    response = urlopen(url)
    CHUNK = 16 * 1024
    with open(filepath , 'wb') as f:
      while True:
        chunk = response.read(CHUNK)
        if not chunk:
          break
        f.write(chunk)
  else:
    print('found {} on disk.'.format(url))
  return pickle.load(open(filepath, 'rb'))


def get_embeddingfile_fasttext(data_dir, language):
  fasttext_urls = {'english': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec', 'finnish': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fi.vec', 'german': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec', 'russian': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ru.vec', 'spanish': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec', 'swedish': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.sv.vec'}
  embeddings_path = os.path.join(data_dir, 'fasttext')
  if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)
  #downloadedfilepath = os.path.join(embeddings_path, language+'.vec')
  embeddings_fpath = os.path.join(embeddings_path, language+'.vec')
  #embeddings_fpath = os.path.join(embeddings_path, member_filename)
  if not os.path.exists(embeddings_fpath):
    url = fasttext_urls[language]
    print('Downloading {} to {}.'.format(url, embeddings_fpath))
    response = urlopen(url)
    CHUNK = 16 * 1024
    with open(embeddings_fpath, 'wb') as f:
      while True:
        chunk = response.read(CHUNK)
        if not chunk:
          break
        f.write(chunk)
    print('Download done.')
  else:
    print('found {} on disk.'.format(embeddings_fpath))
  embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_fpath, binary=False)#, unicode_errors='ignore')
  return embeddings

def get_embeddingfile_kyubyong(data_dir, gdrive_id, language, member_filename):
  embeddings_path = os.path.join(data_dir, 'embeddings')
  if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)
  downloadedfilepath = os.path.join(embeddings_path, language+'.zip')
  embeddings_fpath = os.path.join(embeddings_path, member_filename)
  if language == 'english':
    downloadedfilepath = os.path.join(embeddings_path, language+'.bin.gz')
    embeddings_fpath = downloadedfilepath[:-3]
  if not os.path.exists(embeddings_fpath):
    if not os.path.exists(downloadedfilepath):
      print('Downloading {} to {}.'.format(gdrive_id, downloadedfilepath))
      download_file_from_google_drive(gdrive_id, downloadedfilepath)
      print('Download done.')
    print('Unzipping {} to {}.'.format(downloadedfilepath, embeddings_fpath))
    if language == 'english':
      CHUNK = 16 * 1024
      with gzip.open(downloadedfilepath, 'rb') as f:
        with open(embeddings_fpath, 'wb') as ef:
          while True:
            chunk = f.read(CHUNK)
            if not chunk:
              break
            ef.write(chunk)
    else:
      with zipfile.ZipFile(downloadedfilepath) as zf:
        zf.extractall(path=embeddings_path)
    print('Unzip done.')
  else:
    print('found {} on disk: {}.'.format(gdrive_id, embeddings_fpath))
  if language == 'english':
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_fpath, binary=True)#, unicode_errors='ignore')
  else:
    embeddings = gensim.models.KeyedVectors.load(embeddings_fpath)#, binary=True, unicode_errors='ignore')
  #word2vec.Word2Vec.load_word2vec_format(embeddings_fpath, binary=True)
  return embeddings

def load_wordembeddings(data_dir, languages, fasttext=False):
  polyglot_urls = {'english': 'http://bit.ly/19bSoAS', 'finnish': 'http://bit.ly/19bSmJo', 'german': 'http://bit.ly/19bSoRk', 'russian': 'http://bit.ly/12FHYSM', 'spanish': 'http://bit.ly/19bSokw', 'swedish': 'http://bit.ly/19bTH2P'}
  kyubyong_urls = {'english': ('', 'https://drive.google.com/uc?export=download&confirm=IiDY&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'), 'finnish': ('fi.bin', 'https://drive.google.com/uc?export=download&confirm=IiDY&id=0B0ZXk88koS2KVnFyem4yQkxJUFk'), 'german': ('de.bin', 'https://drive.google.com/uc?export=download&confirm=IiDY&id=0B0ZXk88koS2KLVVLRWt0a3VmbDg'), 'russian': ('ru.bin', 'https://drive.google.com/uc?export=download&confirm=IiDY&id=0B0ZXk88koS2KMUJxZ0w0WjRGdnc'), 'spanish': ('es.bin', 'https://drive.google.com/uc?export=download&confirm=IiDY&id=0B0ZXk88koS2KNGNrTE4tVXRUZFU'), 'swedish': ('sv.bin', 'https://drive.google.com/uc?export=download&confirm=IiDY&id=0B0ZXk88koS2KNk1odTJtNkUxcEk')}
  kyubyong_ids = {'english': ('', '0B7XkCwpI5KDYNlNUTTlSS21pQmM'), 'finnish': ('fi.bin', '0B0ZXk88koS2KVnFyem4yQkxJUFk'), 'german': ('de.bin', '0B0ZXk88koS2KLVVLRWt0a3VmbDg'), 'russian': ('ru.bin', '0B0ZXk88koS2KMUJxZ0w0WjRGdnc'), 'spanish': ('es.bin', '0B0ZXk88koS2KNGNrTE4tVXRUZFU'), 'swedish': ('sv.bin', '0B0ZXk88koS2KNk1odTJtNkUxcEk')}
  # English:
  #   - info: https://code.google.com/archive/p/word2vec/
  #   - txt:  
  #   - bin:  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
  #   - polyglot: http://bit.ly/19bSoAS
  #   - kyubyong: n/a
  #
  # Finnish:
  #   - info: 
  #   - txt:  
  #   - bin:  
  #   - polyglot: http://bit.ly/19bSmJo
  #   - kyubyong: https://drive.google.com/open?id=0B0ZXk88koS2KVnFyem4yQkxJUFk
  #
  # German:
  #   - info: 
  #   - txt:  
  #   - bin:  
  #   - polyglot: http://bit.ly/19bSoRk
  #   - kyubyong: https://drive.google.com/open?id=0B0ZXk88koS2KLVVLRWt0a3VmbDg
  #
  # Russian:
  #   - info: 
  #   - txt:  
  #   - bin:  
  #   - polyglot: http://bit.ly/12FHYSM
  #   - kyubyong: https://drive.google.com/open?id=0B0ZXk88koS2KMUJxZ0w0WjRGdnc
  #
  # Spanish:
  #   - info: http://crscardellino.me/SBWCE/http://crscardellino.me/SBWCE/
  #   - txt:  http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.txt.bz2
  #   - bin:  http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz
  #   - polyglot: http://bit.ly/19bSokw
  #   - kyubyong: https://drive.google.com/open?id=0B0ZXk88koS2KNGNrTE4tVXRUZFU
  #
  # Swedish:
  #   - info: http://demo.spraakdata.gu.se/richard/w2v/
  #   - txt:  
  #   - bin:  http://demo.spraakdata.gu.se/richard/w2v/cbow0_s512_w5.bin
  #   - polyglot: http://bit.ly/19bTH2P
  #   - kyubyong: https://drive.google.com/open?id=0B0ZXk88koS2KNk1odTJtNkUxcEk
  #
  # Polyglot info: https://sites.google.com/site/rmyeid/projects/polyglot
  # Kyobyung info: CBOW. https://github.com/Kyubyong/wordvectors/
  embeddings = {}
  for l in languages: #['english', 'finnish', 'german', 'russian', 'spanish', 'swedish']:
    #embeddings[l] = get_embeddingfile(l, polyglot_urls[l], l+'.pkl')
    if fasttext:
      embeddings[l] = get_embeddingfile_fasttext(data_dir, l)
    else:
      if l not in kyubyong_ids:
        print('{} was not specified in kyubhyong_ids. Returning None.'.format(l))
        return None
      embeddings[l] = get_embeddingfile_kyubyong(data_dir, kyubyong_ids[l][1], l, kyubyong_ids[l][0])
  if not os.path.exists(os.path.join(data_dir, 'embeddings/english.bin')):
    print('Please download the following files, and name them [language].bin, into {}'.format(os.path.join(data_dir, 'embeddings')))
    for l in languages: #kyubyong_urls:
      print('{}: {}'.format(l, kyubyong_urls[l][1]))
  return embeddings

def download_file_from_google_drive(id, destination):
  '''
    Credits to Andrew Hundt!
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
  '''
  def get_confirm_token(response):
    for key, value in response.cookies.items():
      if key.startswith('download_warning'):
        return value

    return None

  def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
      for chunk in response.iter_content(CHUNK_SIZE):
        if chunk: # filter out keep-alive new chunks
          f.write(chunk)

  URL = "https://docs.google.com/uc?export=download"

  session = requests.Session()

  response = session.get(URL, params = { 'id' : id }, stream = True)
  token = get_confirm_token(response)

  if token:
    params = { 'id' : id, 'confirm' : token }
    response = session.get(URL, params = params, stream = True)

  save_response_content(response, destination)    

def randomChoice(l):
  return l[random.randint(0, len(l) - 1)]

