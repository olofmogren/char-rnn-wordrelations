#!/usr/bin/python

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
from urllib.request import urlopen
import urllib3, os


data_url_english    = 'https://raw.githubusercontent.com/en-wl/wordlist/master/alt12dicts/2of12id.txt'


def get_english_data(data_dir):
  d = {}
  vocab = set()
  tags = {}

  language = 'english'
  d[language] = {}
  local_fname = os.path.join(data_dir, '2of12id.txt')
  if not os.path.exists(local_fname):
    with urlopen(data_url_english) as f_web:
      with open(local_fname, 'w') as f_local:
        print('downloading {}'.format(data_url_english))
        for line in f_web:
          f_local.write(line.decode('utf-8'))
  else:
    print('found {} on disk.'.format(data_url_english))

  with open(local_fname, 'r') as f_web:
    for line in f_web:
      #print(line)
      #words = [x.decode('utf-8') for x in line.split()]
      words = line.split()
      #print(words)
      for i in range(len(words)-1,-1,-1):
        if words[i][0] == '~' or words[i][0] == '+' or words[i][0] == '-' or words[i][0] == '!':
          words[i] = words[i][1:]
          if words[i] == '':
            del words[i]
      first = words[0]
      vocab.update(set(first))
      POS = words[1]
      for i in range(len(words)-1,1,-1):
        if words[i][0] == '(' or words[i][-1] ==')':
          del words[i]
        elif words[i][0] == '{' or words[i][-1] =='}':
          del words[i]
        elif words[i] == '|' or words[i] =='/':
          if i+1 < len(words):
            del words[i+1]
          del words[i]
      paradigm = {}
      if POS == 'A:':
        # we don't seem to have inflections of these.
        #print('adjective')
        if len(words) != 4:
          #print('Unexpected length of adjective line: {}, {}. Ignoring.'.format(line, words))
          continue
        paradigm['comp=POS,pos=ADJ'] = first
        add_to_dictset(tags, [('comp', 'POS'),('pos','ADJ')])

        second = words[2]
        vocab.update(set(second))
        if len(second) >= 0:
          paradigm['comp=CMPR,pos=ADJ'] = second
          add_to_dictset(tags, [('comp', 'CMPR'),('pos', 'ADJ')])
        third = words[3]
        vocab.update(set(third))
        if len(third) >= 0:
          paradigm['comp=SPRL,pos=ADJ'] = third
          add_to_dictset(tags, [('comp', 'SPRL')])
        if len(paradigm.keys()) > 1:
          if 'ADJ' not in d[language]:
            d[language]['ADJ'] = []
          d[language]['ADJ'].append(paradigm)
          #print(paradigm)
      elif POS == 'N:':
        paradigm['num=SG,pos=N'] = first
        # most of the time, these have only one inflection, the plural.
        # Sometimes, there is an alternative form in parentheses.
        if len(words) != 3:
          #print('Unexpected length of noun line: {}, {}. Ignoring.'.format(line, words))
          continue
        second = words[2]
        paradigm['num=PL,pos=N'] = second
        add_to_dictset(tags, [('num', 'PL'),('pos', 'N')])
        if len(paradigm.keys()) > 1:
          if 'N' not in d[language]:
            d[language]['N'] = []
          d[language]['N'].append(paradigm)
      elif POS == 'V:':
        paradigm['finite=NFIN,pos=V'] = first
        #print('verb')
        if len(words) != 5 and len(words) != 6:
          #print('Unexpected length of verb line: {}, {}. Ignoring.'.format(line, words))
          continue
        second = words[2]
        paradigm['pos=V,tense=PST'] = second
        # sometimes, there is an extra form in second place. Sometimes not. Indexing from end.
        third = words[-2]
        vocab.update(set(third))
        paradigm['finite=NFIN,pos=V,tense=PRS.PROGR'] = third
        add_to_dictset(tags, [('finite','NFIN'),('pos','V'),('tense','PRS.PROGR')])
        fourth = words[-1]
        vocab.update(set(fourth))
        paradigm['num=SG,per=3,pos=V,tense=PRS'] = fourth
        add_to_dictset(tags, [('num','SG'),('per','3'),('pos','V'),('tense','PRS')])
        if len(paradigm.keys()) > 1:
          if 'V' not in d[language]:
            d[language]['V'] = []
          d[language]['V'].append(paradigm)

  return d, vocab, tags

def add_to_dictset(d, list_of_pairs):
  for key,val in list_of_pairs:
    d.setdefault(key, set())
    d[key].add(val)

def get_english_data_old(data_dir):
  all_relations = {}
  vocab = set()

  language = 'english'
  all_relations[language] = {}
  local_fname = os.path.join(data_dir, '2of12id.txt')
  if not os.path.exists(local_fname):
    with urlopen(data_url_english) as f_web:
      with open(local_fname, 'w') as f_local:
        print('downloading {}'.format(data_url_english))
        for line in f_web:
          f_local.write(line.decode('utf-8'))
  else:
    print('found {} on disk.'.format(data_url_english))

  with open(local_fname, 'r') as f_web:
    for line in f_web:
      #print(line)
      #words = [x.decode('utf-8') for x in line.split()]
      words = line.split()
      #print(words)
      for i in range(len(words)-1,-1,-1):
        if words[i][0] == '~' or words[i][0] == '+' or words[i][0] == '-' or words[i][0] == '!':
          words[i] = words[i][1:]
          if words[i] == '':
            del words[i]
      first = words[0]
      vocab.update(set(first))
      POS = words[1]
      for i in range(len(words)-1,1,-1):
        if words[i][0] == '(' or words[i][-1] ==')':
          del words[i]
        elif words[i][0] == '{' or words[i][-1] =='}':
          del words[i]
        elif words[i] == '|' or words[i] =='/':
          if i+1 < len(words):
            del words[i+1]
          del words[i]
      if POS == 'A:':
        # we don't seem to have inflections of these.
        #print('adjective')
        if len(words) != 4:
          #print('Unexpected length of adjective line: {}, {}. Ignoring.'.format(line, words))
          continue
        second = words[2]
        vocab.update(set(second))
        if len(second) == 0:
          continue
        if 'a_comparative' not in all_relations[language]:
          all_relations[language]['a_comparative'] = []
        all_relations[language]['a_comparative'].append((first,second))
        third = words[3]
        vocab.update(set(third))
        if len(third) == 0:
          continue
        if 'a_superlative' not in all_relations[language]:
          all_relations[language]['a_superlative'] = []
        all_relations[language]['a_superlative'].append((first,third))
        if 'a_comparative_superlative' not in all_relations[language]:
          all_relations[language]['a_comparative_superlative'] = []
        all_relations[language]['a_comparative_superlative'].append((second,third))
      elif POS == 'N:':
        # most of the time, these have only one inflection, the plural.
        # Sometimes, there is an alternative form in parentheses.
        # Currently, we discard these (see for loop above).
        # TODO: test to just include them?
        #print('noun')
        if len(words) != 3:
          #print('Unexpected length of noun line: {}, {}. Ignoring.'.format(line, words))
          continue
        second = words[2]
        if 'n_plural' not in all_relations[language]:
          all_relations[language]['n_plural'] = []
        all_relations[language]['n_plural'].append((first,second))
      elif POS == 'V:':
        #print('verb')
        if len(words) != 5 and len(words) != 6:
          #print('Unexpected length of verb line: {}, {}. Ignoring.'.format(line, words))
          continue
        second = words[2]
        if 'v_imperfect' not in all_relations[language]:
          all_relations[language]['v_imperfect'] = []
        all_relations[language]['v_imperfect'].append((first,second))
        # sometimes, there is an extra form in second place. Sometimes not. Indexing from end.
        third = words[-2]
        if 'v_progressive' not in all_relations[language]:
          all_relations[language]['v_progressive'] = []
        all_relations[language]['v_progressive'].append((first,third))
        fourth = words[-1]
        vocab.update(set(fourth))
        if 'v_presence' not in all_relations[language]:
          all_relations[language]['v_presence'] = []
        all_relations[language]['v_presence'].append((first,fourth))

  tags = set(all_relations[language].keys())

  return all_relations, vocab, tags

