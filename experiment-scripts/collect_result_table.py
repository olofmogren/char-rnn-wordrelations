#!/usr/bin/python3

import os, sys

if len(sys.argv) > 1:
  data_dir = sys.argv[1]
else:
  print('Please specify path.')
  exit()
#data_dir = '/home/mogren/experiments/2017-char-rnn-wordrelations/nov-monolingual-resplit/tie_rel-F0.0-keep0.6/'
print(data_dir)

med_results = {'arabic': '97.38', 'finnish': '97.40', 'georgian': '99.14', 'german': '97.45', 'hungarian': '99.67', 'maltese': '88.17', 'navajo': '96.64', 'russian': '91.00', 'spanish': '98.74', 'turkish': '97.94'}

CP = False

def pretty_print_markdown_table(t, latex_conversion=False, first_pref_suff=None, global_pref_suff=None):
  separator = '&' if latex_conversion else '|'
  beginning_of_line = '' if latex_conversion else '| '
  end_of_line = '\\\\' if latex_conversion else ' |'
  sizes = []
  right_margins = []
  for line in t.split('\n'):
    for i,col in enumerate([c.strip() for c in line.split('|') if c.strip()]):
      if '---' in col:
        if i >= len(right_margins):
          right_margins.append(col[-1] == ':')
        continue
      size = len(col)
      if i >= len(sizes):
        sizes.append(max(size, 5))
      else:
        sizes[i] = max(sizes[i], size)
  lines = []
  for l,line in enumerate(t.split('\n')):
    resultline = []
    for i,col in enumerate([c.strip() for c in line.split('|') if c.strip()]):
      if '---' in col:
        # midrule:
        if latex_conversion:
          if i == 0:
            resultline.append('\midrule')
        else:
          resultline.append(' '+'-'*(sizes[i]-2)+(':' if right_margins[i] else ' '))
      else:
        content = (col.rjust(sizes[i]) if right_margins[i] else col.ljust(sizes[i]))
        if latex_conversion and first_pref_suff is not None and (l == 0 or i == 0):
          content = first_pref_suff[0]+content+first_pref_suff[1]
        elif latex_conversion and global_pref_suff is not None and (l > 0 or i > 0):
          content = global_pref_suff[0]+content+global_pref_suff[1]
        resultline.append(content)
    lines.append(beginning_of_line+(separator.join(resultline))+(end_of_line if 'midrule' not in resultline else ''))
  res = '\n'.join(lines).replace('\midrule\\\\', '\midrule ')
  if latex_conversion:
    res = res.replace('%', '\%')
  return res
        

score_lines = []
#levenshtein_lines = []
for language in os.listdir(data_dir):
  p = os.path.join(data_dir, language)
  score = {}
  levenshtein = {}
  col = 4
  #col_poet = 7
  if language == 'english' or language == 'swedish':
    # for these languages,  we evaluate both directions.
    col = 3
    #col_poet = 8
  for system in ['model', 'copysuffix_baseline', 'lepage_baseline']: #, 'fasttext_baseline', 'fasttext_hybrid_baseline', 'wordembedding_baseline', 'hybrid_baseline']:
    score[system] = float('nan')
    levenshtein[system] = float('nan')
    results_filename = os.path.join(p, 'test-{}.data'.format(system))
    if not os.path.exists(results_filename):
      print('{} not found. Ongoing training?'.format(results_filename))
      continue
    with open(results_filename) as f:
      for line in f:
        if line.startswith('#'):
          continue
        words = line.split(' ')
        if len(words) <= col: # or len(words) <= col_poet:
          print('Line: \'{}\', looks fishy.'.format(line))
        score[system] = float(words[col])
        #score_poet = float(words[col_poet])
    results_filename = os.path.join(p, 'test-levenshteins-{}.data'.format(system))
    if not os.path.exists(results_filename):
      print('{} not found. Ongoing training?'.format(results_filename))
      continue
    with open(results_filename) as f:
      for line in f:
        if line.startswith('#'):
          continue
        words = line.split(' ')
        levenshtein[system] = float(words[col])
        #levenshtein_copysuffix_poet = float(words[col_poet])
  #score_lines.append('| {}{} | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% |'.format(language[0].upper(), language[1:], score['model'], score['copysuffix_baseline'], score['wordembedding_baseline'], score['hybrid_baseline'], score['fasttext_baseline'], score['fasttext_hybrid_baseline']))
  #levenshtein_lines.append('| {}{} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(language[0].upper(), language[1:], levenshtein['model'], levenshtein['copysuffix_baseline'], levenshtein['wordembedding_baseline'], levenshtein['hybrid_baseline'], levenshtein['fasttext_baseline'], levenshtein['fasttext_hybrid_baseline']))
  cp_accuracy = '{:.2f}% |'.format(score['copysuffix_baseline']) if CP else ''
  cp_levenshtein = '{:.2f}% |'.format(score['copysuffix_baseline']) if CP else ''
  score_lines.append('| {}{} | {:.2f}% | {:.2f}% | {} ~ | {:.2f} | {:.2f} | {}'.format(language[0].upper(), language[1:], score['model'], score['lepage_baseline'], cp_accuracy, levenshtein['model'], levenshtein['lepage_baseline'], cp_levenshtein))
  #levenshtein_lines.append('| {}{} | {:.2f} | {:.2f} |'.format(language[0].upper(), language[1:], levenshtein['model'], levenshtein['copysuffix_baseline']))
  #, med_results.get(language, 'n/a')))

#md ='| ~ | {\modelshortname} | CP  | WV | HYB | FT | FTH |\n'
#md += '| --------- | -----------------:| ---:| ---:| ---:| ---:| ---:|\n'
md ='| ~ | {{\modelshortname}} | Lepage | {}~ | {{\modelshortname}} | Lepage   | {}\n'.format('CP  ' if CP else '', 'CP  ' if CP else '')
md += '| --------- | -----------------:| ---:| {} ---:| -----------------:| ---:| {}\n'.format('---:|' if CP else '', '---:|' if CP else '')
md += '\n'.join(sorted([l for l in score_lines]))

print('Scores:')
print('Latex:\n')
print(pretty_print_markdown_table(md, latex_conversion=True, first_pref_suff=('\\bf ',''), global_pref_suff=('','')))
print('\n\nMarkdown:\n')
print(pretty_print_markdown_table(md))



#md ='| ~  | {\modelshortname} | CP |\n'
#md += '| --------- | -----------------:| ---:|\n'
#md += '\n'.join(sorted([l for l in levenshtein_lines]))

#print('Avg Levenshtein:')
#print('Latex:\n')
#print(pretty_print_markdown_table(md, latex_conversion=True, first_pref_suff=('\\bf ',''), global_pref_suff=('','')))
#print('\n\nMarkdown:\n')
#print(pretty_print_markdown_table(md))

