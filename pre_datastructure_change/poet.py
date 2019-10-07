
def poet(source, prediction, relations, language, example_tree):
  '''
    Here: lookup an example tree instead of 'knowing' the relation?
  '''
  tree_results = []
  candidate_trees = set()
  t1 = time.time()
  t4 = t3 = t2 = t1
  # Try out example_tree first. Quick fix.
  res = apply_edit_tree(source, example_tree)
  if res is not None:
    if prediction != res and abs(len(prediction)-len(res)) <= 1:
        if levenshtein(prediction,res) == 1:
          return res, 0.0, 0.0, 0.0
  for i,r in enumerate(relations['train'][language]):
    if len(relations['train'][language]) > 50 or example_tree in relations['train'][language][i]['edittrees']:
      candidate_trees.update(relations['train'][language][i]['edittrees'])
    if len(relations['train'][language]) > 50 or example_tree in relations['train'][language][i]['edittrees_r']:
      candidate_trees.update(relations['train'][language][i]['edittrees_r'])
  t2 = time.time()
  for tree in candidate_trees:
    res = apply_edit_tree(source, tree)
    if res is not None:
      tree_results.append((res, tree))
  t3 = time.time()
  matches = []
  for (res,tree) in tree_results:
    if res == prediction:
      return prediction, t2-t1, t3-t2, time.time()-t3
  for (res,tree) in tree_results:
    if prediction != res and abs(len(prediction)-len(res)) <= 1:
      if levenshtein(prediction,res) == 1:
        #print('found matching tree for source {}, and prediction {}: {}, result:: {}'.format(source, prediction, tree, r))
        matches.append(res)
  t4 = time.time()
  if len(matches):
    return randomChoice(matches), t2-t1, t3-t2, t4-t3
  return prediction, t2-t1, t3-t2, t4-t3

def poet_cheat(source, prediction, relations, language, relation):
  '''
    Cheating. Using relation information that _should_ not be available.
  '''
  tree_results = []
  for tree in relations['train'][language][relation]:
    r = apply_edit_tree(source, tree)
    if r is not None:
      tree_results.append((r, tree))
  matches = []
  for (r,tree) in tree_results:
    if levenshtein(prediction,r) == 1:
      #print('found matching tree for source {}, and prediction {}: {}, result:: {}'.format(source, prediction, tree, r))
      matches.append(r)
  if len(res):
    return randomChoice(res)
  return prediction

def levenshtein(seq1, seq2):
  '''
    Trusting https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    on this one. Just python3-ified it a bit.
  '''
  oneago = None
  thisrow = list(range(1, len(seq2) + 1)) + [0]
  for x in range(len(seq1)):
    twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
    for y in range(len(seq2)):
      delcost = oneago[y] + 1
      addcost = thisrow[y - 1] + 1
      subcost = oneago[y - 1] + (seq1[x] != seq2[y])
      thisrow[y] = min(delcost, addcost, subcost)
  return thisrow[len(seq2) - 1]

def compute_edit_trees(relations):
  p = 'train'
  for l in relations[p]:
    for r in relations[p][l]:
      r['edittrees'] = set()
      r['edittrees_r'] = set()
      for (w1,t1),(w2,t2) in r['wordpairs']:
        r['edittrees'].add(get_edit_tree(w1,w2))
        r['edittrees_r'].add(get_edit_tree(w2,w1))

def apply_edit_tree(source, tree):
  if tree is None:
    return source
  if tree[0] == 'replace':
    if source == tree[1]:
      return tree[2]
    else:
      return None
  if tree[0] == 'edit':
    prefix = apply_edit_tree(source[:tree[1]], tree[2])
    suffix = apply_edit_tree(source[len(source)-tree[3]:], tree[4])
    if prefix is None or suffix is None:
      return None
    return prefix+source[tree[1]:len(source)-tree[3]]+suffix

def get_edit_tree(source, target):
  if len(source) == 0 or len(target) == 0:
    return ('replace', source, target)
  lcs = longest_common_substring(source, target)
  if len(lcs):
    begin_s   = source.find(lcs)
    neg_end_s = len(source)-begin_s-len(lcs)
    begin_t   = target.find(lcs)
    neg_end_t = len(target)-begin_t-len(lcs)
    left_tree = None
    right_tree = None
    if begin_s > 0 or begin_t > 0:
      left_tree = get_edit_tree(source[:begin_s], target[:begin_t])
    if neg_end_s > 0 or neg_end_t > 0:
      right_tree = get_edit_tree(source[len(source)-neg_end_s:], target[len(target)-neg_end_t:])
  else:
    return ('replace', source, target)
    
  return ('edit', begin_s, left_tree, neg_end_s, right_tree)

def longest_common_substring(s1, s2):
   m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
   longest, x_longest = 0, 0
   for x in range(1, 1 + len(s1)):
     for y in range(1, 1 + len(s2)):
       if s1[x - 1] == s2[y - 1]:
         m[x][y] = m[x - 1][y - 1] + 1
         if m[x][y] > longest:
           longest = m[x][y]
           x_longest = x
       else:
         m[x][y] = 0
   return s1[x_longest - longest: x_longest]

def randomChoice(l):
