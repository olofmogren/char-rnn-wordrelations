#!/usr/bin/python3

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import matplotlib

data_dir = os.path.join(sys.argv[1], 'saved_embeddings/validation/')
#data_dir = '/home/mogren/experiments/2017-char-rnn-wordrelations/nov-monolingual-resplit/tie_rel-F0.0-keep0.6/english/saved_embeddings/validation'
data_file_relation   = os.path.join(data_dir,'relation_encoder.dat')
data_file_query      = os.path.join(data_dir,'query_encoder.dat')
data_file_combined   = os.path.join(data_dir,'combined_encoder.dat')
data_file_characters = os.path.join(sys.argv[1],'saved_embeddings/character-embeddings.dat')
data_file_vocab      = os.path.join(sys.argv[1],'vocab.pkl')
# data_file_relation = '/home/mogren/experiments/2017-char-rnn-wordrelations/nov-monolingual-resplit-covreg/tie_rel-F0.0-covreg0.01-keep0.6/english/saved_embeddings/validation/relation_encoder.dat'

print('\n'.join([data_file_relation, data_file_query, data_file_combined]))

#FINNISH:
#pos_to_color = 'case=IN+LAT,num=PL,pos=N'
#pos_to_color = 'case=ON+ESS,num=PL,pos=N'
#pos_to_color = 'pos=ADJ'
pos_to_color = 'num=SG,pos=N-case=ON+ESS,num=PL,pos=N'
#GERMAN:
#pos_to_color = 'gen=NEUT,num=PL,pos=N'
#RUSSIAN:
#pos_to_color = 'case=NOM,num=PL,pos=N'
#SWEDISH,ENGLISH:
#pos_to_color = None

plot_relations  = True
plot_queries    = True
plot_combined   = False
plot_characters = False
css_colors=['AliceBlue','Aqua','Aquamarine','Azure','Bisque','Black','BlanchedAlmond','Blue','BlueViolet','Brown','BurlyWood','CadetBlue','Chartreuse','Chocolate','Coral','CornflowerBlue','Cornsilk','Crimson','Cyan','DarkBlue','DarkCyan','DarkGoldenRod','DarkGray','DarkGrey','DarkGreen','DarkKhaki','DarkMagenta','DarkOliveGreen','DarkOrange','DarkOrchid','DarkRed','DarkSalmon','DarkSeaGreen','DarkSlateBlue','DarkSlateGray','DarkSlateGrey','DarkTurquoise','DarkViolet','DeepPink','DeepSkyBlue','DimGray','DimGrey','DodgerBlue','FireBrick','ForestGreen','Fuchsia','Gainsboro','Gold','GoldenRod','Gray','Green','GreenYellow','HotPink','IndianRed','Indigo','Ivory','Lavender','LavenderBlush','LawnGreen','LightBlue','LightCoral','LightCyan','LightGreen','LightPink','LightSalmon','LightSeaGreen','LightSlateGray','LightSlateGrey','LightSteelBlue','LightYellow','Lime','LimeGreen','Magenta','Maroon','MediumAquaMarine','MediumBlue','MediumOrchid','MediumPurple','MediumSeaGreen','MediumSlateBlue','MediumSpringGreen','MediumTurquoise','MediumVioletRed','MidnightBlue','MistyRose','Moccasin','Navy','OldLace','Olive','OliveDrab','Orange','OrangeRed','Orchid','PapayaWhip','PeachPuff','Peru','Pink','Plum','Purple','RebeccaPurple','Red','RosyBrown','RoyalBlue','SaddleBrown','Salmon','SandyBrown','SeaGreen','SeaShell','Sienna','SkyBlue','SlateBlue','SlateGray','SlateGrey','SpringGreen','SteelBlue','Tan','Teal','Thistle','Tomato','Turquoise','Violet','Yellow','YellowGreen']
#selected_colors=['Black','Blue','DarkGreen','DarkRed','BlueViolet','Red','Chartreuse','Coral','CornflowerBlue','Crimson','Cyan','DarkBlue','DarkCyan','DarkMagenta','DarkOliveGreen','DarkSalmon','DarkSeaGreen','DarkSlateBlue','DarkTurquoise','DarkViolet','DeepPink','DeepSkyBlue','DodgerBlue','FireBrick','FloralWhite','ForestGreen','Fuchsia','Gainsboro','GhostWhite','Gold','GoldenRod','Green','GreenYellow','HoneyDew','HotPink','IndianRed ','Indigo ','Ivory','Khaki','Lavender','LavenderBlush','LawnGreen','LemonChiffon','LightBlue','LightCoral','LightCyan','LightGoldenRodYellow','LightGreen','LightPink','LightSalmon','LightSeaGreen','LightSkyBlue','LightSteelBlue','LightYellow','Lime','LimeGreen','Linen','Magenta','Maroon','MediumAquaMarine','MediumBlue','MediumOrchid','MediumPurple','MediumSeaGreen','MediumSlateBlue','MediumSpringGreen','MediumTurquoise','MediumVioletRed','MidnightBlue','MintCream','MistyRose','Moccasin','NavajoWhite','Navy','OldLace','Olive','OliveDrab','Orange','OrangeRed','Orchid','PaleGoldenRod','PaleGreen','PaleTurquoise','PaleVioletRed','PapayaWhip','PeachPuff','Peru','Pink','Plum','PowderBlue','Purple','RebeccaPurple','RoyalBlue','Salmon','SeaGreen','SeaShell','Sienna','Silver','SkyBlue','SlateBlue','Snow','SpringGreen','SteelBlue','Tan','Teal','Thistle','Tomato','Turquoise','Violet','Wheat','White','WhiteSmoke','Yellow','YellowGreen']
#plt.rc('text', usetex=True)
#plt.rc('font', family='sans-serif', size=22)
#plt.rc('font', size=18)

t = TSNE()
#a = np.loadtxt(open(csv_file, "rb"), delimiter=",")
demo_word1 = []
demo_word2 = []
query_word = []
target_word = []
relation_type = []
listoflists = []
all_relation_types = set()
with open(data_file_relation, 'r') as f:
  line = f.readline()
  while line:
    #print(line)
    if line.startswith('#'):
      line = f.readline()
      continue
    words = line.split(' ')
    demo_word1.append(words[0])
    demo_word2.append(words[1])
    query_word.append(words[2])
    target_word.append(words[3])
    relation_type.append(words[4])
    all_relation_types.add(words[4])
    listoflists.append([float(w) for w in words[5:]])
    line = f.readline()

all_relation_types = sorted(list(all_relation_types))
all_relation_types = [t for t in all_relation_types if 'pos=ADJ' in t]+\
                     [t for t in all_relation_types if 'pos=N' in t]+\
                     [t for t in all_relation_types if 'pos=V' in t]
list_relation_type = [all_relation_types.index(x) for x in relation_type]
np_relation_type = np.array(list_relation_type)
a = np.array(listoflists)
print('a dimensions: {}'.format(a.shape))

format_tag = {'CMPR': 'Comparative', 'POS': 'Positive', 'SPRL': 'Superlative', 'NFIN': 'Infinitive', 'PRS.PROGR': 'Progressive', 'PST': 'Past', 'PL': 'Plural', 'SG': 'Singular', 'PRS': 'Presence'}

def format_relation_types(ts):
  new_ts = []
  for t in ts:
    pos = ''
    source = []
    target = []
    if t.endswith('_r'):
      t2 = t[:-2].split('-')
      s0 = t2[1]
      s1 = t2[0]
    else:
      t2 = t.split('-')
      s0 = t2[0]
      s1 = t2[1]
    l0 = set(s0.split(','))
    l1 = set(s1.split(','))
    l0 = l0.difference(l1)
    l1 = l1.difference(l0)
    for tag in sorted(list(l0)):
      tags = tag.split('=')
      #print(tags[0])
      if tags[0] == 'pos':
        pos = tags[1]
        #print('pos!')
      else:
        key0 = tags[0]
        val0 = format_tag.get(tags[1], tags[1])
        source.append(val0)
    for tag in sorted(list(l1)):
      tags = tag.split('=')
      #print(tags[0])
      if tags[0] == 'pos':
        pos = tags[1]
        #print('pos!')
      else:
        val1 = format_tag.get(tags[1], tags[1])
        target.append(val1)
    if not source:
      if pos == 'V':
        source.append('Infinitive')
      else:
        source.append('Base')
    if not target:
      if pos == 'V':
        target.append('Infinitive')
      else:
        target.append('Base')
    #new_ts.append(('(■) ' if t.endswith('_r') else '(♦) ')+pos+': '+(' '.join(source))+'-'+(' '.join(target)))
    new_ts.append(('(x)  ' if t.endswith('_r') else '(+) ')+pos+': '+(' '.join(source))+'-'+(' '.join(target)))
  new_ts = [x.replace('Singular 3', '3rd Pers. Sg.') for x in new_ts]
  new_ts = [x.replace('Infinitive Progressive', 'Progressive') for x in new_ts]
  return new_ts


selected_colors=['xkcd:cyan', 'xkcd:blue', 'xkcd:beige', 'xkcd:brown', 'xkcd:red', 'xkcd:magenta', 'xkcd:orange', 'xkcd:sky blue', 'xkcd:dark green', 'xkcd:dark blue', 'xkcd:purple', 'xkcd:lilac', 'xkcd:dark pink', 'xkcd:blue green', 'xkcd:green', 'xkcd:purplish blue', 'xkcd:dark violet', 'xkcd:yellow', 'xkcd:pink', 'xkcd:black']
selected_colors += css_colors
#markers = ["x",".","1","2","3","4","8","P","*","+","x","D","d","|","_","o","o","s","p","h","H","X", ",","v","^","<",">"]

if plot_relations:
  print('plot_relations')
  print('Computing T-SNE')
  a_projected = t.fit_transform(a)
  print('a_projected dimensions: {}'.format(a_projected.shape))
  print('Done computing T-SNE.')

  #fig, (ax,lax) = plt.subplots(ncols=2)#, gridspec_kw={"width_ratios":[4,1]})
  recs = []
  color_idx = 0
  for i in range(0,len(all_relation_types)):
    recs.append(mpatches.Circle((0,0),1,fc=selected_colors[int(i/2)%len(selected_colors)]))
    n = (np_relation_type==i).sum()
    if pos_to_color is not None and pos_to_color not in all_relation_types[i]:
      colors = ['xkcd:grey']*n
      mark = '.'
    else:
      colors = [selected_colors[int(color_idx/2)%len(selected_colors)]]*n
      color_idx += 1
      #mark = 's' if all_relation_types[i].endswith('_r') else 'D'
      mark = 'x' if all_relation_types[i].endswith('_r') else '+'
    #plt.scatter(a_projected[np_relation_type==i, 0], a_projected[np_relation_type==i, 1], s=1, c=colors)
    #ax.scatter(a_projected[np_relation_type==i, 0], a_projected[np_relation_type==i, 1], s=25, c=colors, marker=mark)
    plt.scatter(a_projected[np_relation_type==i, 0], a_projected[np_relation_type==i, 1], s=25, c=colors, marker=mark)

  print(all_relation_types)
  #plt.legend(recs,all_relation_types,loc=7)
  if pos_to_color is not None:
    print('coloring: {}'.format(' '.join([t for t in all_relation_types if pos_to_color in t])))
    plt.legend(recs,format_relation_types([t for t in all_relation_types if pos_to_color in t]),
               #loc=7,
               #bbox_to_anchor=(1, 1),
               #bbox_transform=plt.gcf().transFigure
               #loc='right'
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  else:
    plt.legend(recs,format_relation_types(all_relation_types)[:20],
               #loc=7,
               #bbox_to_anchor=(1, 1),
               #bbox_transform=plt.gcf().transFigure
               #loc='right'
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.axis('off')
  #plt.legend(loc=4)
  #plt.axis('tight')
  #plt.tight_layout()
  plt.savefig("/home/mogren/relation-embeddings.pdf", bbox_inches="tight")
  plt.show()

listoflists = []
with open(data_file_query, 'r') as f:
  line = f.readline()
  while line:
    #print(line)
    if line.startswith('#'):
      line = f.readline()
      continue
    words = line.split(' ')
    listoflists.append([float(w) for w in words[5:]])
    line = f.readline()

q = np.array(listoflists)
print('q dimensions: {}'.format(q.shape))
#print('Computing T-SNE')
#q_projected = t.fit_transform(q)
#print('q_projected dimensions: {}'.format(q_projected.shape))
#print('Done computing T-SNE.')
if plot_queries:
  print('plot_queries')
  print('Computing T-SNE')
  q_projected = t.fit_transform(q)
  print('q_projected dimensions: {}'.format(q_projected.shape))
  print('Done computing T-SNE.')

  #fig, (ax,lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":[4,1]})
  recs = []
  for i in range(0,len(all_relation_types)):
    recs.append(mpatches.Circle((0,0),1,fc=selected_colors[int(i/2)%len(selected_colors)]))
    n = (np_relation_type==i).sum()
    colors = [selected_colors[int(i/2)%len(selected_colors)]]*n
    #mark = 's' if all_relation_types[i].endswith('_r') else 'D'
    mark = 'x' if all_relation_types[i].endswith('_r') else '+'
    plt.scatter(q_projected[np_relation_type==i, 0], q_projected[np_relation_type==i, 1], s=25, c=colors, marker=mark)

  print(all_relation_types)
  #plt.legend(recs,all_relation_types,loc=7)
  plt.legend(recs,format_relation_types(all_relation_types)[:20],
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.axis('off')
  #plt.legend(loc=4)
  #plt.axis('tight')
  #plt.tight_layout()
  plt.savefig("/home/mogren/query-embeddings.pdf", bbox_inches="tight")
  plt.show()

if plot_combined:
  print('plot_combined')
  listoflists = []
  with open(data_file_combined, 'r') as f:
    line = f.readline()
    while line:
      #print(line)
      if line.startswith('#'):
        line = f.readline()
        continue
      words = line.split(' ')
      listoflists.append([float(w) for w in words[5:]])
      line = f.readline()

  c = np.array(listoflists)
  print('c dimensions: {}'.format(c.shape))

  # a = relations
  # q = querys
  # c = combined
  print('Concatenating data...')

  # [(n+n+n)*100] -> [(n+n+n)*2]
  catted = np.concatenate((q,a), axis=0)
  print('catted dimensions: {}'.format(catted.shape))
  print('Computing T-SNE')
  catted_projected = t.fit_transform(catted)
  print('catted_projected dimensions: {}'.format(catted_projected.shape))
  print('Done computing T-SNE.')

  # [n*2]
  a_joint_projected = catted_projected[0:a.shape[0],:]
  q_joint_projected = catted_projected[a.shape[0]:,:]
  #c_joint_projected = catted_projected[a.shape[0]+q.shape[0]:,:]

  # [n*2*2]
  aqc_joint_projected = np.stack((a_joint_projected, q_joint_projected), axis=1)
  print(aqc_joint_projected.shape)

  #fig, (ax,lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":[4,1]})
  recs = []
  for i in range(0,len(all_relation_types)):
    recs.append(mpatches.Circle((0,0),1,fc=selected_colors[int(i/2)%len(selected_colors)]))
    n = (np_relation_type==i).sum()
    colors = [selected_colors[int(i/2)%len(selected_colors)]]*n
    #mark = 's' if all_relation_types[i].endswith('_r') else 'D'
    mark = 'x' if all_relation_types[i].endswith('_r') else '+'
    indices = np_relation_type==i
    print(indices)
    for j,b in enumerate(indices):
      if b:
        print(aqc_joint_projected.shape)
        print(j)
        print(b)
        x = aqc_joint_projected[j, :, 0]#.squeeze(axis=0)
        print(x.shape)
        y = aqc_joint_projected[j, :, 1]#.squeeze(axis=0)
        print(y.shape)
        #ax.plot(x, y, s=25, c=colors, marker=mark)
        plt.plot(x, y, marker=mark, color=selected_colors[int(i/2)%len(selected_colors)])

  print(all_relation_types)
  #plt.legend(recs,all_relation_types,loc=7)
  plt.legend(recs,format_relation_types(all_relation_types)[:20],
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.axis('off')
  #plt.legend(loc=4)
  #plt.axis('tight')
  #plt.tight_layout()
  plt.savefig("/home/mogren/combined-embeddings.pdf", bbox_inches="tight")
  plt.show()


e = np.loadtxt(open(data_file_characters , "rb"), delimiter=",")
vocab = pickle.load(open(data_file_vocab, 'rb'))
vowels = 'aoueiy'
vowel_indices = [i for i in range(e.shape[0]) if vocab[i] in vowels]
consonants = 'qwrtpsdfghjklzxcvbnm'
consonant_indices = [i for i in range(e.shape[0]) if vocab[i] in consonants]
other_indices = [i for i in range(e.shape[0]) if i not in consonant_indices and i not in vowel_indices]
#category_names = ['Vowels', 'Consonants', 'Other']
#category_indices = [vowel_indices, consonant_indices, other_indices]
category_names = ['Vowels', 'Consonants']
category_indices = [vowel_indices, consonant_indices]
if plot_characters:
  print('plot_characters')
  print('Computing T-SNE')
  e_projected = t.fit_transform(e)
  print('e_projected dimensions: {}'.format(e_projected.shape))
  print('Done computing T-SNE.')

  fig, (ax,lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios":[4,1]})
  recs = []
  for i in range(0,len(category_names)):
    recs.append(mpatches.Circle((0,0),1,fc=selected_colors[i%len(selected_colors)]))
    n = len(category_indices[i])
    colors = [selected_colors[i%len(selected_colors)]]*n
    mark = 'x'# if all_relation_types[i].endswith('_r') else '+'
    ax.scatter(e_projected[category_indices[i], 0], e_projected[category_indices[i], 1], s=25, c=colors, marker=mark)


  for label, x, y in zip(vocab, e_projected[:, 0], e_projected[:, 1]):
    #print(label)
    print('{} {} {}'.format(label, x, y))
    plt.annotate(
              label,
              xy=(x, y),
              xytext=(x,y))
              #, xytext=(-2, 2),
              #textcoords='offset points', ha='right', va='bottom',
              #bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
              #arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
  print(all_relation_types)
  #plt.legend(recs,all_relation_types,loc=7)
  lax.legend(recs,category_names,
               bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  lax.axis('off')
  #plt.legend(loc=4)
  plt.axis('tight')
  plt.tight_layout()
  plt.savefig("/home/mogren/character-embeddings.pdf", bbox_inches="tight")
  plt.show()
