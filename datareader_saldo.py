import xml.sax.handler
import xml.sax
import sys

from collections import defaultdict
import urllib3, os
from urllib.request import urlopen

data_url_swedish    = 'https://svn.spraakdata.gu.se/sb-arkiv/pub/lmf/saldom/saldom.xml'

skipped_forms = set(['c', 'ci', 'cm', 'sms'])

class SaldoLexiconHandler(xml.sax.handler.ContentHandler):
    def __init__(self, result):
        self.result = result

    def startElement(self, name, attributes):
        if name == 'LexicalEntry':
            self.current = [ None, None, defaultdict(list) ]
        elif name == 'feat' and attributes['att'] == 'partOfSpeech':
            if self.current[1]:
                raise Exception("pos already set!")
            self.current[1] = attributes['val']
        elif name == 'feat' and attributes['att'] == 'writtenForm':
            if self.inFR:
                if self.current[0]:
                    raise Exception("lemma already set!")
                self.current[0] = attributes['val']
            else:
                self.wf = attributes['val']
        elif name == 'feat' and attributes['att'] == 'msd':            
            form_name = attributes['val']
            if self.wf[-1] != '-' and form_name not in skipped_forms:
                self.current[2][form_name].append(self.wf)
        elif name == 'FormRepresentation':
            self.inFR = True
                
    def endElement(self, name):
        if name == 'LexicalEntry':
            self.result.append(self.current)
        elif name == 'FormRepresentation':
            self.inFR = False

def read_saldom_lexicon(lexicon_file):
    parser = xml.sax.make_parser()
    result = []
    parser.setContentHandler(SaldoLexiconHandler(result))
    with open(lexicon_file, 'r') as f:
      parser.parse(f)
    return result

def to_form_str(forms, form):
    if form not in forms:
        return '-'
    else:
        return '/'.join(sorted(forms[form]))

def get_saldo_data(data_dir):
    d = {}
    vocab = set()
    tags = {}

    language = 'swedish'
    d[language] = {}
    d[language]['V'] = []
    d[language]['ADJ'] = []
    d[language]['N'] = []


    saldo_filename = os.path.join(data_dir, 'saldom.xml')
    if os.path.exists(saldo_filename):
      print('found {} on disk.'.format(data_url_swedish))
    else:
      print('downloading {}'.format(data_url_swedish))
      with urlopen(data_url_swedish) as f_web:
        with open(saldo_filename , 'w') as f:
          for line in f_web:
            #f.write(line.decode('utf-8')+'\n')
            f.write(line.decode('utf-8'))

    lexicon_data = read_saldom_lexicon(saldo_filename)

    forms_for_pos = defaultdict(set)
    
    for lemma, pos, forms in lexicon_data:
        if pos in ['nn', 'av', 'vb']:
            forms_for_pos[pos].update(forms)
            
    forms_for_pos = { pos: sorted(forms) for pos, forms in forms_for_pos.items() }

    #print(forms_for_pos['av'])
    #print(forms_for_pos['vb'])
    #print(forms_for_pos['nn'])
    
    #advocera        vb      advocera advocera advoceras advocerar advoceras - - advocerandes advocerande advocerade advocerades - - advocerades advocerade advocerades advocerade advocerades advocerade advocerades advocerade advocerats advocerat advocerads advocerad advocerat advocerats
    #skivbroms       nn      skivbromsarnas skivbromsarna skivbromsars skivbromsar skivbromsens skivbromsen skivbroms skivbroms
    #klyvbar av      - klyvbarares klyvbarare klyvbaras klyvbara klyvbares klyvbare klyvbaras klyvbara klyvbaras klyvbara klyvbarts klyvbart klyvbars klyvbar klyvbarastes klyvbaraste klyvbarastes klyvbaraste klyvbarasts klyvbarast
    
    
    for lemma, pos, forms in lexicon_data:
        if pos in forms_for_pos:
            form_str = ' '.join(to_form_str(forms, f) for f in forms_for_pos[pos])
            #print(forms)
            #print('{}\t{}\t{}'.format(lemma, pos, form_str))
            paradigm = {}
            if pos == 'nn':
              singular = to_form_str(forms, forms_for_pos[pos][7]).split('/')[0]
              plural = to_form_str(forms, forms_for_pos[pos][3]).split('/')[0]
              for w in [singular, plural]:
                vocab.update(set(w))
              if singular != '-':
                paradigm['num=SG,pos=N'] = singular
                add_to_dictset(tags, [('num', 'SG'),('pos', 'N')])
              if plural != '-':
                paradigm['num=PL,pos=N'] = plural
                add_to_dictset(tags, [('num', 'PL'),('pos', 'N')])
              if len(paradigm.keys()) > 1:
                d[language]['N'].append(paradigm)

            elif pos == 'vb':
              infinitive = to_form_str(forms, forms_for_pos[pos][0]).split('/')[0]
              imperfect= to_form_str(forms, forms_for_pos[pos][9]).split('/')[0]
              presence = to_form_str(forms, forms_for_pos[pos][3]).split('/')[0]
              passive = to_form_str(forms, forms_for_pos[pos][2]).split('/')[0]
              for w in [infinitive, imperfect, presence, passive]:
                vocab.update(set(w))
              if infinitive != '-':
                paradigm['finite=NFIN,pos=V,tense=NA,voice=ACT'] = infinitive
                add_to_dictset(tags, [('finite', 'NFIN'),('pos', 'V'),('tense', 'NA'),('voice', 'ACT')])
              if imperfect != '-':
                paradigm['pos=V,tense=PST,voice=ACT'] = imperfect
                add_to_dictset(tags, [('tense', 'PST'),('pos', 'V'),('voice', 'ACT')])
              if presence != '-':
                paradigm['pos=V,tense=PRS,voice=ACT'] = presence
                add_to_dictset(tags, [('tense', 'PRS'),('pos', 'V'),('voice', 'ACT')])
              if passive != '-':
                paradigm['finite=NFIN,pos=V,tense=NA,voice=PASS'] = passive
                add_to_dictset(tags, [('finite', 'NFIN'),('voice', 'PASS'),('pos', 'V'),('tense', 'NA')])
              if len(paradigm.keys()) > 1:
                d[language]['V'].append(paradigm)

            elif pos == 'av':
              base = to_form_str(forms, forms_for_pos[pos][14]).split('/')[0]
              comparative = to_form_str(forms, forms_for_pos[pos][2]).split('/')[0]
              superlative = to_form_str(forms, forms_for_pos[pos][20]).split('/')[0]
              for w in [base, comparative, superlative]:
                vocab.update(set(w))
              if base != '-':
                paradigm['comp=POS,pos=ADJ'] = base
                add_to_dictset(tags, [('comp', 'POS'),('pos', 'ADJ')])
              if comparative != '-':
                paradigm['comp=CMPR,pos=ADJ'] = comparative
                add_to_dictset(tags, [('comp', 'CMPR'),('pos', 'ADJ')])
              if superlative != '-':
                paradigm['comp=SPRL,pos=ADJ'] = superlative
                add_to_dictset(tags, [('comp', 'SPRL'),('pos', 'ADJ')])
              if len(paradigm.keys()) > 1:
                d[language]['ADJ'].append(paradigm)

    return d, vocab, tags

def add_to_dictset(d, list_of_pairs):
  for key,val in list_of_pairs:
    d.setdefault(key, set())
    d[key].add(val)


def get_saldo_data_old(infile):
    relations = {}
    relations['n_plural']      = []
    relations['v_imperfect']   = []
    relations['v_progressive'] = []
    relations['v_presence']    = []
    relations['a_comparative'] = []
    relations['a_superlative'] = []
    relations['a_comparative_superlative'] = []

    lexicon_data = read_saldom_lexicon(infile)

    forms_for_pos = defaultdict(set)
    
    for lemma, pos, forms in lexicon_data:
        if pos in ['nn', 'av', 'vb']:
            forms_for_pos[pos].update(forms)
            
    forms_for_pos = { pos: sorted(forms) for pos, forms in forms_for_pos.items() }

    #print(forms_for_pos['av'])
    #print(forms_for_pos['vb'])
    #print(forms_for_pos['nn'])
    
    #advocera        vb      advocera advocera advoceras advocerar advoceras - - advocerandes advocerande advocerade advocerades - - advocerades advocerade advocerades advocerade advocerades advocerade advocerades advocerade advocerats advocerat advocerads advocerad advocerat advocerats
    #skivbroms       nn      skivbromsarnas skivbromsarna skivbromsars skivbromsar skivbromsens skivbromsen skivbroms skivbroms
    #klyvbar av      - klyvbarares klyvbarare klyvbaras klyvbara klyvbares klyvbare klyvbaras klyvbara klyvbaras klyvbara klyvbarts klyvbart klyvbars klyvbar klyvbarastes klyvbaraste klyvbarastes klyvbaraste klyvbarasts klyvbarast
    
    
    for lemma, pos, forms in lexicon_data:
        if pos in forms_for_pos:
            form_str = ' '.join(to_form_str(forms, f) for f in forms_for_pos[pos])
            #print(forms)
            #print('{}\t{}\t{}'.format(lemma, pos, form_str))
            if pos == 'nn':
              first = to_form_str(forms, forms_for_pos[pos][7]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][3]).split('/')[0]
              if first != '-' and second != '-':
                relations['n_plural'].append((first, second))

            elif pos == 'vb':
              first = to_form_str(forms, forms_for_pos[pos][0]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][9]).split('/')[0]
              if first != '-' and second != '-':
                relations['v_imperfect'].append((first, second))

              first = to_form_str(forms, forms_for_pos[pos][0]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][3]).split('/')[0]
              if first != '-' and second != '-':
                relations['v_presence'].append((first, second))

              first = to_form_str(forms, forms_for_pos[pos][0]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][2]).split('/')[0]
              if first != '-' and second != '-':
                relations['v_progressive'].append((first, second))

            elif pos == 'av':
              first = to_form_str(forms, forms_for_pos[pos][14]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][2]).split('/')[0]
              if first != '-' and second != '-':
                relations['a_comparative'].append((first, second))

              first = to_form_str(forms, forms_for_pos[pos][14]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][20]).split('/')[0]
              if first != '-' and second != '-':
                relations['a_superlative'].append((first, second))
              
              first = to_form_str(forms, forms_for_pos[pos][2]).split('/')[0]
              second = to_form_str(forms, forms_for_pos[pos][20]).split('/')[0]
              if first != '-' and second != '-':
                relations['a_comparative_superlative'].append((first, second))

    return relations

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == '-':
        infile = sys.stdin
    else:
        infile = sys.argv[1]
    lexicon_data = read_saldom_lexicon(infile)

    forms_for_pos = defaultdict(set)
    
    for lemma, pos, forms in lexicon_data:
        if pos in ['nn', 'av', 'vb']:
            forms_for_pos[pos].update(forms)
            
    forms_for_pos = { pos: sorted(forms) for pos, forms in forms_for_pos.items() }

    #print(forms_for_pos['av'])
    #print(forms_for_pos['vb'])
    #print(forms_for_pos['nn'])
    
    #for lemma, pos, forms in lexicon_data:
    #    if pos in forms_for_pos:
    #        form_str = ' '.join(to_form_str(forms, f) for f in forms_for_pos[pos])
    #        print(forms)
    #        print('{}\t{}\t{}'.format(lemma, pos, form_str))




