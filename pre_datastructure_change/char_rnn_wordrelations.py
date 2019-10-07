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
import unicodedata, string, random, datetime, time, math, random, os, argparse, sys, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from subprocess import Popen,PIPE

default_datadir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
default_savedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model-parameters')

parser = argparse.ArgumentParser()

#TRAINING
parser.add_argument("--max_iterations", type=int, default=20000, help="Max iterations")
parser.add_argument("--early_stopping_threshold", type=int, default=6000, help="Early stopping threshold.")
parser.add_argument("--ensemble_size", type=int, default=1, help="Ensemble size")
parser.add_argument("--disable_gpu", action="store_true", help="Disable GPU usage.")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--beam_size", type=int, default=1, help="Beam size. Only size=1 works. Debugging needed for size > 1.")
parser.add_argument("--keep_probability", type=float, default=1.0, help="Dropout keep probability.")
parser.add_argument("--drop_char_p", type=float, default=0.0, help="Drop characters with probability p.")
parser.add_argument("--uniform_sampling", action="store_true", help="Uniform sampling of training data.")
parser.add_argument("--prio_relation", type=str, default=None, help="Prio relation. Train more on this.")
parser.add_argument("--prio_p", type=float, default=0.9, help="Prio relation fraction. How much of the time to train on --prio_relation (if specified).")
parser.add_argument("--id_prob", type=float, default=0.0, help="Sample identity relation probability during training.")

# MODEL
parser.add_argument("--rnn_depth", type=int, default=2, help="RNN depth. Can be overidden for decoder by using --decoder_rnn_depth.")
parser.add_argument("--decoder_rnn_depth", type=int, default=None, help="Decoder RNN depth. Default: using size specified by --rnn_depth.")
parser.add_argument("--hidden_size", type=int, default=100, help="RNN hidden size for all RNN parts. Can be overridden for decoder with --decoder_hidden_size.")
parser.add_argument("--decoder_hidden_size", type=int, default=None, help="Decoder RNN hidden size. Default: using size specified by --hidden_size.")
parser.add_argument("--decoder_extra_layer", action="store_true", help="Adds an extra fully connected layer before decoder output.")
parser.add_argument("--encoder_extra_layer", action="store_true", help="Adds an extra fully connected layer after fc_combined before decoder input.")
parser.add_argument("--embedding_size", type=int, default=100, help="Embedding size")
parser.add_argument("--learning_rate", type=float, default=0.005, help="Learning rate")
parser.add_argument("--input_shortcut", action="store_true", help="Input shortcut. Feed input into decoder.")
parser.add_argument("--disable_attention", action="store_true", help="Disable the attention mechanism.")
parser.add_argument("--disable_relation_input", action="store_true", help="Disable the relation input mechanism.")
parser.add_argument("--softmax_relation", action="store_true", help="Enable the relation softmax output to be used as relation input to decoder.")
parser.add_argument("--rel_shortcut", action="store_true", help="Feed relation encoder outputs to output softmax.")
parser.add_argument("--rel_shortcut_rnn", action="store_true", help="Feed relation encoder outputs to decoder RNN.")
parser.add_argument("--rel_bottleneck_size", type=int, default=None, help="Relation encoder output is squashed through a bottleneck. Specify its size.")
parser.add_argument("--tie_rel_weights", action="store_true", help="Tie the weights on the two RNNs encoding the relation.")
parser.add_argument("--tie_all_encoder_weights", action="store_true", help="Tie the weights on the three RNNs encoding the relation and query. Overrides --tie_rel_weights.")
parser.add_argument("--bidirectional_encoders", action="store_true", help="Enable bidirectional encoders.")
parser.add_argument("--attend_to_relation", action="store_true", help="Append relation output to query encoder outputs. Lets the attention mechanism choose when to use it.")
parser.add_argument("--reverse_target", action='store_true', help="Reverse target generation.")

#LOSSES
parser.add_argument("--pad_loss_correction", action="store_true", help="Loss weight zero for pad tokens.")
parser.add_argument("--enable_relation_loss", action="store_true", help="Enable the relation classification loss.")
parser.add_argument("--enable_query_tags_loss", action="store_true", help="Disable the tags classification loss.")
parser.add_argument("--tags_loss_fraction", type=float, default=1.0, help="Tags classification used as auxilliary training for a random selection of training examples. Default 1.0 (all examples).")
parser.add_argument("--bitags_loss", action='store_true', help="Tags loss will be separate for each of the two words in relation encoder.")
parser.add_argument("--disable_language_loss", action="store_true", help="Disable the language classification loss. Applicable only if running with more than one language.")
parser.add_argument("--l2_reg", type=float, default=0.00005, help="L2 regularization factor.")
parser.add_argument("--l1_reg", type=float, default=0.0, help="L1 regularization factor.")
parser.add_argument("--cov_reg", type=float, default=0.0, help="Covariance regularization factor.")

# DATASET
parser.add_argument("--workshop_relation_selection", action="store_true", help="Use only the relations that were reported in the SCLeM paper draft. (Affects the format of data stored).")
parser.add_argument("--test_words", type=str, default=None, help="Comma-separated list of test words. Any paradigm with any word in list will removed from train and validation set (for this session), and explicitly tested at test-time. (Does not affect the format of data stored).")

parser.add_argument("--save_dir", type=str, default=default_savedir, help="Path to directory where the model parameters can be saved and loaded.")
parser.add_argument("--data_dir", type=str, default=default_datadir, help="Path to directory where the dataset can be saved and loaded.")
parser.add_argument("--languages", type=str, default='english', help="Comma-separated list of languages. Implemented: english,swedish,arabic,finnish,georgian,german,hungarian,maltese,navajo,russian,spanish,turkish. Or \'all\'")

# EVALUATION
parser.add_argument("--test_only", action="store_true", help="Disables the training procedure. Tries to load model and then test the best_iteration.")
parser.add_argument("--interactive", action='store_true', help="Interactive mode.")
parser.add_argument("--remove_id_tests", action='store_true', help="Remove tests in validation and test sets where demo relation have two identical words.")
parser.add_argument("--native_vocab", action='store_true', help="Use only characters from English dataset.")
parser.add_argument("--poet", action='store_true', help="Use variant of POET postprocessing (Kann, Schutze, 2016).")
parser.add_argument("--save_embeddings", action='store_true', help="Save embedding vectors to file.")
parser.add_argument("--disable_self_relation_test", action='store_true', help="Allow evaluation step to use query-target as demo relations when evaluating, if only one word-pair exist in test set.")

args = parser.parse_args()

if not args.disable_gpu:
  #print('CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
  #if os.environ['CUDA_VISIBLE_DEVICES'] is None or int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
  try:
    import gridengine
    gridengine.start_job(1, 30)
    print('Started grid engine job.')
  except ImportError:
    print('Failed to start grid engine job. (Don\'t worry).')

random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# In the spell-checker dataset, the longest word is 24 characters.
# Most words are significantly shorter.
max_sequence_len = 30
use_cuda = False

teacher_forcing_ratio = .5

if not args.disable_gpu:
  use_cuda = torch.cuda.is_available()

languages            = ['english']
supported_languages  = ['english', 'swedish', 'arabic', 'finnish', 'georgian', 'german', 'hungarian', 'maltese', 'navajo', 'russian', 'spanish', 'turkish']
all_relation_labels  = []
relations            = {}
flattened_train_set  = {}
all_tags             = {}
vocab                = []
vocab_size           = -1
reverse_vocab        = {}
num_relation_classes = None
# all letters will be further populated from dataset in prepare_data().
#all_characters          = string.ascii_letters + " .,;'-ÅåÄäÖöÜüß"
all_characters          = ''
native_characters       = {}

test_words = []

def print_len_stats():
  maxlen = 0
  minlen = 100
  lens = {}
  for p in relations:
    for l in relations:
      for k in relations[l]:
        for (w1,t1),(w2,t2) in relations[p][l][k]:
          lens[len(w1)] = lens.get(len(w1), 0)+1
          lens[len(w2)] = lens.get(len(w2), 0)+1
          maxlen = max(maxlen, len(w1))
          maxlen = max(maxlen, len(w2))
          minlen = min(minlen, len(w1))
          minlen = min(minlen, len(w2))
  print('wordlens: min {}, max{}.'.format(minlen, maxlen))
  l = list(lens.keys())
  l.sort()
  for ln in l:
    print('len: {}, num: {}.'.format(ln, lens[ln]))

def initialize_vocab(_vocab=None):
  global vocab, vocab_size, reverse_vocab
  if _vocab is not None:
    vocab = _vocab
    print('Found vocab: \'{}\''.format(''.join(vocab)))
  else:
    special_tokens = []
    special_tokens.append('<PAD>')
    special_tokens.append('<BOS>')
    special_tokens.append('<EOS>')
    special_tokens.append('<UNK>')
    special_tokens.append('<DRP>')
    if args.native_vocab:
      biglist = []
      for l in languages:
        for i in range(len(native_characters[l])):
          biglist.append(native_characters[l][i])
      biglist = sorted(biglist)
      vocab += biglist
    else:
      for i in range(len(all_characters)):
        vocab.append(all_characters[i])
    vocab = special_tokens+sorted(list(set(vocab)))
    print('Constructed vocab: \'{}\''.format(''.join(vocab)))
  for i in range(len(vocab)):
    reverse_vocab[vocab[i]] = i
  vocab_size = len(vocab)
  print('Vocab size: {}'.format(vocab_size))
  
# turn a unicode string to plain ascii

#def unicodeToAscii(s):
#  return ''.join(
#    c for c in unicodedata.normalize('NFD', s)
#    if unicodedata.category(c) != 'Mn'
#    and c in all_characters
#  )

#print(unicodeToAscii('Ślusàrski'))

def line_to_index_tensor(lines, pad_before=True, append_bos_eos=False, reverse=False, drop_char_p=0.0):
  if reverse:
    lines = [l[::-1] for l in lines]
  seqlen = max([len(l) for l in lines])
  seqlen = min(seqlen, max_sequence_len)
  if append_bos_eos:
    seqlen += 2
  tensor = torch.zeros(len(lines), seqlen).long()
  tensor += reverse_vocab['<PAD>']
  for b in range(len(lines)):
    begin_pos = 0
    if pad_before:
      begin_pos = max(0,seqlen-len(lines[b]))
    else:
      begin_pos = 0
    if append_bos_eos:
      begin_pos += 1
      tensor[b][begin_pos-1] = reverse_vocab['<BOS>']
    for li, letter in enumerate(lines[b]):
      idx = li+begin_pos
      if idx >= seqlen:
        break
      if drop_char_p>0.0 and random.random() < drop_char_p:
        tensor[b][idx] = reverse_vocab['<DRP>']
      tensor[b][idx] = reverse_vocab.get(letter, reverse_vocab['<UNK>'])
    if append_bos_eos:
      eos_pos = min(seqlen-1,begin_pos+len(lines[b]))
      tensor[b][eos_pos] = reverse_vocab['<EOS>']
  if use_cuda:
    tensor = tensor.cuda()
  return tensor

class AttentionModule(nn.Module):
  def __init__(self, encoder_hidden_size, decoder_hidden_size):
    super(AttentionModule, self).__init__()
    self.encoder_hidden_size    = encoder_hidden_size
    self.decoder_hidden_size    = decoder_hidden_size
    self.sigmoid        = nn.Sigmoid()
    self.tanh           = nn.Tanh()
    self.attn           = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size * 2)
    self.attn2          = nn.Linear(self.encoder_hidden_size * 2, 1)
    self.attn_combine   = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.decoder_hidden_size)
    self.softmax        = nn.Softmax()
    if use_cuda:
      self.sigmoid.cuda()
      self.tanh.cuda()
      self.attn.cuda()
      self.attn2.cuda()
      self.attn_combine.cuda()
      self.softmax.cuda()

  def forward(self, hidden, decoder_out, encoder_states):
    #attention mechanism:
    # hidden is shape [depth, batch, encoder_hidden_size].
    # We use only the top level hidden state: [batch, encoder_hidden_size]
    attention_weights = []
    for i in range(encoder_states.size()[0]):
      attention_weights.append(self.attn2(self.tanh(self.attn(torch.cat((hidden, encoder_states[i]), 1)))))
    attention_weights=torch.stack(attention_weights, dim=1)
    attention_weights=attention_weights.squeeze(dim=2)
    attention_weights = self.softmax(attention_weights)
    attention_weights = attention_weights.view(hidden.size()[0],1,-1)[:,:,:encoder_states.size()[0]]
    encoder_states_batchfirst = encoder_states.permute(1,0,2)
    attention_applied = torch.bmm(attention_weights,encoder_states_batchfirst)

    attention_applied = attention_applied.view(hidden.size()[0], -1)
    attention_out = torch.cat((decoder_out, attention_applied), 1)
    attention_out = self.attn_combine(attention_out)
    attention_out = attention_out.view(hidden.size()[0], -1)
    attention_out = self.tanh(attention_out)
    return attention_out, attention_applied

class RnnRelationModel(nn.Module):
  def __init__(self, vocab_size, num_relation_classes, embedding_size, encoder_hidden_size, decoder_hidden_size, encoder_depth, decoder_depth, disable_attention, disable_relation_input, enable_relation_loss, disable_language_loss, all_tags, num_languages, decoder_extra_layer, bidirectional_encoders=False):
    super(RnnRelationModel, self).__init__()

    self.embedding_size         = embedding_size
    self.encoder_hidden_size            = encoder_hidden_size
    self.decoder_hidden_size            = decoder_hidden_size
    self.num_relation_classes   = num_relation_classes
    self.encoder_depth                  = encoder_depth
    self.decoder_depth                  = decoder_depth
    self.disable_attention      = disable_attention
    self.disable_relation_input = disable_relation_input
    self.all_tags               = all_tags
    self.num_languages          = num_languages
    self.enable_relation_loss   = enable_relation_loss 
    self.disable_language_loss  = disable_language_loss 
    self.decoder_extra_layer    = decoder_extra_layer
    self.first                  = True
    self.bidirectional_encoders = bidirectional_encoders
    self.encoder_num_directions = 2 if self.bidirectional_encoders else 1
    self.forward_dim            = 0
    self.reverse_dim            = 1

    self.embedding      = nn.Embedding(vocab_size, embedding_size)

    # hidden dimensions: num_layers * num_directions, batch, hidden_size

    self.rnn_demo1      = nn.GRU(input_size=embedding_size, hidden_size=encoder_hidden_size, num_layers=encoder_depth, bidirectional=self.bidirectional_encoders)
    if args.tie_rel_weights or args.tie_all_encoder_weights:
      #for v in dir(self.rnn_demo1):
      #  if v.startswith('weight') or v.startswith('bias'):
      #    setattr(self.rnn_demo2, v, getattr(self.rnn_demo1, v))
      #self.rnn_demo2.weight = self.rnn_demo1.weight
      self.rnn_demo2      = self.rnn_demo1
    else:
      self.rnn_demo2      = nn.GRU(input_size=embedding_size, hidden_size=encoder_hidden_size, num_layers=encoder_depth, bidirectional=self.bidirectional_encoders)

    if args.tie_all_encoder_weights:
      #for v in dir(self.rnn_demo1):
      #  if v.startswith('weight') or v.startswith('bias'):
      #    setattr(self.rnn_query, v, getattr(self.rnn_demo1, v))
      self.rnn_query     = self.rnn_demo1
    else:
      self.rnn_query     = nn.GRU(input_size=embedding_size, hidden_size=encoder_hidden_size, num_layers=encoder_depth, bidirectional=self.bidirectional_encoders)

    fc_rel_size = encoder_hidden_size
    if args.rel_bottleneck_size is not None:
      fc_rel_size = args.rel_bottleneck_size
    self.fc_rel         = nn.Linear(encoder_hidden_size*self.encoder_num_directions*2, fc_rel_size)

    relation_encoder_size = fc_rel_size
    # rnn output dimensions: seq_len, batch, hidden_size * num_directions

    if self.enable_relation_loss or args.softmax_relation:
      self.rel_out        = nn.Linear(fc_rel_size, num_relation_classes)
      if args.softmax_relation:
        relation_encoder_size = num_relation_classes

    if not self.disable_language_loss and self.num_languages:
      self.lang_out_rel   = nn.Linear(fc_rel_size, self.num_languages)
      self.lang_out_query = nn.Linear(encoder_hidden_size*self.encoder_num_directions, self.num_languages)

    total_len           = 0
    self.tag_out_q       = {}
    self.tag_out_rel1       = {}
    self.tag_out_rel2       = {}
    for t in sorted(self.all_tags.keys()):
      self.tag_out_q[t]   = nn.Linear(encoder_hidden_size*self.encoder_num_directions, len(all_tags[t]))
      if args.bitags_loss:
        self.tag_out_rel1[t]   = nn.Linear(encoder_hidden_size*self.encoder_num_directions, len(all_tags[t]))
        self.tag_out_rel2[t]   = nn.Linear(encoder_hidden_size*self.encoder_num_directions, len(all_tags[t]))
      else:
        self.tag_out_rel1[t]   = nn.Linear(fc_rel_size, len(all_tags[t]))
        self.tag_out_rel2[t]   = nn.Linear(fc_rel_size, len(all_tags[t]))
      total_len          += len(all_tags[t])
    if args.softmax_relation:
      relation_encoder_size = total_len*2

    #print('Relation encoder size: {}'.format(relation_encoder_size))

    if disable_relation_input:
      relation_encoder_size = 0

    self.fc_combined      = nn.Linear(relation_encoder_size+encoder_hidden_size*self.encoder_num_directions, decoder_hidden_size*decoder_depth)
    if args.encoder_extra_layer:
      self.fc_combined2      = nn.Linear(decoder_hidden_size*decoder_depth, decoder_hidden_size*decoder_depth)
      
    self.tanh             = nn.Tanh()

    # Zero initial state:
    # dimensions: num_layers * num_directions, batch, hidden_size
    self.hidden_initial   = Variable(torch.zeros(encoder_depth*self.encoder_num_directions, 1, self.encoder_hidden_size), requires_grad=False)

    decoder_input_size    = (embedding_size if not args.input_shortcut else embedding_size*2)
    if args.rel_shortcut_rnn:
      decoder_input_size += relation_encoder_size
    self.rnn_decoder      = nn.GRU(input_size=decoder_input_size, hidden_size=decoder_hidden_size, num_layers=decoder_depth)
    if args.rel_shortcut:
      if args.decoder_extra_layer:
        self.linear_extra   = nn.Linear(decoder_hidden_size+relation_encoder_size, decoder_hidden_size+relation_encoder_size)
      self.linear         = nn.Linear(decoder_hidden_size+relation_encoder_size, vocab_size)
    else:
      if args.decoder_extra_layer:
        self.linear_extra   = nn.Linear(decoder_hidden_size, decoder_hidden_size)
      self.linear         = nn.Linear(decoder_hidden_size, vocab_size)
    self.logsoftmax       = nn.LogSoftmax()

    self.dropout          = nn.Dropout(p=(1.0-args.keep_probability))

    if not disable_attention:
      self.attention_query = AttentionModule(encoder_hidden_size*self.encoder_num_directions, decoder_hidden_size)

    if use_cuda:
      self.embedding.cuda()
      self.rnn_demo1.cuda()
      self.rnn_demo2.cuda()
      self.rnn_query.cuda()
      self.rnn_decoder.cuda()
      self.fc_rel.cuda()
      if self.enable_relation_loss or args.softmax_relation:
        self.rel_out.cuda()
      if not self.disable_language_loss and self.num_languages:
        self.lang_out_rel.cuda()
        self.lang_out_query.cuda()
      for t in sorted(self.all_tags.keys()):
        self.tag_out_q[t].cuda()
        self.tag_out_rel1[t].cuda()
        self.tag_out_rel2[t].cuda()
      self.fc_combined.cuda()
      if args.encoder_extra_layer:
        self.fc_combined2.cuda()
      self.tanh.cuda()
      self.hidden_initial = self.hidden_initial.cuda()
      self.linear.cuda()
      self.logsoftmax.cuda()
      self.dropout.cuda()
      if not disable_attention:
        self.attention_query.cuda()

  # input = demo1
  def forward(self, demo1, demo2, query, target, teacher_forcing_r=0.0, return_embeddings=False):

    embeddings = None
    if return_embeddings:
      embeddings = {}

    batch_size = demo1.size()[0]
    # word dimensons: [batch, seqlen]
    #print(demo1.size())
    h_init = self.hidden_initial.repeat(1, batch_size , 1)
    #print('{}, {}, {}'.format(demo1.size(), demo2.size(), query.size()))
    demo1_emb = self.embedding(demo1).permute(1,0,2)
    #print(demo1.size())
    #print(self.hidden_initial.size())
    #print(h_init.size())
    #print(demo1_emb.size())
    #demo1_emb = demo1_emb.contiguous()
    demo1_o, demo1_h = self.rnn_demo1(demo1_emb, h_init)
    demo2_emb = self.embedding(demo2).permute(1,0,2)
    demo2_o, demo2_h = self.rnn_demo2(demo2_emb, h_init)
    query_emb = self.embedding(query).permute(1,0,2)
    query_o, query_h = self.rnn_query(query_emb, h_init)
    if self.bidirectional_encoders:
      query_o_ends = torch.cat((query_o[-1,:,0:self.encoder_hidden_size], query_o[0,:,self.encoder_hidden_size:]), 1)
    else:
      query_o_ends = query_o[-1]

    if return_embeddings:
      embeddings['query_encoder'] = query_o_ends.data

    lang_out_query_head = None
    if not self.disable_language_loss and self.num_languages:
      lang_out_query_head = self.logsoftmax(self.lang_out_query(query_o_ends))
    # if bidirectional, we look at last output of forward RNN, and first output of backward RNN.
    if self.bidirectional_encoders:
      demo1_o_ends = torch.cat((demo1_o[-1,:,0:self.encoder_hidden_size], demo1_o[0,:,self.encoder_hidden_size:]), 1)
      demo2_o_ends = torch.cat((demo2_o[-1,:,0:self.encoder_hidden_size], demo2_o[0,:,self.encoder_hidden_size:]), 1)
    else:
      demo1_o_ends = demo1_o[-1]
      demo2_o_ends = demo2_o[-1]
    rels_out = self.dropout(torch.cat((demo1_o_ends, demo2_o_ends), 1))#, query_o_ends), 1)
    rel_encoder_out = self.dropout(self.tanh(self.fc_rel(rels_out))) #.clamp(min=0)
    if return_embeddings:
      embeddings['relation_encoder'] = rel_encoder_out.data
    lang_out_rel_head = None
    if not self.disable_language_loss and self.num_languages:
      lang_out_rel_head = self.logsoftmax(self.lang_out_rel(rel_encoder_out))
    relation_classification_head = None
    if self.enable_relation_loss:
      relation_classification_head = self.logsoftmax(self.rel_out(rel_encoder_out))
      if args.softmax_relation:
        rel_encoder_out = torch.exp(relation_classification_head)

    tag_head_q    = {}
    tag_head_rel1 = {}
    tag_head_rel2 = {}
    for t in sorted(self.all_tags.keys()):
      tag_head_q[t] = self.logsoftmax(self.tag_out_q[t](query_o_ends))
      if args.bitags_loss:
        # if bidirectional, we look at last output of forward RNN, and first output of backward RNN.
        tag_head_rel1[t] = self.logsoftmax(self.tag_out_rel1[t](demo1_o_ends))
        tag_head_rel2[t] = self.logsoftmax(self.tag_out_rel2[t](demo2_o_ends))
      else:
        tag_head_rel1[t] = self.logsoftmax(self.tag_out_rel1[t](rel_encoder_out))
        tag_head_rel2[t] = self.logsoftmax(self.tag_out_rel2[t](rel_encoder_out))
    if args.softmax_relation:
      # feed the output from the tag prediction as input to fc_combined
      rel_encoder_out = torch.cat([torch.exp(tag_head_rel1[t]) for t in sorted(self.all_tags.keys())]+[torch.exp(tag_head_rel2[t]) for t in sorted(self.all_tags.keys())], dim=1)
    
    if self.disable_relation_input:
      encoders_out = query_o_ends
    else:
      encoders_out = torch.cat((rel_encoder_out, query_o_ends), 1)
    encoders_out = self.dropout(self.tanh(self.fc_combined(encoders_out))) #.clamp(min=0)
    if args.encoder_extra_layer:
      encoders_out = self.dropout(self.tanh(self.fc_combined2(encoders_out)))

    if return_embeddings:
      embeddings['combined_encoder'] = encoders_out.data

    if args.attend_to_relation:
      rel_3d = rel_encoder_out.view(1, rel_encoder_out.size(0), rel_encoder_out.size(1))
      #print(rel_3d.size())
      if query_o.size(2) <= rel_encoder_out.size(1):
        padded = rel_3d[:,:,:query_o.size(2)]
        if query_o.size(2) < rel_encoder_out.size(1):
          if self.first:
            print('Warning: query_o.size(2) (={}) < rel_encoder_out.size(1) (={}). Will do ugly folding compression on information from relation encoder!'.format(query_o.size(2), rel_encoder_out.size(1)))
            self.first = False
          for i in range(query_o.size(2),rel_encoder_out.size(1), query_o.size(2)):
            # ugly compression. fold it.
            length = min(query_o.size(2),rel_3d.size(2)-i)
            new_padded = padded.clone()
            new_padded[:,:,:length] = padded[:,:,:length]+rel_3d[:,:,i:i+length]
            padded = new_padded
      else:
        padding = torch.zeros(1).expand(1, rel_encoder_out.size(0), query_o.size(2)-rel_encoder_out.size(1))
        print('{} {} {}'.format(1, rel_encoder_out.size(0), query_o.size(2)-rel_encoder_out.size(1)))
        print(padding.size())
        padded = torch.cat([rel_3d, padding], dim=2)
      query_o = torch.cat([query_o, padded], dim=0)

    r = random.random()
    use_teacher_forcing = True if r < teacher_forcing_r else False
    #print('teacher forcing: {} ({},{})'.format(use_teacher_forcing, r, teacher_forcing_ratio))
    
    #print('encoders_out: {}'.format(encoders_out.size()))
    hidden = torch.stack(torch.chunk(encoders_out, chunks=self.decoder_depth, dim=1), dim=0)

    def get_emb(symbol):
      v = Variable(torch.zeros(demo1.size()[0], 1).long()+reverse_vocab[symbol], requires_grad=False)
      v = v.cuda() if use_cuda else v
      return self.embedding(v).permute(1,0,2).contiguous()

    # BEAM SIZE == 1 IS THE TESTED CODE:
    if args.beam_size == 1:
      output_classes = []
      choices = [] #only for debugging
      if use_teacher_forcing:
        # contiguous is needed for the view below. Only once per batch.
        dec_in_emb_seq = self.embedding(target).permute(1,0,2).contiguous()
        # add sequence dimension with len 1:
        input_emb = dec_in_emb_seq[0].view(1, -1, self.embedding_size)
      else:
        input_emb = get_emb('<BOS>')
      pad_embedded = get_emb('<PAD>')
      for i in range(max_sequence_len):
        # We use the GRU as a (deep) GRUCell, one step at a time:
        if args.input_shortcut:
          q_emb_i = query_emb[i].contiguous().view(1, query_emb.size(1), query_emb.size(2)) if i < query_emb.size(0) else pad_embedded
          input_emb = torch.cat((input_emb, q_emb_i), dim=2)
        if args.rel_shortcut_rnn:
          input_emb = torch.cat((input_emb, rel_encoder_out.view(1, rel_encoder_out.size(0), rel_encoder_out.size(1))), dim=2)
        outputs, hidden = self.rnn_decoder(input_emb, hidden)
        if self.disable_attention:
          output = outputs[-1]
        else:
          # hidden from top cell, last (only) output, and decoder_outputs.
          output, weighted_sum = self.attention_query(hidden[-1], outputs[-1], query_o)
            
        #print(output.size())
        if args.rel_shortcut:
          output = torch.cat([output, rel_encoder_out], dim=1)

        if self.decoder_extra_layer:
          output = self.tanh(self.linear_extra(output))
        output_classes.append(self.logsoftmax(self.linear(output)))
        topv, topi = output_classes[-1].data.topk(1)
        choices.append(topi[0][0])
        if use_teacher_forcing:
          if i+1 < dec_in_emb_seq.size()[0]:
            # add sequence dimension with len 1:
            input_emb = dec_in_emb_seq[i+1].view(1, demo1.size()[0], self.embedding_size)
          else:
            input_emb = get_emb('<EOS>')
        else:
          if use_cuda:
            dec_in = Variable(torch.cuda.LongTensor(topi))
          else:
            dec_in = Variable(torch.LongTensor(topi))
          input_emb = self.embedding(dec_in).permute(1,0,2)

      output_classes = torch.stack(output_classes, dim=0)

      return output_classes, relation_classification_head, tag_head_rel1, tag_head_rel2, tag_head_q, lang_out_rel_head, lang_out_query_head, embeddings

    else:
      #BEAM SEARCH:
      # NOT FINISHED. WOULD NEED TO IMPEMENT --input_shortcut AND DEBUG.

      #print(hidden.size())
      beam_size           = 5
      beam_logprobs       = Variable(torch.zeros(batch_size, beam_size), requires_grad=False)
      beam_logprobs       = beam_logprobs.cuda() if use_cuda else beam_logprobs
      # [batch, beam, sequence, vocab]:
      beam_output_classes = Variable(torch.zeros(batch_size, beam_size, max_sequence_len, self.vocab_size), requires_grad=False)
      beam_output_classes = beam_output_classes.cuda() if use_cuda else beam_output_classes 
      beam_decoded        = Variable(torch.zeros(batch_size, beam_size, max_sequence_len).long(), requires_grad=False)
      beam_decoded        = beam_decoded.cuda() if use_cuda else beam_decoded
      beam_input          = Variable(torch.zeros(batch_size, beam_size, 1).long(), requires_grad=False)
      beam_input          = beam_input.cuda() if use_cuda else beam_input
      # [depth (2), batch, beam, encoder_hidden_size]:
      beam_hidden         = hidden.unsqueeze(dim=2).repeat(1,1,beam_size,1)
      for b in range(beam_size):
        if use_teacher_forcing:
          dec_in = target[:,0]
        else:
          dec_in = Variable(torch.zeros(batch_size , 1).long()+reverse_vocab['<BOS>'], requires_grad=False)
          dec_in = dec_in.cuda() if use_cuda else dec_in
        print(dec_in)
        beam_input[:,b,:] = dec_in

      for si in range(max_sequence_len):
        print('seq_index: {}'.format(si))
        input()
        # the folowing will be indexed something like this:
        #   - current_beam_idx*beam_size+next_beam_idx
        beam_next_step_classes      = Variable(torch.zeros(batch_size , beam_size, self.vocab_size), requires_grad=False)
        beam_next_step_classes      = beam_next_step_classes.cuda() if use_cuda else beam_next_step_classes
        beam_next_step_logprobs     = Variable(torch.zeros(batch_size , beam_size*beam_size), requires_grad=False)
        beam_next_step_logprobs     = beam_next_step_logprobs.cuda() if use_cuda else beam_next_step_logprobs
        beam_next_step_topis        = Variable(torch.zeros(batch_size , beam_size*beam_size).long(), requires_grad=False)
        beam_next_step_topis        = beam_next_step_topis.cuda() if use_cuda else beam_next_step_topis
        for b in range(beam_size):
          print('  beam_index: {}'.format(b))
          input_emb = self.embedding(beam_input[:,b,:]).permute(1,0,2)
          #print(hidden[0,0,:30])
          # We use the GRU as a (deep) GRUCell, one step at a time:
          outputs, new_hidden = self.rnn_decoder(input_emb, beam_hidden[:,:,b,:].contiguous())
          beam_hidden[:,:,b,:] = new_hidden
          if self.disable_attention:
            output = outputs[-1]
          else:
            # hidden from top cell, last (only) output, and decoder_outputs.
            output, weighted_sum = self.attention_query(new_hidden[-1], outputs[-1], query_o)
              
          #print(output.size())
          output_logprobs = self.logsoftmax(self.linear(output))
          #beam_next_step_logprobs.append(output_logprobs)
          topv, topi = output_logprobs.data.topk(beam_size)
          #print(beam_logprobs)
          #print(beam_logprobs[:,b:b+1].expand(batch_size,beam_size))
          #print(topv)
          print('    {}'.format(topv))
          print('    beam_next_step_logprobs[:,{}:{}]: {}'.format(b*beam_size,(b+1)*beam_size, beam_next_step_logprobs[:,b*beam_size:(b+1)*beam_size]))
          beam_next_step_logprobs[:,b*beam_size:(b+1)*beam_size] = torch.add(topv,beam_logprobs[:,b:b+1].expand(batch_size,beam_size).data) # retains beam dimension of size 1, will be broadcasted when added to topv.
          print('    beam_next_step_logprobs[:,{}:{}]: {}'.format(b*beam_size,(b+1)*beam_size, beam_next_step_logprobs[:,b*beam_size:(b+1)*beam_size]))
          beam_next_step_topis[:,b*beam_size:(b+1)*beam_size] = topi
          beam_next_step_classes[:,b,:] = output_logprobs
        topv, topi = beam_next_step_logprobs.topk(beam_size)
        # topv, topi.size(): [batch, beam_size]
        # now need to figure out where topis are in the tensors (see indexing above)
        #print(topi)
        beam_id = torch.div(topi.data,beam_size)
        #print(beam_id)
        #beam_id = beam_id.cuda() if use_cuda else beam_id
            
        beam_logprobs       = topv
        #update beam_output_classes:
        #print(beam_output_classes)
        #print(beam_id)
        new_beam_output_classes = []
        new_beam_inputs = []
        new_beam_hidden = []
        #new_beam_decoded = []
        for batch in range(batch_size):
          print(beam_id[batch])
          print(beam_next_step_topis[batch])
          batch_beam_output_classes = beam_output_classes[batch].clone()
          batch_beam_output_classes = batch_beam_output_classes[beam_id[batch]]
          batch_beam_next_step_classes = beam_next_step_classes[batch]
          #print(beam_id[batch])
          #print(batch_beam_next_step_classes)
          #print(batch_beam_next_step_classes[beam_id[batch]])
          #print(batch_beam_output_classes)
          batch_beam_output_classes[:,si,:] = batch_beam_next_step_classes[beam_id[batch]]
          new_beam_output_classes.append(batch_beam_output_classes)
          batch_beam_next_step_topis = beam_next_step_topis[batch]

          # [depth, batch, beam, encoder_hidden_size] -> [batch, beam, depth, encoder_hidden_size]
          batch_beam_hidden = beam_hidden.permute(1,2,0,3)[batch]
          new_beam_hidden.append(batch_beam_hidden[beam_id[batch]].permute(1,0,2))

          if use_teacher_forcing:
            dec_in = target[batch,si+1]
            new_beam_inputs.apend(dec_in)
          else:
            new_beam_inputs.append(batch_beam_next_step_topis[topi.data[batch]])
          #new_beam_decoded.append(batch_beam_next_step_topis[topi.data[batch]])
          beam_decoded[batch,:,si] = batch_beam_next_step_topis[topi.data[batch]]
        beam_output_classes = torch.stack(new_beam_output_classes)
        beam_hidden         = torch.stack(new_beam_hidden, dim=1)
        beam_input          = torch.stack(new_beam_inputs)
        beam_input = beam_input.unsqueeze(dim=2)
        print(' beam_output_classes: {}'.format(beam_output_classes))
        print(' beam_input: {}'.format(beam_input))

      topv, topi = beam_logprobs.topk(1)
      beam_decoded_l = []
      output_classes = []
      for batch in range(batch_size):
        batch_beam_decoded = beam_decoded[batch]
        beam_decoded_l.append(torch.squeeze(batch_beam_decoded[topi.data[batch]], dim=0))
        batch_beam_output_classes = beam_output_classes[batch]
        output_classes.append(torch.squeeze(batch_beam_output_classes[topi.data[batch]], dim=0))
      beam_decoded = torch.stack(beam_decoded_l, dim=1)
      output_classes = torch.stack(output_classes, dim=1)

      return output_classes, relation_classification_head, tag_head_rel1, tag_head_rel2, tag_head_q, lang_out_rel_head, lang_out_query_head, beam_decoded, embeddings

def covarNorm(h):
  s = covar(h)
  reg = safenorm(s, 1) #- s.trace()    # - 0.9*s.trace().norm(1/2)
  n = (s.size(0) * s.size(1))# - s.size(0)  # Normalization constant, i.e. number of terms summed over.
  return reg / n

def covar(a):
  # Compute the sample covariance of a.
  am = a.sub(a.mean(0).expand_as(a))  # Remove mean
  s = am.t().mm(am) / (a.size(0)-1)   # Compute covariance matrix
  return s

def safenorm(input, n):
  input[input == 0] = 1e-30  # Fix subgradient error in torch.norm by removing zeros. min 1.2*-38
  return input.norm(n)

def poet(source, prediction, trees, language, example_tree):
  '''
    Here: lookup an example tree instead of 'knowing' the relation?
  '''
  tree_results = []
  relation_choices = []
  t1 = time.time()
  for relation in trees[language]:
    if example_tree in trees[language][relation]:
      relation_choices.append(relation)
  t2 = time.time()
  for relation in relation_choices:
    for tree in trees[language][relation]:
      r = apply_edit_tree(source, tree)
      if r is not None:
        tree_results.append((r, tree))
  t3 = time.time()
  matches = []
  for (r,tree) in tree_results:
    if r == prediction:
      return prediction, t2-t1, t3-t2, time.time()-t3
  for (r,tree) in tree_results:
    if prediction != r and abs(len(prediction)-len(r)) <= 1:
      if levenshtein(prediction,r) == 1:
        #print('found matching tree for source {}, and prediction {}: {}, result:: {}'.format(source, prediction, tree, r))
        matches.append(r)
  t4 = time.time()
  if len(matches):
    return randomChoice(matches), t2-t1, t3-t2, t4-t3
  return prediction, t2-t1, t3-t2, t4-t3

def poet_cheat(source, prediction, trees, language, relation):
  '''
    Cheating. Using relation information that _should_ not be available.
  '''
  tree_results = []
  for tree in trees[language][relation]:
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

def get_edit_trees(relations):
  trees = {}
  p = 'train'
  for l in relations[p]:
    trees[l] = {}
    for r in relations[p][l]:
      trees[l][r] = set()
      trees[l][r+'_r'] = set()
      for (w1,t1),(w2,t2) in relations[p][l][r]:
        trees[l][r].add(get_edit_tree(w1,w2))
        trees[l][r+'_r'].add(get_edit_tree(w2,w1))
  return trees

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
  return l[random.randint(0, len(l) - 1)]

def relation_label_to_class(label, all_relation_labels, reverse=False):
  c = all_relation_labels.index(label)*2
  if reverse:
    c += 1
  return c
def relation_class_to_label(c, all_relation_labels):
  relation_label = all_relation_labels[c//2]
  if c % 2 != 0:
    relation_label += '_r'
  return relation_label

def get_pair_of_pairs(partition='train', generator=None, batch_size=None, relation=None, return_strings=False):
  '''
    returns random pair of pairs from dataset partition=partition.
    If arguments are specified, then does not sample randomly.
    If position is supplied, (some) other parameters are iterated.
  '''
  #each argument specified as None, will result in a random choice.
  #print(list(all_relations.keys()))
  dw1s, dw2s, qws, tws = [], [], [], []
  if batch_size == None:
    batch_size = args.batch_size
  relation_class = torch.zeros(batch_size).long()
  # this could be stored in the data files. hack for backwards compatibility.
  tag_classes1 = {}
  tag_classes2 = {}
  for t in all_tags:
    tag_classes1[t] = torch.zeros(batch_size).long()
    tag_classes2[t] = torch.zeros(batch_size).long()
  language_class = torch.zeros(batch_size).long()
  reversings = []
  for b in range(batch_size):
    if generator is None:
      language_choice = randomChoice(languages)
      if relation == 'id':
        relation_type, (demo_word1, demo_tag1), (demo_word2, demo_tag2) = randomChoice(flattened_train_set[language_choice])
        relation_type = relation
        if randomChoice([True, False]):
          (demo_word1, demo_tag1) = (demo_word2, demo_tag2)
        else:
          (demo_word2, demo_tag2) = (demo_word1, demo_tag1)
        relation_type, (w1, t1), (w2, t2) = randomChoice(flattened_train_set[language_choice])
        if randomChoice([True, False]):
          (query_word, query_tag) = (w1, t1)
        else:
          (query_word, query_tag) = (w2, t2)
        (target_word, target_tag) = (query_word, query_tag)
      else:
        if args.uniform_sampling:
          relation_type, (demo_word1, demo_tag1), (demo_word2, demo_tag2) = randomChoice(flattened_train_set[language_choice])
        else:
          relation_type = randomChoice(sorted(relations[partition][language_choice].keys()))
          if relation:
            relation_type = relation
          (demo_word1, demo_tag1), (demo_word2, demo_tag2) = randomChoice(relations[partition][language_choice][relation_type])
        (query_word, query_tag), (target_word, target_tag) = randomChoice(relations[partition][language_choice][relation_type])
      #relation_class[b] = all_relation_labels.index(relation_type)*2
      relation_class[b] = relation_label_to_class(relation_type, all_relation_labels, reverse=False)
    else:
      try:
        language_choice, relation_type, demo_pair, query_pair, reverse = next(generator)
        (demo_word1, demo_tag1), (demo_word2, demo_tag2) = demo_pair
        (query_word, query_tag), (target_word, target_tag) = query_pair
        #relation_class[b] = all_relation_labels.index(relation_type)*2
        relation_class[b] = relation_label_to_class(relation_type, all_relation_labels, reverse=False)
        if reverse:
          relation_class[b] += 1
      except:
        if b > 0 and b < batch_size-1:
          # trim the batch, make it shorter.
          relation_class = relation_class[:b] # adjust size, if we exited loop prematurely due to lack of data.
          for t in all_tags:
            tag_classes1[t] = tag_classes1[t][:b]
            tag_classes2[t] = tag_classes2[t][:b]
          language_class = language_class[:b] # adjust size, if we exited loop prematurely due to lack of data.
        break
      #print('{}, {}, {}, {}'.format(language_choice, relation_type, b, demo_word1))
    language_class[b] = languages.index(language_choice)
    #tag_classes_l1, tag_classes_l2 = get_tag_pair(relation_type)
    tag_classes_l1 = get_tag_indices(demo_tag1)
    tag_classes_l2 = get_tag_indices(demo_tag2)
    for t in all_tags:
      tag_classes1[t][b] = tag_classes_l1[t]
      tag_classes2[t][b] = tag_classes_l2[t]

    #language_index = args.languages.split(',').index(language_choice)
    #relation_class[b] = language_index*len(all_relation_labels)*2+all_relation_labels.index(relation_type)*2
    #print('relation_type: {}'.format(relation_type))
    if generator is None and random.choice([True, False]):
      # reversing the relation.
      # plus one for reverse.
      #relation_class[b] = all_relation_labels.index(relation_type)*2+1
      relation_class[b] = relation_label_to_class(relation_type, all_relation_labels, reverse=True)
      tmp = demo_word2
      demo_word2 = demo_word1
      demo_word1 = tmp
      tmp = target_word
      target_word = query_word
      query_word = tmp
    dw1s.append(demo_word1)
    dw2s.append(demo_word2)
    qws.append(query_word)
    tws.append(target_word)

  if len(dw1s) <= 0:
    return None, None, None, None, None, None, None, None

  drop_char_p=0.0
  if partition == 'train' and args.drop_char_p > 0.0:
    drop_char_p = args.drop_char_p
  relation_class = Variable(relation_class, requires_grad=False)
  for t in all_tags:
    tag_classes1[t] = Variable(tag_classes1[t], requires_grad=False)
    tag_classes2[t] = Variable(tag_classes2[t], requires_grad=False)
  language_class = Variable(language_class, requires_grad=False)
  if return_strings:
    return dw1s, dw2s, qws, tws, relation_class, tag_classes1, tag_classes2, language_class
  dw1t = Variable(line_to_index_tensor(dw1s, pad_before=True, drop_char_p=drop_char_p), requires_grad=False)
  dw2t = Variable(line_to_index_tensor(dw2s, pad_before=True, drop_char_p=drop_char_p), requires_grad=False)
  qwt = Variable(line_to_index_tensor(qws, pad_before=True, drop_char_p=drop_char_p), requires_grad=False)
  twt = Variable(line_to_index_tensor(tws, pad_before=False, append_bos_eos=True, reverse=args.reverse_target), requires_grad=False)
  return dw1t, dw2t, qwt, twt, relation_class, tag_classes1, tag_classes2, language_class

def pair_generator(partition, disable_self_relation_test=False):
  for l in languages:
    for r in sorted(relations[partition][l].keys()):
      for pair_i in range(len(relations[partition][l][r])):
        for reverse in [False, True]:
          size = len(relations[partition][l][r])
          if size == 1 and disable_self_relation_test:
            continue
          (p1w1,p1w2) = relations[partition][l][r][(pair_i-1)%size]
          (p2w1,p2w2) = relations[partition][l][r][pair_i]
          if reverse:
            sw = p1w2
            p1w2 = p1w1
            p1w1 = sw
            sw = p2w2
            p2w2 = p2w1
            p2w1 = sw
          yield l, r, (p1w1,p1w2), (p2w1,p2w2), reverse

def get_tag_pair(relation_type):
  tag_classes1 = {}
  tag_classes2 = {}
  tags = relation_type.split('-')
  tags1 = tags[0].split(',')
  tags1_d = {}
  for t in tags1:
    tp = t.split('=')
    tags1_d[tp[0]] = tp[1]
  tags2 = tags[1].split(',')
  tags2_d = {}
  for t in tags2:
    tp = t.split('=')
    tags2_d[tp[0]] = tp[1]
  for t in all_tags:
    tag1 = tags1_d.get(t, 'nil')
    tag2 = tags2_d.get(t, 'nil')
    tag_classes1[t] = all_tags[t].index(tag1)
    tag_classes2[t] = all_tags[t].index(tag2)

  return tag_classes1, tag_classes2

def get_tag_indices(tag_d):
  tag_classes = {}
  for t in all_tags:
    tagval = tag_d.get(t, 'nil')
    tag_classes[t] = all_tags[t].index(tagval)
  return tag_classes

def train(demo_word1, demo_word2, query_word, target_word, relation_class, tags1, tags2, language_class, model, criterion, class_criterion, tfr=teacher_forcing_ratio, optimizer=None, return_embeddings=False):
  if optimizer is not None:
    for m in range(len(optimizer)):
      optimizer[m].zero_grad()

  total_loss = 0.0
  total_rel_loss_val = 0.0
  total_tags_loss_val = 0.0
  ensemble_outputs = []

  for m in range(len(model)):
    #if len(model) > 1:
    #  #print('model {}'.format(m))
    outputs, relation_classification_output, tags_output1, tags_output2, tags_output_q, lang_out_rel_head, lang_out_query_head, embeddings = model[m](demo_word1, demo_word2, query_word, target_word, tfr, return_embeddings=(return_embeddings or args.cov_reg != 0.0))
    ensemble_outputs.append(outputs)

    target = target_word.permute(1,0)[1:,:]
    loss = 0.0
    pad_target = Variable(torch.zeros(outputs.size()[1]).long()+reverse_vocab['<PAD>'], requires_grad=False)
    if use_cuda:
      pad_target = pad_target.cuda()
    for i in range(outputs.size()[0]):
      if i < len(target):
        t = target[i]
      else:
        t = pad_target
      loss += criterion(outputs[i], t)
    loss = loss / outputs.size()[0]

    model_loss = loss
    model_rel_loss_val = 0.0
    if relation_class is not None and args.enable_relation_loss:
      relation_classification_loss = class_criterion(relation_classification_output, relation_class)
      model_loss += relation_classification_loss
      model_rel_loss_val = relation_classification_loss.data[0]

    if args.l1_reg > 0.0:
      for parameter in model.parameters():
        model_loss += args.l1_reg*torch.sum(torch.abs(parameter))

    if args.cov_reg > 0.0:
      for embedding in embeddings:
        model_loss += args.cov_reg*covarNorm(embeddings[embedding])
      if not return_embeddings:
        embeddings = None

    model_tags_loss_val = 0.0
    enable_tags_loss = random.random() < args.tags_loss_fraction
    if tags1 is not None and tags2 is not None and enable_tags_loss:
      for t in all_tags.keys():
        if args.enable_query_tags_loss:
          tags_classification_loss = class_criterion(tags_output_q[t], tags1[t])
          model_loss += tags_classification_loss 
        tags_classification_loss = class_criterion(tags_output1[t], tags1[t])
        model_loss += tags_classification_loss 
        model_tags_loss_val += tags_classification_loss.data[0]
        tags_classification_loss = class_criterion(tags_output2[t], tags2[t])
        model_loss += tags_classification_loss
        model_tags_loss_val += tags_classification_loss.data[0]

    lang_loss_val = 0.0
    if language_class is not None and not args.disable_language_loss and len(languages):
      language_classification_loss = class_criterion(lang_out_rel_head, language_class)
      model_loss += language_classification_loss
      lang_loss_val = language_classification_loss.data[0]
      language_classification_loss = class_criterion(lang_out_query_head, language_class)
      model_loss += language_classification_loss
      lang_loss_val += language_classification_loss.data[0]

    if optimizer is not None:
      model_loss.backward()
      optimizer[m].step()
    total_loss += model_loss
    total_rel_loss_val += model_rel_loss_val
    total_tags_loss_val += model_tags_loss_val

  total_loss /= len(model)
  total_rel_loss_val /= len(model)
  total_tags_loss_val /= len(model)

  return ensemble_outputs, total_loss.data[0], total_rel_loss_val, total_tags_loss_val, embeddings

def predict(demo_word1, demo_word2, query_word, target_word, model, criterion, class_criterion, return_embeddings=False):
  return train(demo_word1, demo_word2, query_word, target_word, None, None, None, None, model, criterion, class_criterion, tfr=0.0, return_embeddings=return_embeddings)


def topi(x):
  topv, topi = x.data.topk(1)
  return topi

def to_scalar(var):
  # returns a python float
  return var.view(-1)[0]
  #return var.view(-1).data.tolist()[0]

def word_tensor_to_string(t, handle_special_tokens=False):
  word = ''
  for o in range(t.size()[0]):
    index = to_scalar(t[o].data)
    if index == reverse_vocab['<PAD>']:
      continue
    if handle_special_tokens:
      if index == reverse_vocab['<EOS>']:
        break
      elif index == reverse_vocab['<BOS>']:
        continue
    word += vocab[index]
  return word

def prediction_tensor_to_string(t):
  word = ''
  for o in t:
    index = to_scalar(topi(o))
    if index == reverse_vocab['<EOS>']:
      break
    elif index == reverse_vocab['<PAD>']:
      continue
    elif index == reverse_vocab['<BOS>']:
      continue
    word += vocab[index]
  return word

def do_save_embeddings(demo_word1_s, demo_word2_s, query_word_s, target_word_s, embeddings, partition, relationtype, overwrite=True):
  for k in embeddings:
    filename = os.path.join(os.path.join(os.path.join(args.save_dir, 'saved_embeddings'), partition), k+'.dat')
    if not os.path.exists(os.path.dirname(filename)):
      os.makedirs(os.path.dirname(filename))
    with open(filename, 'w' if overwrite else 'a') as f:
      if overwrite:
        f.write('# demo_word1 demo_word2 query_word target_word relation_type embedding ({}D)\n'.format(embeddings[k].size(1)))
      for b in range(len(demo_word1_s)):
        f.write('{} {} {} {} {} {}\n'.format(demo_word1_s[b], demo_word2_s[b], query_word_s[b], target_word_s[b], relationtype[b], ' '.join(['{}'.format(embeddings[k][b,i]) for i in range(embeddings[k].size(1))])))

def evaluate(partition, model, criterion, class_criterion, system=None, edit_trees=None, save_embeddings=False, disable_self_relation_test=False):
  if model is not None:
    for m in model:
      m.train(False)
  printstrings_correct = []
  printstrings_incorrect = []
  printstrings_prio = []
  printstrings_nouns = []
  evaluation_steps = 0
  evaluation_batch_size = args.batch_size
  loss_sum = 0.0
  counts = {}
  counts_exact_correct = {}
  levenshteins = {}
  suffixes = ['']
  if edit_trees is not None:
    suffixes.append('_poet')
  for suffix in suffixes:
    counts['tot'+suffix] = 0
    counts['tot_fw'+suffix] = 0
    counts_exact_correct['tot'+suffix] = 0
    counts_exact_correct['tot_fw'+suffix] = 0
    levenshteins['tot'+suffix] = []
    levenshteins['tot_fw'+suffix] = []
    
    for language in languages:
      counts[language[:3]+'_tot'+suffix] = 0
      counts[language[:3]+'_tot_fw'+suffix] = 0
      counts_exact_correct[language[:3]+'_tot'+suffix] = 0
      counts_exact_correct[language[:3]+'_tot_fw'+suffix] = 0
      levenshteins[language[:3]+'_tot'+suffix] = []
      levenshteins[language[:3]+'_tot_fw'+suffix] = []
      for relation in sorted(relations[partition][language].keys()):
        counts[language[:3]+'_'+relation+suffix] = 0
        counts[language[:3]+'_'+relation+'_r'+suffix] = 0
        counts_exact_correct[language[:3]+'_'+relation+suffix] = 0
        counts_exact_correct[language[:3]+'_'+relation+'_r'+suffix] = 0
        levenshteins[language[:3]+'_'+relation+suffix] = []
        levenshteins[language[:3]+'_'+relation+'_r'+suffix] = []
  demo_word1 = 1
  generator = pair_generator(partition, disable_self_relation_test)
  #iteration = 0
  first = True
  while demo_word1 is not None:
    #if iteration %100 == 0:
    #print(iteration)
    #iteration += 1
    sys.stdout.flush()
    if model is not None:
      demo_word1, demo_word2, query_word, target_word, relation_class, tags1, tags2, language_class = get_pair_of_pairs(partition=partition, generator=generator, batch_size=evaluation_batch_size)
      if demo_word1 is None or len(demo_word1.size()) <= 0:
        break
      demo_word1_s  = [word_tensor_to_string(demo_word1[b]) for b in range(demo_word1.size(0))]
      demo_word2_s  = [word_tensor_to_string(demo_word2[b]) for b in range(demo_word2.size(0))]
      query_word_s  = [word_tensor_to_string(query_word[b]) for b in range(query_word.size(0))]
      target_word_s = [word_tensor_to_string(target_word[b], handle_special_tokens=True) for b in range(target_word.size(0))]
      #print(target_word_s[0])
      if use_cuda:
        relation_class = relation_class.cuda()
      ensemble_val_outputs, val_loss, val_class_loss, val_tags_loss, embeddings = predict(demo_word1, demo_word2, query_word, target_word, model, criterion, class_criterion, return_embeddings=save_embeddings)
      if save_embeddings:
        do_save_embeddings(demo_word1_s, demo_word2_s, query_word_s, target_word_s, embeddings, partition, [relation_class_to_label(relation_class[b].view(-1).data[0], all_relation_labels) for b in range(relation_class.size(0))], overwrite=first)
      first = False

      val_outputs_strings = ensemble_voting(ensemble_val_outputs)
      loss_sum += val_loss
    elif system=='copysuffix':
      demo_word1_s, demo_word2_s, query_word_s, target_word_s, relation_class, tags1, tags2, language_class = get_pair_of_pairs(partition=partition, generator=generator, batch_size=evaluation_batch_size, return_strings=True)
      if demo_word1_s is None or len(demo_word1_s) <= 0:
        break
      val_outputs_strings = [copysuffix_baseline(demo_word1_s[b], demo_word2_s[b], query_word_s[b]) for b in range(len(demo_word1_s))]
    sys.stdout.flush()
    evaluation_steps += 1
    
    #t1s = []
    #t2s = []
    #t3s = []
    for b in range(len(target_word_s)):
      if args.remove_id_tests and demo_word1_s[b] == demo_word2_s[b]:
        continue
      language = languages[language_class[b].view(-1).data[0]]
      relation_label = relation_class_to_label(relation_class[b].view(-1).data[0], all_relation_labels)
      #relation_label = all_relation_labels[relation_class[b].view(-1).data[0]//2]
      #if relation_class[b].view(-1).data[0] % 2 != 0:
      #  relation_label += '_r'
      prediction = val_outputs_strings[b]
      use_poet_choices = [False]
      if edit_trees is not None:
        use_poet_choices.append(True)
      for use_poet in use_poet_choices:
        suffix = '_poet' if use_poet else ''
        if use_poet:
          poet_prediction, t1, t2, t3 = poet(query_word_s[b], prediction, edit_trees, language, get_edit_tree(demo_word1_s[b], demo_word2_s[b]))
          #t1s.append(t1)
          #t2s.append(t2)
          #t3s.append(t3)
          #if poet_prediction != prediction:
          #  print('query: {}, prediction: {}, poet: {}'.format(query_word_s[b], prediction, poet_prediction))
          prediction = poet_prediction
        if target_word_s[b] == prediction:
          counts_exact_correct['tot'+suffix] += 1
          counts_exact_correct[language[:3]+'_tot'+suffix] += 1
          counts_exact_correct[language[:3]+'_'+relation_label+suffix] += 1
          if not relation_label.endswith('_r'):
            counts_exact_correct['tot_fw'+suffix] += 1
            counts_exact_correct[language[:3]+'_tot_fw'+suffix] += 1

        #levenshteins
        levenshteins['tot'+suffix].append(levenshtein(target_word_s[b], prediction))
        levenshteins[language[:3]+'_tot'+suffix].append(levenshtein(target_word_s[b], prediction))
        levenshteins[language[:3]+'_'+relation_label+suffix].append(levenshtein(target_word_s[b], prediction))
        if not relation_label.endswith('_r'):
          levenshteins['tot_fw'+suffix].append(levenshtein(target_word_s[b], prediction))
          levenshteins[language[:3]+'_tot_fw'+suffix].append(levenshtein(target_word_s[b], prediction))

        # counts
        counts['tot'+suffix] += 1
        counts[language[:3]+'_tot'+suffix] += 1
        counts[language[:3]+'_'+relation_label+suffix] += 1
        if not relation_label.endswith('_r'):
          counts['tot_fw'+suffix] += 1
          counts[language[:3]+'_tot_fw'+suffix] += 1

      ###### ONLY FOR PRINTING ######
      # Going without POET here, for now.
      printstring = format_wordline(demo_word1_s[b], demo_word2_s[b], query_word_s[b], target_word_s[b], val_outputs_strings[b], prediction)
      if target_word_s[b] == val_outputs_strings[b] or target_word_s[b] == prediction:
        printstrings_correct.append(printstring)
      else:
        printstrings_incorrect.append(printstring)
      if args.prio_relation is not None and (args.prio_relation == relation_label or args.prio_relation+'_r' == relation_label):
        printstrings_prio.append(printstring)
      if ('pos=N' in relation_label):
        printstrings_nouns.append(printstring)
      ###### PRINTING PREPARATIONS DONE ######
      
    #if len(t1s) and len(t2s) and len(t3s):
    #  print('times: {} {} {}'.format(0.0 if len(t1s) == 0 else sum(t1s)/len(t1s), 0.0 if len(t2s) == 0 else sum(t2s)/len(t2s), 0.0 if len(t3s) == 0 else sum(t3s)/len(t3s)))

  if model is not None:
    print('******** Samples from {} set (correctly predicted, {}):  **********'.format(partition[:3], len(printstrings_correct)))
    print(format_wordline())
    for l in random.sample(printstrings_correct, min(len(printstrings_correct),10)):
      print(l)
    print('*********  Incorrect ({}): *********'.format(len(printstrings_incorrect)))
    for l in printstrings_incorrect:
      print(l)
    if args.prio_relation:
      print('********* Prio relation ({}): *********'.format(len(printstrings_prio)))
      for l in printstrings_prio:
        print(l)
    if len(printstrings_nouns):
      print('********* Nouns ({}): *********'.format(len(printstrings_nouns)))
      for l in printstrings_nouns:
        print(l)
  if model is not None:
    for m in model:
      m.train(False)
  fraction_exact_correct = {}
  for k in counts_exact_correct:
    fraction_exact_correct[k] = counts_exact_correct[k]/counts[k] if counts[k] > 0 else 0.0
  avg_levenshteins = {}
  for k in levenshteins:
    avg_levenshteins[k] = sum(levenshteins[k])/len(levenshteins[k])
  count = counts['tot']
  return loss_sum, evaluation_steps, fraction_exact_correct, avg_levenshteins, count

def ensemble_voting(ensemble_outputs):
  return_strings = []
  for b in range(ensemble_outputs[0].size(1)):
    candidate_strings = {}
    for m in range(len(ensemble_outputs)):
      s = prediction_tensor_to_string(ensemble_outputs[m].permute(1,0,2)[b])
      candidate_strings[s] = candidate_strings.get(s, 0)+1
    maximum = 0
    if len(ensemble_outputs) > 1 and len(candidate_strings):
      print(candidate_strings)
    maximum_choices = []
    for s in candidate_strings:
      if candidate_strings[s] > maximum:
        maximum = candidate_strings[s]
        maximum_choices = [s]
      elif candidate_strings[s] == maximum:
        maximum_choices.append(s)
    return_strings.append(randomChoice(maximum_choices))
  return return_strings

def copysuffix_baseline(demo_word1, demo_word2, query_word):
  suffix1 = ''
  suffix2 = ''
  for i in range(min(len(demo_word1), len(demo_word2))):
    if demo_word1[i] != demo_word2[i]:
      #if len(demo_word1) > len(demo_word2):
      #  return query_word+demo_word1[i:]
      #else
      #  return query_word+demo_word2[i:]
      i = i-1
      break
  suffix1 = demo_word1[i+1:]
  suffix2 = demo_word2[i+1:]
  if query_word.endswith(suffix1):
    return query_word[:len(query_word)-len(suffix1)]+suffix2
  else:
    return query_word+suffix2
    
def test_words_evaluation(model, criterion, class_criterion):
  if len(test_words)%4 != 0:
    print('Cannot test with these test words. Please specify comma-separated quadruples with demo_word1,demo_word2,query_word,target_word.')
  else:
    quadruple = []
    correct_count = 0
    total_count = 0
    print('--test_words:')
    print(format_wordline())
    for word in test_words:
      quadruple.append(word)
      if len(quadruple) < 4:
        continue
      else:
        for demo_swap in [True, False]:
          for forward in [True, False]:
            local_quadruple = quadruple
            if demo_swap:
              local_quadruple = [local_quadruple[2], local_quadruple[3], local_quadruple[0], local_quadruple[1]]
            if not forward:
              local_quadruple = [local_quadruple[1], local_quadruple[0], local_quadruple[3], local_quadruple[2]]
            dw1 = Variable(line_to_index_tensor([local_quadruple[0]], pad_before=True), requires_grad=False)
            dw2 = Variable(line_to_index_tensor([local_quadruple[1]], pad_before=True), requires_grad=False)
            qw = Variable(line_to_index_tensor([local_quadruple[2]], pad_before=True), requires_grad=False)
            tw = Variable(line_to_index_tensor([local_quadruple[3]], pad_before=False, append_bos_eos=True), requires_grad=False)
            #print('tw'+str(tw))
            relation_class = None
            ensemble_outputs, _, _, _, _ = predict(dw1, dw2, qw, tw, model, criterion, class_criterion)
            #print('prediction step:')
            #print(outputs.size())
            prediction = ensemble_voting(ensemble_outputs)[0]
            #print('prediction step done')
            print(format_wordline(local_quadruple[0], local_quadruple[1], local_quadruple[2], local_quadruple[3], prediction))
            sys.stdout.flush()
            correct_count += 1 if prediction == local_quadruple[3] else 0
            total_count += 1
        quadruple = []
    print('test_words accuracy: {}'.format(correct_count/total_count))

def format_wordline(dw1='Rel1', dw2='Rel2', qw='Query', tw='Target', p='Prediction', poet_prediction=None):
  correct = tw==p or tw ==poet_prediction
  correct_str = ' ' if dw1 == 'Rel1' else '?' if tw == 'N/A' else '✓' if correct else '✗'
  return '{} {} {} {} {} {} {}'.format(correct_str, dw1.ljust(16), dw2.ljust(16), qw.ljust(16), tw[::-1].ljust(16) if args.reverse_target else tw.ljust(16), p[::-1].ljust(16) if args.reverse_target else p.ljust(16), poet_prediction.ljust(16) if poet_prediction is not None else '')

def format_results(iteration, loss_sum, steps, score, training_loss='n/a', factor=100):
  header = '# 0: iteration\n# 1: tr_loss\n# 2: val_loss\n# '
  if training_loss != 'n/a':
    training_loss = '{:.5f}'.format(training_loss)
  line = '{} {} {:.5f}'.format(iteration, training_loss, loss_sum/steps)
  # begin with 'tot' before loop below.
  field_i = 3
  for k in ['tot', 'tot_fw']:
    header += ' {}: {}\n#'.format(field_i, k.ljust(6))
    field_i += 1
    line += ' {:.5f}'.format(factor*score[k])
    print('{} {:.2f}'.format(k.rjust(14), factor*score[k]))
    #printheader = header
    #printline = line
  for tot in [True, False]:
    for k in sorted(score.keys()):
      if k in ['tot', 'tot_fw']:
        continue
      if (tot and 'tot' in k) or (not tot and not 'tot' in k):
        header += '{}: {}\n# '.format(field_i, k.replace('_tot', '').ljust(6))
        field_i += 1
        line += ' {:.5f}'.format(factor*score[k])
        if field_i < 3:
          print('{} {:.2f}'.format(k.rjust(14), factor*score[k]))
        #elif field_i == 3:
        #  print('...')
      #if tot and 'tot' in k:
      #  #printheader += ' {}'.format(k.replace('_tot', '').ljust(6))
      #  #printline += ' {:.2f}'.format(factor*score[k])
  return header,line

def main():
  global languages,relations,num_relation_classes,all_relation_labels,all_characters,native_characters,all_tags,test_words,flattened_train_set,vocab
  n_iters          = args.max_iterations
  print_every      = 200 # 5000
  plot_every       = 200
  save_every       = 200
  begin_iteration  = 0
  current_accuracy = -1.0
  best_accuracy    = -1.0
  best_iteration   = -1
  last_iteration   = -1
  last_saved_iteration = -1

  command = ' '.join(sys.argv)
  print(command)

  p = Popen(["/usr/bin/git","log","--pretty=format:\"%H\"","-1"], stdout=PIPE, stderr=PIPE)
  res_string = ""
  res_out,res_err = p.communicate()
  print("{}: git commit: {}".format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), res_out.decode()))

  if args.languages == 'all':
    languages = supported_languages
  else:
    languages = args.languages.split(',')
  if len(languages) <= 1:
    if args.disable_language_loss:
      print('--disable_language_loss was specified, but the language loss is always disabled when running only one language.')
    args.disable_language_loss = True

  if args.uniform_sampling:
    print('Uniform sampling of training data.')
    if args.prio_relation:
      print('(Uniform sampling disables prio_relation: {}).'.format(args.prio_relation))
  elif args.prio_relation:
    print('prio_relation: {}'.format(args.prio_relation))
  if args.ensemble_size > 1:
    print('Ensemble of {} models.'.format(args.ensemble_size))

  if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)
  if args.save_dir:
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'command.txt'), 'a') as f:
      f.write(command+'\n')

  if args.workshop_relation_selection:
    import datapreparer_workshop as datapreparer
  else:
    import datapreparer
  relations,num_relation_classes,all_relation_labels,all_characters,native_characters,all_tags,test_words,flattened_train_set = datapreparer.prepare_data(args.data_dir, args.id_prob, args.test_words)
  if args.save_dir and os.path.exists(os.path.join(args.save_dir, 'vocab.pkl')):
      with open(os.path.join(args.save_dir, 'vocab.pkl'), 'rb') as f:
        _vocab = pickle.load(f)
        initialize_vocab(_vocab=_vocab)
  else:
    initialize_vocab()
    if args.save_dir:
      with open(os.path.join(args.save_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

  print('num relation classes: {}'.format(num_relation_classes))
  print('languages: {}'.format(languages))

  edit_trees = None
  if args.poet:
    print('POET: Creating edit tree data base.')
    edit_trees = get_edit_trees(relations)
    print('POET: Creating edit tree data base done.')

  model = []
  for m in range(args.ensemble_size):
    dec_hidden_size = args.hidden_size
    if args.decoder_hidden_size is not None:
      dec_hidden_size = args.decoder_hidden_size
    dec_rnn_depth = args.rnn_depth
    if args.decoder_rnn_depth is not None:
      dec_rnn_depth = args.decoder_rnn_depth
    model.append(RnnRelationModel(vocab_size, num_relation_classes, args.embedding_size, args.hidden_size, dec_hidden_size, args.rnn_depth, dec_rnn_depth, args.disable_attention, args.disable_relation_input, args.enable_relation_loss, args.disable_language_loss, all_tags, len(languages), args.decoder_extra_layer, args.bidirectional_encoders))

  if args.save_dir:
    print('save_dir: {}'.format(args.save_dir))
    if os.path.exists(os.path.join(args.save_dir, 'current-iteration.txt')):
      with open(os.path.join(args.save_dir, 'current-iteration.txt'), 'r') as f:
        for l in f:
          if l:
            print('iteration: {}'.format(l))
            begin_iteration = int(l)
            begin_iteration += 1
          else:
            print('empty line!')
      try:
        with open(os.path.join(args.save_dir, 'best-iteration.txt'), 'r') as f:
          for l in f:
            if l:
              best_iteration = int(l)
        with open(os.path.join(args.save_dir, 'best-accuracy.txt'), 'r') as f:
          for l in f:
            if l:
              best_accuracy = float(l)
      except:
        pass

    last_iteration = begin_iteration - 1
  else:
    print('Created model with fresh parameters.')

  if use_cuda:
    for m in range(len(model)):
      model[m] = model[m].cuda()

  # Keep track of losses for plotting
  current_loss = 0
  all_losses = []

  def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

  start = time.time()

  loss_weights = torch.ones(vocab_size)
  if args.pad_loss_correction:
    loss_weights[reverse_vocab['<PAD>']] = 0.0
  if use_cuda:
    loss_weights = loss_weights.cuda()
  criterion = nn.NLLLoss(weight=loss_weights)
  class_criterion = nn.NLLLoss()
  optimizer = []
  for m in range(len(model)):
    optimizer.append(optim.Adam(model[m].parameters(), lr = args.learning_rate, weight_decay = args.l2_reg))

  iter=begin_iteration-1
  if not args.test_only and not args.interactive:
    if not args.save_dir:
      print('About to load saved parameters from best model, but no save_dir arguent!')
      raise('About to load saved parameters from best model, but no save_dir arguent!')
    print('Loading model parameters (iteration {}) from {}.'.format(last_iteration, args.save_dir))
    if os.path.exists(os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(0, last_iteration))):
      for m in range(len(model)):
        filename = os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(m, last_iteration))
        if os.path.exists(filename):
          model[m].load_state_dict(torch.load(filename))
          last_saved_iteration = last_iteration
        else:
          print('Could not load file: {}'.format(filename))
    elif args.ensemble_size == 1 and os.path.exists(os.path.join(args.save_dir, 'parameters-{}.torch'.format(last_iteration))):
      print('Loading model parameters (iteration {}) from (legacy file) {}.'.format(last_iteration, args.save_dir))
      model[0].load_state_dict(torch.load(os.path.join(args.save_dir, 'parameters-{}.torch'.format(last_iteration))))
      last_saved_iteration = last_iteration
    else:
      print('No model parameters found at location: {}. Will proceed with freshly initialized parameters and try to save to this location.'.format(args.save_dir))
    for iter in range(begin_iteration, n_iters + 1):
      # get random example:
      if args.prio_relation and random.random() < args.prio_p:
        demo_word1, demo_word2, query_word, target_word, relation_class, tags1, tags2, language_class = get_pair_of_pairs('train', relation=args.prio_relation)
      elif args.id_prob and random.random() < args.id_prob:
        demo_word1, demo_word2, query_word, target_word, relation_class, tags1, tags2, language_class = get_pair_of_pairs('train', relation='id')
      else:
        demo_word1, demo_word2, query_word, target_word, relation_class, tags1, tags2, language_class = get_pair_of_pairs('train')
      if use_cuda:
        relation_class = relation_class.cuda()
        for t in all_tags:
          tags1[t] = tags1[t].cuda()
          tags2[t] = tags2[t].cuda()
        language_class = language_class.cuda()
      ensemble_outputs, loss, class_loss, tags_loss, _ = train(demo_word1, demo_word2, query_word, target_word, relation_class, tags1, tags2, language_class, model, criterion, class_criterion, tfr=teacher_forcing_ratio, optimizer=optimizer)
      current_loss += loss

      # Print iter number, loss, name and prediction
      if iter % print_every == 0 and iter > 0:
        print('train: step {} ({}%, {}) loss: {:.4f} rel (leg) loss {:.4f}, tags loss {:.4f}'.format(iter, iter / n_iters * 100, timeSince(start), loss, class_loss, tags_loss))

        vt = time.time()
        val_loss_sum, validation_steps, fraction_exact_correct, avg_levenshteins, count = evaluate('validation', model, criterion, class_criterion, edit_trees=edit_trees, save_embeddings=False, disable_self_relation_test=args.disable_self_relation_test)
        print('{}: validation loss: {:.4f}, accuracy (exact match): {:.2f} (fw {:.2f}). Total validation tests: {} time for evaluation: {:.1f}'.format(iter, val_loss_sum/validation_steps, 100*fraction_exact_correct['tot'], 100*fraction_exact_correct['tot_fw'], count, time.time()-vt))
        if args.prio_relation:
          print('prio_relation ({}): {}'.format(args.prio_relation, ' '.join(['{}: {:.2f}'.format(x.replace('_'+args.prio_relation, ''), 100*fraction_exact_correct[x]) for x in fraction_exact_correct if x.endswith(args.prio_relation)])))
        sys.stdout.flush()
        current_accuracy = fraction_exact_correct['tot']
        sys.stdout.flush()
        
        if args.test_words:
          test_words_evaluation(model, criterion, class_criterion)
        
        if args.save_dir:
          print_header = False
          progress_file = os.path.join(args.save_dir, 'progress.data')
          if not os.path.exists(progress_file):
            print_header = True
          with open(progress_file, 'a') as f:
            header,line = format_results(iter, val_loss_sum, validation_steps, fraction_exact_correct, loss)
            if print_header:
              f.write(header+'\n')
            f.write(line+'\n')
          print_header = False
          progress_file = os.path.join(args.save_dir, 'progress-levenshteins.data')
          if not os.path.exists(progress_file):
            print_header = True
          with open(progress_file, 'a') as f:
            header,line = format_results(iter, val_loss_sum, validation_steps, avg_levenshteins, loss, factor=1)
            if print_header:
              f.write(header+'\n')
            f.write(line+'\n')
      # Add current loss avg to list of losses
      if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
      if iter % save_every == 0 and iter > 0:
        time_to_stop=False
        if args.save_dir:
          print('Saving model parameters (iteration {}) to {}.'.format(iter, args.save_dir))
          for m in range(len(model)):
            torch.save(model[m].state_dict(), os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(m, iter)))
          with open(os.path.join(args.save_dir, 'current-iteration.txt'), 'w') as f:
            f.write(str(iter)+'\n')
        if current_accuracy > best_accuracy:
          print('Updating best_accuracy, best_iteration.')
          best_accuracy = current_accuracy
          time_to_stop = iter-best_iteration > args.early_stopping_threshold
          last_best_iteration = best_iteration
          best_iteration = iter
          if args.save_dir:
            with open(os.path.join(args.save_dir, 'best-iteration.txt'), 'w') as f:
              f.write(str(iter)+'\n')
            with open(os.path.join(args.save_dir, 'best-accuracy.txt'), 'w') as f:
              f.write(str(current_accuracy)+'\n')
            if last_best_iteration > -1 and last_best_iteration != last_saved_iteration:
              print('Removing old parameter files: \'{}\'.'.format(last_best_iteration))
              for m in range(len(model)):
                filename = os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(m, last_best_iteration))
                if os.path.exists(filename):
                  os.remove(filename)
                else:
                  print('Could not remove file: {}'.format(filename))
        if args.save_dir:
          if last_saved_iteration > -1 and last_saved_iteration != best_iteration:
            print('Removing old parameter files: \'{}\'.'.format(last_saved_iteration))
            for m in range(len(model)):
              filename = os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(m, last_saved_iteration))
              if os.path.exists(filename):
                os.remove(filename)
              else:
                print('Could not remove file: {}'.format(filename))
          last_saved_iteration = iter
          print('Done saving.')
        if time_to_stop:
          print('Early stopping!')
          break
  
  if args.save_dir:
    if os.path.exists(os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(0, best_iteration))) \
       or (args.ensemble_size == 1 and os.path.exists(os.path.join(args.save_dir, 'parameters-{}.torch'.format(best_iteration)))):
      print('Loading model parameters (iteration {}) from {}.'.format(best_iteration, args.save_dir))
      for m in range(len(model)):
        sys.stdout.flush()
        if os.path.exists(os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(m, best_iteration))):
          model[m].load_state_dict(torch.load(os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(m, best_iteration))))
        elif m == 0:
          model[m].load_state_dict(torch.load(os.path.join(args.save_dir, 'parameters-{}.torch'.format(best_iteration))))
      if args.test_words:
        test_words_evaluation(model, criterion, class_criterion)
      if args.interactive:
        print('Interactive mode. Enter space-separated word triplet, A B C, such that A is to B as C is to what?')
        while True:
          line = input('> ')
          words = line.split(' ')
          if len(words) != 3:
            print('Please input a space-separated list of exactly three words.')
            continue
          dw1 = Variable(line_to_index_tensor([words[0]], pad_before=True), requires_grad=False)
          dw2 = Variable(line_to_index_tensor([words[1]], pad_before=True), requires_grad=False)
          qw = Variable(line_to_index_tensor([words[2]], pad_before=True), requires_grad=False)
          tw = Variable(line_to_index_tensor(['N/A'], pad_before=False, append_bos_eos=True), requires_grad=False)

          relation_class = None
          ensemble_outputs, _, _, _, _ = predict(dw1, dw2, qw, tw, model, criterion, class_criterion)
          #print('prediction step:')
          #print(outputs.size())
          prediction = ensemble_voting(ensemble_outputs)[0]
          #print('prediction step done')
          print(format_wordline())
          print(format_wordline(word_tensor_to_string(dw1[0]), word_tensor_to_string(dw2[0]), word_tensor_to_string(qw[0]), 'N/A', prediction))
          #print('{} is to {} as {} is to {}.'.format(word_tensor_to_string(dw1[0]), word_tensor_to_string(dw2[0]), word_tensor_to_string(qw[0]), prediction))
          #print('{} is to {} as {} is to {}.'.format(words[0], words[1], words[2], prediction))
          sys.stdout.flush()
      else:
        # time for test!
        for system in ['copysuffix', 'model']:
          print(system)
          tt = time.time()
          if system == 'model':
            if args.save_embeddings:
              test_loss_sum, test_steps, fraction_exact_correct, avg_levenshteins, count = evaluate('validation', model, criterion, class_criterion, edit_trees=edit_trees, save_embeddings=args.save_embeddings, disable_self_relation_test=args.disable_self_relation_test)
            test_loss_sum, test_steps, fraction_exact_correct, avg_levenshteins, count = evaluate('test', model, criterion, class_criterion, edit_trees=edit_trees, save_embeddings=args.save_embeddings, disable_self_relation_test=args.disable_self_relation_test)
          else:
            test_loss_sum, test_steps, fraction_exact_correct, avg_levenshteins, count = evaluate('test', None, None, None, system=system, edit_trees=edit_trees, save_embeddings=args.save_embeddings, disable_self_relation_test=args.disable_self_relation_test)
          print('time for test: {:.1f}'.format(time.time()-tt))
          
          print_header = False
          results_file = os.path.join(args.save_dir, 'test-{}.data'.format(system))
          if not os.path.exists(results_file):
            print_header = True
          with open(results_file, 'a') as f:
            header,line = format_results(best_iteration, test_loss_sum, test_steps, fraction_exact_correct)
            if print_header:
              f.write(header+'\n')
            f.write(line+'\n')
          
          print_header = False
          results_file = os.path.join(args.save_dir, 'test-levenshteins-{}.data'.format(system))
          if not os.path.exists(results_file):
            print_header = True
          with open(results_file, 'a') as f:
            header,line = format_results(best_iteration, test_loss_sum, test_steps, avg_levenshteins, factor=1)
            if print_header:
              f.write(header+'\n')
            f.write(line+'\n')
    else:
      print('Error: Tried to load the best model (from iteration: {}), but it was not found at \'{}\'.'.format(best_iteration, os.path.join(args.save_dir, 'parameters-{}-{}.torch'.format(0, best_iteration))))
  else:
    print('Error: No --save_dir provided.')


if __name__ == '__main__':
  main()
