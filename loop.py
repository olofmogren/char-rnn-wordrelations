#!/usr/bin/python3

import random, subprocess, os

donefilename = '/home/mogren/experiments/2017-char-rnn-wordrelations/oct/search-corrected-reverse/done.txt'

commands = []
for softmax in ['--softmax_relation', '']:
  for shortcut in ['--rel_outputs_shortcut', '']:
    for tied in ['--tie_rel_weights', '--tie_all_encoder_weights', '']:
      for tags_loss_fraction in ['--tags_loss_fraction 0.5', '--tags_loss_fraction 0.5 --enable_query_tags_loss', '--tags_loss_fraction 0.25', '--tags_loss_fraction 0.0', '--enable_query_tags_loss', '']:
        #for enable_query_tags_loss in ['--enable_query_tags_loss', '']:
        for disable_language_loss in ['--disable_language_loss']:
          for language in ['english', 'swedish', 'turkish', 'finnish']:
            for embedding_size in ['100']:
              for hidden_size in ['100']:
                for sampling in ['--uniform_sampling', '']:
                  for attend_to_relation in ['--attend_to_relation', '']:
                    for id_prob in ['--id_prob 0.3', '--id_prob 0.1', '']:
                      for reverse_target in ['--reverse_target', '']:
                        filename = softmax+shortcut+tied+tags_loss_fraction+disable_language_loss+sampling+attend_to_relation+id_prob+reverse_target
                        filename = filename.replace(' ', '_')
                        filename_new = filename
                        first = True
                        while filename_new != filename or first:
                          filename = filename_new
                          filename_new = filename.replace('--', '-')
                          first = False
                        filename = filename_new

                        filename_old = filename
                        filename = filename.replace('softmax_relation', 'softmax').replace('rel_outputs_shortcut', 'shortcut').replace('tie_rel_weights', 'tie_rel').replace('tie_all_encoder_weights', 'tie_all').replace('tags_loss_fraction', 'F').replace('enable_query_tags_loss', 'querytagsloss').replace('-disable_language_loss', '')

                        path_old = '/home/mogren/experiments/2017-char-rnn-wordrelations/oct/search-corrected-reverse/single{}'.format(filename_old)
                        path = '/home/mogren/experiments/2017-char-rnn-wordrelations/oct/search-corrected-reverse/single{}'.format(filename)
                        full_path_old = '/home/mogren/experiments/2017-char-rnn-wordrelations/oct/search-corrected-reverse/single{}/{}-{}-{}'.format(filename_old, language, embedding_size, hidden_size)
                        full_path = '/home/mogren/experiments/2017-char-rnn-wordrelations/oct/search-corrected-reverse/single{}/{}-{}-{}'.format(filename, language, embedding_size, hidden_size)
                        if os.path.exists(path_old):
                          print('old pathname exists: {}\nwill rename to {}'.format(path_old, path))
                          os.rename(path_old, path)
                        #else:
                        #  print('does not exist: {}\nwould instead use {}'.format(path_old, path))


                        commands.append('/usr/bin/python3 char_rnn_wordrelations.py --save_dir {} --language {} --embedding_size {} --hidden_size {} {} {} {} {} {}'.format(full_path, language, embedding_size, hidden_size, softmax, shortcut, tied, tags_loss_fraction, disable_language_loss))

done = []
todo = commands
random.shuffle(commands)
for command in commands:
  print(command)
  #print([w for w in command.split(' ') if w])
  commandlist = [w for w in command.split(' ') if w]
  next_is_directory = False
  for i in range(len(commandlist)):
    if next_is_directory:
      directory = commandlist[i]
      break
    elif commandlist[i] == '--save_dir':
      next_is_directory = True
  if os.path.exists(directory):
    print('directory exists ({}). will continue with another one.'.format(directory))
    continue
  retcode = subprocess.run(commandlist).returncode
  if retcode != 0:
    print(retcode)
    print('done: {}'.format([x+'\n' for x in done]))
    print('todo: {}'.format([x+'\n' for x in todo]))
    exit()
  else:
    with open(donefilename, 'a') as f:
      f.write(command+'\n')
    done.append(command)
    todo.remove(command)

