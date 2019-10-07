#!/bin/bash

# Base experiment:

for l in english finnish german russian swedish; do ./char_rnn_wordrelations.py --language $l --tie_rel_weights --tags_loss_fraction 0.0 --data_dir data-uncollapsed --dataset_partitions_without_singles train,validation,test --disable_self_relation_test --hidden_size 50 --save_dir /home/mogren/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-h50/$l; done

./char_rnn_wordrelations.py --language english --tie_rel_weights --tags_loss_fraction 0.0 --data_dir data-uncollapsed --dataset_partitions_without_singles train --disable_self_relation_test --disable_attention --hidden_size 50 --save_dir /home/mogren/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/--disable_attention-h50/english

./char_rnn_wordrelations.py --language english --tie_rel_weights --tags_loss_fraction 0.0 --data_dir data-uncollapsed --dataset_partitions_without_singles train --disable_self_relation_test --disable_relation_input --hidden_size 50 --save_dir /home/mogren/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/--disable_relation_input-h50/english

./char_rnn_wordrelations.py --language english --tie_rel_weights --tags_loss_fraction 0.0 --data_dir data-uncollapsed --dataset_partitions_without_singles train,validation,test --disable_self_relation_test --rel_shortcut --hidden_size 50 --save_dir /home/mogren/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/relshortc/english

./char_rnn_wordrelations.py --language english --tie_rel_weights --tags_loss_fraction 0.0 --data_dir data-uncollapsed --dataset_partitions_without_singles train,validation,test --disable_self_relation_test --attend_to_relation --hidden_size 50 --save_dir /home/mogren/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/attendtorelation/english

./char_rnn_wordrelations.py --language english --tie_rel_weights --tags_loss_fraction 0.0 --data_dir data-uncollapsed --dataset_partitions_without_singles train --disable_self_relation_test --disable_demosource_input --hidden_size 50 --save_dir /home/mogren/experiments/2017-char-rnn-wordrelations/feb_resubmission/tie_rel-F0.0-keep1.0-ablations/--disable_demosource_input-h50/english

./char_rnn_wordrelations.py --language english --tie_rel_weights --tags_loss_fraction 0.0 --data_dir data-uncollapsed --dataset_partitions_without_singles train --disable_self_relation_test --disable_demosource_input --hidden_size 50 --save_dir /home/mogren/experiments/2017-char-rnn-wordrelations/feb_resubmission/tie_rel-F0.0-keep1.0-ablations/--disable_demosource_input-h50/english

#Plotting:

./plot_training_trajectory.py ~/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-h50/english/progress.data "NMAS" ~/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/attendtorelation/english/progress.data "Attend to relation" ~/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/attendtorelation-disablecombined/english/progress.data "Attend to relation & No FC combined" ~/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/reverseeverything/english/progress.data "Reversed words" ~/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/dropout0.5/english/progress.data "Dropout" ~/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-ablations/relshortc/english/progress.data "Relation shortcut" ~/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-h50-auxilliarytagsloss/english/progress.data "Auxilliary classification"

#Interactive
l=english
./char_rnn_wordrelations.py --language $l --tie_rel_weights --data_dir data-uncollapsed --dataset_partitions_without_singles train,validation,test --disable_self_relation_test --hidden_size 50 --interactive --save_dir /home/mogren/experiments/2017-char-rnn-wordrelations/nov-final1/tie_rel-F0.0-keep1.0-h50/$l;
