#!/bin/bash

EXPERIMENT_PATH=~/experiments/

LABEL=paper-experiments; for l in german russian finnish; do mkdir -p $EXPERIMENT_PATH/$LABEL/$l && ./char_rnn_wordrelations.py --language $l --tie_rel_weights --tags_loss_fraction 0.0 --data_dir data-uncollapsed-2019 --dataset_partitions_without_singles train,validation,test --disable_self_relation_test --hidden_size 50 --embedding_size 100 --save_dir $EXPERIMENT_PATH/$LABEL/$l --print_validation_predictions > $EXPERIMENT_PATH/$LABEL/$l/log; done


