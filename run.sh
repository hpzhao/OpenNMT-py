#!/bin/bash

src_train='data/IWSLT2014/train.de-en.de'
src_valid='data/IWSLT2014/valid.de-en.de'
src_test='data/IWSLT2014/test.de-en.de'

tgt_train='data/IWSLT2014/train.de-en.en'
tgt_valid='data/IWSLT2014/valid.de-en.en'
tgt_test='data/IWSLT2014/test.de-en.en'

tgt_train_distill='result/ensembleX3.train'
tf_prob='data/tf.prob.pkl'
sampling_prob='data/sampling.prob.pkl'

# train the single model
#python train.py -data data/teacher -save_model ./model/model_1 -rnn_size 256 -epochs 50 -batch_size 64 -seed 1 -gpuid 1 > log/model_1.log 2>&1 &
#python train.py -data data/teacher -save_model ./model/model_2 -rnn_size 256 -epochs 50 -batch_size 64 -seed 2 -gpuid 2 > log/model_2.log 2>&1 &
#python train.py -data data/teacher -save_model ./model/model_3 -rnn_size 256 -epochs 50 -batch_size 64 -seed 3 -gpuid 3 > log/model_3.log 2>&1 &

# preprocess distilling data

#python preprocess.py -use_vocab data/teacher.vocab.pt -train_src $src_train -train_tgt $tgt_train_distill -valid_src $src_valid -valid_tgt $tgt_valid  -shuffle 0 -save_data data/student_sampling -distill_prob $sampling_prob
#python preprocess.py -use_vocab data/teacher.vocab.pt -train_src $src_train -train_tgt $tgt_train -valid_src $src_valid -valid_tgt $tgt_valid  -shuffle 0 -save_data data/student_tf -distill_prob $tf_prob
python train.py -data data/student_tf -save_model ./model/student_tf -rnn_size 256 -epochs 50 -batch_size 64 -seed 1 -gpuid 1 -distill -topK 5 > log/student_tf.log 2>&1 &
python train.py -data data/student_sampling -save_model ./model/student_sampling -rnn_size 256 -epochs 50 -batch_size 64 -seed 1 -gpuid 3 -distill -topK 5 > log/student_sampling.log 2>&1 &
