#!/bin/bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/teacher -shuffle 0
#python train.py -data data/Distill -save_model ./model/distill_gen_model -rnn_size 256 -epochs 50 -batch_size 16 -seed 1 -gpuid 2 -prob data/generate_prob.pkl -topK 3
#python train.py -data data/IWSLT2014 -save_model ./model/distill_gold_model -rnn_size 256 -epochs 50 -batch_size 16 -seed 1 -gpuid 3 -prob data/gold_prob.pkl -topK 3
#python train.py -data data/IWSLT2014 -save_model ./model/distill_gold_model_top1 -rnn_size 256 -epochs 50 -batch_size 64 -seed 1 -gpuid 2 -prob data/gold_prob.pkl -topK 1
# python train.py -data data/Distill -save_model ./model/distill_gen_model_top1 -rnn_size 256 -epochs 50 -batch_size 32 -seed 1 -gpuid 5 -prob data/generate_prob.pkl -topK 1


# train the single model
python train.py -data data/teacher -save_model ./model/model_1 -rnn_size 256 -epochs 50 -batch_size 64 -seed 1 -gpuid 0 > log/model_1.log 2>&1 &
python train.py -data data/teacher -save_model ./model/model_2 -rnn_size 256 -epochs 50 -batch_size 64 -seed 2 -gpuid 1 > log/model_2.log 2>&1 &
python train.py -data data/teacher -save_model ./model/model_3 -rnn_size 256 -epochs 50 -batch_size 64 -seed 3 -gpuid 3 > log/model_3.log 2>&1 &
