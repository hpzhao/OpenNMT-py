#!/bin/bash
# train the single model
python train.py -data data/teacher -save_model ./model/model_1 -rnn_size 256 -epochs 50 -batch_size 64 -seed 1 -gpuid 1 > log/model_1.log 2>&1 &
python train.py -data data/teacher -save_model ./model/model_2 -rnn_size 256 -epochs 50 -batch_size 64 -seed 2 -gpuid 2 > log/model_2.log 2>&1 &
python train.py -data data/teacher -save_model ./model/model_3 -rnn_size 256 -epochs 50 -batch_size 64 -seed 3 -gpuid 3 > log/model_3.log 2>&1 &
