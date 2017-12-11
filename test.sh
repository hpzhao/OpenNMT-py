#!/bin/bash
src_valid='data/IWSLT2014/valid.de-en.de'
src_test='data/IWSLT2014/test.de-en.de'

python translate.py -model model/model_1.pt -src $src_test -output result/model_1_03.test  -gpu 1  > log_1 2>&1 &
python translate.py -model model/model_1.pt -src $src_valid -output result/model_1_03.valid -gpu 1 > log_2 2>&1 &
python translate.py -model model/model_2.pt -src $src_test -output result/model_2_03.test -gpu 2  > log_3 2>&1 &
python translate.py -model model/model_2.pt -src $src_valid -output result/model_2_03.valid -gpu 2 > log_4 2>&1 &
python translate.py -model model/model_3.pt -src $src_test -output result/model_3_03.test -gpu 3  > log_5 2>&1 &
python translate.py -model model/model_3.pt -src $src_valid -output result/model_3_03.valid -gpu 3 > log_6 2>&1 &
