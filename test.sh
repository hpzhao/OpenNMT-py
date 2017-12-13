#!/bin/bash
src_train='data/IWSLT2014/train.de-en.de'
src_valid='data/IWSLT2014/valid.de-en.de'
src_test='data/IWSLT2014/test.de-en.de'

tgt_train='data/IWSLT2014/train.de-en.en'

models='model/model_1.pt model/model_2.pt model/model_3.pt'

# single model translate
#python translate.py -model model/model_1.pt -src $src_test -output result/model_1_03.test  -gpu 1  > /dev/null 2>&1 &
#python translate.py -model model/model_1.pt -src $src_valid -output result/model_1_03.valid -gpu 1 > /dev/null 2>&1 &
#python translate.py -model model/model_2.pt -src $src_test -output result/model_2_03.test -gpu 2  > /dev/null 2>&1 &
#python translate.py -model model/model_2.pt -src $src_valid -output result/model_2_03.valid -gpu 2 > /dev/null 2>&1 &
#python translate.py -model model/model_3.pt -src $src_test -output result/model_3_03.test -gpu 3  > /dev/null 2>&1 &
#python translate.py -model model/model_3.pt -src $src_valid -output result/model_3_03.valid -gpu 3 > /dev/null 2>&1 &

# ensemble model translate
#python translate.py -models $models -src $src_valid -output result/ensembleX3.valid -gpu 1 > /dev/null 2>&1 &
#python translate.py -models $models -src $src_test -output result/ensembleX3.test -gpu 2 > /dev/null 2>&1 &

#python translate.py -models $models -src $src_train -output result/ensembleX3.train -dump_prob data/sampling.prob.pkl -gpu 2 > /dev/null 2>&1 &
#python translate.py -models $models -src $src_train -tgt $tgt_train -dump_prob data/tf.prob.pkl -gpu 3 > /dev/null 2>&1 &

# ensemble model translate
python translate.py -models model/student_tf.pt -src $src_valid -output result/student_tf.valid -gpu 1 > /dev/null 2>&1 &
python translate.py -models model/student_tf.pt -src $src_test -output result/student_tf.test -gpu 1 > /dev/null 2>&1 &

python translate.py -models model/student_sampling.pt -src $src_valid -output result/student_sampling.valid -gpu 3 > /dev/null 2>&1 &
python translate.py -models model/student_sampling.pt -src $src_test -output result/student_sampling.test -gpu 3 > /dev/null 2>&1 &
