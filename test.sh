#!/bin/bash
#python translate.py -model a.pt -src data/src-train.txt -output tgt-train-gen.txt -beam_size 1 -batch_size 1 -gpu 3
#python translate.py -model model/distill_gold_model_top1.pt -src data/src-val.txt -output distill_gold_valid_top1.txt -beam_size 1 -batch_size 1  -gpu 1
#python translate.py -model model/distill_gold_model_top1.pt -src data/src-test.txt -output distill_gold_test_top1.txt -beam_size 1 -batch_size 1  -gpu 1
#python translate.py -model model/distill_gen_model.pt -src data/src-test.txt -output distill_gen_test.txt -beam_size 1 -batch_size 1  -gpu 2
#python translate.py -model model/distill_gold_model.pt -src data/src-val.txt -output distill_gold_valid.txt -beam_size 1 -batch_size 1  -gpu 2
#python translate.py -model model/distill_gold_model.pt -src data/src-test.txt -output distill_gold_test.txt -beam_size 1 -batch_size 1  -gpu 2
#python translate.py -model model/model_1.pt -src data/src-test.txt -output result/model_1.test.txt -beam_size 1 -batch_size 1 -gpu 0
#python translate.py -model model/model_1.pt -src data/src-val.txt -output result/model_1.valid.txt -beam_size 1 -batch_size 1 -gpu 0
#python translate.py -model model/model_2.pt -src data/src-test.txt -output result/model_2.test.txt -beam_size 1 -batch_size 1 -gpu 0
#python translate.py -model model/model_2.pt -src data/src-val.txt -output result/model_2.valid.txt -beam_size 1 -batch_size 1 -gpu 0
#python translate.py -model model/model_3.pt -src data/src-test.txt -output result/model_3.test.txt -beam_size 1 -batch_size 1 -gpu 0
#python translate.py -model model/model_3.pt -src data/src-val.txt -output result/model_3.valid.txt -beam_size 1 -batch_size 1 -gpu 0

#python translate.py -model model/model_1.pt -src data/src-val.txt -output mode_1.re.valid.txt  -beam_size 1 -batch_size 1 -replace_unk -gpu 0 > log_1 2>&1 &
#python translate.py -model model/model_2.pt -src data/src-val.txt -output mode_2.re.valid.txt  -beam_size 1 -batch_size 1 -gpu 0 -replace_unk > log_2 2>&1 &
#python translate.py -model model/model_3.pt -src data/src-val.txt -output mode_3.re.valid.txt  -beam_size 1 -batch_size 1 -gpu 0 -replace_unk > log_3 2>&1 &

python translate.py -model model/model_1.pt -src data/src-test.txt -output mode_1.re.test.txt  -beam_size 1 -batch_size 1 -gpu 0 -replace_unk > log_4 2>&1 &
python translate.py -model model/model_2.pt -src data/src-test.txt -output mode_2.re.test.txt  -beam_size 1 -batch_size 1 -gpu 0 -replace_unk > log_5 2>&1 &
python translate.py -model model/model_3.pt -src data/src-test.txt -output mode_3.re.test.txt  -beam_size 1 -batch_size 1 -gpu 0 -replace_unk > log_6 2>&1 &
