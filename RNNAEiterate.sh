#!/bin/bash
hid=("80" "100")
nBin=("50" "120" "200")
sig=("0.5")
for h in ${hid[*]}
do
	for nb in ${nBin[*]}
	do
                for s in ${sig[*]}
                do
		        python3 combined_ae_nn_classifier_main.py -e 10 -b 128 -be 0.6 -ds DCASE2 -lS 20 -lE 60 -hE 10 -nb "$nb" -a 3 -d -fe -nf 2048 -hp 512 -l 2 -hid "$h" -lr 0.1 -s "$s" -nl 1 -nhid 50 -nlr 0.1 -el 4 RNN_AE train --loop
                done
	done
done
