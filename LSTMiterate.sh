#!/bin/bash
hid=("25" "50" "100")
layer=("1" "2" "3")
for h in ${hid[*]}
do
	for l in ${layer[*]}
	do
		python3 nn_main.py -e 80 -b 128 -be 0.5 -ds DCASE2 -hE 100 -nb 50 -a 0 -d -fe -l "$l" -hid "$h" -lr 0.1 -s 0.0 LSTM train
	done
done
