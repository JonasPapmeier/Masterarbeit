#!/bin/bash
sigma=("0.2" "0.5" "0.75")
layer=("1" "2" "3")
for s in ${sigma[*]}
do
	for l in ${layer[*]}
	do
		python3 nn_main.py -e 80 -b 128 -be 0.5 -ds DCASE2 -lE 11 -hE 1 -nb 50 -a 0 -d -fe -l "$l" -hid 750 -lr 0.1 -s "$s" FF train
	done
done
