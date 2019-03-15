#!/bin/bash
exampleLenght=("11" "5" "1")
hid=("5" "3" "15" "50")
layer=("1" "2" "3")
for le in ${exampleLenght[*]}
do
	for l in ${layer[*]}
	do
		for h in ${hid[*]}
		do
			python3 nn_main.py -e 80 -b 128 -be 0.1 -ds DCASE2 -lE "$le" -hE 1 -nb 50 -a 0 -d -fe -l "$l" -hid "$h" -lr 0.1 -s 0.0 FF_AE train
		done
	done
done
