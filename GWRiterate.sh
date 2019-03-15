#!/bin/bash
maxAge=("10")
iThres=("0.6" "0.7" "0.8")
for ma in ${maxAge[*]}
do
    for it in ${iThres[*]}
    do
        python3 combined_ae_gwr_main.py -gmo models/RNN_AE_lay2_nod100_lr0.1_s0.5DCASE2_sceLen20.0_sampLen60_sampHop10_ev5_mel_fft2048_hop512_bin200_aug3_lpTrue_dFalse_feFalse_predD0 -e 10 -b 128 -be 0.0 -ds DCASE2 -lS 20 -lE 60 -hE 10 -nf 2048 -hp 512 -nb 200 -a 2 -d -fe -it "$it" -ma "$ma" -eb 0.3 -en 0.0003 -ht 0.1 -el 4 RNN_AE validate
        #python3 gwr_main.py -e 10 -b 128 -be 0.0 -ds DCASE2 lS 20 -lE 1 -hE 1 -nb 50 -a 2 -d -fe -it "$it" -ma "$ma" -eb 0.3 -en 0.0003 -ht 0.1 train
    done
done
