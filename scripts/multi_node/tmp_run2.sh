#!/bin/bash
# run on 3 and 4

# sleep 3h..
sleep 10800

# 4h
./batch_1201_toC_30b_4096.sh

# 2h
./batch_1201_coder_30b_4096.sh

# 2h
./batch_1201_toB_30b_4096.sh




