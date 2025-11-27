#!/bin/bash

./batch_1126_toB_sc8-15min.sh

cd /mnt/debugger/hjb/node1/blitz-infer-pack
git restore exps/blitz-run/configs/config-stubs.json
git checkout tmp-merge-simulator

# back
cd -
./batch_1126_toB_sc8-15min-pad.sh
