#!/bin/bash
for timestep in $(seq 0 500 5000000); do
    newnumber='000000000'${timestep}      # get number, pack with zeros
    newnumber=${newnumber:(-9)}       # the last seven characters
    bash getLine.bash raw_data/${newnumber}.txt 128 | tee -a Mx_xface_center_128.txt
done
