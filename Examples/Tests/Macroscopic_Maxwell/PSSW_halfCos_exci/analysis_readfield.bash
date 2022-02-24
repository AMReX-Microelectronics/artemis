#!/bin/bash
for timestep in $(seq 0 500 5000000 ); do
    newnumber='000000000'${timestep}      # get number, pack with zeros
    newnumber=${newnumber:(-9)}       # the last seven characters
    ./WritePlotfileToASCII3d.gnu.ex infile=diags/plt${newnumber} | tee raw_data/${newnumber}.txt
done
