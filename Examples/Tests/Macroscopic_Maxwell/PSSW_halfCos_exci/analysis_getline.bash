#!/bin/bash
FILE="$1"
LINE_NO=$2
i=0
while read line; do
  i=$(( i + 1 ))
  case $i in $LINE_NO) echo "$line"; break;; esac
done <"$FILE"

### how to run this script: 
### $ bash squashTime.bash FILE LINE_NO