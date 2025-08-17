#!/bin/bash

# List of identifiers
patches=(
    "exp17"
    "exp26"
    "exp27"
    "exp38"
)

# Iterate over each patch and run eval_patch.sh
for patch in "${patches[@]}"; 
do
    ./eval_patch.sh "$patch"
done
