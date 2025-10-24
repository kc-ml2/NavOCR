#!/bin/bash

set -e

source ./env.sh

# Iterate over each dataset
for ID in "${DATA_LIST[@]}"
do
    echo "▶ Running NavOCR for $ID"
    INPUT_ROOT="$DATA_DIR/$ID" OUTPUT_ROOT="$RESULTS_DIR/$ID" python3 src/run_NavOCR.py
done

echo "Completed."