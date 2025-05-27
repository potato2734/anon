#!/bin/bash
echo "Starting mln.cpp"
cd ./mln || { echo "Failed to change directory to ./mln"; exit 1; }

# Compile the C++ program
g++ -O3 mln.cpp -o mln -lpthread

echo "Finished mln.cpp"
# Move back to the starting point
cd .. || { echo "Failed to change back to the original directory"; exit 1; }

export PYTHONPATH=$(pwd)/..
GPU=$1
ITER=$2
DATA=$3
KGE=$4
TH_RULE=$5
TH_TRIPLET=$6
W=$7
SEED=$8

python -u ./run.py -gpu $GPU \
    --iterations $ITER \
    --dataset $DATA \
    --kge_model $KGE \
    --threshold_of_rule $TH_RULE \
    --mln_threshold_of_triplet $TH_TRIPLET \
    --weight $W \
    -seed $SEED \