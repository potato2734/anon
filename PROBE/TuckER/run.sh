#!/bin/sh

export PYTHONPATH=$(pwd)/..
python -u -c 'import torch; print(torch.__version__)'


GPU_DEVICE=$1
DATASET=$2
MAX_STEPS=$3
BATCH_SIZE=$4
LEARNING_RATE=$5 
DR=$6
ENTITY_DIM=$7 
RELATION_DIM=$8 
ID=$9
HD1=${10} 
HD2=${11} 
LABEL_SMOOTHING=${12} 

SAVE=results

# Collect additional arguments dynamically
EXTRA_ARGS="${@:13}"  # Captures all arguments from the 14th onward

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --dataset "$DATASET" \
    --save $SAVE \
    --num_iterations $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --dr $DR \
    --edim $ENTITY_DIM \
    --rdim $RELATION_DIM \
    --input_dropout $ID \
    --hidden_dropout1 $HD1 \
    --hidden_dropout2 $HD2 \
    --label_smoothing $LABEL_SMOOTHING \
    $EXTRA_ARGS  # Include additional arguments dynamically



