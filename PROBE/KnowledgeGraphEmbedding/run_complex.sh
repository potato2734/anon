#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models
export PYTHONPATH=$(pwd)/..
# The first four parameters must be provided
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

FULL_DATA_PATH=$DATA_PATH/$DATASET


# Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
GAMMA=$9
ALPHA=${10}
LEARNING_RATE=${11}
MAX_STEPS=${12}
TEST_BATCH_SIZE=${13}
L2=${14}

SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$L2(r)"_"$ALPHA(a)"-"$SAVE_ID"

# Collect additional arguments dynamically
EXTRA_ARGS="${@:15}"  # Captures all arguments from the 14th onward

if [ $MODE == "train" ]
then

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
    --cuda -save_model  \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE -r $L2\
    $EXTRA_ARGS

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE

elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE $EXTRA_ARGS # Pass additional arguments

else
   echo "Unknown MODE" $MODE
fi
