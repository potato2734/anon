#!/bin/bash
START_DIR=$(pwd)
DATA=$1
SMOOTHING_p=$2
SMOOTHING_pp=$3
SEED=$4

OUTPUT_FILE=../data/"$DATA"/mined_rules_"$SEED(seed)".txt
export PYTHONPATH=$(pwd)/..
echo "Switching to 'miner' directory..."
cd miner || { echo "Error: 'miner' directory not found!"; exit 1; }
g++ -O3 rnnlogic.h rnnlogic.cpp main.cpp -o rnnlogic -lpthread
# Run the first command
echo "Running RNNLogic cpp..."
./rnnlogic -data-path ../data/"$DATA" \
            -max-length 3 \
            -threads 30 \
            -lr 0.01 \
            -wd 0.0005 \
            -temp 100 \
            -iterations 1 \
            -top-n 0 \
            -top-k 0 \
            -top-n-out 0 \
            -output-file $OUTPUT_FILE \
            -seed $SEED

# Check if the first command succeeded
if [ $? -ne 0 ]; then
    echo "Error: The first command failed. Exiting..."
    exit 1
fi

# echo "Switching to 'src' directory..."
cd ../src || { echo "Error: 'src' directory not found!"; exit 1; }
# cd ./src
# Run the second command
echo "Running Python script..."
python run_rnnlogic.py --config ../config/"$DATA".yaml --smoothing_p $SMOOTHING_p --smoothing_pp $SMOOTHING_pp --rule_file $OUTPUT_FILE --seed $SEED

# Check if the second command succeeded
if [ $? -ne 0 ]; then
    echo "Error: The second command failed. Exiting..."
    exit 1
fi

echo "Both commands executed successfully."
cd "$START_DIR"