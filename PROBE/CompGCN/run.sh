export PYTHONPATH=$(pwd)/..
DATA=$1
GPU=$2
BATCH=$3
LR=$4
NUM_LAYER=$5
DIM=$6
DROPOUT=$7
SEED=$8
echo "Start training..."
python -u ./src/comp.py \
    --data $DATA \
    --gpu $GPU \
    --batch $BATCH \
    --lr $LR \
    --num_layer $NUM_LAYER \
    --dim $DIM \
    --dropout $DROPOUT \
    --seed $SEED \