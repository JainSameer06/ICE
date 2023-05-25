#!/bin/bash

SETTING=crossval
ASPECT="$1"
MODE="$2"
FOLD_GROUP_SEED=1
TEST_SAMPLE_SEED=1

DATA_DIR=../prepared_data_uniform/fs${FOLD_GROUP_SEED}_ts${TEST_SAMPLE_SEED}/${SETTING}/${MODE}
RESULTS_DIR=../results_final_uniform_1536/fs${FOLD_GROUP_SEED}_ts${TEST_SAMPLE_SEED}/${SETTING}/${MODE}/${ASPECT}

for fold_idx in {0..23}
do
    bash run.sh crossval ${ASPECT} ${MODE} ${fold_idx} 1 uniform
    sleep 30
    # The second last argument is ORDER_SEED, which is not significant for complete cross-validation
done

# RESULTS_DIR=../results_final_uniform_1536/fs${FOLD_GROUP_SEED}_ts${TEST_SAMPLE_SEED}/${SETTING}/single/temp
python aggregateMetaEval.py \
    --aspect ${ASPECT} \
    --results_dir ${RESULTS_DIR}