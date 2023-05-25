#!/bin/bash

ASPECT="$1"
MODE="$2"   # Can take values "single" and "multi", corresponding to single or multiple examples per article. We are just using "single" now, so it can be hardcoded too.
ORDER_SEED="$3" # Always 1
SAMPLING_PROCEDURE="$4" # uniform or stratified
TASK=summarization
FOLD_GROUP_SEED=1
TEST_SAMPLE_SEED=1

DATA_DIR=../unidata
RESULTS_DIR=../turboresults
if [ ${SAMPLING_PROCEDURE} = "uniform" ];
then
    EXAMPLES=${DATA_DIR}/train_examples_0_${ASPECT}.json #change
else
    EXAMPLES=${DATA_DIR}/train_examples_0_${ASPECT}.json #change
fi

RESULTS_DIR=${RESULTS_DIR}/${SAMPLING_PROCEDURE}/${ASPECT}
TASK_PROMPTS=../task_prompts.json

FOLD_FOR_NAMING=1536
TEST_DATA=${DATA_DIR}/unidata_1536_processed.json #change

python inContextEvaluator.py \
    --example_file ${EXAMPLES} \
    --test_data_file ${TEST_DATA} \
    --results_dir ${RESULTS_DIR} \
    --task_prompt_file ${TASK_PROMPTS} \
    --aspect ${ASPECT} \
    --mode ${MODE} \
    --seed ${ORDER_SEED} \
    --fold_id ${FOLD_FOR_NAMING} \
    --sample \
    --sample_size 2 \
    --max_tokens 5 \
    --num_candidates 1 \
    --task ${TASK} \
    --exclude_description