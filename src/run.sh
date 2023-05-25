#!/bin/bash

SETTING="$1"
ASPECT="$2"
MODE="$3"
FOLD_ID="$4"
ORDER_SEED="$5"
SAMPLING_PROCEDURE="$6"
TASK=summarization
FOLD_GROUP_SEED=1
TEST_SAMPLE_SEED=1

if [ ${SAMPLING_PROCEDURE} = "uniform" ];
then
    DATA_DIR=../prepared_data_uniform/fs${FOLD_GROUP_SEED}_ts${TEST_SAMPLE_SEED}/${SETTING}/${MODE}
    RESULTS_DIR=../results_final_uniform_1536/fs${FOLD_GROUP_SEED}_ts${TEST_SAMPLE_SEED}/${SETTING}/${MODE}/${ASPECT}
else
    DATA_DIR=../prepared_data/fs${FOLD_GROUP_SEED}_ts${TEST_SAMPLE_SEED}/${SETTING}/${MODE}
    RESULTS_DIR=../results_final_bucketed_1536/fs${FOLD_GROUP_SEED}_ts${TEST_SAMPLE_SEED}/${SETTING}/${MODE}/${ASPECT}
fi

# EXAMPLES=${DATA_DIR}/train_examples_${FOLD_ID}_${ASPECT}_${MODE}.json
EXAMPLES=${DATA_DIR}/train_examples_0_${ASPECT}_${MODE}.json
TASK_PROMPTS=../task_prompts.json

if [ ${SETTING} = "ice" ];
then 
    FOLD_FOR_NAMING=${FOLD_ID}
    TEST_DATA=${DATA_DIR}/test_examples_combined.json
else
    TEST_FOLD=$(((FOLD_ID + 1) % 25))
    FOLD_FOR_NAMING=${TEST_FOLD}
    TEST_DATA=${DATA_DIR}/test_examples_${TEST_FOLD}.json
fi

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
    --sample_size 100 \
    --max_tokens 5 \
    --num_candidates 1 \
    --task ${TASK} \
    --exclude_description