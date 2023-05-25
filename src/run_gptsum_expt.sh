#!/bin/bash

ASPECT="$1"
MODE="$2"
FOLD_ID="$3"
ORDER_SEED="$4"
TASK=summarization
FOLD_GROUP_SEED=1
TEST_SAMPLE_SEED=1

DATA_DIR=../data_gptsummary_expt/fs${FOLD_GROUP_SEED}
RESULTS_DIR=../results_gptsummary_expt/fs${FOLD_GROUP_SEED}/t0/${ASPECT}

EXAMPLES=${DATA_DIR}/train_examples_0_${ASPECT}_single.json
TASK_PROMPTS=../task_prompts.json

FOLD_FOR_NAMING=${FOLD_ID}
TEST_DATA=${DATA_DIR}/t0_summaries_converted.json


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