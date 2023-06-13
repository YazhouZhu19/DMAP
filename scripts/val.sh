#!/bin/bash
# test a model to segment abdominal/cardiac MRI
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='CHAOST2'
#DATASET='CMR'
NWORKER=16
RUNS=1
ALL_EV=(0 1 2 3 4) # 5-fold cross validation (0, 1, 2, 3, 4)
TEST_LABEL=[1,2,3,4]
###### Training configs ######
NSTEP=35000
DECAY=0.98

MAX_ITER=50 # defines the size of an epoch
SNAPSHOT_INTERVAL=1000 # interval for saving snapshot
SEED=2025

N_PART=3 # defines the number of chunks for evaluation
ALL_SUPP=(2) # CHAOST2: 0-4, CMR: 0-7
echo ========================================================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="test_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./results"

  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi
  for SUPP_IDX in "${ALL_SUPP[@]}"
  do
    # RELOAD_PATH='please feed the absolute path to the trained weights here' # path to the reloaded model
    RELOAD_MODEL_PATH="...${EVAL_FOLD}/3/snapshots/30000.pth"
    python3 test.py with \
    mode="test" \
    dataset=$DATASET \
    num_workers=$NWORKER \
    n_steps=$NSTEP \
    eval_fold=$EVAL_FOLD \
    max_iters_per_load=$MAX_ITER \
    supp_idx=$SUPP_IDX \
    test_label=$TEST_LABEL \
    seed=$SEED \
    n_part=$N_PART \
    reload_model_path=$RELOAD_MODEL_PATH \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR
    done
done

