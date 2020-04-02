#!/bin/bash
REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-0}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

TASK='udpos'
export CUDA_VISIBLE_DEVICES=$GPU
LANGS='af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh'
NUM_EPOCHS=10
MAX_LENGTH=128
LR=2e-5

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ]; then
  MODEL_TYPE="xlmr"
fi

DATA_DIR=$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAX_LENGTH}/
OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCH}-MaxLen${MAXL}/"  
mkdir -p $OUTPUT_DIR
python3 $REPO/third_party/run_tag.py \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_gpu_train_batch_size 8 \
  --save_steps 500 \
  --seed 1 \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_predict \
  --do_predict_dev \
  --evaluate_during_training \
  --predict_langs $LANGS \
  --gradient_accumulation_steps 4 \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --save_only_best_checkpoint $LC

