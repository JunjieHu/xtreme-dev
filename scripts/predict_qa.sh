#!/bin/bash
# Script to obtain predictions using a trained model on XQuAD, TyDi QA, and MLQA.
REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
MODEL_TYPE=${2:-bert}
MODEL_PATH=${3}
TGT=${4:-xquad}
GPU=${5:-0}
DATA_DIR=${6:-"$REPO/download/"}

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path ${MODEL_PATH} does not exist."
  exit
fi

DIR=${DATA_DIR}/${TGT}/
PREDICTIONS_DIR=${REPO}/predictions
PRED_DIR=${PREDICTIONS_DIR}/$TGT/
mkdir -p "${PRED_DIR}"

if [ $TGT == 'xquad' ]; then
  langs=( en es de el ru tr ar vi th zh hi )
elif [ $TGT == 'mlqa' ]; then
  langs=( en es de ar hi vi zh )
elif [ $TGT == 'tydiqa' ]; then
  langs=( en ar bn fi id ko ru sw te )
fi

echo "************************"
echo ${MODEL}
echo "************************"

echo
echo "Predictions on $TGT"
for lang in ${langs[@]}; do
  echo "  $lang "
  if [ $TGT == 'xquad' ]; then
    TEST_FILE=${DIR}/xquad.$lang.json
  elif [ $TGT == 'mlqa' ]; then
    TEST_FILE=${DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json
  elif [ $TGT == 'tydiqa' ]; then
    TEST_FILE=${DIR}/tydiqa-goldp-v1.1-dev/tydiqa.$lang.dev.json
  fi

  CUDA_VISIBLE_DEVICES=${CUDA} python third_party/run_squad.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --do_eval \
    --do_lower_case \
    --eval_lang ${lang} \
    --predict_file "${TEST_FILE}" \
    --output_dir "${PRED_DIR}" &> /dev/null
done

