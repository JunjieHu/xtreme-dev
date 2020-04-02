#!/bin/bash
# Script to evaluate the predictions of a trained model on XQuAD, TyDi QA, and MLQA.
REPO=$PWD
DIR=${REPO}/download
XQUAD_DIR=${DIR}/xquad
MLQA_DIR=${DIR}/mlqa
TYDIQA_DIR=${DIR}/tydiqa

EVAL_SQUAD=${DIR}/squad/evaluate-v1.1.py
EVAL_MLQA=${MLQA_DIR}/mlqa_evaluation_v1.py

PREDICTIONS_DIR=${REPO}/predictions
XQUAD_PRED_DIR=${PREDICTIONS_DIR}/xquad
MLQA_PRED_DIR=${PREDICTIONS_DIR}/mlqa
TYDIQA_PRED_DIR=${PREDICTIONS_DIR}/tydiqa

for pred_path in ${PREDICTIONS_DIR} ${XQUAD_PRED_DIR} ${MLQA_PRED_DIR} ${TYDIQA_PRED_DIR}; do
  if [ ! -d ${pred_path} ]
then
  echo "Predictions path ${pred_path} does not exist."
  exit
fi
done

echo
echo "XQuAD"
for lang in en es de el ru tr ar vi th zh hi; do
  echo -n "  $lang "
  TEST_FILE=${XQUAD_DIR}/xquad.$lang.json
  PRED_FILE=${XQUAD_PRED_DIR}/predictions_${lang}_.json
  python "${EVAL_SQUAD}" "${TEST_FILE}" "${PRED_FILE}"
done

echo
echo "MLQA"
for lang in en es de ar hi vi zh; do
 echo -n "  $lang "
 TEST_FILE=${MLQA_DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json
 PRED_FILE=${MLQA_PRED_DIR}/predictions_${lang}_.json
 python "${EVAL_MLQA}" "${TEST_FILE}" "${PRED_FILE}" ${lang}
done

echo "TyDi QA Gold Passage"
for lang in en ar bn fi id ko ru sw te; do
  echo -n "  $lang "
  TEST_FILE=${TYDIQA_DIR}/tydiqa-goldp-v1.1-dev/tydiqa.$lang.dev.json
  PRED_FILE=${TYDIQA_PRED_DIR}/predictions_${lang}_.json
  python "${EVAL_SQUAD}" "${TEST_FILE}" "${PRED_FILE}"
done
