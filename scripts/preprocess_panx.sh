#!/bin/bash
REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
DATA_DIR=${2:-"$REPO/download/"}

TASK='panx'
MAXL=128
LANGS="ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu"
LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xom-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

SAVE_DIR="$DATA_DIR/$TASK/${TASK}_processed_maxlen${MAXL}"
mkdir -p $SAVE_DIR
python3 $REPO/utils_preprocess.py \
  --data_dir $DATA_DIR/$TASK/ \
  --task panx_tokenize \
  --model_name_or_path $MODEL \
  --model_type $MODEL_TYPE \
  --max_len $MAXL \
  --output_dir $SAVE_DIR \
  --languages $LANGS $LC
if [ ! -f $SAVE_DIR/labels.txt ]; then
  cat $SAVE_DIR/*/*.${MODEL} | cut -f 2 | grep -v "^$" | sort | uniq > $SAVE_DIR/labels.txt
fi
