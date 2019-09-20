#!/usr/bin/env bash

export BERT_BASE_DIR=./uncased_L-12_H-768_A-12

module load python_gpu/3.6.4 cuda/9.0.176

bsub -n 4 -W 4:00 -R "rusage[mem=1024, ngpus_excl_p=1]" \
python run_classifier.py \
  --task_name=IMDB \
  --data_dir=aclImdb \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_BASE_DIR/arrays_bert_model.ckpt.npz \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --output_dir=./out_imdb \
  --do_train=false \
  --do_eval=false \
  --do_resume=true \
  --do_experiment=true \