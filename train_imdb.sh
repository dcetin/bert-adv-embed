#!/usr/bin/env bash

export BERT_BASE_DIR=./uncased_L-12_H-768_A-12
export GLUE_DIR=./glue_data

# tensorboard --logdir /data/outputs &

bsub -n 6 -W 4:00 -R "rusage[mem=1024, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" \
python run_classifier.py \
  --task_name=IMDB \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/IMDB \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_BASE_DIR/arrays_bert_model.ckpt.npz \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=1 \
  --output_dir=./out_imdb \
  $@

  # --max_seq_length=128 \
  # --train_batch_size=32 \
  # --learning_rate=2e-5 \
  # --num_train_epochs=3 \