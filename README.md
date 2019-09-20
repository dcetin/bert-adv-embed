# Adversarial embeddings for BERT

Built on top of the [Chainer reimplementation](https://github.com/soskek/bert-chainer) of the Google Research's [original TensorFlow implementation](https://github.com/google-research/bert). IMDB loader and processor functions taken from [this branch](https://github.com/hsm207/bert/tree/imdb).

## Requirements

- Python (3.6.4)
- Chainer (6.0.0)
- CuPy (6.1.0)

## Installation

Install packages if they are not already present.
```bash
pip install cupy-cuda90 --no-cache-dir --user
pip install chainer --user
```

Clone and enter the repository.
```bash
# cd /cluster/scratch/nethzid
git clone https://github.com/dcetin/bert-chainer.git
cd bert-chainer
# module load python_cpu/3.6.4 cuda/9.0.176
```

Download and load the pretrained TensorFlow BERT checkpoints.

```bash
wget 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
unzip uncased_L-12_H-768_A-12.zip
export BERT_BASE_DIR=./uncased_L-12_H-768_A-12
python convert_tf_checkpoint_to_chainer.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --npz_dump_path $BERT_BASE_DIR/arrays_bert_model.ckpt.npz
rm uncased_L-12_H-768_A-12.zip
```

Download and extract the IMDB dataset.
```bash
wget 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
tar -xzf aclImdb_v1.tar.gz
python create_imdb_dataset.py
rm aclImdb_v1.tar.gz
```

Download the model checkpoint, if hasn't done before.
```bash
wget 'https://n.ethz.ch/~dcetin/download/model_snapshot_iter_2343_max_seq_length_128.npz' -P base_models
```

## Usage

Example command (can be found in train_imdb.sh as well) to run the experiments block of the code. Change last four options accordingly for the desired usage.
```bash
# module load python_gpu/3.6.4 cuda/9.0.176
# bsub -n 4 -W 4:00 -R "rusage[mem=1024, ngpus_excl_p=1]" \
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
```
