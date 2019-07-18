# Clean Adversarial Examples

Classification and adversarial stuff are based on [Neural Networks for Text Classification](https://github.com/chainer/chainer/tree/master/examples/text_classification) and [Interpretable Adversarial Perturbation](https://github.com/dcetin/interpretable-adv), respectively. Please refer to `README_base.md` and `README_iAdv.md` for a quick glance.

## Directory

### Current files
- `do.sh` contains useful commands.
- `train.py` trains and evaluates the model.
- `nets.py` implements the RNN model.
- `utils.py` implements data loader.
- `data/imdb/preprocess.py` data preprocessing.

### Other stuff
- `run.py` runs models on online data.
- `text_datasets.py` legacy data loader.
- `nlp_utils.py` legacy data preprocessing.
- `demo.py` is where experimental code is dumped.
- `README.base` belongs to simplistic classifier.

## Prepare data

```bash
cd data/imdb/
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -O imdb.tar.gz
tar -xf imdb.tar.gz
python preprocess.py prepare_imdb
```