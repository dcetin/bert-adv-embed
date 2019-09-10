# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import logging
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

from distutils.util import strtobool

import chainer
from chainer import functions as F
from chainer import training
from chainer.training import extensions
import numpy as np

_logger = logging.getLogger(__name__)

import random
import pickle
import utils
import visualize
from chainer.backends.cuda import to_cpu
# import pdb; pdb.set_trace()

def get_arguments():
    parser = argparse.ArgumentParser(description='Arxiv')

    # Required parameters
    parser.add_argument(
        '--init_checkpoint', '--load_model_file', required=True,
        help="Initial checkpoint (usually from a pre-trained BERT model)."
        " The model array file path.")
    parser.add_argument(
        '--data_dir', required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument(
        '--bert_config_file', required=True,
        help="The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
    parser.add_argument(
        '--task_name', required=True,
        help="The name of the task to train.")
    parser.add_argument(
        '--vocab_file', required=True,
        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument(
        '--output_dir', required=True,
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        '--gpu', '-g', type=int, default=0,
        help="The id of gpu device to be used [0-]. If -1 is given, cpu is used.")

    # Other parameters
    parser.add_argument(
        '--do_lower_case', type=strtobool, default='True',
        help="Whether to lower case the input text. Should be True for uncased models and False for cased models.")
    parser.add_argument(
        '--max_seq_length', type=int, default=128,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument(
        '--do_train', type=strtobool, default='False',
        help="Whether to run training.")
    parser.add_argument(
        '--do_eval', type=strtobool, default='False',
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        '--train_batch_size', type=int, default=32,
        help="Total batch size for training.")
    parser.add_argument(
        '--eval_batch_size', type=int, default=8,
        help="Total batch size for eval.")
    parser.add_argument(
        '--learning_rate', type=float, default=5e-5,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        '--num_train_epochs', type=float, default=3.0,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        '--warmup_proportion', type=float, default=0.1,
        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument(
        '--save_checkpoints_steps', type=int, default=1000,
        help="How often to save the model checkpoint.")
    parser.add_argument(
        '--iterations_per_loop', type=int, default=1000,
        help="How many steps to make in each estimator call.")

    # drk
    parser.add_argument(
        '--do_resume', type=strtobool, default='False',
        help="Whether to resume training from a checkpoint.")
    parser.add_argument(
        '--do_experiment', type=strtobool, default='False',
        help="Whether to experiment before exiting.")
    parser.add_argument(
        '--adv_epsilon', type=float, default=0.6,
        help="Adversarial perturbation coefficient.")
    parser.add_argument(
        '--random_seed', dest='random_seed', type=int, default=1234, 
        help='Random seed.')

    # These args are NOT used in this port.
    parser.add_argument('--use_tpu', type=strtobool, default='False')
    parser.add_argument('--tpu_name')
    parser.add_argument('--tpu_zone')
    parser.add_argument('--gcp_project')
    parser.add_argument('--master')
    parser.add_argument('--num_tpu_cores', type=int, default=8)

    args = parser.parse_args()
    return args

FLAGS = get_arguments()

# Seed the generators
xp = chainer.backends.cuda.cupy if FLAGS.gpu >= 0 else np
np.random.seed(FLAGS.random_seed)
xp.random.seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)
os.environ["CHAINER_SEED"] = str(FLAGS.random_seed)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = np.array(input_ids, 'i')
        self.input_mask = np.array(input_mask, 'i')
        self.segment_ids = np.array(segment_ids, 'i')
        self.label_id = np.array([label_id], 'i')  # shape changed


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type,
                              tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ImdbProcessor(DataProcessor):
  """Processor for the IMDB data set (custom)."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv"), quotechar='"'), "train")

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the test set."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv"), quotechar='"'), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and test sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class Converter(object):
    """Converts examples to features, and then batches and to_gpu."""

    def __init__(self, label_list, max_seq_length, tokenizer):
        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.label_map = {}
        for (i, label) in enumerate(label_list):
            self.label_map[label] = i

    def __call__(self, examples, gpu):
        return self.convert_examples_to_features(examples, gpu)

    def convert_examples_to_features(self, examples, gpu):
        """Loads a data file into a list of `InputBatch`s.

        Args:
          examples: A list of examples (`InputExample`s).
          gpu: int. The gpu device id to be used. If -1, cpu is used.

        """
        max_seq_length = self.max_seq_length
        tokenizer = self.tokenizer
        label_map = self.label_map

        features = []
        for (exidx, example) in enumerate(examples):
            # momoize
            if getattr(example, 'tokens_a', None):
                tokens_a = tokenizer.tokenize(example.text_a)
            else:
                tokens_a = tokenizer.tokenize(example.text_a)
                example.tokens_a = tokens_a

            tokens_b = None
            if example.text_b:
                # memoize
                if getattr(example, 'tokens_b', None):
                    tokens_b = tokenizer.tokenize(example.text_b)
                else:
                    tokens_b = tokenizer.tokenize(example.text_b)
                    example.tokens_b = tokens_b

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            label_id = label_map[example.label]
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id))

        return self.make_batch(features, gpu)

    def make_batch(self, features, gpu):
        """Creates a concatenated batch from a list of data and to_gpu."""

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_id)

        def stack_and_to_gpu(data_list):
            sdata = F.pad_sequence(
                data_list, length=None, padding=0).array
            return chainer.dataset.to_device(gpu, sdata)

        batch_input_ids = stack_and_to_gpu(all_input_ids).astype('i')
        batch_input_mask = stack_and_to_gpu(all_input_mask).astype('f')
        batch_input_segment_ids = stack_and_to_gpu(all_segment_ids).astype('i')
        batch_input_label_ids = stack_and_to_gpu(
            all_label_ids).astype('i')[:, 0]  # shape should be (batch_size, )
        return (batch_input_ids, batch_input_mask,
                batch_input_segment_ids, batch_input_label_ids)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# New functions

def adv_FGSM_k(model, data, epsilon=0.6, k=1, train=False, sparsity_keep=None, xp=np):
    """
    k-step FGSM on word embeddings.

    Args:
        model: Neural network model.
        data: Output tuple of the converter.
        epsilon: Norm coefficient of the perturbation.
        k: Number of successive perturbation steps.
        train: If False, disables dropout, for evaluation purposes.
        sparsity_keep: Ratio of highest L2 norm perturbations to be kept.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        Perturbed embedding variable.
    """

    # Cannot seem to backprop on eval mode, so this seems like a possible workaround
    # -------------------------------------------------------------------------------
    org_dropout = (model.output_dropout, 
        model.bert.dropout_prob, 
        model.bert.encoder.hidden_dropout_prob)
    attn_dropout = []
    for layer_idx in range(model.bert.encoder.num_hidden_layers):
            layer_name = "layer_%d" % layer_idx
            layer = getattr(model.bert.encoder, layer_name)
            attn_dropout.append(layer.attention.attention_probs_dropout_prob)
            if train is False:
                layer.attention.attention_probs_dropout_prob = 0
    if train is False:
        model.output_dropout = 0.0
        model.bert.dropout_prob = 0.0
        model.bert.encoder.hidden_dropout_prob = 0.0
    # -------------------------------------------------------------------------------

    input_ids, input_mask, segment_ids, label_id = data
    org_embed_lookup = model.bert.get_word_embeddings(input_ids, input_mask, segment_ids) # var: (64, 128, 768)
    word_embed_lookup = org_embed_lookup

    for i in range(k):
        pooled_out = model.bert(input_ids, input_mask, segment_ids, 
            feed_word_embeddings=True, input_word_embeddings=word_embed_lookup)
        pred_logits = model.get_logits_from_output(pooled_out)
        loss_eval = F.softmax_cross_entropy(pred_logits, label_id, normalize=True)
        model.cleargrads()
        adv_g = chainer.grad([loss_eval], [model.bert.word_embed_lookup])[0] # var: (64, 128, 768)

        with chainer.using_config('train', False):
            def sentence_level_norm(grads):
                batchsize, embed_dim, maxlen = grads.shape
                grads = F.reshape(grads, (batchsize, embed_dim * maxlen))
                grads = F.normalize(grads, axis=1)
                grads = F.reshape(grads, (batchsize, embed_dim, maxlen))
                return grads

            adv_p = epsilon * sentence_level_norm(adv_g) # sentence-level L_2
            word_embed_lookup = word_embed_lookup + adv_p

    if sparsity_keep is not None:
        with chainer.using_config('train', False):
            def sparsify_perturb(data, mask, keep_prob=0.75, xp=np):
                seqlens = xp.sum(mask, axis=1)
                keep_counts = xp.floor(seqlens * keep_prob)
                norms = xp.linalg.norm(data, axis=2)
                thresh = data.shape[1] - keep_counts[:, xp.newaxis]
                sort_ids = xp.argsort(xp.argsort(norms, axis=-1))
                sparse_mask = (sort_ids > thresh).astype(int)
                data[sparse_mask == 0] = 0
                return data

            sparse_perturbation = sparsify_perturb(
                word_embed_lookup.data - org_embed_lookup.data, input_mask, keep_prob=sparsity_keep, xp=xp)
            word_embed_lookup.data = org_embed_lookup.data + sparse_perturbation

    # Restore dropout before returning
    # -------------------------------------------------------------------------------
    (model.output_dropout, 
        model.bert.dropout_prob, 
        model.bert.encoder.hidden_dropout_prob) = org_dropout
    for layer_idx in range(model.bert.encoder.num_hidden_layers):
            layer_name = "layer_%d" % layer_idx
            layer = getattr(model.bert.encoder, layer_name)
            layer.attention.attention_probs_dropout_prob = attn_dropout[layer_idx]
    # -------------------------------------------------------------------------------

    return word_embed_lookup

def evaluate_fn(eval_iter, device, model, converter, adversarial=False, sparsity_keep=None, adv_k=1, epsilon=0.6):
    """
    

    Args:
        eval_iter: 
        model: Neural network model.
        epsilon: Norm coefficient of the perturbation.
        k: Number of successive perturbation steps.
        train: If False, disables dropout, for evaluation purposes.
        sparsity_keep: Ratio of highest L2 norm perturbations to be kept.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        
    """    
    xp = model.bert.xp
    eval_losses = []
    eval_accuracies = []
    for test_batch in eval_iter:

        data = converter(test_batch, device)
        input_ids, input_mask, segment_ids, label_id = data

        if adversarial:
            adv_word_embed_lookup = adv_FGSM_k(model, data, 
                epsilon=epsilon, k=adv_k, train=False, sparsity_keep=sparsity_keep, xp=xp)

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():

                if adversarial:
                    # Adversarial evaluation: manual feed the embeds to bert
                    pooled_out = model.bert(input_ids, input_mask, segment_ids, 
                        feed_word_embeddings=True, input_word_embeddings=adv_word_embed_lookup)
                    pred_logits = model.get_logits_from_output(pooled_out)
                else:
                    # Standard evaluation: the whole pipeline
                    pred_logits = model(input_ids, input_mask, segment_ids, label_id, return_logits=True)

                # Calculate the loss
                loss_eval = F.softmax_cross_entropy(pred_logits, label_id, normalize=True)
                eval_losses.append(to_cpu(loss_eval.array))
                # Calculate the accuracy
                accuracy = F.accuracy(pred_logits, label_id)
                accuracy.to_cpu()
                eval_accuracies.append(accuracy.array)
    eval_iter.reset()

    return np.mean(eval_losses), np.mean(eval_accuracies)

def get_adv_example(model, emb, adv, inp):
    # Get embedding matrix
    embed_mat = model.bert.word_embeddings.W.data
    norm_embed = utils.mat_normalize(embed_mat, xp=xp)

    # Calculate L2 norms and cosine nearest neighbors...
    # ... for the original embeddings
    emb_cos_nn = utils.get_seq_nn(model, emb, norm_embed=norm_embed, xp=xp)
    emb_l2_norm = xp.linalg.norm(emb, axis=1).tolist()
    # ... for the adversarial embeddings
    adv_cos_nn = utils.get_seq_nn(model, adv, norm_embed=norm_embed, xp=xp)
    adv_l2_norm = xp.linalg.norm(adv, axis=1).tolist()
    # ... for the adversarial perturbations (also the L2-nns)
    per = adv-emb
    per_cos_nn = []
    per_l2_nn = []
    for wi in range(len(per)):
        per_w = per[wi]
        org_w = emb[wi]
        rel_norm_embed = utils.mat_normalize(embed_mat - org_w, xp=xp)
        nn_dir_w = utils.get_vec_nn(model, per_w, k=1, norm_embed=rel_norm_embed, xp=xp)
        per_cos_nn.append(nn_dir_w)
        per_l2_nn.append(utils.to_sent(utils.nn_vec_L2(embed_mat - org_w, per_w, k=1, return_vals=False, xp=xp), model.unvocab))
    per_cos_nn = ' '.join(per_cos_nn)
    per_l2_nn = ' '.join(per_l2_nn)
    per_l2_norm = xp.linalg.norm(per, axis=1).tolist()

    # Cosine similarity of original and perturbed embeddings
    cos_emb_adv = utils.cosine_seq(emb, adv, xp=xp)

    # Print L2 nearest neighbors of the original embeddings
    '''
    print('L2 nearest neighbors of the original input:')
    inp_words = emb_cos_nn.split(' ')
    for wi in range(len(per)):
        org_w = emb[wi]
        print(inp_words[wi] + ': ' + utils.to_sent(utils.nn_vec_L2(embed_mat, org_w, k=10, return_vals=False, xp=xp), model.unvocab))
    print('\n')
    '''

    # Print the nearest neighbors and norms

    # print('Original (cosine) nearest neighbors')
    # print(emb_cos_nn)
    # print('Adversarial (cosine) nearest neighbors')
    # print(adv_cos_nn)
    # print('Perturbation (L2) nearest neighbors')
    # print(per_l2_nn)
    # print('Perturbation (cosine) nearest neighbors')
    # print(per_cos_nn)
    # print('Cosine similarity of original and perturbed sequences')
    # print(cos_emb_adv)

    # print(emb_l2_norm)
    # print(adv_l2_norm)
    # print(per_l2_norm)
    # print('\n')

    # Experiment with "sentiment-carrying" dimensions 
    '''
    vec_good = embed_mat[tokenizer.vocab['good']]
    vec_bad = embed_mat[tokenizer.vocab['bad']]
    norm_good = utils.vec_normalize(vec_good, xp=xp)
    norm_bad = utils.vec_normalize(vec_bad, xp=xp)
    vec_NegativeDir = vec_bad - vec_good
    norm_NegativeDir = utils.vec_normalize(vec_NegativeDir, xp=xp)
    per_normed = utils.mat_normalize(per, xp=xp)
    adv_normed = utils.mat_normalize(adv, xp=xp)
    emb_normed = utils.mat_normalize(emb, xp=xp)
    per_NegativeDir = xp.matmul(per_normed, norm_NegativeDir).tolist()
    adv_good = [round(x, 4) for x in xp.matmul(adv_normed, norm_good).tolist()]
    adv_bad = [round(x, 4) for x in xp.matmul(adv_normed, norm_bad).tolist()]
    emb_good = [round(x, 4) for x in xp.matmul(emb_normed, norm_good).tolist()]
    emb_bad = [round(x, 4) for x in xp.matmul(emb_normed, norm_bad).tolist()]
    print(list(zip(per_cos_nn.split(' '), per_NegativeDir)))
    print(list(zip(adv_cos_nn.split(' '), adv_good)))
    print(list(zip(adv_cos_nn.split(' '), adv_bad)))
    print(list(zip(emb_cos_nn.split(' '), emb_good)))
    print(list(zip(emb_cos_nn.split(' '), emb_bad)))
    '''

    #  Prepare the data
    emb_cos_nn, adv_cos_nn, per_cos_nn = emb_cos_nn.split(' '), adv_cos_nn.split(' '), per_cos_nn.split(' ')
    emb_cos_nn, adv_cos_nn, per_cos_nn = np.asarray(emb_cos_nn), np.asarray(adv_cos_nn), np.asarray(per_cos_nn)
    emb_l2_norm, adv_l2_norm, per_l2_norm = np.asarray(emb_l2_norm), np.asarray(adv_l2_norm), np.asarray(per_l2_norm)
    cos_emb_adv = np.asarray(cos_emb_adv.tolist())

    # Pack the data
    data = {}
    data['emb_cos_nn'] = emb_cos_nn
    data['adv_cos_nn'] = adv_cos_nn
    data['per_cos_nn'] = per_cos_nn
    data['emb_l2_norm'] = emb_l2_norm
    data['adv_l2_norm'] = adv_l2_norm
    data['per_l2_norm'] = per_l2_norm
    data['cos_emb_adv'] = cos_emb_adv
    return data

def proj_adv_FGSM_k(model, data, epsilon=0.6, k=1, train=False, sparsity_keep=None, 
    token_budget=0.15, redundancy=True, max_step=50, verbose=False, xp=np):
    '''
    k-step FGSM on word embeddings, returns perturbed embedding variable.
    '''
    # Cannot seem to backprop on eval mode, so this seems like a possible workaround
    # -------------------------------------------------------------------------------
    org_dropout = (model.output_dropout, 
        model.bert.dropout_prob, 
        model.bert.encoder.hidden_dropout_prob)
    attn_dropout = []
    for layer_idx in range(model.bert.encoder.num_hidden_layers):
            layer_name = "layer_%d" % layer_idx
            layer = getattr(model.bert.encoder, layer_name)
            attn_dropout.append(layer.attention.attention_probs_dropout_prob)
            if train is False:
                layer.attention.attention_probs_dropout_prob = 0
    if train is False:
        model.output_dropout = 0.0
        model.bert.dropout_prob = 0.0
        model.bert.encoder.hidden_dropout_prob = 0.0
    # -------------------------------------------------------------------------------

    # Save the embedding matrix
    embed_mat = model.bert.word_embeddings.W.data
    norm_embed = utils.mat_normalize(embed_mat, xp=xp)

    # Save the original embeddings
    input_ids, input_mask, segment_ids, label_id = data
    org_embed_lookup = model.bert.get_word_embeddings(input_ids, input_mask, segment_ids)
    word_embed_lookup = org_embed_lookup

    # Prediction on original data
    pred_logits = model(input_ids, input_mask, segment_ids, label_id, return_logits=True)
    prediction_test = xp.argmax(pred_logits.data, axis=1)
    std = (prediction_test == label_id)

    # Set parameters for iteration over Projected FGSM steps
    step = 0
    token_budget = int(token_budget * input_ids.size)
    retval = None

    while True:
        for i in range(k):
            pooled_out = model.bert(input_ids, input_mask, segment_ids, 
                feed_word_embeddings=True, input_word_embeddings=word_embed_lookup)
            pred_logits = model.get_logits_from_output(pooled_out)
            loss_eval = F.softmax_cross_entropy(pred_logits, label_id, normalize=True)
            model.cleargrads()
            adv_g = chainer.grad([loss_eval], [model.bert.word_embed_lookup])[0] # var: (64, 128, 768)

            with chainer.using_config('train', False):
                def sentence_level_norm(grads):
                    batchsize, embed_dim, maxlen = grads.shape
                    grads = F.reshape(grads, (batchsize, embed_dim * maxlen))
                    grads = F.normalize(grads, axis=1)
                    grads = F.reshape(grads, (batchsize, embed_dim, maxlen))
                    return grads

                adv_p = epsilon * sentence_level_norm(adv_g) # sentence-level L_2
                word_embed_lookup = word_embed_lookup + adv_p

        if sparsity_keep is not None:
            with chainer.using_config('train', False):
                def sparsify_perturb(data, mask, keep_prob=0.75, xp=np):
                    seqlens = xp.sum(mask, axis=1)
                    keep_counts = xp.floor(seqlens * keep_prob)
                    norms = xp.linalg.norm(data, axis=2)
                    thresh = data.shape[1] - keep_counts[:, xp.newaxis]
                    sort_ids = xp.argsort(xp.argsort(norms, axis=-1))
                    sparse_mask = (sort_ids > thresh).astype(int)
                    data[sparse_mask == 0] = 0
                    return data

                sparse_perturbation = sparsify_perturb(
                    word_embed_lookup.data - org_embed_lookup.data, input_mask, keep_prob=sparsity_keep, xp=xp)
                word_embed_lookup.data = org_embed_lookup.data + sparse_perturbation

        # Project the adversarial embedding into its nearest neighboring sequence and test it
        step += 1
        with chainer.using_config('train', False):

            if verbose:
                print('Step {}'.format(step))
            adv_test_x = word_embed_lookup

            # Calculate cosine nearest neighbors
            adv_cos_nn_ids = utils.get_seq_nn(model, adv_test_x.data[0], norm_embed=norm_embed, get_ids=True, xp=xp)
            adv_cos_nn_ids = adv_cos_nn_ids[xp.newaxis,:]

            # Note the differences between original and adversarial-to-be input
            diff_ids = [(e, (w1,w2)) for (e, (w1,w2)) in enumerate(zip(input_ids[0].tolist(), adv_cos_nn_ids[0].tolist())) if w1 != w2]
            diff_tokens = [(model.unvocab[w1],model.unvocab[w2]) for (e, (w1,w2)) in diff_ids]

            # See if Projected FGSM changes any ordinary tokens to special tokens or vice versa
            illegal = [(e, (w1,w2)) for (e, (w1,w2)) in diff_ids if (w1 < 999 or w2 < 999)]
            if len(illegal)  > 0:
                # Revert illegal changes
                for (e, x) in illegal:
                    adv_cos_nn_ids[0,e] = input_ids[0,e]
                # Reiterate over differences
                diff_ids = [(e, (w1,w2)) for (e, (w1,w2)) in enumerate(zip(input_ids[0].tolist(), adv_cos_nn_ids[0].tolist())) if w1 != w2]
                diff_tokens = [(model.unvocab[w1],model.unvocab[w2]) for (e, (w1,w2)) in diff_ids]

            if step > max_step or len(diff_tokens) > token_budget:
                break

            # Test only if at least one input token has been changed
            if len(diff_tokens) > 0:
                if verbose:
                    print('Changed ' + ', '.join([ x + ' to ' +  y for (x,y) in diff_tokens]))

                # Prediction on adversarial data
                pred_logits = model(adv_cos_nn_ids, input_mask, segment_ids, label_id, return_logits=True)
                prediction_adv_test = xp.argmax(pred_logits.data, axis=1)

                # Compare predictions and get the indices of the adversarial examples
                adv = (prediction_adv_test == label_id)
                res = ~adv & std
                res_ids = xp.where(res.astype(int) == 1)[0].tolist()
                if verbose:
                    print('Adversarial: {}, label is {}, pred. changed {} -> {}'.format(
                    res[0], label_id[0], prediction_test[0], prediction_adv_test[0]))

                # Exit if an adversarial example is found or the maximum step limit is reached
                if res[0]:

                    # NAIVE REDUNDANCY CHECK
                    if redundancy:
                        while True:
                            new_diff_ids = []
                            for (e, (w1,w2)) in diff_ids:
                                temp_ids = xp.copy(adv_cos_nn_ids)
                                temp_ids[0,e] = input_ids[0,e]
                                pred_logits = model(temp_ids, input_mask, segment_ids, label_id, return_logits=True)
                                prediction_adv_test = xp.argmax(pred_logits.data, axis=1)
                                adv = (prediction_adv_test == label_id)
                                res = ~adv & std
                                if res[0]:
                                    if verbose:
                                        print((e, (w1,w2)), 'is redundant')
                                    adv_cos_nn_ids[0,e] = input_ids[0,e]
                                else:
                                    new_diff_ids.append((e, (w1,w2)))
                            if new_diff_ids == diff_ids:
                                break
                            else:
                                diff_ids = new_diff_ids

                    retval = adv_cos_nn_ids
                    break
            else:
                if verbose:
                    print('Changed nothing.')

            if verbose:
                print(' ')

    # Restore dropout before returning
    # -------------------------------------------------------------------------------
    (model.output_dropout, 
        model.bert.dropout_prob, 
        model.bert.encoder.hidden_dropout_prob) = org_dropout
    for layer_idx in range(model.bert.encoder.num_hidden_layers):
            layer_name = "layer_%d" % layer_idx
            layer = getattr(model.bert.encoder, layer_name)
            layer.attention.attention_probs_dropout_prob = attn_dropout[layer_idx]
    # -------------------------------------------------------------------------------

    if verbose:
        if retval is None:
            if step > max_step:
                print('Step limit ({}) exceeded.\n'.format(max_step))
            elif len(diff_tokens) > token_budget:
                print('Token budget ({}) exceeded.\n'.format(token_budget))
        else:
            print('Adversarial text found with {} changes (<{}).\n'.format(len(diff_ids), token_budget))
    return retval

def adv_demo_by_index(model, eval_examples, sequence_offset, epsilon, adv_k=1, prefix='', 
    sparsity_keep=None, proj=True, create_table=False, early_return=True, return_data=False, xp=np):

    retval = True

    with chainer.using_config('train', False):
        data = model.converter([eval_examples[sequence_offset]], FLAGS.gpu)
        input_ids, input_mask, segment_ids, label_id = data

        # Prediction on original data
        pred_logits = model(input_ids, input_mask, segment_ids, label_id, return_logits=True)
        prediction_test = xp.argmax(pred_logits.data, axis=1)
        std = (prediction_test == label_id)

        # Check if the original prediction is correct or wrong
        if not std[0]:
            if early_return:
                return None
            else:
                retval = None

    if proj:
        adv_test_x_ids = proj_adv_FGSM_k(model, data, k=adv_k, epsilon=epsilon, train=False, sparsity_keep=sparsity_keep, verbose=True, xp=xp)

        # Check if an adversarial example is found
        if adv_test_x_ids is None:
            if early_return:
                return False
            else:
                retval = False
    else:
        adv_test_x_embed = adv_FGSM_k(model, data, k=adv_k, epsilon=epsilon, train=False, sparsity_keep=sparsity_keep, xp=xp)

    with chainer.using_config('train', False):

        # Prediction on adversarial data
        if proj:
            # ...using adversarial text
            pred_logits = model(adv_test_x_ids, input_mask, segment_ids, label_id, return_logits=True)
            prediction_adv_test = xp.argmax(pred_logits.data, axis=1)
        else:
            # ...using adversarial embeddings
            pooled_out = model.bert(input_ids, input_mask, segment_ids, 
                feed_word_embeddings=True, input_word_embeddings=adv_test_x_embed)
            pred_logits = model.get_logits_from_output(pooled_out)
            prediction_adv_test = xp.argmax(pred_logits.data, axis=1)

        # Compare predictions and get the indices of the adversarial examples
        adv = (prediction_adv_test == label_id)
        res = ~adv & std
        res_ids = xp.where(res.astype(int) == 1)[0].tolist()

        if not res[0]:
            if early_return:
                return False
            else:
                retval = False

        if create_table:
            # Save the embedding arrays
            embed_x = model.bert.get_word_embeddings(input_ids, input_mask, segment_ids)
            embed_x = embed_x.data
            
            # Get adversarial embeddings if using adversarial text as input
            if proj:
                adv_test_x = model.bert.get_word_embeddings(adv_test_x_ids, input_mask, segment_ids)
            else:
                adv_test_x = adv_test_x_embed
            adv_test_x = adv_test_x.data

            # Print indices and accuracies over the current batch
            print('groundtruth: {}'.format(label_id))
            print('predictions: {}'.format(prediction_test))
            print('adv.  preds: {}'.format(prediction_adv_test))

            # Only one example
            seq_idx = 0

            # Print sequence metadata
            seqlen = int(sum(input_mask[seq_idx].tolist()))
            print('seq_offset: {} of length {}'.format(sequence_offset, seqlen))

            # Get the matrices
            emb = embed_x[seq_idx]
            adv = adv_test_x[seq_idx]
            inp = input_ids[seq_idx]

            # Crop the sequences to avoid printing the padding
            emb = emb[:seqlen]
            adv = adv[:seqlen]
            inp = input_ids[:seqlen]

            data = get_adv_example(model, emb, adv, inp)
            metadata = {}
            metadata['epsilon'] = str(epsilon)
            metadata['adv_k'] = str(adv_k)
            metadata['label'] = str(prediction_test[seq_idx]) + 'to' + str(prediction_adv_test[seq_idx])
            metadata['name'] = FLAGS.output_dir + prefix + str(sequence_offset)

            # pik_file = 'example_adv_data.pickle'
            # with open(os.path.join('./', pik_file), 'wb') as handle:
            #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

            visualize.create_adv_table(data, metadata, folder=FLAGS.output_dir, save_norms=True)

        if return_data:
            return retval, data
        else:
            return retval

def summary_statistics(model, eval_examples, verbose=False, xp=np):

    k = 6
    m = 5
    n = 10000

    cosnn_cosines_list = []
    random_cosines_list = []
    random_eucs_list = []
    eucnn_eucs_list = []
    eucnn_cosines_list = []
    cosnn_eucs_list = []
    per_cosine_list = []
    per_euc_list = []

    dataset_len = len(eval_examples)
    embed_mat = model.bert.word_embeddings.W.data
    norm_embed = utils.mat_normalize(embed_mat, xp=xp)
    eval_examples_array = np.array(eval_examples)
    if verbose:
        print('dataset_len: {}'.format(dataset_len))

    counter = 0

    for i in range(n):

        # Pick a random sequence
        seq_idx = np.random.randint(dataset_len, size=1)[0]
        data = model.converter([eval_examples[seq_idx]], FLAGS.gpu)
        input_ids, input_mask, segment_ids, label_id = data
        if verbose:
            print('seq_idx: {}'.format(seq_idx))

        # Pick a random token from that sequence
        seqlen = int(sum(input_mask[0].tolist()))
        tok_offset = np.random.randint(seqlen, size=1)[0]
        tok_idx = input_ids[0, tok_offset].tolist()
        tok_word = model.unvocab[tok_idx]
        if verbose:
            print('tok_offset: {}'.format(tok_offset))
            print('tok_idx: {}'.format(tok_idx))
            print('tok_word: {}'.format(tok_word))

        # Get k Euclidean nearest neighbors of the word
        eucnn_tok_ids, eucnn_eucs = utils.nn_vec_L2(embed_mat, embed_mat[tok_idx], k=k, return_vals=True, xp=xp)
        eucnn_words = [model.unvocab[w] for w in eucnn_tok_ids.tolist()]
        if verbose:
            print('eucnn_words: {}'.format(eucnn_words))
            print('eucnn_eucs: {}'.format(eucnn_eucs))

        # Get k cosine nearest neighbors of the word
        cosnn_words, cosnn_cosines = utils.get_vec_nn(model, tok_idx, k=k, return_vals=True, norm_embed=None, xp=xp)
        cosnn_tok_ids = [model.vocab[x] for x in cosnn_words.split(' ')]
        if verbose:
            print('cosnn_words: {}'.format(cosnn_words))
            print('cosnn_cosines: {}'.format(cosnn_cosines))

        # Calculate the Euclidean distances w.r.t. cosine nearest neighbors
        cosnn_eucs = xp.linalg.norm(embed_mat[cosnn_tok_ids] - embed_mat[tok_idx], axis=1)
        if verbose:
            print('cosnn_eucs: {}'.format(cosnn_eucs))

        # Calculate the cosine similarities w.r.t. Euclidean nearest neighbors
        eucnn_cosines = xp.matmul(norm_embed[eucnn_tok_ids], norm_embed[tok_idx])
        if verbose:
            print('eucnn_cosines: {}'.format(eucnn_cosines))

        # Calculate the cosine similarity and Euclidean distance of th
        adv = adv_FGSM_k(model, data, k=1, epsilon=0.6, train=False, sparsity_keep=None, xp=xp).data[0]
        per_cosine = utils.cosine_vec(adv[tok_offset], embed_mat[tok_idx], xp=xp)
        per_euc = xp.linalg.norm(adv[tok_offset] - embed_mat[tok_idx])
        if verbose:
            print('per_cosine: {}'.format(per_cosine))
            print('per_euc: {}'.format(per_euc))

        # Pick m random sequences
        seq_ids = np.random.randint(dataset_len, size=m)
        pls = eval_examples_array[seq_ids]
        data_rand = model.converter(pls, FLAGS.gpu)
        input_ids, input_mask, segment_ids, label_id = data_rand
        if verbose:
            print('seq_ids: {}'.format(seq_ids))

        # Pick a random token from each of those sequences
        seqlens = xp.sum(input_mask, axis=1).astype(int).tolist()
        rand_tok_offsets = [np.random.randint(seqlen, size=1)[0] for seqlen in seqlens]
        rand_tok_ids = [input_ids[i, tok_offset].tolist() for (i,tok_offset) in enumerate(rand_tok_offsets)]
        rand_tok_words = [model.unvocab[rti] for rti in rand_tok_ids]
        rand_tok_ids = xp.array(rand_tok_ids)
        if verbose:
            print('rand_tok_offsets: {}'.format(rand_tok_offsets))
            print('rand_tok_ids: {}'.format(rand_tok_ids))
            print('rand_tok_words: {}'.format(rand_tok_words))

        # Calculate cosine similarity w.r.t. each sampled token
        random_cosines = xp.matmul(norm_embed[rand_tok_ids], norm_embed[tok_idx])
        if verbose:
            print('random_cosines: {}'.format(random_cosines))

        # Calculate Euclidean distance w.r.t. each sampled token
        random_eucs = xp.linalg.norm(embed_mat[rand_tok_ids] - embed_mat[tok_idx], axis=1)
        if verbose:
            print('random_eucs: {}'.format(random_eucs))

        # Append new entries to respective lists
        random_cosines_list.append(xp.array(random_cosines))
        random_eucs_list.append(xp.array(random_eucs))
        cosnn_cosines_list.append(xp.array(cosnn_cosines))
        eucnn_eucs_list.append(xp.array(eucnn_eucs))
        eucnn_cosines_list.append(xp.array(eucnn_cosines))
        cosnn_eucs_list.append(xp.array(cosnn_eucs))
        per_cosine_list.append(xp.array(per_cosine))
        per_euc_list.append(xp.array(per_euc))

        if verbose:
            print(' ')

        counter += 1

    random_cosines = xp.asnumpy(xp.stack(random_cosines_list))
    random_eucs = xp.asnumpy(xp.stack(random_eucs_list))
    cosnn_cosines = xp.asnumpy(xp.stack(cosnn_cosines_list))
    eucnn_eucs = xp.asnumpy(xp.stack(eucnn_eucs_list))
    eucnn_cosines = xp.asnumpy(xp.stack(eucnn_cosines_list))
    cosnn_eucs = xp.asnumpy(xp.stack(cosnn_eucs_list))
    per_cosine = xp.asnumpy(xp.stack(per_cosine_list))
    per_euc = xp.asnumpy(xp.stack(per_euc_list))

    pik_file = 'summary_data_10000_5_5.pickle'
    with open(os.path.join('./', pik_file), 'wb') as handle:
        pickle.dump(random_cosines, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(random_eucs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(cosnn_cosines, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(eucnn_eucs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(eucnn_cosines, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(cosnn_eucs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(per_cosine, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(per_euc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    visualize.summary_histogram('cosine', cosnn_cosines, random_cosines, eucnn_cosines, per_cosine, density=True, folder=FLAGS.output_dir)
    visualize.summary_histogram('euclidean', cosnn_eucs, random_eucs, eucnn_eucs, per_euc, density=True, folder=FLAGS.output_dir)



def main():
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "imdb": ImdbProcessor,
    }

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_experiment:
        raise ValueError("At least one of `do_train` or `do_eval` "
                         "or `do_experiment` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    # TODO: use special Adam from "optimization.py"
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    bert = modeling.BertModel(config=bert_config)
    model = modeling.BertClassifier(bert, num_labels=len(label_list))
    chainer.serializers.load_npz(FLAGS.init_checkpoint, model, ignore_names=['output/W', 'output/b'])
    converter = Converter(label_list, FLAGS.max_seq_length, tokenizer)

    # Assign processing variables to model
    # -------------------------------------------------------------------------------
    unvocab = {}
    for key, value in tokenizer.vocab.items():
        unvocab[value] = key
    model.vocab = tokenizer.vocab
    model.unvocab = unvocab
    model.converter = converter
    # -------------------------------------------------------------------------------

    if FLAGS.do_resume:
        chainer.serializers.load_npz('./base_models/model_snapshot_iter_2343_max_seq_length_128.npz', model)
        # chainer.serializers.load_npz('./base_models/model_snapshot_iter_2343_max_seq_length_256.npz', model)
        # chainer.serializers.load_npz('./base_models/model_snapshot_iter_4687_max_seq_length_512.npz', model)

    if FLAGS.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(FLAGS.gpu).use()
        model.to_gpu()

    if FLAGS.do_train:
        # Adam with weight decay only for 2D matrices
        optimizer = optimization.WeightDecayForMatrixAdam(
            alpha=1.,  # ignore alpha. instead, use eta as actual lr
            eps=1e-6, weight_decay_rate=0.01)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(1.))

        train_iter = chainer.iterators.SerialIterator(
            train_examples, FLAGS.train_batch_size)
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer,
            converter=converter,
            device=FLAGS.gpu)
        trainer = training.Trainer(
            updater, (num_train_steps, 'iteration'), out=FLAGS.output_dir)

        # learning rate (eta) scheduling in Adam
        lr_decay_init = FLAGS.learning_rate * \
            (num_train_steps - num_warmup_steps) / num_train_steps
        trainer.extend(extensions.LinearShift(  # decay
            'eta', (lr_decay_init, 0.), (num_warmup_steps, num_train_steps)))
        trainer.extend(extensions.WarmupShift(  # warmup
            'eta', 0., num_warmup_steps, FLAGS.learning_rate))
        trainer.extend(extensions.observe_value(
            'eta', lambda trainer: trainer.updater.get_optimizer('main').eta),
            trigger=(50, 'iteration'))  # logging

        trainer.extend(extensions.snapshot_object(
            model, 'model_snapshot_iter_{.updater.iteration}.npz'),
            trigger=(num_train_steps, 'iteration'))
        trainer.extend(extensions.LogReport(
            trigger=(50, 'iteration')))
        trainer.extend(extensions.PrintReport(
            ['iteration', 'main/loss',
             'main/accuracy', 'elapsed_time']))
        # trainer.extend(extensions.ProgressBar(update_interval=10))        

        trainer.run()

    if FLAGS.do_eval:
        eval_examples = processor.get_test_examples(FLAGS.data_dir)
        test_iter = chainer.iterators.SerialIterator(eval_examples, FLAGS.train_batch_size * 2, repeat=False, shuffle=False)

        # test_adv_loss, test_adv_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter, 
        #     adv_k=1, epsilon=FLAGS.adv_epsilon, adversarial=True)
        # print('[test  adv] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))

        # test_loss, test_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter)
        # print('[test ] loss:{:.04f} acc:{:.04f}'.format(test_loss, test_acc))

        # test_adv_loss, test_adv_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter, 
        #     adv_k=1, epsilon=0.6, adversarial=True)
        # print('[test 1x0.6] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))

        # test_adv_loss, test_adv_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter, 
        #     adv_k=1, epsilon=0.6, adversarial=True, sparsity_keep=0.25)
        # print('[test 1x0.6[0.25]] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))

        # test_adv_loss, test_adv_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter, 
        #     adv_k=3, epsilon=0.2, adversarial=True)
        # print('[test 3x0.2] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))

        test_adv_loss, test_adv_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter, 
            adv_k=3, epsilon=0.2, adversarial=True, sparsity_keep=0.25)
        print('[test 3x0.2[0.25]] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))

    if FLAGS.do_experiment:

        eval_examples = processor.get_test_examples(FLAGS.data_dir)
        test_iter = chainer.iterators.SerialIterator(eval_examples, FLAGS.train_batch_size, repeat=False, shuffle=False)
        xp = model.bert.xp

        # print('Some example nearest neighbors:')
        # utils.example_nn(model, return_vals=True, xp=xp)
        # print('Some example analogies:')
        # print("Given 'king + woman - man', 'sad + good - bad', 'walked + went - walk', 'paris + rome - france', expecting queen, happy, go, italy")
        # utils.analogy(model, pos_words=None, neg_words=None, return_vals=True, xp=xp)

        def adv_demo_iterate(model, eval_iter, epsilon, adv_k=1, max_examples=1, max_seq_len=50, prefix='', sparsity_keep=None, xp=np):
            
            num_examples = 0
            for test_batch in eval_iter:

                # Get the current batch of data and perturb the embeddings
                data = model.converter(test_batch, FLAGS.gpu)
                input_ids, input_mask, segment_ids, label_id = data
                adv_test_x = adv_FGSM_k(model, data, k=adv_k, epsilon=epsilon, train=False, sparsity_keep=sparsity_keep, xp=xp)

                with chainer.using_config('train', False):

                    # Prediction on original data
                    pred_logits = model(input_ids, input_mask, segment_ids, label_id, return_logits=True)
                    prediction_test = xp.argmax(pred_logits.data, axis=1)

                    # Prediction on adversarial data
                    pooled_out = model.bert(input_ids, input_mask, segment_ids, 
                        feed_word_embeddings=True, input_word_embeddings=adv_test_x)
                    pred_logits = model.get_logits_from_output(pooled_out)
                    prediction_adv_test = xp.argmax(pred_logits.data, axis=1)

                    # Compare predictions and get the indices of the adversarial examples
                    std = (prediction_test == label_id)
                    adv = (prediction_adv_test == label_id)
                    res = ~adv & std
                    res_ids = xp.where(res.astype(int) == 1)[0].tolist()

                    # Save the embedding arrays
                    embed_x = model.bert.get_word_embeddings(input_ids, input_mask, segment_ids)
                    embed_x = embed_x.data
                    adv_test_x = adv_test_x.data

                    # Print indices and accuracies over the current batch
                    '''
                    print('groundtruth: {}'.format(label_id))
                    print('predictions: {}'.format(prediction_test))
                    print('adv.  preds: {}'.format(prediction_adv_test))
                    print('std: {}'.format(std.astype(int)))
                    print('adv: {}'.format(adv.astype(int)))
                    print('res: {}'.format(res.astype(int)))
                    print('std acc: {}'.format(std.mean()))
                    print('adv acc: {}'.format(adv.mean()))
                    '''

                    if len(res_ids) > 0:

                        # seqlen_list = [int(sum(x.tolist())) for x in input_mask]
                        # shortest = min(input_mask, key=lambda x:int(sum(x.tolist())))
                        # print('shortest', seqlen_list, shortest)
                        # for seq_idx in [shortest]:

                        # for seq_idx in range(res.shape[0]): # Iterate over ALL sequences in the batch, not the adversarial ones
                        # for seq_idx in [0]: # Get just one example
                        for seq_idx in res_ids: # Iterate over all adversarial sequences

                            # Iterate until finding a sufficiently short sequence
                            seqlen = int(sum(input_mask[seq_idx].tolist()))
                            if max_seq_len is not None and seqlen > max_seq_len:
                                continue

                            # Print sequence metadata
                            sequence_offset = eval_iter.current_position + seq_idx - (FLAGS.train_batch_size * 2)
                            num_examples += 1
                            print('Visualizing example {}:'.format(num_examples))
                            print('seq_offset: {} of length {}'.format(sequence_offset, seqlen))
                            print('groundtruth:', label_id[seq_idx], 'original:', prediction_test[seq_idx], 'adversarial:', prediction_adv_test[seq_idx])

                            # Get the matrices
                            emb = embed_x[seq_idx]
                            adv = adv_test_x[seq_idx]
                            inp = input_ids[seq_idx]

                            # Crop the sequences to avoid printing the padding
                            emb = emb[:seqlen]
                            adv = adv[:seqlen]
                            inp = input_ids[:seqlen]

                            data = get_adv_example(model, emb, adv, inp)
                            metadata = {}
                            metadata['epsilon'] = str(epsilon)
                            metadata['adv_k'] = str(adv_k)
                            metadata['label'] = str(prediction_test[seq_idx]) + 'to' + str(prediction_adv_test[seq_idx])
                            metadata['name'] = FLAGS.output_dir + prefix + str(sequence_offset)

                            visualize.create_adv_table(data, metadata, folder=FLAGS.output_dir, save_norms=True)

                            # Exit function if enough examples are created
                            if num_examples >= max_examples:
                                eval_iter.reset()
                                return
            
            # Reset the iterator before returning
            eval_iter.reset()
            return

        def ugly_adv_iterator(model, eval_examples, xp=np):
            emb_counter = 0
            proj_counter = 0
            all_counter = 0
            true_pred_counter = 0

            for ex_index in range(20000,20030):

                all_counter += 1
                emb = adv_demo_by_index(model, eval_examples, ex_index, 
                    epsilon=0.2, adv_k=3, prefix='_025sparse', create_table=True, sparsity_keep=0.25, proj=False, xp=xp)
                proj = False

                if emb is not None:
                    for epsilon in [3.0,5.0,7.0,10.0]:
                        proj = proj or adv_demo_by_index(model, eval_examples, ex_index, 
                            epsilon=7.0, adv_k=1, prefix='_025sparse', create_table=True, sparsity_keep=0.25, xp=xp)

                    true_pred_counter += 1
                    if emb:
                        emb_counter += 1
                    if proj:
                        proj_counter += 1

            print('emb_counter: {}'.format(emb_counter))
            print('proj_counter: {}'.format(proj_counter))
            print('all_counter: {}'.format(all_counter))
            print('true_pred_counter: {}'.format(true_pred_counter))

        def PCA_stats():
            # Set the iterators
            from sklearn.decomposition import PCA
            adv_k = 1
            epsilon = 0.6
            sparsity_keep = None
            batch_size = FLAGS.train_batch_size * 2

            train_examples = processor.get_train_examples(FLAGS.data_dir)
            test_examples = processor.get_test_examples(FLAGS.data_dir)

            train_iter = chainer.iterators.SerialIterator(train_examples, batch_size, repeat=False, shuffle=False)
            test_iter = chainer.iterators.SerialIterator(test_examples, batch_size, repeat=False, shuffle=False)

            # Collect embeddings and outputs on training data
            train_outputs = []
            train_adv_outputs = []
            # train_embeds = []
            # train_adv_embeds = []
            for train_batch in train_iter:
                data = model.converter(train_batch, FLAGS.gpu)
                input_ids, input_mask, segment_ids, label_id = data
                adv_train_x = adv_FGSM_k(model, data, k=adv_k, epsilon=epsilon, train=False, sparsity_keep=sparsity_keep, xp=xp)
                with chainer.using_config('train', False):
                    pooled_out = model.bert.get_pooled_output(input_ids, input_mask, segment_ids).data
                    train_outputs.append(xp.asnumpy(pooled_out))
                    # embed_lookup = model.bert.word_embed_lookup.data
                    # embed_lookup = embed_lookup.reshape(-1, embed_lookup.shape[-1])
                    # train_embeds.append(xp.asnumpy(embed_lookup))
                    # train_adv_embeds.append(xp.asnumpy(adv_train_x.data.reshape(-1, embed_lookup.shape[-1])))
                    pooled_adv_out = model.bert(input_ids, input_mask, segment_ids, feed_word_embeddings=True, input_word_embeddings=adv_train_x).data
                    train_adv_outputs.append(xp.asnumpy(pooled_adv_out))
            train_outputs = np.concatenate(train_outputs, axis=0)
            print(train_outputs.shape)
            train_adv_outputs = np.concatenate(train_adv_outputs, axis=0)
            print(train_adv_outputs.shape)
            # train_embeds = np.concatenate(train_embeds, axis=0)
            # print(train_embeds.shape)
            # train_adv_embeds = np.concatenate(train_adv_embeds, axis=0)
            # print(train_adv_embeds.shape)

            # Collect embeddings and outputs on test data
            test_outputs = []
            test_adv_outputs = []
            # test_embeds = []
            # test_adv_embeds = []
            for test_batch in test_iter:
                data = model.converter(test_batch, FLAGS.gpu)
                input_ids, input_mask, segment_ids, label_id = data
                adv_test_x = adv_FGSM_k(model, data, k=adv_k, epsilon=epsilon, train=False, sparsity_keep=sparsity_keep, xp=xp)
                with chainer.using_config('train', False):
                    pooled_out = model.bert.get_pooled_output(input_ids, input_mask, segment_ids).data
                    test_outputs.append(xp.asnumpy(pooled_out))
                    # embed_lookup = model.bert.word_embed_lookup.data
                    # embed_lookup = embed_lookup.reshape(-1, embed_lookup.shape[-1])
                    # test_embeds.append(xp.asnumpy(embed_lookup))
                    # test_adv_embeds.append(xp.asnumpy(adv_test_x.data.reshape(-1, embed_lookup.shape[-1])))
                    pooled_adv_out = model.bert(input_ids, input_mask, segment_ids, feed_word_embeddings=True, input_word_embeddings=adv_test_x).data
                    test_adv_outputs.append(xp.asnumpy(pooled_adv_out))
            test_outputs = np.concatenate(test_outputs, axis=0)
            print(test_outputs.shape)
            test_adv_outputs = np.concatenate(test_adv_outputs, axis=0)
            print(test_adv_outputs.shape)
            # test_embeds = np.concatenate(test_embeds, axis=0)
            # print(test_embeds.shape)
            # test_adv_embeds = np.concatenate(test_adv_embeds, axis=0)
            # print(test_adv_embeds.shape)

            # Save the intermediary data
            pik_file = 'pca_intermediary.pickle'
            with open(os.path.join('./', pik_file), 'wb') as handle:
                pickle.dump(train_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(train_adv_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(test_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(test_adv_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(train_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(train_adv_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(test_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # pickle.dump(test_adv_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)

            pik_file = 'pca_intermediary.pickle'
            with open(os.path.join('./', pik_file), 'rb') as handle:
                train_outputs = pickle.load(handle)
                train_adv_outputs = pickle.load(handle)
                test_outputs = pickle.load(handle)
                test_adv_outputs = pickle.load(handle)
                # train_embeds = pickle.load(handle)
                # train_adv_embeds = pickle.load(handle)
                # test_embeds = pickle.load(handle)
                # test_adv_embeds = pickle.load(handle)

            # Do PCA on train outputs
            pca_outputs = PCA()
            pca_outputs.fit(train_outputs)
            print(pca_outputs.explained_variance_ratio_)
            pca_outputs_params = pca_outputs.get_params()

            train_outputs_transformed = pca_outputs.transform(train_outputs)
            print(train_outputs_transformed.shape)
            train_adv_outputs_transformed = pca_outputs.transform(train_adv_outputs)
            print(train_adv_outputs_transformed.shape)
            test_outputs_transformed = pca_outputs.transform(test_outputs)
            print(test_outputs_transformed.shape)
            test_adv_outputs_transformed = pca_outputs.transform(test_adv_outputs)
            print(test_adv_outputs_transformed.shape)

            pik_file = 'pca_outputs_train.pickle'
            with open(os.path.join('./', pik_file), 'wb') as handle:
                pickle.dump(pca_outputs_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(train_outputs_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(train_adv_outputs_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(test_outputs_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(test_adv_outputs_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Do PCA on train embeddings
            # pca_embeds = PCA()
            # pca_embeds.fit(train_embeds)
            # print(pca_embeds.explained_variance_ratio_)
            # pca_embeds_params = pca_embeds.get_params()

            # train_embeds_transformed = pca_embeds.transform(train_embeds)
            # print(train_embeds_transformed.shape)
            # train_adv_embeds_transformed = pca_embeds.transform(train_adv_embeds)
            # print(train_adv_embeds_transformed.shape)
            # test_embeds_transformed = pca_embeds.transform(test_embeds)
            # print(test_embeds_transformed.shape)
            # test_adv_embeds_transformed = pca_embeds.transform(test_adv_embeds)
            # print(test_adv_embeds_transformed.shape)

            # pik_file = 'pca_embeds_train.pickle'
            # with open(os.path.join('./', pik_file), 'wb') as handle:
            #     pickle.dump(pca_embeds_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(train_embeds_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(train_adv_embeds_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(test_embeds_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #     pickle.dump(test_adv_embeds_transformed, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Read and plot the output data
            pik_file = 'pca_outputs_train.pickle'
            with open(os.path.join('./', pik_file), 'rb') as handle:
                pca_outputs_params = pickle.load(handle)
                train_outputs_transformed = pickle.load(handle)
                train_adv_outputs_transformed = pickle.load(handle)
                test_outputs_transformed = pickle.load(handle)
                test_adv_outputs_transformed = pickle.load(handle)

            train_outputs_scores = np.mean(np.abs(train_outputs_transformed), axis=0)
            train_adv_outputs_scores = np.mean(np.abs(train_adv_outputs_transformed), axis=0)
            test_outputs_scores = np.mean(np.abs(test_outputs_transformed), axis=0)
            test_adv_outputs_scores = np.mean(np.abs(test_adv_outputs_transformed), axis=0)

            print(train_outputs_scores)
            print(train_adv_outputs_scores)
            print(test_outputs_scores)
            print(test_adv_outputs_scores)

            visualize.PCA_score_plot(train_outputs_scores, train_adv_outputs_scores, 
                test_outputs_scores, test_adv_outputs_scores, 'pooled_out', folder=FLAGS.output_dir)

            # Read and plot the embedding data
            # pik_file = 'pca_embeds_train.pickle'
            # with open(os.path.join('./', pik_file), 'rb') as handle:
            #     pca_embeds_params = pickle.load(handle)
            #     train_embeds_transformed = pickle.load(handle)
            #     train_adv_embeds_transformed = pickle.load(handle)
            #     test_embeds_transformed = pickle.load(handle)
            #     test_adv_embeds_transformed = pickle.load(handle)

            # train_embeds_scores = np.mean(np.abs(train_embeds_transformed), axis=0)
            # train_adv_embeds_scores = np.mean(np.abs(train_adv_embeds_transformed), axis=0)
            # test_embeds_scores = np.mean(np.abs(test_embeds_transformed), axis=0)
            # test_adv_embeds_scores = np.mean(np.abs(test_adv_embeds_transformed), axis=0)

            # print(train_embeds_scores)
            # print(train_adv_embeds_scores)
            # print(test_embeds_scores)
            # print(test_adv_embeds_scores)

            # visualize.PCA_score_plot(train_embeds_scores, train_adv_embeds_scores, 
            #     test_embeds_scores, test_adv_embeds_scores, 'embed_lookup', folder=FLAGS.output_dir)

        # adv_demo_by_index(model, eval_examples, 58, epsilon=0.6, adv_k=1, prefix='_normal', sparsity_keep=None, proj=False, create_table=True, xp=xp)
        # summary_statistics(model, eval_examples, verbose=False, xp=xp)


if __name__ == "__main__":
    main()
