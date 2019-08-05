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

import utils
from chainer.backends.cuda import to_cpu
import sys
from visualize import create_plots

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
        '--adv_epsilon', type=float, default=0.5,
        help="Adversarial perturbation coefficient.")

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
        for (ex_index, example) in enumerate(examples):
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

def adv_FGSM(model, data, epsilon=0.5, train=False):

    # Cannot seem to backprop on eval mode, so this seems like a possible workaround
    org_dropout = (model.output_dropout, 
        model.bert.dropout_prob, 
        model.bert.encoder.hidden_dropout_prob, 
        model.bert.encoder.attention_probs_dropout_prob)
    if train is False:
        model.output_dropout = 0.0
        model.bert.dropout_prob = 0.0
        model.bert.encoder.hidden_dropout_prob = 0.0
        model.bert.encoder.attention_probs_dropout_prob = 0.0

    input_ids, input_mask, segment_ids, label_id = data
    word_embed_lookup = model.bert.get_word_embeddings(input_ids, input_mask, segment_ids) # var: (64, 128, 768)
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
        perturbed = word_embed_lookup + adv_p

    # Restore dropout before returning
    (model.output_dropout, 
        model.bert.dropout_prob, 
        model.bert.encoder.hidden_dropout_prob, 
        model.bert.encoder.attention_probs_dropout_prob) = org_dropout

    return perturbed

def adv_FGSM_k(model, data, epsilon=0.5, k=3, train=False):

    # Cannot seem to backprop on eval mode, so this seems like a possible workaround
    org_dropout = (model.output_dropout, 
        model.bert.dropout_prob, 
        model.bert.encoder.hidden_dropout_prob, 
        model.bert.encoder.attention_probs_dropout_prob)
    if train is False:
        model.output_dropout = 0.0
        model.bert.dropout_prob = 0.0
        model.bert.encoder.hidden_dropout_prob = 0.0
        model.bert.encoder.attention_probs_dropout_prob = 0.0

    input_ids, input_mask, segment_ids, label_id = data
    word_embed_lookup = model.bert.get_word_embeddings(input_ids, input_mask, segment_ids) # var: (64, 128, 768)


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


    # Restore dropout before returning
    (model.output_dropout, 
        model.bert.dropout_prob, 
        model.bert.encoder.hidden_dropout_prob, 
        model.bert.encoder.attention_probs_dropout_prob) = org_dropout

    return word_embed_lookup

def evaluate_fn(eval_iter, device, model, converter, adversarial=False, k=1, epsilon=0.5):
    eval_losses = []
    eval_accuracies = []
    for test_batch in eval_iter:

        data = converter(test_batch, device)
        input_ids, input_mask, segment_ids, label_id = data

        if adversarial:
            if k == 1:
                adv_word_embed_lookup = adv_FGSM(model, data, epsilon=epsilon, train=False)
            else:
                adv_word_embed_lookup = adv_FGSM_k(model, data, epsilon=epsilon, k=k, train=False)

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

def get_seq_nn(model, seq, unvocab, norm_embed=None, project=False, xp=np):
    if norm_embed is None:
        embed_mat = model.bert.word_embeddings.W.data
        norm_embed = utils.mat_normalize(embed_mat, xp=xp)
    seq_norm = utils.mat_normalize(seq, xp=xp)
    seq_nn = xp.matmul(norm_embed, seq_norm.T)
    seq_nn = xp.argmax(seq_nn, axis=0)

    if project:
        units = norm_embed[seq_nn]
        return xp.multiply(units, seq)
    else:
        return utils.to_sent(seq_nn, unvocab)

def get_vec_nn(model, inp, unvocab, vocab, k=10, return_vals=False, norm_embed=None, xp=np):
    embed_mat = model.bert.word_embeddings.W.data
    if norm_embed is None:
        norm_embed = utils.mat_normalize(embed_mat, xp=xp)

    if type(inp) == str:            # input is a word
        norm_forw = utils.vec_normalize(embed_mat[vocab[inp]], xp=xp)
    elif type(inp) == int:          # input is a vocabulary index
        norm_forw = utils.vec_normalize(embed_mat[inp], xp=xp)
    elif type(inp) == xp.ndarray:   # input is an embedding vector
        norm_forw = utils.vec_normalize(inp, xp=xp)
    else:
        logger.error('Unsupported input format for nearest neighbors.')
    
    max_idx = utils.nn_vec(norm_embed, norm_forw, k=k, normalize=False, xp=xp, return_vals=return_vals)
    if return_vals:
        words = utils.to_sent(max_idx[0], unvocab)
        return words, [round(x, 5) for x in max_idx[1].tolist()]
    else:
        words = utils.to_sent(max_idx, unvocab)
        return words

def analogy(model, unvocab, vocab, pos_words=None, neg_words=None, return_vals=False, xp=np):
    '''
    Analogy on word embeddings examples.
    '''
    embed_mat = model.bert.word_embeddings.W.data
    norm_embed = utils.mat_normalize(embed_mat, xp=xp)
    if pos_words is None or neg_words is None:
        # Expecting queen, happy, go, italy
        word_list = [ (['king', 'woman'],['man']), (['sad', 'good'],['bad']), 
        (['walked', 'went'],['walk']), (['paris', 'rome'],['france']) ]
    else:
        word_list = [ (pos_words, neg_words) ]
    for pos_words, neg_words in word_list:
        pos_embed = [embed_mat[vocab[word]] for word in pos_words]
        neg_embed = [embed_mat[vocab[word]] for word in neg_words]
        pos_embed = xp.sum(xp.stack(pos_embed, axis=0), axis=0)
        neg_embed = xp.sum(xp.stack(neg_embed, axis=0), axis=0)
        forw = pos_embed - neg_embed
        if return_vals:
            nns, vals = get_vec_nn(model, forw, unvocab, vocab, xp=xp, norm_embed=norm_embed, return_vals=True)
            print(' '.join([nn + ' (' + str(val) + ')' for (nn,val) in list(zip(nns.split(' '), vals))]))
        else:
            print(get_vec_nn(model, forw, unvocab, vocab, xp=xp, norm_embed=norm_embed))
    print('\n')

def example_nn(model, unvocab, vocab, return_vals=False, xp=np):
        '''
        Nearest negihbor examples.
        '''
        embed_mat = model.bert.word_embeddings.W.data
        norm_embed = utils.mat_normalize(embed_mat, xp=xp)
        word_list = ['good', 'this', 'that', 'awesome', 'bad', 'wrong']
        for word in word_list:
            if return_vals:
                nns, vals = get_vec_nn(model, word, unvocab, vocab, xp=xp, norm_embed=norm_embed, return_vals=True)
                exstr = ' '.join([nn + ' (' + str(val) + ')' for (nn,val) in list(zip(nns.split(' '), vals))])
                print(word + ': ' + exstr)
            else:
                print(word + ': ' + get_vec_nn(model, word, unvocab, vocab, xp=xp, norm_embed=norm_embed))
        print('\n')


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
    chainer.serializers.load_npz(
        FLAGS.init_checkpoint, model,
        ignore_names=['output/W', 'output/b'])

    converter = Converter(label_list, FLAGS.max_seq_length, tokenizer)

    if FLAGS.do_resume:
        # chainer.serializers.load_npz('./base_out_imdb/model_snapshot_iter_781.npz', model)
        chainer.serializers.load_npz('./base_out3_imdb/model_snapshot_iter_2343.npz', model)        

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

        test_loss, test_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter)
        print('[test ] loss:{:.04f} acc:{:.04f}'.format(test_loss, test_acc))
        test_adv_loss, test_adv_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter, epsilon=FLAGS.adv_epsilon, adversarial=True)
        print('[test  adv] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))
        # test_adv_loss, test_adv_acc = evaluate_fn(test_iter, FLAGS.gpu, model, converter, k=3, epsilon=(FLAGS.adv_epsilon/3), adversarial=True)
        # print('[test adv3] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))

    # if you wanna see some output arrays for debugging
    if FLAGS.do_experiment:

        eval_examples = processor.get_test_examples(FLAGS.data_dir)
        test_iter = chainer.iterators.SerialIterator(eval_examples, FLAGS.train_batch_size * 2, repeat=False, shuffle=False)
        xp = model.bert.xp
        unvocab = {}
        for key, value in tokenizer.vocab.items():
            unvocab[value] = key

        print('Some example nearest neighbors:')
        example_nn(model, unvocab, tokenizer.vocab, return_vals=True, xp=xp)
        print('Some example analogies:')
        print("Given 'king + woman - man', 'sad + good - bad', 'walked + went - walk', 'paris + rome - france', expecting queen, happy, go, italy")
        analogy(model, unvocab, tokenizer.vocab, pos_words=None, neg_words=None, return_vals=True, xp=xp)

        def projection_demo(model, eval_iter, converter, epsilon, adv_k=1, xp=np):

            # Number of examples to generate
            max_examples = 1
            # Maximum sequence length to choose sequences from
            max_seq_len = 90

            num_examples = 0
            for bi, test_batch in enumerate(eval_iter):

                data = converter(test_batch, FLAGS.gpu)
                input_ids, input_mask, segment_ids, label_id = data

                if adv_k == 1:
                    adv_test_x = adv_FGSM(model, data, epsilon=epsilon, train=False)
                else:
                    adv_test_x = adv_FGSM_k(model, data, k=adv_k, epsilon=epsilon, train=False)

                # print('bi {} batchsize {} pos {}'.format(bi, len(test_batch), eval_iter.current_position))

                # Sparsification here on the masked input?

                with chainer.using_config('train', False):

                    embed_x = model.bert.get_word_embeddings(input_ids, input_mask, segment_ids)

                    # Prediction on original data
                    pred_logits = model(input_ids, input_mask, segment_ids, label_id, return_logits=True)
                    prediction_test = xp.argmax(pred_logits.data, axis=1)

                    # Prediction on adversarial data
                    pooled_out = model.bert(input_ids, input_mask, segment_ids, 
                        feed_word_embeddings=True, input_word_embeddings=adv_test_x)
                    pred_logits = model.get_logits_from_output(pooled_out)
                    prediction_adv_test = xp.argmax(pred_logits.data, axis=1)

                    std = (prediction_test == label_id)
                    adv = (prediction_adv_test == label_id)
                    res = ~adv & std
                    res_ids = xp.where(res.astype(int) == 1)[0].tolist()

                    adv_test_x = adv_test_x.data
                    embed_x = embed_x.data

                    # print('groundtruth: {}'.format(label_id))
                    # print('predictions: {}'.format(prediction_test))
                    # print('adv.  preds: {}'.format(prediction_adv_test))
                    # print('std: {}'.format(std.astype(int)))
                    # print('adv: {}'.format(adv.astype(int)))
                    # print('res: {}'.format(res.astype(int)))
                    # print('std acc: {}'.format(std.mean()))
                    # print('adv acc: {}'.format(adv.mean()))

                    # Iterate over all adversarial sequences, or pick the shortest one
                    if len(res_ids) > 0:
                        # shortest = min(res_ids, key=lambda x:int(input_ids[x].size))
                        # for seq_idx in [shortest]:
                        for seq_idx in res_ids:

                            seqlen = int(sum(input_mask[seq_idx].tolist()))

                            # Iterate until finding a sufficiently short sequence
                            if max_seq_len is not None and seqlen > max_seq_len:
                                continue

                            sequence_offset = eval_iter.current_position + seq_idx - (FLAGS.train_batch_size * 2)

                            num_examples += 1
                            print('Visualizing example {}:'.format(num_examples))
                            print('seq_offset: {} of length {}'.format(sequence_offset, seqlen))

                            # Vectorised version
                            embed_mat = model.bert.word_embeddings.W.data
                            norm_embed = utils.mat_normalize(embed_mat, xp=xp)

                            emb = embed_x[seq_idx]
                            adv = adv_test_x[seq_idx]
                            inp = input_ids[seq_idx]

                            emb = emb[:seqlen]
                            adv = adv[:seqlen]
                            inp = input_ids[:seqlen]

                            emb_cos_nn = get_seq_nn(model, emb, unvocab, norm_embed=norm_embed, xp=xp)
                            emb_l2_norm = xp.linalg.norm(emb, axis=1).tolist()

                            adv_cos_nn = get_seq_nn(model, adv, unvocab, norm_embed=norm_embed, xp=xp)
                            adv_l2_norm = xp.linalg.norm(adv, axis=1).tolist()

                            per = adv-emb
                            per_cos_nn = []
                            # per_l2_nn = []
                            for wi in range(len(per)):
                                per_w = per[wi]
                                org_w = emb[wi]
                                rel_norm_embed = utils.mat_normalize(embed_mat - org_w, xp=xp)
                                nn_dir_w = get_vec_nn(model, per_w, unvocab, tokenizer.vocab, k=1, norm_embed=rel_norm_embed, xp=xp)
                                per_cos_nn.append(nn_dir_w)
                                # per_l2_nn.append(utils.to_sent(utils.nn_vec_L2(embed_mat - org_w, per_w, k=1, return_vals=False, xp=xp), unvocab))
                            per_cos_nn = ' '.join(per_cos_nn)
                            # per_l2_nn = ' '.join(per_l2_nn)
                            per_l2_norm = xp.linalg.norm(per, axis=1).tolist()

                            print('L2 nearest neighbors of the original input:')
                            inp_words = emb_cos_nn.split(' ')
                            for wi in range(len(per)):
                                org_w = emb[wi]
                                print(inp_words[wi] + ': ' + utils.to_sent(utils.nn_vec_L2(embed_mat, org_w, k=10, return_vals=False, xp=xp), unvocab))
                            print('\n')

                            print('Original (cosine) nearest neighbors')
                            print(emb_cos_nn)
                            print('Adversarial (cosine) nearest neighbors')
                            print(adv_cos_nn)
                            print('Perturbation (cosine) nearest neighbors')
                            print(per_cos_nn)
                            # print('Perturbation (L2) nearest neighbors')
                            # print(per_l2_nn)
                            # print(emb_l2_norm)
                            # print(adv_l2_norm)
                            # print(per_l2_norm)
                            print('\n')


                            # vec_good = embed_mat[tokenizer.vocab['good']]
                            # vec_bad = embed_mat[tokenizer.vocab['bad']]
                            # norm_good = utils.vec_normalize(vec_good, xp=xp)
                            # norm_bad = utils.vec_normalize(vec_bad, xp=xp)
                            # vec_NegativeDir = vec_bad - vec_good
                            # norm_NegativeDir = utils.vec_normalize(vec_NegativeDir, xp=xp)
                            # per_normed = utils.mat_normalize(per, xp=xp)
                            # adv_normed = utils.mat_normalize(adv, xp=xp)
                            # emb_normed = utils.mat_normalize(emb, xp=xp)
                            # per_NegativeDir = xp.matmul(per_normed, norm_NegativeDir).tolist()
                            # adv_good = [round(x, 4) for x in xp.matmul(adv_normed, norm_good).tolist()]
                            # adv_bad = [round(x, 4) for x in xp.matmul(adv_normed, norm_bad).tolist()]
                            # emb_good = [round(x, 4) for x in xp.matmul(emb_normed, norm_good).tolist()]
                            # emb_bad = [round(x, 4) for x in xp.matmul(emb_normed, norm_bad).tolist()]
                            # print(list(zip(per_cos_nn.split(' '), per_NegativeDir)))
                            # print(list(zip(adv_cos_nn.split(' '), adv_good)))
                            # print(list(zip(adv_cos_nn.split(' '), adv_bad)))
                            # print(list(zip(emb_cos_nn.split(' '), emb_good)))
                            # print(list(zip(emb_cos_nn.split(' '), emb_bad)))

                            for coeff in [1.0,5.0,12.0]:
                                per = adv-emb
                                print(coeff)
                                per = coeff * per
                                per_cos_nn = []
                                # per_l2_nn = []
                                for wi in range(len(per)):
                                    per_w = per[wi]
                                    org_w = emb[wi]
                                    rel_norm_embed = utils.mat_normalize(embed_mat - org_w, xp=xp)
                                    nn_dir_w = get_vec_nn(model, per_w, unvocab, tokenizer.vocab, k=1, norm_embed=rel_norm_embed, xp=xp)
                                    per_cos_nn.append(nn_dir_w)
                                    # per_l2_nn.append(utils.to_sent(utils.nn_vec_L2(embed_mat - org_w, per_w, k=1, return_vals=False, xp=xp), unvocab))
                                per_cos_nn = ' '.join(per_cos_nn)
                                # per_l2_nn = ' '.join(per_l2_nn)
                                per_l2_norm = xp.linalg.norm(per, axis=1).tolist()
                                print('Perturbation (cosine) nearest neighbors')
                                print(per_cos_nn)


                            
                            data = emb_cos_nn, adv_cos_nn, per_cos_nn, emb_l2_norm, adv_l2_norm, per_l2_norm
                            metadata = {}
                            metadata['name'] = FLAGS.output_dir + str(sequence_offset)
                            metadata['epsilon'] = str(epsilon)
                            metadata['adv_k'] = str(adv_k)
                            metadata['label'] = str(prediction_test[seq_idx]) + 'to' + str(prediction_adv_test[seq_idx])
                            create_plots(data, metadata, folder=FLAGS.output_dir)

                            if num_examples >= max_examples:
                                eval_iter.reset()
                                return
            
            eval_iter.reset()
        projection_demo(model, test_iter, converter, FLAGS.adv_epsilon, adv_k=1, xp=xp)
        projection_demo(model, test_iter, converter, FLAGS.adv_epsilon/3, adv_k=3, xp=xp)

        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                data = test_iter.__next__()
                input_ids, input_mask, segment_ids, label_id = converter(data, FLAGS.gpu)
                # pooled_out = model.bert.get_pooled_output(input_ids, input_mask, segment_ids).data
                pre_embedding_out = model.bert.get_word_embeddings(input_ids, input_mask, segment_ids).data
                embedding_out = model.bert.get_embedding_output(input_ids, input_mask, segment_ids).data

                embed_mat = model.bert.word_embeddings.W.data
                norm_embed = utils.mat_normalize(embed_mat, xp=xp)
                ex_sent = utils.to_sent(input_ids[0], unvocab)
                nns = utils.to_sent(utils.nn_vec(embed_mat, embed_mat[tokenizer.vocab['bad']], k=20, xp=xp), unvocab)
                # import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
