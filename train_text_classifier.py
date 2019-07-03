#!/usr/bin/env python
import argparse
import datetime
import json
import os

import chainer
from chainer import training

import nets
from nlp_utils import convert_seq
import text_datasets

from chainer.training import extensions
from chainer import serializers
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import chainer.functions as F
import numpy as np
import sys

def main():
    current_datetime = '{}'.format(datetime.datetime.today())
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--dataset', '-data', default='imdb.binary',
                        choices=['dbpedia', 'imdb.binary', 'imdb.fine',
                                 'TREC', 'stsa.binary', 'stsa.fine',
                                 'custrev', 'mpqa', 'rt-polarity', 'subj'],
                        help='Name of dataset.')
    parser.add_argument('--model', '-model', default='cnn',
                        choices=['cnn', 'rnn', 'bow'],
                        help='Name of encoder model type.')
    parser.add_argument('--char-based', action='store_true')
    parser.add_argument('--testing', dest='testing', action='store_true')
    parser.set_defaults(testing=False)
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    device = chainer.get_device(args.device)
    device.use()

    # Load a dataset
    if args.dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            fine_grained=args.dataset.endswith('.fine'),
            char_based=args.char_based)
    # elif args.dataset == 'dbpedia':
    #     train, valid, vocab = text_datasets.get_dbpedia(
    #         char_based=args.char_based)
    # elif args.dataset in ['TREC', 'stsa.binary', 'stsa.fine',
    #                       'custrev', 'mpqa', 'rt-polarity', 'subj']:
    #     train, valid, vocab = text_datasets.get_other_text_dataset(
    #         args.dataset, char_based=args.char_based)

    # Split train into train and valid
    pos_train = train[:12500]
    neg_train = train[12500:]
    valid = pos_train[11250:] + neg_train[11250:]
    train = pos_train[:11250] + neg_train[:11250]

    if args.testing:
        train = train[:100]
        test = test[:100]

    print('Device: {}'.format(device))
    print('# train data: {}'.format(len(train)))
    print('# valid data: {}'.format(len(valid)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))

    # Setup a model
    if args.model == 'rnn':
        Encoder = nets.RNNEncoder
    elif args.model == 'cnn':
        Encoder = nets.CNNEncoder
    elif args.model == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=args.layer, n_vocab=len(vocab),
                      n_units=args.unit, dropout=args.dropout)
    # model = nets.TextClassifier(encoder, n_class)
    model = nets.BasicNetwork(encoder, n_class)
    model.to_device(device)  # Copy the model to the device

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    best_valid_acc = 0
    train_accuracies = []

    if args.resume is not None:
        checkpoint_epoch = args.resume
        chainer.serializers.load_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(checkpoint_epoch)), model)
        chainer.serializers.load_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(checkpoint_epoch)), optimizer)
    else:
        checkpoint_epoch = 0

    # Training loop
    while train_iter.epoch < args.epoch - int(checkpoint_epoch):

        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()
        train_data = convert_seq(train_batch, device)
        train_x = train_data['xs']
        train_y = train_data['ys']
        train_y = F.concat(train_y, axis=0)

        # Calculate the prediction of the network
        prediction_train = model(train_x)

        # Calculate the loss with softmax_cross_entropy
        loss = F.softmax_cross_entropy(prediction_train, train_y)

        accuracy = F.accuracy(prediction_train, train_y)
        accuracy.to_cpu()
        train_accuracies.append(accuracy.array)

        # Calculate the gradients in the network
        model.cleargrads()
        loss.backward()

        # Update all the trainable parameters
        optimizer.update()
        # --------------------- until here ---------------------

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

            # Display training loss and accuracy
            print('epoch:{:02d} train_loss:{:.04f} '.format(
                train_iter.epoch + int(checkpoint_epoch), float(to_cpu(loss.array))), end='')
            print('train_accuracy:{:.04f} '.format(
                np.mean(train_accuracies)), end='')
            sys.stdout.flush()
            train_accuracies = []

            valid_losses = []
            valid_accuracies = []
            for valid_batch in valid_iter:
                valid_data = convert_seq(valid_batch, device)
                valid_x = valid_data['xs']
                valid_y = valid_data['ys']
                valid_y = F.concat(valid_y, axis=0)

                # Forward the valid data
                prediction_valid = model(valid_x)

                # Calculate the loss
                loss_valid = F.softmax_cross_entropy(prediction_valid, valid_y)
                valid_losses.append(to_cpu(loss_valid.array))

                # Calculate the accuracy
                accuracy = F.accuracy(prediction_valid, valid_y)
                accuracy.to_cpu()
                valid_accuracies.append(accuracy.array)

            valid_iter.reset()

            print('valid_loss:{:.04f} valid_accuracy:{:.04f}'.format(
                np.mean(valid_losses), np.mean(valid_accuracies)))

            # Checkpointing
            cur_valid_acc = np.mean(valid_accuracies)
            if cur_valid_acc > best_valid_acc:
                best_valid_acc = cur_valid_acc
                chainer.serializers.save_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(train_iter.epoch + int(checkpoint_epoch))), model)
                chainer.serializers.save_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(train_iter.epoch + int(checkpoint_epoch))), optimizer)
                chainer.serializers.save_npz(os.path.join(args.out, 'best_model.npz'), model)
                chainer.serializers.save_npz(os.path.join(args.out, 'best_state.npz'), optimizer)

            # Predict on test
            test_losses = []
            test_accuracies = []
            for test_batch in test_iter:
                test_data = convert_seq(test_batch, device)
                test_x = test_data['xs']
                test_y = test_data['ys']
                test_y = F.concat(test_y, axis=0)

                # Forward the test data
                prediction_test = model(test_x)

                # Calculate the loss
                loss_test = F.softmax_cross_entropy(prediction_test, test_y)
                test_losses.append(to_cpu(loss_test.array))

                # Calculate the accuracy
                accuracy = F.accuracy(prediction_test, test_y)
                accuracy.to_cpu()
                test_accuracies.append(accuracy.array)

            test_iter.reset()

            print('test_loss:{:.04f} test_accuracy:{:.04f}'.format(
                np.mean(test_losses), np.mean(test_accuracies)))

    chainer.serializers.save_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(train_iter.epoch + int(checkpoint_epoch))), model)
    chainer.serializers.save_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(train_iter.epoch + int(checkpoint_epoch))), optimizer)

    # Save vocabulary and model's setting
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    vocab_path = os.path.join(args.out, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    model_path = os.path.join(args.out, 'best_model.npz')
    state_path = os.path.join(args.out, 'best_state.npz')
    model_setup = args.__dict__
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['state_path'] = state_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)

if __name__ == '__main__':
    main()
