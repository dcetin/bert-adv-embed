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
from chainer.backends import cuda
from chainer.backends.cuda import to_cpu
import chainer.functions as F
import numpy as np
import sys
import pickle
from nets import sequence_embed
from os import listdir
import logging
import logging.config
# import pdb; pdb.set_trace()

def imdb_loader(tempdir, testing=False):
    '''
    An ugly function for loading the IMDB data.
    '''
    if (os.path.exists(os.path.join(tempdir, 'imdb_train.pickle'))
        and os.path.exists(os.path.join(tempdir, 'imdb_test.pickle'))
        and os.path.exists(os.path.join(tempdir, 'imdb_vocab.pickle'))):

        with open(os.path.join(tempdir, 'imdb_train.pickle'), 'rb') as handle:
            train = pickle.load(handle)
        with open(os.path.join(tempdir, 'imdb_test.pickle'), 'rb') as handle:
            test = pickle.load(handle)
        with open(os.path.join(tempdir, 'imdb_vocab.pickle'), 'rb') as handle:
            vocab = pickle.load(handle)
        with open(os.path.join(tempdir, 'imdb_unvocab.pickle'), 'rb') as handle:
            unvocab = pickle.load(handle)
    else:
        train, test, vocab = text_datasets.get_imdb(fine_grained=False, char_based=False)

        unvocab = {}
        for key, value in vocab.items():
            unvocab[value] = key

        if not testing:
            with open(os.path.join(tempdir, 'imdb_train.pickle'), 'wb') as handle:
                pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(tempdir, 'imdb_test.pickle'), 'wb') as handle:
                pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(tempdir, 'imdb_vocab.pickle'), 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(tempdir, 'imdb_unvocab.pickle'), 'wb') as handle:
                pickle.dump(unvocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Split train into train and valid
    pos_train = train[:12500]
    neg_train = train[12500:]
    valid = pos_train[11250:] + neg_train[11250:]
    train = pos_train[:11250] + neg_train[:11250]

    if testing:
        train = train[:100]
        valid = valid[:100]
        test = test[:100]

    return train, valid, test, vocab, unvocab

def adv_FGSM(model, encoder, test_x, test_y, epsilon=0.01):
    '''
    Applies FGSM on embeddings once.
    '''
    embed_x = sequence_embed(encoder.embed, test_x, encoder.dropout)
    prediction_test = model.embed_forward(embed_x)
    loss_test = F.softmax_cross_entropy(prediction_test, test_y)
    model.cleargrads()
    adv_g = chainer.grad([loss_test], model.embed_inputs)
    adv_p = [epsilon * F.sign(x) for x in adv_g]
    perturbed = [x+p for x, p in zip(embed_x, adv_p)]
    return perturbed

def adv_FGSM_k(model, encoder, test_x, test_y, epsilon=0.01, k=1):
    '''
    Applies FGSM on embeddings several times.
    '''
    for i in range(k):
        test_x = adv_FGSM(model, encoder, test_x, test_y, epsilon=0.01)
    return test_x

def evaluate_fn(eval_iter, device, model, encoder, adversarial=False):
    '''
    Evaluates the model on one epoch of the iterator.Supports standard and adversarial evaluation.
    '''
    test_losses = []
    test_accuracies = []
    for test_batch in eval_iter:
        test_data = convert_seq(test_batch, device)
        test_x = test_data['xs']
        test_y = test_data['ys']
        test_y = F.concat(test_y, axis=0)

        if adversarial:
            test_x = adv_FGSM(model, encoder, test_x, test_y)

        with chainer.using_config('train', False):
            # Forward the test data
            if adversarial: # Adversarial evaluation
                prediction_test = model.embed_forward(test_x)
            else: # Standard evaluation
                prediction_test = model(test_x)
            # Calculate the loss
            loss_test = F.softmax_cross_entropy(prediction_test, test_y)
            test_losses.append(to_cpu(loss_test.array))

            # Calculate the accuracy
            accuracy = F.accuracy(prediction_test, test_y)
            accuracy.to_cpu()
            test_accuracies.append(accuracy.array)

    eval_iter.reset()

    return np.mean(test_losses), np.mean(test_accuracies)

def main():

    # Parse arguments
    current_datetime = '{}'.format(datetime.datetime.today())
    datasets = ['dbpedia', 'imdb.binary', 'imdb.fine','TREC', 'stsa.binary', 'stsa.fine', 'custrev', 'mpqa', 'rt-polarity', 'subj']
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    # Learning
    parser.add_argument('--batchsize', '-b', type=int, default=64, help='Number of images in each mini-batch.')
    parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of sweeps over the dataset to train.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    # Output
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result.')
    parser.add_argument('--temp', default='temp', help='Temporary directory.')
    # Model
    parser.add_argument('--model', '-model', default='rnn', choices=['cnn', 'rnn', 'bow'], help='Name of encoder model type.')
    parser.add_argument('--resume', '-r', type=int, default=None, help='Resume the training from snapshot.')
    parser.add_argument('--unit', '-u', type=int, default=300, help='Number of units.')
    parser.add_argument('--layer', '-l', type=int, default=1, help='Number of layers of RNN or MLP following CNN.')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate.')
    # Adversarial
    parser.add_argument('--adv_lambda', type=float, default=1, help='Adversarial training coefficient.')
    parser.add_argument('--adv_train', dest='adv_train', action='store_true')
    # Testing
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--testing', dest='testing', action='store_true')
    # Device
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--device', '-d', type=str, default='-1', help='Device specifier. Either ChainerX device '
        'specifier or an integer. If non-negative integer, CuPy arrays with specified device id are used. If negative integer, NumPy arrays are used')
    args = parser.parse_args()

    # Seed the generators
    xp = cuda.cupy if args.device >= 0 else np
    xp.random.seed(args.random_seed)
    os.environ["CHAINER_SEED"] = str(args.random_seed)

    # Create directories
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    if not os.path.exists(args.temp):
        os.makedirs(args.temp)

    # Get device
    device = chainer.get_device(args.device)
    device.use()

    # Set up the logger (debug, info, warn, error, critical)
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('simpleLogger')
    fh = logging.FileHandler(os.path.join(args.out, 'output.log'))
    fh.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(fh)
    logger.info(json.dumps(args.__dict__, indent=2))

    # Load data
    train, valid, test, vocab, unvocab = imdb_loader(args.temp, args.testing)

    # Log metadata
    logger.info('Device: {}'.format(device))
    logger.info('# train data: {}'.format(len(train)))
    logger.info('# valid data: {}'.format(len(valid)))
    logger.info('# test  data: {}'.format(len(test)))
    logger.info('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    logger.info('# class: {} \n'.format(n_class))

    # Setup the model
    if args.model == 'rnn':
        Encoder = nets.RNNEncoder
    else:
        logger.error('Code only supports RNNEncoder for now.')
    encoder = Encoder(n_layers=args.layer, n_vocab=len(vocab),
                      n_units=args.unit, dropout=args.dropout)
    model = nets.BasicNetwork(encoder, n_class)
    model.to_device(device) # Copy the model to the device

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up data iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Resume training from a checkpoint
    if args.resume is not None:
        chainer.serializers.load_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(args.resume)), model)
        chainer.serializers.load_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(args.resume)), optimizer)
    else:
        args.resume = 0

    best_valid_acc = -1
    best_valid_epoch = -1
    train_accuracies = []
    train_adv_accuracies = []
    chainer.config.train = True
    logger.info('Started training from epoch: {} \n'.format(args.resume))

    # Training loop
    while train_iter.epoch < args.epoch - args.resume:

        # Get the training batch
        train_batch = train_iter.next()
        train_data = convert_seq(train_batch, device)
        train_x = train_data['xs']
        train_y = train_data['ys']
        train_y = F.concat(train_y, axis=0)

        # Calculate the prediction and loss of the network
        prediction_train = model(train_x)
        loss = F.softmax_cross_entropy(prediction_train, train_y)
        accuracy = F.accuracy(prediction_train, train_y)
        accuracy.to_cpu()
        train_accuracies.append(accuracy.array)

        # Adversarial training
        if args.adv_train:
            perturbed = adv_FGSM(model, encoder, train_x, train_y)
            prediction_train = model.embed_forward(perturbed)
            loss_adv = F.softmax_cross_entropy(prediction_train, train_y)
            loss = loss + args.adv_lambda * loss_adv
            accuracy = F.accuracy(prediction_train, train_y)
            accuracy.to_cpu()
            train_adv_accuracies.append(accuracy.array)

        # Calculate gradients and update all trainable parameters
        model.cleargrads()
        loss.backward()
        optimizer.update()

        # Check valid accuracy after every epoch
        if train_iter.is_new_epoch:

            # Display training loss and accuracy
            logger.info('EPOCH:{:02d} train_loss:{:.04f} train_accuracy:{:.04f} '.format(
                train_iter.epoch + args.resume, float(to_cpu(loss.array)), np.mean(train_accuracies)))
            if args.adv_train:
                logger.info('train_adv_accuracy:{:.04f} '.format(np.mean(train_adv_accuracies)))
            train_accuracies = []

            # Evaluation
            valid_loss, valid_acc = evaluate_fn(valid_iter, device, model, encoder)
            logger.info('valid_loss:{:.04f} valid_accuracy:{:.04f}'.format(valid_loss, valid_acc))
            adv_valid_loss, adv_valid_acc = evaluate_fn(valid_iter, device, model, encoder, adversarial=True)
            logger.info('adv. valid_loss:{:.04f} adv. valid_accuracy:{:.04f} \n'.format(adv_valid_loss, adv_valid_acc))

            # Checkpointing
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_valid_epoch = train_iter.epoch
                chainer.serializers.save_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(train_iter.epoch + args.resume)), model)
                chainer.serializers.save_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(train_iter.epoch + args.resume)), optimizer)
                chainer.serializers.save_npz(os.path.join(args.out, 'best_model.npz'), model)
                chainer.serializers.save_npz(os.path.join(args.out, 'best_state.npz'), optimizer)

    # Save the last epoch and load the best checkpoint
    logger.info('Loading the model with best validation accuracy from checkpoint {} \n'.format(best_valid_epoch))
    chainer.serializers.save_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(train_iter.epoch + args.resume)), model)
    chainer.serializers.save_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(train_iter.epoch + args.resume)), optimizer)
    chainer.serializers.load_npz(os.path.join(args.out, 'best_model.npz'), model)
    chainer.serializers.load_npz(os.path.join(args.out, 'best_state.npz'), optimizer)

    # Evaluation
    test_loss, test_acc = evaluate_fn(test_iter, device, model, encoder)
    logger.info('test_loss:{:.04f} test_accuracy:{:.04f}'.format(test_loss, test_acc))
    adv_loss, adv_acc = evaluate_fn(test_iter, device, model, encoder, adversarial=True)
    logger.info('adv. test_loss:{:.04f} adv. test_accuracy:{:.04f}'.format(adv_loss, adv_acc))

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
