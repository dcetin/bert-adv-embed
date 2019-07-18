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
import random
import data_utils
from utils import mat_normalize
# import pdb; pdb.set_trace()

def imdb_loader(tempdir, max_vocab_size=None, testing=False, write=True, sato=True):
    '''
    An ugly custom function for loading the IMDB data.
    '''
    pik_file = 'imdb_data.pickle'

    if os.path.exists(os.path.join(tempdir, pik_file)):
        with open(os.path.join(tempdir, pik_file), 'rb') as handle:
            train = pickle.load(handle)
            valid = pickle.load(handle)
            test = pickle.load(handle)
            unsup = pickle.load(handle)
            vocab = pickle.load(handle)
            unvocab = pickle.load(handle)

    else:
        if sato:
            sato_vocab_obj, sato_dataset, sato_lm_data, sato_t_vocab = data_utils.load_dataset_imdb(
                include_pretrain=1, lower=0,
                min_count=1, ignore_unk=1,
                use_semi_data=0, add_labeld_to_unlabel=1)
            (sato_train_x, sato_train_x_len, sato_train_y,
             sato_dev_x, sato_dev_x_len, sato_dev_y,
             sato_test_x, sato_test_x_len, sato_test_y) = sato_dataset
            sato_vocab, sato_vocab_count = sato_vocab_obj
            sato_semi_train_x, sato_semi_train_x_len = sato_lm_data
            # print('train_vocab_size:', sato_t_vocab)
            sato_vocab_inv = dict([(widx, w) for w, widx in sato_vocab.items()])
            # print('vocab_inv:', len(sato_vocab_inv))

            train = list(zip(sato_train_x, [np.asarray([x]) for x in sato_train_y]))
            valid = list(zip(sato_dev_x, [np.asarray([x]) for x in sato_dev_y]))
            test = list(zip(sato_test_x, [np.asarray([x]) for x in sato_test_y]))
            vocab = sato_vocab
            unsup = sato_semi_train_x

        else:
            train, valid, test, unsup, vocab = text_datasets.get_imdb(fine_grained=False, char_based=False, max_vocab_size=None)

        unvocab = {}
        for key, value in vocab.items():
            unvocab[value] = key

        if write and testing is False:
            with open(os.path.join(tempdir, pik_file), 'wb') as handle:
                pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(unsup, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(unvocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if testing:
        train = train[:100]
        valid = valid[:100]
        test = test[:100]

    return train, valid, test, unsup, vocab, unvocab

def adv_FGSM(model, test_x, test_y, epsilon=1.0, norm='l2'):
    '''
    Applies FGSM on embeddings once.
    '''
    embed_x = sequence_embed(model.embed, test_x, model.dropout)
    prediction_test = model(embed_x, feed_embed=True)
    loss_test = F.softmax_cross_entropy(prediction_test, test_y, normalize=True)
    model.cleargrads()
    adv_g = chainer.grad([loss_test], model.embed_inputs)

    if norm == 'l2':        # L_2-norm constraint
        adv_p = [epsilon * F.normalize(x, axis=1) for x in adv_g]
    elif norm == 'linf':    # L_infty-norm constraint
        adv_p = [epsilon * F.sign(x) for x in adv_g]

    perturbed = [x+p for x, p in zip(embed_x, adv_p)]
    return perturbed

def evaluate_fn(eval_iter, device, model, adversarial=False, epsilon=1.0):
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
            test_x = adv_FGSM(model, test_x, test_y, epsilon)

        with chainer.using_config('train', False):
            # Forward the test data
            if adversarial: # Adversarial evaluation
                prediction_test = model(test_x, feed_embed=True)
            else: # Standard evaluation
                prediction_test = model(test_x)
            # Calculate the loss
            loss_test = F.softmax_cross_entropy(prediction_test, test_y, normalize=True)
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
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    # Learning
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of images in each mini-batch.')
    parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of sweeps over the dataset to train.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--alpha_decay', type=float, default=0.9998, help='Learning rate decay.')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--exp_decay', type=bool, default=True, help='Exponential learning rate decay.')
    # Output
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result.')
    parser.add_argument('--temp', default='temp', help='Temporary directory.')
    # Model
    parser.add_argument('--embed_size', type=int, default=256, help='Size of the embedding layer.')
    parser.add_argument('--rnn_units', type=int, default=1024, help='Number of units of the RNN layer.')
    parser.add_argument('--hidden_units', type=int, default=30, help='Number of units of the hidden layer.')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers of RNN or MLP following CNN.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    # Adversarial
    parser.add_argument('--adv_lambda', type=float, default=1.0, help='Adversarial training loss weight.')
    parser.add_argument('--adv_train', dest='adv_train', action='store_true')
    parser.add_argument('--adv_epsilon', type=float, default=5.0, help='Adversarial training perturbation scale.')
    # Testing
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--testing', dest='testing', action='store_true', help='Loads a small portion of data for debugging.')
    # Checkpointing
    parser.add_argument('--resume', '-r', type=int, default=None, help='Resume the training from snapshot.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Skips training (resume option is ignored).')
    # Device
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--device', '-d', type=str, default='-1', help='Device specifier. Either ChainerX device '
        'specifier or an integer. If non-negative integer, CuPy arrays with specified device id are used. If negative integer, NumPy arrays are used')
    args = parser.parse_args()

    # Seed the generators
    xp = cuda.cupy if args.device >= 0 else np
    xp.random.seed(args.random_seed)
    random.seed(args.random_seed)
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
    # logger.info(json.dumps(args.__dict__, indent=2))
    logger.info(args.__dict__)

    # Load data (e.g. max_vocab_size=20000 for a small vocab)
    train, valid, test, unsup, vocab, unvocab = imdb_loader(args.temp, testing=args.testing)
    # def dataset_stats(data):
    #     seqlens = [x[0].size -1 for x in data]
    #     print('Num. examples:\t', len(data))
    #     print('Min. sequence length:\t', np.min(seqlens))
    #     print('Max. sequence length:\t', np.max(seqlens))
    #     print('Avg. sequence length:\t', np.average(seqlens))
    # [dataset_stats(x) for x in [train,valid,test,unsup]]

    # Log metadata
    logger.info('Device: {}'.format(device))
    logger.info('# train data: {}'.format(len(train)))
    logger.info('# valid data: {}'.format(len(valid)))
    logger.info('# test  data: {}'.format(len(test)))
    logger.info('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    logger.info('# class: {} \n'.format(n_class))

    # Setup the model
    model = nets.classifierModel(n_class, n_layers=args.layers, n_vocab=len(vocab),
        n_units=args.rnn_units, embed_size=args.embed_size, hidden_units=args.hidden_units, dropout=args.dropout)
    model.to_device(device) # Copy the model to the device

    # Set the model environment
    model.vocab = vocab
    model.unvocab = unvocab

    # Setup an optimizer
    base_alpha = args.learning_rate
    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

    # Set up data iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Log nearest neighbors
    norm_embed = mat_normalize(model.embed.W.data, xp=xp)
    logger.debug('good: ' + model.nn_word('good', xp=xp, norm_embed=norm_embed))
    logger.debug('this: ' + model.nn_word('this', xp=xp, norm_embed=norm_embed))
    logger.debug('that: ' + model.nn_word('that', xp=xp, norm_embed=norm_embed))
    logger.debug('awesome: ' + model.nn_word('awesome', xp=xp, norm_embed=norm_embed))
    logger.debug('bad: ' + model.nn_word('bad', xp=xp, norm_embed=norm_embed))
    logger.debug('wrong: ' + model.nn_word('wrong', xp=xp, norm_embed=norm_embed))

    # Resume training from a checkpoint
    if args.resume is not None:
        # if args.resume.isnumeric():
        chainer.serializers.load_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(args.resume)), model)
        chainer.serializers.load_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(args.resume)), optimizer)
        # else:
        #     logger.error('Can only load checkpoints via specifying epochs.')
    else:
        args.resume = 0

    # Training phase
    if not args.evaluate:

        best_valid_acc = -1
        best_valid_epoch = -1
        train_accuracies = []
        train_adv_accuracies = []
        chainer.config.train = True
        logger.info('Started training from epoch: {} \n'.format(args.resume))

        model.cleargrads()
        global_step = 0.0

        # Training loop
        while train_iter.epoch < args.epoch - args.resume:

            global_step += 1.0

            # Get the training batch
            train_batch = train_iter.next()
            train_data = convert_seq(train_batch, device)
            train_x = train_data['xs']
            train_y = train_data['ys']
            train_y = F.concat(train_y, axis=0)

            # Calculate the prediction and loss of the network
            prediction_train = model(train_x)
            loss = F.softmax_cross_entropy(prediction_train, train_y, normalize=True)
            accuracy = F.accuracy(prediction_train, train_y)
            accuracy.to_cpu()
            train_accuracies.append(accuracy.array)

            # Adversarial training
            if args.adv_train:
                perturbed = adv_FGSM(model, train_x, train_y, epsilon=args.adv_epsilon)
                prediction_train = model(perturbed, feed_embed=True)
                loss_adv = F.softmax_cross_entropy(prediction_train, train_y, normalize=True)
                loss = loss + args.adv_lambda * loss_adv
                accuracy = F.accuracy(prediction_train, train_y)
                accuracy.to_cpu()
                train_adv_accuracies.append(accuracy.array)

            # Calculate gradients and update all trainable parameters
            model.cleargrads()
            loss.backward()
            optimizer.update()

            # Learning rate decay
            if args.alpha_decay > 0.0:
                if args.exp_decay:
                    optimizer.hyperparam.alpha = (base_alpha) * (args.alpha_decay ** global_step)
                else:
                    optimizer.hyperparam.alpha *= args.alpha_decay

            # Check valid accuracy after every epoch
            if train_iter.is_new_epoch:

                # Display training loss and accuracy
                logger.info('EPOCH:{:02d} train_loss:{:.04f} train_accuracy:{:.04f} '.format(
                    train_iter.epoch + args.resume, float(to_cpu(loss.array)), np.mean(train_accuracies)))
                if args.adv_train:
                    logger.info('train_adv_accuracy:{:.04f} '.format(np.mean(train_adv_accuracies)))
                train_accuracies = []

                logger.info('alpha:{:.04f} global_step:{:.04f} '.format(optimizer.hyperparam.alpha, global_step))

                # Evaluation on validation data
                valid_loss, valid_acc = evaluate_fn(valid_iter, device, model)
                logger.info('valid_loss:{:.04f} valid_accuracy:{:.04f}'.format(valid_loss, valid_acc))
                adv_valid_loss, adv_valid_acc = evaluate_fn(valid_iter, device, model, adversarial=True, epsilon=args.adv_epsilon)
                logger.info('adv. valid_loss:{:.04f} adv. valid_accuracy:{:.04f}'.format(adv_valid_loss, adv_valid_acc))

                # Evaluation on test data
                # test_loss, test_acc = evaluate_fn(test_iter, device, model)
                # logger.info('test_loss:{:.04f} test_accuracy:{:.04f}'.format(test_loss, test_acc))
                # adv_loss, adv_acc = evaluate_fn(test_iter, device, model, adversarial=True, epsilon=args.adv_epsilon)
                # logger.info('adv. test_loss:{:.04f} adv. test_accuracy:{:.04f}'.format(adv_loss, adv_acc))

                logger.info('\n')

                # Checkpointing
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_valid_epoch = train_iter.epoch
                    chainer.serializers.save_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(train_iter.epoch + args.resume)), model)
                    chainer.serializers.save_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(train_iter.epoch + args.resume)), optimizer)
                    chainer.serializers.save_npz(os.path.join(args.out, 'best_model.npz'), model)
                    chainer.serializers.save_npz(os.path.join(args.out, 'best_state.npz'), optimizer)

                # Before the new training epoch
                model.cleargrads()
                chainer.config.train = True
        
        # Save the last epoch
        chainer.serializers.save_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(train_iter.epoch + args.resume)), model)
        chainer.serializers.save_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(train_iter.epoch + args.resume)), optimizer)

    # Load the best checkpoint
    # logger.info('Loading the model with best validation accuracy from checkpoint {} \n'.format(best_valid_epoch))
    chainer.serializers.load_npz(os.path.join(args.out, 'best_model.npz'), model)
    chainer.serializers.load_npz(os.path.join(args.out, 'best_state.npz'), optimizer)

    norm_embed = mat_normalize(model.embed.W.data, xp=xp)
    logger.debug('good: ' + model.nn_word('good', xp=xp, norm_embed=norm_embed))
    logger.debug('this: ' + model.nn_word('this', xp=xp, norm_embed=norm_embed))
    logger.debug('that: ' + model.nn_word('that', xp=xp, norm_embed=norm_embed))
    logger.debug('awesome: ' + model.nn_word('awesome', xp=xp, norm_embed=norm_embed))
    logger.debug('bad: ' + model.nn_word('bad', xp=xp, norm_embed=norm_embed))
    logger.debug('wrong: ' + model.nn_word('wrong', xp=xp, norm_embed=norm_embed))

    # Evaluation
    # valid_loss, valid_acc = evaluate_fn(valid_iter, device, model)
    # logger.info('valid_loss:{:.04f} valid_accuracy:{:.04f}'.format(valid_loss, valid_acc))
    # adv_valid_loss, adv_valid_acc = evaluate_fn(valid_iter, device, model, adversarial=True, epsilon=args.adv_epsilon)
    # logger.info('adv. valid_loss:{:.04f} adv. valid_accuracy:{:.04f}'.format(adv_valid_loss, adv_valid_acc))
    # test_loss, test_acc = evaluate_fn(test_iter, device, model)
    # logger.info('test_loss:{:.04f} test_accuracy:{:.04f}'.format(test_loss, test_acc))
    # adv_loss, adv_acc = evaluate_fn(test_iter, device, model, adversarial=True, epsilon=args.adv_epsilon)
    # logger.info('adv. test_loss:{:.04f} adv. test_accuracy:{:.04f}'.format(adv_loss, adv_acc))

    # Save vocabulary and model's setting
    if not args.evaluate:
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
