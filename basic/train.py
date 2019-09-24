#!/usr/bin/env python

import argparse
import datetime
import json
import os
import sys
import pickle
import random
import logging
import logging.config
import numpy as np

import chainer
import chainer.functions as F
from chainer import serializers
from chainer.backends import cuda
from chainer.backends.cuda import to_cpu

import nets
import data_utils
from visualize import create_plots

import utils

# import pdb; pdb.set_trace()

def imdb_loader(tempdir, max_vocab_size=None, testing=False, cache=True):
    '''
    An ugly custom function for loading the IMDB data.
    '''
    pik_file = 'imdb_data.pickle'

    if cache and os.path.exists(os.path.join(tempdir, pik_file)):
        with open(os.path.join(tempdir, pik_file), 'rb') as handle:
            train = pickle.load(handle)
            valid = pickle.load(handle)
            test = pickle.load(handle)
            unsup = pickle.load(handle)
            vocab = pickle.load(handle)
            unvocab = pickle.load(handle)

    else:
        vocab_obj, dataset, lm_data, t_vocab = data_utils.load_dataset_imdb(
            include_pretrain=1, lower=0,
            min_count=1, ignore_unk=1,
            use_semi_data=0, add_labeld_to_unlabel=1)
        (train_x, train_x_len, train_y,
         dev_x, dev_x_len, dev_y,
         test_x, test_x_len, test_y) = dataset
        vocab, vocab_count = vocab_obj
        semi_train_x, semi_train_x_len = lm_data
        # print('train_vocab_size:', t_vocab)
        vocab_inv = dict([(widx, w) for w, widx in vocab.items()])
        # print('vocab_inv:', len(vocab_inv))

        train = list(zip(train_x, train_y, train_x_len))
        valid = list(zip(dev_x, dev_y, dev_x_len))
        test = list(zip(test_x, test_y, test_x_len))
        unsup = lm_data

        unvocab = {}
        for key, value in vocab.items():
            unvocab[value] = key

        if cache and testing is False:
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

def adv_FGSM(model, xs, ys, xlens=None, epsilon=5.0, train=False):
    '''
    Applies FGSM on embeddings once.
    '''
    # Cannot seem to backprop on eval mode, so this seems like a possible workaround
    org_dropout = model.dropout
    if train is False:
        model.dropout = 0.0

    if xlens is None:
        xlens = [x.shape[0] for x in xs]
    embed_x = model(xs, xlens=xlens, return_embed=True)
    prediction_test = model(embed_x, feed_embed=True)
    loss_test = F.softmax_cross_entropy(prediction_test, ys, normalize=True)
    model.cleargrads()
    adv_g = chainer.grad([loss_test], model.embedded)

    with chainer.using_config('train', False):
        def sentence_level_norm(grads, lengths):
            # grads: list of (seqlen, embed_dim)
            maxlen = np.max(lengths)
            grads = F.pad_sequence(grads, length=maxlen, padding=0.0)
            batchsize, embed_dim, maxlen = grads.shape
            grads = F.reshape(grads, (batchsize, embed_dim * maxlen))
            grads = F.normalize(grads, axis=1)
            grads = F.reshape(grads, (batchsize, embed_dim, maxlen))
            grads = F.split_axis(grads, batchsize, axis=0)
            grads = [g[0, :l, :] for g, l in zip(grads, lengths)]
            return grads

        adv_p = [epsilon * x for x in sentence_level_norm(adv_g, xlens)] # sentence-level L_2
        # adv_p = [epsilon * F.normalize(x, axis=1) for x in adv_g] # word-level L_2
        # adv_p = [epsilon * F.sign(x) for x in adv_g] # L_infty-norm constraint

    perturbed = [(x+p).data for x, p in zip(embed_x, adv_p)]

    # Restore dropout before returning
    model.dropout = org_dropout

    return perturbed

def evaluate_fn(eval_iter, device, model, adversarial=False, epsilon=5.0, xp=np):
    '''
    Evaluates the model on one epoch of the iterator.Supports standard and adversarial evaluation.
    '''
    eval_losses = []
    eval_accuracies = []
    for eval_batch in eval_iter:
        eval_x, eval_y, eval_x_len = map(list, zip(*eval_batch))
        eval_x = [chainer.dataset.to_device(device, x) for x in eval_x]
        eval_y = chainer.dataset.to_device(device, xp.asarray(eval_y))

        if adversarial:
            adv_eval_x = adv_FGSM(model, eval_x, eval_y, xlens=eval_x_len, epsilon=epsilon, train=False)

        with chainer.using_config('train', False):
            # Forward the eval data
            if adversarial: # Adversarial evaluation
                prediction_eval = model(adv_eval_x, feed_embed=True)
            else: # Standard evaluation
                prediction_eval = model(eval_x, xlens=eval_x_len)
            # Calculate the loss
            loss_eval = F.softmax_cross_entropy(prediction_eval, eval_y, normalize=True)
            eval_losses.append(to_cpu(loss_eval.array))

            # Calculate the accuracy
            accuracy = F.accuracy(prediction_eval, eval_y)
            accuracy.to_cpu()
            eval_accuracies.append(accuracy.array)

    eval_iter.reset()

    return np.mean(eval_losses), np.mean(eval_accuracies)

def example_nn(model, logger, return_vals=False, xp=np):
    '''
    Nearest negihbor examples.
    '''
    norm_embed = model.get_norm_embed(xp=xp)
    word_list = ['good', 'this', 'that', 'awesome', 'bad', 'wrong']
    for word in word_list:
        if return_vals:
            nns, vals = model.get_vec_nn(word, xp=xp, norm_embed=norm_embed, return_vals=True)
            exstr = ' '.join([nn + ' (' + str(val) + ')' for (nn,val) in list(zip(nns.split(' '), vals))])
            logger.debug(word + ': ' + exstr)
        else:
            logger.debug(word + ': ' + model.get_vec_nn(word, xp=xp, norm_embed=norm_embed))
    logger.debug('\n')

def analogy(model, logger, pos_words=None, neg_words=None, return_vals=False, xp=np):
    '''
    Analogy on word embeddings examples.
    '''
    if pos_words is None or neg_words is None:
        # Expecting queen, happy, go, Italy
        word_list = [ (['king', 'woman'],['man']), (['sad', 'good'],['bad']), 
        (['walked', 'went'],['walk']), (['Paris', 'Rome'],['France']) ]
    else:
        word_list = [ (pos_words, neg_words) ]
    for pos_words, neg_words in word_list:
        pos_embed = [model.embed(xp.array([model.vocab[word]])).data[0] for word in pos_words]
        neg_embed = [model.embed(xp.array([model.vocab[word]])).data[0] for word in neg_words]
        pos_embed = xp.sum(xp.stack(pos_embed, axis=0), axis=0)
        neg_embed = xp.sum(xp.stack(neg_embed, axis=0), axis=0)
        forw = pos_embed - neg_embed
        norm_embed = model.get_norm_embed(xp=xp)
        if return_vals:
            nns, vals = model.get_vec_nn(forw, xp=xp, norm_embed=norm_embed, return_vals=True)
            logger.debug(' '.join([nn + ' (' + str(val) + ')' for (nn,val) in list(zip(nns.split(' '), vals))]))
        else:
            logger.debug(model.get_vec_nn(forw, xp=xp, norm_embed=norm_embed))
    logger.debug('\n')

def main():

    # Parse arguments
    current_datetime = '{}'.format(datetime.datetime.today())
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    # Learning
    parser.add_argument('--batchsize', '-b', type=int, default=32, help='Number of images in each mini-batch.')
    parser.add_argument('--epoch', '-e', type=int, default=15, help='Number of sweeps over the dataset to train.')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='Gradient clipping.')
    # Scheduling
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--alpha_decay', type=float, default=0.9998, help='Learning rate decay.')
    parser.add_argument('--exp_decay', type=bool, default=True, help='Exponential learning rate decay.')
    parser.add_argument('--resume_lr', type=bool, default=True, help='Resume learning rate according the schedule.')
    # Output
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result.')
    parser.add_argument('--temp', default='temp', help='Temporary directory.')
    parser.add_argument('--file_log', type=bool, default=True, help='Logging to file.')
    # Pretrained model
    parser.add_argument('--load_pretrained', type=bool, default=True, help='Load the pretrained model.')
    parser.add_argument('--pretrained_path', default='temp/pretrained.pkl', help='Location of the pretrained weights.')
    # Model
    parser.add_argument('--embed_size', type=int, default=256, help='Size of the embedding layer.')
    parser.add_argument('--rnn_units', type=int, default=1024, help='Number of units of the RNN layer.')
    parser.add_argument('--hidden_units', type=int, default=30, help='Number of units of the hidden layer.')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers of RNN or MLP following CNN.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    # Adversarial
    parser.add_argument('--adv_train', dest='adv_train', action='store_true', help='Adversarial training.')
    parser.add_argument('--adv_lambda', type=float, default=1.0, help='Adversarial training loss weight.')
    parser.add_argument('--adv_epsilon', type=float, default=5.0, help='Adversarial training perturbation scale.')
    # Testing
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--testing', dest='testing', action='store_true', help='Loads a small portion of data for debugging.')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Runs Chainer in debug mode.')
    # Checkpointing
    parser.add_argument('--resume', '-r', type=int, default=None, help='Resume the training from snapshot.')
    parser.add_argument('--eval_epoch', type=int, default=0, help='Checkpoint to be evaluated, 0 means best epoch.')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Skips training and evaluates the model.')
    parser.add_argument('--save_all', dest='save_all', action='store_true', help='Saves checkpoint after every epoch.')
    # Device
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--device', '-d', type=str, default='-1', help='Device specifier. Either ChainerX device specifier '
        'or an integer. If non-negative integer, CuPy arrays with specified device id are used. If negative integer, NumPy arrays are used')
    args = parser.parse_args()

    # Seed the generators
    xp = cuda.cupy if args.device >= 0 else np
    xp.random.seed(args.random_seed)
    random.seed(args.random_seed)
    os.environ["CHAINER_SEED"] = str(args.random_seed)

    # Debug mode
    if args.debug:
        chainer.set_debug(True)

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
    if args.file_log is True:
        fh = logging.FileHandler(os.path.join(args.out, 'output.log'))
        fh.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(fh)
    logger.info(args.__dict__)

    # Load data
    train, valid, test, unsup, vocab, unvocab = imdb_loader(args.temp, testing=args.testing)
    n_class = len(set([int(d[1]) for d in train]))

    # Log metadata
    if not args.evaluate and args.resume is None:
        # logger.info('Device: {}'.format(device))
        # logger.info('# train data: {}'.format(len(train)))
        # logger.info('# valid data: {}'.format(len(valid)))
        # logger.info('# test  data: {}'.format(len(test)))
        # logger.info('# vocab: {}'.format(len(vocab)))
        # logger.info('# class: {} \n'.format(n_class))
        # def dataset_stats(data):
        #     seqlens = [x[0].size -1 for x in data]
        #     print('Num. examples:\t', len(data))
        #     print('Min. sequence length:\t', np.min(seqlens))
        #     print('Max. sequence length:\t', np.max(seqlens))
        #     print('Avg. sequence length:\t', np.average(seqlens))
        # [dataset_stats(x) for x in [train,valid,test,unsup]]
        pass

    # Setup the model and copy the model to the device
    model = nets.classifierModel(n_class, n_layers=args.layers, n_vocab=len(vocab),
        n_units=args.rnn_units, embed_size=args.embed_size, hidden_units=args.hidden_units, dropout=args.dropout)
    model.to_device(device)

    # Set the model environment
    model.vocab = vocab
    model.unvocab = unvocab
    model.logger = logger

    # Log nearest neighbors
    # example_nn(model, logger, xp=xp)
    # analogy(model, logger, xp=xp)

    # Load pretrained embedding and weights
    if args.load_pretrained:
        logger.info('Loading pretrained model weights')
        model.load_pretrained(args.pretrained_path)
        # Log nearest neighbors
        # example_nn(model, logger, xp=xp)
        # analogy(model, logger, xp=xp)

    # Setup an optimizer
    base_alpha = args.learning_rate
    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
    global_step = 0.0

    # Set up data iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Resume training from a checkpoint
    if args.resume is not None:
        logger.info('Loading model checkpoint from epoch {}'.format(args.resume))
        chainer.serializers.load_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(args.resume)), model)
        chainer.serializers.load_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(args.resume)), optimizer)
        if args.resume_lr is True and args.alpha_decay > 0.0:
            global_step = 664.0 * args.resume
            if args.exp_decay:
                optimizer.hyperparam.alpha = (base_alpha) * (args.alpha_decay ** global_step)
            else:
                optimizer.hyperparam.alpha *= args.alpha_decay ** args.resume
        # Log nearest neighbors
        example_nn(model, logger, xp=xp)
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

        # Training loop
        while train_iter.epoch < args.epoch - args.resume:
            global_step += 1.0

            # Get the training batch
            train_batch = train_iter.next()
            train_x, train_y, train_x_len = map(list, zip(*train_batch))
            train_x = [chainer.dataset.to_device(device, x) for x in train_x]
            train_y = chainer.dataset.to_device(device, xp.asarray(train_y))

            # Calculate the prediction and loss of the network
            prediction_train = model(train_x, xlens=train_x_len)
            org_loss = F.softmax_cross_entropy(prediction_train, train_y, normalize=True)
            accuracy = F.accuracy(prediction_train, train_y)
            accuracy.to_cpu()
            train_accuracies.append(accuracy.array)

            # Adversarial training
            if args.adv_train:
                if args.load_pretrained or (~args.load_pretrained and (args.resume > 0 or train_iter.epoch > 1)):
                    perturbed = adv_FGSM(model, train_x, train_y, xlens=train_x_len, epsilon=args.adv_epsilon, train=True)
                    prediction_train = model(perturbed, feed_embed=True)
                    adv_loss = F.softmax_cross_entropy(prediction_train, train_y, normalize=True)
                    loss = org_loss + args.adv_lambda * adv_loss
                    accuracy = F.accuracy(prediction_train, train_y)
                    accuracy.to_cpu()
                    train_adv_accuracies.append(accuracy.array)
            else:
                loss = org_loss

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
                logger.info('EPOCH:{:02d} global_step:{:.04f} alpha:{:.08f} '.format(train_iter.epoch + args.resume, global_step, optimizer.hyperparam.alpha))
                logger.info('[train] loss:{:.04f} acc:{:.04f} '.format(float(to_cpu(org_loss.array)), np.mean(train_accuracies)))
                train_accuracies = []

                # Standard evaluation
                valid_loss, valid_acc = evaluate_fn(valid_iter, device, model, xp=xp)
                logger.info('[valid] loss:{:.04f} acc:{:.04f}'.format(valid_loss, valid_acc))
                test_loss, test_acc = evaluate_fn(test_iter, device, model, xp=xp)
                logger.info('[test ] loss:{:.04f} acc:{:.04f}'.format(test_loss, test_acc))

                # Adversarial evalation
                if args.adv_train:
                    logger.info('[train adv] loss:{:.04f} acc:{:.04f} '.format(float(to_cpu(adv_loss.array)), np.mean(train_adv_accuracies)))
                valid_adv_loss, valid_adv_acc = evaluate_fn(valid_iter, device, model, adversarial=True, epsilon=args.adv_epsilon, xp=xp)
                logger.info('[valid adv] loss:{:.04f} acc:{:.04f}'.format(valid_adv_loss, valid_adv_acc))
                # test_adv_loss, test_adv_acc = evaluate_fn(test_iter, device, model, adversarial=True, epsilon=args.adv_epsilon, xp=xp)
                # logger.info('[test  adv] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))

                logger.info('\n')

                # Checkpointing
                if args.save_all or valid_acc > best_valid_acc:
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

    # Load the evaluation checkpoint, best is default
    if args.eval_epoch == 0:
        logger.info('Loading the model with best valid. accuracy for evaluation')
        chainer.serializers.load_npz(os.path.join(args.out, 'best_model.npz'), model)
        chainer.serializers.load_npz(os.path.join(args.out, 'best_state.npz'), optimizer)
    else:
        logger.info('Loading model checkpoint from epoch {} for evaluation'.format(args.eval_epoch))
        chainer.serializers.load_npz(os.path.join(args.out, 'model_epoch_{}.npz'.format(args.eval_epoch)), model)
        chainer.serializers.load_npz(os.path.join(args.out, 'state_epoch_{}.npz'.format(args.eval_epoch)), optimizer)

    # Log nearest neighbors
    # example_nn(model, logger, xp=xp)
    # analogy(model, logger, xp=xp)

    # Playground
    def projection_demo(model, eval_iter, logger, xp=np):

        # Number of examples to generate
        max_examples = 1
        # Maximum sequence length to choose sequences from
        max_seq_len = 20

        num_examples = 0
        for bi, test_batch in enumerate(eval_iter):
            test_x, test_y, test_x_len = map(list, zip(*test_batch))
            test_x = [chainer.dataset.to_device(device, x) for x in test_x]
            test_y = chainer.dataset.to_device(device, xp.asarray(test_y))

            adv_test_x = adv_FGSM(model, test_x, test_y, xlens=test_x_len, epsilon=args.adv_epsilon, train=False)

            # logger.debug('bi {} batchsize {} pos {}'.format(bi, len(test_batch), eval_iter.current_position))

            with chainer.using_config('train', False):
                prediction_test = model(test_x, xlens=test_x_len, argmax=True)
                embed_x = model.embedded
                embed_x = [x.data for x in embed_x]
                prediction_adv_test = model(adv_test_x, feed_embed=True, argmax=True)
                std = (prediction_test == test_y)
                adv = (prediction_adv_test == test_y)
                res = ~adv & std
                res_ids = xp.where(res.astype(int) == 1)[0].tolist()

                # logger.debug('groundtruth: {}'.format(test_y.data))
                # logger.debug('predictions: {}'.format(prediction_test))
                # logger.debug('adv.  preds: {}'.format(prediction_adv_test))
                # logger.debug('std: {}'.format(std.astype(int)))
                # logger.debug('adv: {}'.format(adv.astype(int)))
                # logger.debug('res: {}'.format(res.astype(int)))
                # logger.debug('std acc: {}'.format(std.mean()))
                # logger.debug('adv acc: {}'.format(adv.mean()))

                # Iterate over all adversarial sequences, or pick the shortest one
                if len(res_ids) > 0:
                    # shortest = min(res_ids, key=lambda x:int(test_x[x].size))
                    # for seq_idx in [shortest]:
                    for seq_idx in res_ids:

                        # Iterate until finding a sufficiently short sequence
                        if test_x[seq_idx].size > max_seq_len:
                            continue

                        sequence_offset = eval_iter.current_position + seq_idx - 32

                        num_examples += 1
                        logger.debug('Visualizing example {}:'.format(num_examples))
                        logger.debug('seq_offset: {} of length {}'.format(sequence_offset, test_x[seq_idx].size))

                        # Vectorised version
                        norm_embed = model.get_norm_embed(xp=xp)

                        emb = embed_x[seq_idx]
                        emb_nn = model.get_seq_nn(emb, norm_embed=norm_embed, xp=xp)
                        emb_norm = xp.linalg.norm(emb, axis=1).tolist()

                        adv = adv_test_x[seq_idx]
                        adv_nn = model.get_seq_nn(adv, norm_embed=norm_embed, xp=xp)
                        adv_norm = xp.linalg.norm(adv, axis=1).tolist()

                        per = adv-emb
                        per_nn = []
                        for wi in range(len(per)):
                            per_w = per[wi]
                            org_w = emb[wi]
                            rel_norm_embed = utils.mat_normalize(model.embed.W.data - org_w, xp=xp)
                            nn_dir_w = model.get_vec_nn(per_w, k=1, norm_embed=rel_norm_embed, xp=xp)
                            per_nn.append(nn_dir_w)
                        per_nn = ' '.join(per_nn)
                        per_norm = xp.linalg.norm(per, axis=1).tolist()

                        logger.debug(emb_nn)
                        logger.debug(adv_nn)
                        logger.debug(per_nn)
                        # logger.debug(emb_norm)
                        # logger.debug(adv_norm)
                        # logger.debug(per_norm)
                        logger.debug('\n')


                        vec_good = model.embed(xp.array([model.vocab['good']])).data[0]
                        vec_bad = model.embed(xp.array([model.vocab['bad']])).data[0]
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

                        print(list(zip(per_nn.split(' '), per_NegativeDir)))

                        print(list(zip(adv_nn.split(' '), adv_good)))
                        print(list(zip(adv_nn.split(' '), adv_bad)))
                        print(list(zip(emb_nn.split(' '), emb_good)))
                        print(list(zip(emb_nn.split(' '), emb_bad)))



                        data = emb_nn, adv_nn, per_nn, emb_norm, adv_norm, per_norm
                        metadata = {}
                        metadata['name'] = args.out + str(sequence_offset) + '_eps' + str(args.adv_epsilon)
                        metadata['epsilon'] = str(args.adv_epsilon)
                        create_plots(data, metadata, folder=args.out)

                        if num_examples >= max_examples:
                            eval_iter.reset()
                            return
        
        eval_iter.reset()
    projection_demo(model, test_iter, logger, xp=xp)

    # Standard evaluation
    # valid_loss, valid_acc = evaluate_fn(valid_iter, device, model, xp=xp)
    # logger.info('[valid] loss:{:.04f} acc:{:.04f}'.format(valid_loss, valid_acc))
    # test_loss, test_acc = evaluate_fn(test_iter, device, model, xp=xp)
    # logger.info('[test ] loss:{:.04f} acc:{:.04f}'.format(test_loss, test_acc))

    # # Adversarial evaluation
    # valid_adv_loss, valid_adv_acc = evaluate_fn(valid_iter, device, model, adversarial=True, epsilon=args.adv_epsilon, xp=xp)
    # logger.info('[valid adv] loss:{:.04f} acc:{:.04f}'.format(valid_adv_loss, valid_adv_acc))
    # test_adv_loss, test_adv_acc = evaluate_fn(test_iter, device, model, adversarial=True, epsilon=args.adv_epsilon, xp=xp)
    # logger.info('[test  adv] loss:{:.04f} acc:{:.04f}'.format(test_adv_loss, test_adv_acc))

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
