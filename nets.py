import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

import utils

class classifierModel(chainer.Chain):

    """A classifier using a LSTM-RNN Encoder with Word Embedding.
      This model encodes a sentence sequentially using LSTM. and classifies it into classes.

    Args:
        n_class (int): The number of classes to be predicted.
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer.
        embed_size (int): The number of units of word embedding.
        hidden_units (int): The number of units the hidden layer.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_class, n_layers, n_vocab, n_units, embed_size, hidden_units=30, dropout=0.1):
        super(classifierModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed_size) # , ignore_label=-1
            self.encoder = L.NStepLSTM(n_layers, embed_size, n_units, dropout)
            self.hidden = L.Linear(n_units, hidden_units)
            self.output = L.Linear(hidden_units, n_class)

        self.embed_size = embed_size
        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout

        # Forget gate bias => 1.0
        # MEMO: Values 1 and 5 reference the forget gate.
        for w in self.encoder:
            w.b1.data[:] = 1.0
            w.b5.data[:] = 1.0

    # 
    def __call__(self, xs, softmax=False, argmax=False, feed_embed=False, return_embed=False):
        """
        Forward step
        Args:
            xs (list): List of ndarray, each containing sequence of vocab. indices.
            softmax (bool): Return softmax result instead of logits.
            argmax (bool): Return the predicted classes.
            feed_embed (bool): Feed continuous embeddings rather than discrete indices as input.
            return_embed (bool): Return embedding output instead of final classifier output.
        """
        if feed_embed:
            self.embedded = xs
        else:
            # Efficient embedding function for variable-length sequences:
            # equal to [F.dropout(self.embed(x), ratio=self.dropout) for x in xs]
            x_len = [len(x) for x in xs]
            x_section = np.cumsum(x_len[:-1])
            ex = self.embed(F.concat(xs, axis=0))
            ex = F.dropout(ex, ratio=self.dropout)
            self.embedded = F.split_axis(ex, x_section, 0)

        if return_embed:
            return self.embedded

        last_h, last_c, ys = self.encoder(None, None, self.embedded)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        self.encoded = last_h[-1]
        hidden_outputs = self.hidden(self.encoded)
        hidden_outputs = F.relu(hidden_outputs)
        hidden_outputs = F.dropout(hidden_outputs, ratio=self.dropout)
        concat_outputs = self.output(hidden_outputs)

        if softmax:
            return F.softmax(concat_outputs).array
        elif argmax:
            return self.xp.argmax(concat_outputs.array, axis=1)
        else:
            return concat_outputs

    def get_vec_nn(self, inp, k=10, return_vals=False, norm_embed=None, xp=np):
        if norm_embed is None:
            norm_embed = utils.mat_normalize(self.embed.W.data, xp=xp)

        if type(inp) == str:            # input is a word
            norm_forw = utils.vec_normalize(self.embed(xp.array([self.vocab[inp]])).data[0], xp=xp)
        elif type(inp) == int:          # input is a vocabulary index
            norm_forw = utils.vec_normalize(self.embed(xp.array([inp])).data[0], xp=xp)
        elif type(inp) == xp.ndarray:   # input is an embedding vector
            norm_forw = utils.vec_normalize(inp, xp=xp)
        else:
            logger.error('Unsupported input format for nearest neighbors.')
        
        max_idx = utils.nn_vec(norm_embed, norm_forw, k=k, normalize=False, xp=xp, return_vals=return_vals)
        if return_vals:
            words = utils.to_sent(max_idx[0], self.unvocab)
            return words, [round(x, 5) for x in max_idx[1].tolist()]
        else:
            words = utils.to_sent(max_idx, self.unvocab)
            return words

    def get_seq_nn(self, seq, norm_embed=None, project=False, xp=np):
        if norm_embed is None:
            norm_embed = utils.mat_normalize(self.embed.W.data, xp=xp)
        seq_norm = utils.mat_normalize(seq, xp=xp)
        seq_nn = xp.matmul(norm_embed, seq_norm.T)
        seq_nn = xp.argmax(seq_nn, axis=0)

        if project:
            units = norm_embed[seq_nn]
            return xp.multiply(units, seq)
        else:
            return utils.to_sent(seq_nn, self.unvocab)

