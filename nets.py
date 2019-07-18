import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs

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

    # Forward step
    def __call__(self, xs, softmax=False, argmax=False, feed_embed=False):
        if feed_embed:
            self.embed_inputs = xs
            exs = [F.dropout(x, ratio=self.dropout) for x in xs]
        else:
            exs = sequence_embed(self.embed, xs, self.dropout)
            # exs = [F.dropout(self.embed(x), ratio=self.dropout) for x in xs]

        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        concat_outputs = last_h[-1]
        self.encoded = concat_outputs
        # concat_encodings = F.dropout(self.encoded, ratio=self.dropout)
        # concat_outputs = self.output(concat_encodings)
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

