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

    """A classifier using a given encoder.

     This chain encodes a sentence and classifies it into classes.

     Args:
         encoder (Link): A callable encoder, which extracts a feature.
             Input is a list of variables whose shapes are
             "(sentence_length, )".
             Output is a variable whose shape is "(batchsize, n_units)".
         n_class (int): The number of classes to be predicted.

     """

    def __init__(self, encoder, n_class, hidden_units=30, dropout=0.1):
        super(classifierModel, self).__init__()
        with self.init_scope():
            self.encoder = encoder

            self.hidden = L.Linear(encoder.out_units, hidden_units)
            self.output = L.Linear(hidden_units, n_class)
            # self.output = L.Linear(encoder.out_units, n_class)
            
        self.dropout = dropout

    # Actual forward step
    def forward(self, xs, softmax=False, argmax=False):
        self.encoded = self.encoder(xs)
        
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

    # Function to forward cont. embeddings, rather than one-hot encodings
    def embed_forward(self, xs, softmax=False, argmax=False):
        self.embed_inputs = xs
        self.encoded = self.encoder.embed_forward(xs)
        
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

class RNNEncoder(chainer.Chain):

    """A LSTM-RNN Encoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer.
        embed_size (int): The number of units of word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, embed_size, dropout=0.1):
        super(RNNEncoder, self).__init__()
        with self.init_scope():
            # self.embed = L.EmbedID(n_vocab, embed_size, initialW=chainer.initializers.Uniform(.25))
            self.embed = L.EmbedID(n_vocab, embed_size)
            self.encoder = L.NStepLSTM(n_layers, embed_size, n_units, dropout)

        self.embed_size = embed_size
        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout

        # Forget gate bias => 1.0
        # MEMO: Values 1 and 5 reference the forget gate.
        for w in self.encoder:
            w.b1.data[:] = 1.0
            w.b5.data[:] = 1.0

    # Actual forward step
    def forward(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        concat_outputs = last_h[-1]
        return concat_outputs

    # Alternative forward step, which is slower
    def alt_forward(self, xs):
        exs = [F.dropout(self.embed(x), ratio=self.dropout) for x in xs]
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        concat_outputs = last_h[-1]
        return concat_outputs

    # Function to forward cont. embeddings, rather than one-hot encodings
    def embed_forward(self, xs):
        exs = [F.dropout(x, ratio=self.dropout) for x in xs]
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        concat_outputs = last_h[-1]
        return concat_outputs

