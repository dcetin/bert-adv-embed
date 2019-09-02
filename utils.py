import numpy as np

# Basic functions

def vec_normalize(vec, epsilon=1e-12, xp=np):
    """
    Normalizes the given vector.

    Args:
        vec: A vector.
        epsilon: A small constant for numerical stability in division.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        Matrix with unit norm columns.
    """
    return vec / (xp.linalg.norm(vec) + epsilon)

def mat_normalize(mat, epsilon=1e-12, xp=np):
    """
    Normalizes the columns of the given matrix.

    Args:
        mat: A two dimensional array.
        epsilon: A small constant for numerical stability in division.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        Matrix with unit norm columns.
    """
    return mat / (xp.linalg.norm(mat, axis=1)[:, xp.newaxis] + epsilon)

def to_sent(x, unvocab, tokenized=False):
    """
    Converts the given input into a sentence.

    Args:
        x: A list of vocabulary indices.
        unvocab: Dictionary mapping vocabulary indices to token strings.
        tokenized: Returns a list of strings if set True, one joined string if False.

    Returns:
        A sequence of tokens formatted as strings.
    """
    if type(x) == tuple:
        return to_sent(x[0], unvocab, tokenized)
    if type(x) == int:
        return to_sent([x], unvocab, tokenized)
    if type(x) == list:
        toks = [unvocab[w] for w in x]
        if tokenized:
            return toks
        else:
            return ' '.join(toks)
    return to_sent(x.tolist(), unvocab, tokenized)

def nn_vec_cos(mat, vec, k=1, normalize=True, return_vals=False, xp=np):
    """
    Computes nearest neighbors (in terms of cosine similarity) 
    of the given vector in the column space of the given matrix.

    Args:
        mat: Matrix of vectors to be compared.
        vec: Input vector.
        k: Number of neighbors.
        normalize: If True, normalizes the input arrays.
        return_vals: If True, returns cosine similarity values as well.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        List of nearest neighboring vectors.
    """
    if normalize:
        mat = mat_normalize(mat, xp=xp)
        vec = vec_normalize(vec, xp=xp)
    cosines = xp.matmul(mat, vec)

    if k==1:    # Return the most similar element
        nns = xp.argmax(cosines)
    else:       # Calculate nearest neighbors
        nns = xp.argsort(-cosines)[:k]
    
    if return_vals:
        return nns, cosines[nns]
    else:
        return nns

def nn_vec_L2(mat, vec, k=1, return_vals=False, xp=np):
    """
    Computes nearest neighbors (in terms of Euclidean distance) 
    of the given vector in the column space of the given matrix.

    Args:
        mat: Matrix of vectors to be compared.
        vec: Input vector.
        k: Number of neighbors.
        return_vals: If True, returns cosine similarity values as well.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        List of nearest neighboring vectors.
    """
    norms = xp.linalg.norm(mat - vec, axis=1)

    if k==1:    # Return the most similar element
        nns = xp.argmin(norms)
    else:       # Calculate nearest neighbors
        nns = xp.argsort(norms)[:k]
    
    if return_vals:
        return nns, norms[nns]
    else:
        return nns

def cosine_vec(vec1, vec2, epsilon=1e-12, xp=np):
    """
    Returns the cosine similarity of two vectors.

    Args:
        vec1: First input vector.
        vec2: Second input vector.
        epsilon: A small constant for numerical stability in division.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        Cosine similarity of the input vectors.
    """
    vec1 = vec_normalize(vec1, epsilon=epsilon, xp=xp)
    vec2 = vec_normalize(vec2, epsilon=epsilon, xp=xp)
    cos = xp.clip(xp.dot(vec1 * vec2), 0.0, 1.0)
    return cos

def cosine_seq(seq1, seq2, epsilon=1e-12, xp=np):
    """
    Returns the piecewise cosine similarities of two sequences.

    Args:
        vec1: First input vector.
        vec2: Second input vector.
        epsilon: A small constant for numerical stability in division.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        Cosine similarity of the input sequences.
    """
    assert seq1.shape == seq2.shape
    seq1 = mat_normalize(seq1, epsilon=epsilon, xp=xp)
    seq2 = mat_normalize(seq2, epsilon=epsilon, xp=xp)
    cos = xp.clip(xp.sum(seq1 * seq2, axis=1), 0.0, 1.0)
    return cos

# Model-specific functions

def get_vec_nn(model, vec, k=10, return_vals=False, norm_embed=None, xp=np):
    """
    Returns cosine nearest neighbors of given vector 
    in the word embedding lookup of the model.

    Args:
        model: Neural network model.
        vec: Input vector.
        k: Number of neighbors.
        return_vals: If True, returns cosine similarity values as well.
        norm_embed: Normalized embedding matrix.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        List of nearest neighboring vectors.
    """
    vocab = model.vocab
    unvocab = model.unvocab
    embed_mat = model.bert.word_embeddings.W.data
    if norm_embed is None:
        norm_embed = mat_normalize(embed_mat, xp=xp)

    if type(vec) == str:            # input is a word
        norm_forw = vec_normalize(embed_mat[vocab[vec]], xp=xp)
    elif type(vec) == int:          # input is a vocabulary index
        norm_forw = vec_normalize(embed_mat[vec], xp=xp)
    elif type(vec) == xp.ndarray:   # input is an embedding vector
        norm_forw = vec_normalize(vec, xp=xp)
    else:
        logger.error('Unsupported input format for nearest neighbors.')
    
    max_idx = nn_vec_cos(norm_embed, norm_forw, k=k, normalize=False, xp=xp, return_vals=return_vals)
    if return_vals:
        words = to_sent(max_idx[0], unvocab)
        return words, [round(x, 5) for x in max_idx[1].tolist()]
    else:
        words = to_sent(max_idx, unvocab)
        return words

def get_seq_nn(model, seq, norm_embed=None, project=False, get_ids=False, xp=np):
    """
    Returns cosine nearest neighbors of given sequence 
    of vectors in the word embedding lookup of the model.

    Args:
        model: Neural network model.
        seq: Input vector sequence.
        k: Number of neighbors.
        norm_embed: Normalized embedding matrix.
        project: If True, returns projection of given seq. onto nearest unit vectors.
        get_ids: If True, returns vocabulary indices rather than tokens.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        Nearest neighboring sequence of tokens.
    """
    vocab = model.vocab
    unvocab = model.unvocab
    if norm_embed is None:
        embed_mat = model.bert.word_embeddings.W.data
        norm_embed = mat_normalize(embed_mat, xp=xp)
    seq_norm = mat_normalize(seq, xp=xp)
    seq_nn = xp.matmul(norm_embed, seq_norm.T)
    seq_nn = xp.argmax(seq_nn, axis=0)

    if project:
        units = norm_embed[seq_nn]
        return xp.multiply(units, seq)
    elif get_ids:
        return seq_nn
    else:
        return to_sent(seq_nn, unvocab)

# Exploratory functions

def analogy(model, pos_words=None, neg_words=None, return_vals=False, xp=np):
    """
    Relational analogies on word embeddings, similar to implementation in gensim package.

    Args:
        model: Neural network model.
        pos_words: A list of words whose embeddings to be summed.
        neg_words: A list of words whose embeddings to be subtracted from the sum.
        return_vals: If True, returns cosine similarity values as well.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        A sequence of tokens formatted as strings.
    """
    vocab = model.vocab
    unvocab = model.unvocab
    embed_mat = model.bert.word_embeddings.W.data
    norm_embed = mat_normalize(embed_mat, xp=xp)
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
            nns, vals = get_vec_nn(model, forw, xp=xp, norm_embed=norm_embed, return_vals=True)
            print(' '.join([nn + ' (' + str(val) + ')' for (nn,val) in list(zip(nns.split(' '), vals))]))
        else:
            print(get_vec_nn(model, forw, xp=xp, norm_embed=norm_embed))
    print(' ')

def example_nn(model, return_vals=False, word_list=None, xp=np):
    """
    Nearest neighbor examples.

    Args:
        model: Neural network model.
        return_vals: If True, returns cosine similarity values as well.
        word_list: A list of tokens, its default is from Sato's codebase.
        xp: Matrix library, np for numpy and cp for cupy.

    Returns:
        A sequence of tokens formatted as strings.
    """
    vocab = model.vocab
    unvocab = model.unvocab
    embed_mat = model.bert.word_embeddings.W.data
    norm_embed = mat_normalize(embed_mat, xp=xp)
    if word_list is None:
        word_list = ['good', 'this', 'that', 'awesome', 'bad', 'wrong']
    for word in word_list:
        if return_vals:
            nns, vals = get_vec_nn(model, word, xp=xp, norm_embed=norm_embed, return_vals=True)
            exstr = ' '.join([nn + ' (' + str(val) + ')' for (nn,val) in list(zip(nns.split(' '), vals))])
            print(word + ': ' + exstr)
        else:
            print(word + ': ' + get_vec_nn(model, word, xp=xp, norm_embed=norm_embed))
    print('\n')
