# Cannot import from this file, just copy and paste the functions to use them

def nearest_demo(encoder, eval_iter, device, idx=3, idxx=0):
        '''
        nearest_demo(encoder, valid_iter, device)
        '''

        epsilon = 1e-10

        # Run in eval mode
        chainer.config.train = False

        # Embedding matrix
        embed_data = encoder.embed.W.data # (20000, 300)
        norm_embed_data = embed_data / (xp.linalg.norm(embed_data, axis=1)[:, xp.newaxis] + epsilon)

        # Get one sample batch
        for test_batch in eval_iter:
            test_data = convert_seq(test_batch, device)
            test_x = test_data['xs']
            test_y = test_data['ys']
            test_y = F.concat(test_y, axis=0)
            break
        eval_iter.reset()

        # Get an example sequence
        print('Sequence index:', idx)
        exx = test_x[idx] # Example sequence: (n,)
        exy = test_y.data[idx] # Example label: ()
        exm = sequence_embed(encoder.embed, test_x, encoder.dropout)[idx].data # Example sequence embeddings: (n, 300)

        # Get an example word from the sequence
        print('Getting the', idxx, 'th word of the sequence')
        print('...which is', unvocab[exx[idxx]], 'with vocab id', exx[idxx])
        print(' ')

        print('Lookup embedding and forwarded output (shown partially) as a sanity check:')
        forw = exm[idxx]
        org = embed_data[exx[idxx]]
        print(org[:15])
        print(forw[:15])
        norm_forw = forw / (xp.linalg.norm(forw) + epsilon)
        norm_org = org / (xp.linalg.norm(org) + epsilon)
        print('... which has a cosine similarity of', xp.matmul(norm_org, norm_forw))
        print(' ')

        # Check most similar word
        cos_mat = xp.matmul(norm_embed_data, norm_forw)
        max_idx = xp.argmax(cos_mat)
        print(unvocab[max_idx], 'with vocab id', max_idx, 'is the most similar word to', unvocab[exx[idxx]])
        print('Similarity is shown alongside two random examples as a sanity check:')
        print(cos_mat[max_idx-1:max_idx+2])
        print(' ')
        
        # Calculate nearest neighbors
        print('Top 5 nearest neighbors w.r.t. cosine similarity:')
        nns = xp.argsort(cos_mat)[::-1]
        for i in nns[:5]:
            print(i, unvocab[i], cos_mat[i])
        print('Bottom 5 nearest neighbors w.r.t. cosine similarity:')
        nns = xp.argsort(cos_mat)
        for i in nns[:5]:
            print(i, unvocab[i], cos_mat[i])