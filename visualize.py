import numpy as np
import matplotlib as mpl
_ = mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os
import pickle

def plotter_helper(df, annot_nn, md, data, folder, norms=False):
    # vmin = math.floor(df.values.min())
    # vmin = df.values.min()
    vmin = 0.0
    # vmax = math.ceil(df.values.max())
    vmax = df.values.max()
    seqlen = data['emb_cos_nn'].size

    size_h = seqlen / 4.0
    size_w = 8

    fig = plt.figure(figsize=(size_w, size_h))
    if norms:
        main_plot = sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', vmin=vmin, vmax=vmax, cbar=False)
        # main_plot = sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    else:
        main_plot = sns.heatmap(df, annot=annot_nn, fmt='', cmap='YlOrRd', vmin=vmin, vmax=vmax, cbar=False)
        # main_plot = sns.heatmap(df, annot=annot_nn, fmt='', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    main_plot.set_xticklabels(main_plot.get_xticklabels(), rotation=0)

    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=mpl.colors.Normalize(vmin=vmin,vmax=vmax))
    cbaxes = fig.add_axes([0.97, 0.1, 0.03, 0.8])
    sm.set_array([])
    cb = plt.colorbar(sm ,cbaxes)

    ax2 = main_plot.twinx()
    ax2.set_ylim((-0.5,seqlen-0.5))
    ax2.set_yticks(np.arange(0.0,seqlen,1.0))
    ax2.invert_yaxis()
    ax2.set_yticklabels(np.around(data['cos_emb_adv'], 3))

    if norms:
        plt.title(r'Norms, $\epsilon = $' + md['adv_k'] + 'x' + md['epsilon'] + ' ' + md['label'])
        plt.savefig(os.path.join(folder, md['name'] + '_adv_k' + md['adv_k'] + '_eps' + md['epsilon'] + '_norms.png'), bbox_inches='tight')
    else:
        plt.title(r'Nearest neighbors, $\epsilon = $' + md['adv_k'] + 'x' + md['epsilon'] + ' ' + md['label'])
        plt.savefig(os.path.join(folder, md['name'] + '_adv_k' + md['adv_k'] + '_eps' + md['epsilon'] + '_nns.png'), bbox_inches='tight')
    plt.clf()

def create_plots(data, metadata, folder='temp', save_norms=True):
    # General stuff
    md = metadata
    annot_nn = np.stack([data['emb_cos_nn'], data['per_cos_nn'], data['adv_cos_nn']], axis=1)
    df = pd.DataFrame({'original': data['emb_l2_norm'], 'perturbation': data['per_l2_norm'], 'adversarial': data['adv_l2_norm']},
        index=np.arange(data['emb_l2_norm'].size))
    df = df[['original', 'perturbation', 'adversarial']]

    plotter_helper(df, annot_nn, md, data, folder)
    plotter_helper(df, annot_nn, md, data, folder, save_norms)

if __name__ == "__main__":
    pik_file = 'example_adv_data.pickle'
    with open(os.path.join('./', pik_file), 'rb') as handle:
        data = pickle.load(handle)
        metadata = pickle.load(handle)
    metadata['name'] = 'example'
    create_plots(data, metadata, folder='out_imdb', save_norms=True)