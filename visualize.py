import numpy as np
import matplotlib as mpl
_ = mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os

def get_sample_data():
    emb_cos_nn = 'The acrobatics mixed with haunting music make one spectacular show The costumes are vibrant and the performances will just boggle your mind Simply amazing <eos>'
    adv_cos_nn = 'The acrobatics mixed with haunting music make one spectacular show The costumes are vibrant and the performances will just boggle your mind Simply amazing <eos>'
    per_cos_nn = 'boring worst worst bad REASON boring bad worst Work boring Redline hoping Jamon Hayek hoe korean waaayyyy <eos> persevered Lemurians aforesaid rhea blinkered lurks Owain'
    emb_norm = np.asarray([14.54380989, 16.79619217, 16.74661064, 15.71444988, 16.02031708, 14.22260571, 
    16.02468872, 17.50171471, 15.14425087, 15.76819897, 14.54380989, 17.31795311, 
    15.84649944, 16.0454464, 16.64375877, 16.30159569, 16.69027901, 16.3413887, 
    16.00512886, 16.42050171, 16.24193573, 16.22417259, 15.48325539, 16.31344986, 
    15.53515816])
    adv_norm = np.asarray([14.59287834, 16.85625458, 16.9739933, 15.72620773, 15.94856358, 14.32188797, 
    16.04281044, 17.51702118, 15.10220909, 15.84380054, 14.54627991, 17.33802986, 
    15.87647247, 16.0267849, 16.5944252, 16.31761169, 16.65192032, 16.32099915, 
    15.95104218, 16.46372032, 16.21652031, 16.23713493, 15.54756737, 16.30161285, 
    15.57370663])
    per_norm = np.asarray([0.95932889, 1.61985123, 2.32326317, 0.63780189, 1.04259896, 1.23247135, 
    0.70824063, 0.32651392, 0.51064724, 0.6613425, 0.32552367, 0.50906086, 
    0.45659554, 0.791655, 0.394712, 0.25209203, 0.70917529, 0.64503556, 
    0.64334869, 1.01625991, 0.59936994, 0.76404864, 0.91365868, 0.81399775, 
    0.26255974])
    return emb_cos_nn, adv_cos_nn, per_cos_nn, emb_norm, adv_norm, per_norm

def process_data(data):

    emb_cos_nn, adv_cos_nn, per_cos_nn, emb_norm, adv_norm, per_norm = data
    if type(emb_cos_nn) == str:
        emb_cos_nn, adv_cos_nn, per_cos_nn = emb_cos_nn.split(' '), adv_cos_nn.split(' '), per_cos_nn.split(' ')
    if type(emb_norm) == list:
        emb_norm, adv_norm, per_norm = np.asarray(emb_norm), np.asarray(adv_norm), np.asarray(per_norm)
    else:
        emb_norm, adv_norm, per_norm = np.asarray(emb_norm.tolist()), np.asarray(adv_norm.tolist()), np.asarray(per_norm.tolist())
    emb_cos_nn, adv_cos_nn, per_cos_nn = np.asarray(emb_cos_nn), np.asarray(adv_cos_nn), np.asarray(per_cos_nn)
    return emb_cos_nn, adv_cos_nn, per_cos_nn, emb_norm, adv_norm, per_norm

def create_plots(data, metadata, folder='temp'):
    md = metadata
    emb_cos_nn, adv_cos_nn, per_cos_nn, emb_norm, adv_norm, per_norm = process_data(data)
    annot_nn = np.stack([emb_cos_nn, per_cos_nn, adv_cos_nn], axis=1)
    df = pd.DataFrame({'original': emb_norm, 'perturbation': per_norm, 'adversarial': adv_norm}, index=np.arange(emb_norm.size))
    df = df[['original', 'perturbation', 'adversarial']]
    # vmin = math.floor(df.values.min())
    # vmax = math.ceil(df.values.max())
    vmin = df.values.min()
    vmax = df.values.max()

    size_h = emb_cos_nn.size / 4.0
    size_w = 8

    plt.figure(figsize=(size_w, size_h))
    nn_plot = sns.heatmap(df, annot=annot_nn, fmt='', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    nn_plot.set_xticklabels(nn_plot.get_xticklabels(), rotation=0)
    plt.title(r'Nearest neighbors, $\epsilon = $' + md['adv_k'] + 'x' + md['epsilon'] + ' ' + md['label'])
    plt.savefig(os.path.join(folder, md['name'] + '_adv_k' + md['adv_k'] + '_eps' + md['epsilon'] + '_nns.png'))
    plt.clf()

    plt.figure(figsize=(size_w, size_h))
    norm_plot = sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', vmin=vmin, vmax=vmax)
    norm_plot.set_xticklabels(norm_plot.get_xticklabels(), rotation=0)
    plt.title(r'Norms, $\epsilon = $' + md['adv_k'] + 'x' + md['epsilon'] + ' ' + md['label'])
    plt.savefig(os.path.join(folder, md['name'] + '_adv_k' + md['adv_k'] + '_eps' + md['epsilon'] + '_norms.png'))
    plt.clf()

def main():
    data = get_sample_data()
    dummy = {'name': 'dummy', 'epsilon': '42', 'label': '1to0', 'adv_k': '1'}
    create_plots(data, dummy)

if __name__ == '__main__':
    main()