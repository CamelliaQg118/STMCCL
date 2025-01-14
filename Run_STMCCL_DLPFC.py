import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import metrics
import scipy as sp
import numpy as np
import torch
import copy
import os
import STMCCL
import warnings
import utils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


ARI_list = []
random_seed = 42
STMCCL.fix_seed(random_seed)
os.environ['R_HOME'] = 'D:/R/R-4.3.3/R-4.3.3'
os.environ['R_USER'] = 'D:/Anaconda3/Anaconda3202303/envs/STMCCL/Lib/site-packages/rpy2'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

dataset = 'DLPFC'
slice = '151674'
platform = '10X'
file_fold = os.path.join('../Data', platform, dataset, slice)
adata, adata_X = utils.load_data(dataset, file_fold)#加载特征及标签
df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
adata = utils.label_process_DLPFC(adata, df_meta)

savepath = '../Result/STMCCL/DLPFC/test1/' + str(slice) + '_4/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
n_clusters = 5 if dataset in ['151669', '151670', '151671', '151672'] else 7

adata, adj, edge_index, smooth_fea = utils.graph_build(adata, adata_X, dataset)
# print('adj', adj)
# print('edge_index', edge_index)
# print('adj_np', adj_np)

stmccl_net = STMCCL.stmccl(adata.obsm['X_pca'], adata, adj, edge_index, smooth_fea, n_clusters, dataset, device=device)
tool = None

if tool == 'mclust':
    emb = stmccl_net.train()
    adata.obsm['STMCCL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STMCCL.mclust_R(adata, n_clusters, use_rep='STMCCL', key_added='STMCCL', random_seed=random_seed)
elif tool == 'leiden':
    emb = stmccl_net.train()
    adata.obsm['STMCCL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STMCCL.leiden(adata, n_clusters, use_rep='STMCCL', key_added='STMCCL', random_seed=random_seed)
elif tool == 'louvain':
    emb = stmccl_net.train()
    adata.obsm['STMCCL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    STMCCL.louvain(adata, n_clusters, use_rep='STMCCL', key_added='STMCCL', random_seed=random_seed)
else:
    emb, idx = stmccl_net.train()
    print("emb", emb)
    adata.obs['STMCCL'] = idx
    adata.obsm['STMCCL'] = emb
    adata.obs['ground_truth'] = df_meta['layer_guess']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

print("adata", adata)

new_type = utils.refine_label(adata, radius=10, key='STMCCL')
adata.obs['STMCCL'] = new_type
ARI = metrics.adjusted_rand_score(adata.obs['ground_truth'], adata.obs['STMCCL'])
NMI = metrics.normalized_mutual_info_score(adata.obs['ground_truth'], adata.obs['STMCCL'])
adata.uns["ARI"] = ARI
adata.uns["NMI"] = NMI
print('===== Project: {}_{} ARI score: {:.4f}'.format(str(dataset), str(slice), ARI))
print('===== Project: {}_{} NMI score: {:.4f}'.format(str(dataset), str(slice), NMI))
print(str(slice))
print(n_clusters)
ARI_list.append(ARI)

#绘制基本图
plt.rcParams["figure.figsize"] = (3, 3)
title = "Manual annotation (" + dataset + "#" + slice + ")"
sc.pl.spatial(adata, img_key="hires", color=['ground_truth'], title=title, show=False)
plt.savefig(savepath + 'Manual Annotation.jpg', bbox_inches='tight', dpi=300)
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 4))
sc.pl.spatial(adata, color='ground_truth', ax=axes[0], show=False)
sc.pl.spatial(adata, color=['STMCCL'], ax=axes[1], show=False)
axes[0].set_title("Manual annotation (" + dataset + "#" + slice + ")")
axes[1].set_title('STMCCL_Clustering: (ARI=%.4f)' % ARI)

# 将两张图片拼接起来
plt.subplots_adjust(wspace=0.5)  # 增加两幅图之间的水平间距
plt.subplots_adjust(hspace=0.5)  # 增加两幅图之间的垂直间距,需放在保存图片前否则没有效果
plt.savefig(savepath + 'STMCCL.jpg', dpi=300)  # 存储图片注意要在plot前面


sc.pp.neighbors(adata, use_rep='STMCCL', metric='cosine')
sc.tl.umap(adata)
sc.pl.umap(adata, color='STMCCL', title='STMCCL', show=False)
plt.savefig(savepath + 'umap.jpg', bbox_inches='tight', dpi=300)

for ax in axes:
    ax.set_aspect(1)
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)


title = 'STMCCL:{}_{} ARI={:.4f} NMI={:.4f}'.format(str(dataset), str(slice), adata.uns['ARI'], adata.uns['NMI'])
sc.pl.spatial(adata, img_key="hires", color=['STMCCL'], title=title, show=False)
plt.savefig(savepath + 'STMCCL_NMI_ARI.tif', bbox_inches='tight', dpi=300)

plt.rcParams["figure.figsize"] = (3, 3)
sc.tl.paga(adata, groups='STMCCL')
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2, show=False)
plt.savefig(savepath + 'STMCCL_PAGA_domain.tif', bbox_inches='tight', dpi=300)

sc.tl.paga(adata, groups='ground_truth')
sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=title, legend_fontoutline=2, show=False)
plt.savefig(savepath + 'STMCCL_PAGA_ground_truth.tif', bbox_inches='tight', dpi=300)
