import os
import ot
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from sklearn import metrics
from functools import partial

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def adata_hvg(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] ==True]
    sc.pp.scale(adata)
    return adata


def adata_hvg1(adata):
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] ==True]
    sc.pp.scale(adata)
    return adata


def adata_hvg_process(adata):
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.scale(adata)
    return adata


def adata_hvg_slide(adata):
    # sc.pp.filter_genes(adata, min_cells=50)
    # sc.pp.normalize_total(adata, target_sum=1e6)
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    # adata = adata[:, adata.var['highly_variable'] ==True]
    # sc.pp.scale(adata)
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)
    return adata


def load_data(dataset, file_fold):
    if dataset == "DLPFC":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        # print("adata", adata)
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    elif dataset == "Human_Breast_Cancer":
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    elif dataset == "Adult_Mouse_Brain_Section_1":
        adata = sc.read_visium(file_fold, count_file='V1_Adult_Mouse_Brain_Coronal_Section_1_filtered_feature_bc_matrix.h5', load_images=True)
        # print('adata', adata)
        adata.var_names_make_unique()
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    elif dataset == 'MOB':
        savepath = '../Result/MOB_Stereo/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        counts_file = os.path.join(file_fold, 'RNA_counts.tsv')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0).T
        counts.index = [f'Spot_{i}' for i in counts.index]
        adata = sc.AnnData(counts)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()

        pos_file = os.path.join(file_fold, 'position.tsv')
        coor_df = pd.read_csv(pos_file, sep='\t')
        coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.loc[:, ['x', 'y']]
        # print('adata.obs_names', adata.obs_names)
        coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
        adata.obs['x'] = coor_df['x'].tolist()
        adata.obs['y'] = coor_df['y'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        print(adata)

        # hires_image = os.path.join(file_fold, 'crop1.png')
        # adata.uns["spatial"] = {}
        # adata.uns["spatial"][dataset] = {}
        # adata.uns["spatial"][dataset]['images'] = {}
        # adata.uns["spatial"][dataset]['images']['hires'] = imread(hires_image)

        # label_file = pd.read_csv(os.path.join(file_fold, 'Cell_GetExp_gene.txt'), sep='\t', header=None)
        # used_barcode = label_file[0]

        barcode_file = pd.read_csv(os.path.join(file_fold, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = barcode_file[0]
        adata = adata[used_barcode]
        adata.var_names_make_unique()

        adata.obs['total_exp'] = adata.X.sum(axis=1)
        fig, ax = plt.subplots()
        sc.pl.spatial(adata, color='total_exp', spot_size=40, show=False, ax=ax)
        ax.invert_yaxis()
        plt.savefig(savepath + 'STMCCL_stereo_MOB1.jpg', dpi=600)

        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)
        # print("adata是否降维", adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))

    elif dataset == 'MOB_V2':
        savepath = '../Result/MOB_Slide/_1/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        # counts_file = os.path.join(input_dir, '')
        # coor_file = os.path.join(input_dir, '')
        counts_file = os.path.join(file_fold, 'Puck_200127_15.digital_expression.txt')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        adata = sc.AnnData(counts.T)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()
        print(adata)

        coor_file = os.path.join(file_fold, 'Puck_200127_15_bead_locations.csv')
        coor_df = pd.read_csv(coor_file, index_col=0)
        # coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.set_index('barcode')
        coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
        adata.obs['x'] = coor_df['xcoord'].tolist()
        adata.obs['y'] = coor_df['ycoord'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        sc.pp.calculate_qc_metrics(adata, inplace=True)

        # print("adata", adata)
        # plt.rcParams["figure.figsize"] = (6, 5)
        # # Original tissue area, some scattered spots
        # sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", s=6, show=False, save='_MOB01_slide.png')
        # plt.title('')
        # plt.axis('off')
        # plt.savefig(savepath + 'STNMAE_MOBV2.jpg', dpi=300)

        barcode_file = pd.read_csv(os.path.join(file_fold, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = barcode_file[0]
        adata = adata[used_barcode]
        adata.var_names_make_unique()
        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_process(adata)
        # print("adata是否降维", adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))
    elif dataset == 'hip':
        savepath = '../Result/hip/_1/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        # counts_file = os.path.join(input_dir, '')
        # coor_file = os.path.join(input_dir, '')
        counts_file = os.path.join(file_fold, 'Puck_200115_08.digital_expression.txt')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        adata = sc.AnnData(counts.T)
        adata.X = csr_matrix(adata.X, dtype=np.float32)
        adata.var_names_make_unique()
        print(adata)

        coor_file = os.path.join(file_fold, 'Puck_200115_08_bead_locations.csv')
        coor_df = pd.read_csv(coor_file, index_col=0)
        # coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.set_index('barcode')
        coor_df = coor_df.loc[adata.obs_names, ['xcoord', 'ycoord']]
        adata.obs['x'] = coor_df['xcoord'].tolist()
        adata.obs['y'] = coor_df['ycoord'].tolist()
        adata.obsm["spatial"] = coor_df.to_numpy()
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        print(adata)
        plt.rcParams["figure.figsize"] = (6, 5)
        sc.pl.embedding(adata, basis="spatial", color="log1p_total_counts", s=6, show=False)
        plt.title('prime')
        plt.axis('off')
        plt.savefig(savepath + 'STMCCL_hip_1.jpg', dpi=600)

        adata.layers['count'] = adata.X.toarray()
        adata = adata_hvg_slide(adata)
        # print("adata是否降维", adata)
        print("adata", adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
        adata_X = torch.FloatTensor(np.array(adata_X))


    elif dataset == 'ISH':
        adata = sc.read(file_fold + '/STARmap_20180505_BY3_1k.h5ad')
        # print(adata)
        adata.obs['x'] = adata.obs["X"]
        adata.obs['y'] = adata.obs["Y"]
        adata.layers['count'] = adata.X
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X
    else:
        platform = '10X'
        file_fold = os.path.join('../Data', platform, dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5')
        adata.obs['x'] = adata.obs["array_row"]
        adata.obs['y'] = adata.obs["array_col"]
        df_meta = pd.read_csv(os.path.join('../Data', dataset,  'metadata.tsv'), sep='\t', header=None, index_col=0)
        adata.obs['layer_guess'] = df_meta['layer_guess']
        df_meta.columns = ['over', 'ground_truth']
        adata.obs['ground_truth'] = df_meta.iloc[:, 1]#同理获取获取数据

        adata.var_names_make_unique()
        adata = adata_hvg(adata)
        sc.pp.scale(adata)
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

    return adata, adata_X


def label_process_DLPFC(adata, df_meta):
    labels = df_meta["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground.replace('WM', '0', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    return adata


def label_process_HBC(adata, df_meta):
    labels = df_meta["ground_truth"].copy()
    # print("labels", labels)
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()
    ground = ground.replace('DCIS/LCIS_1', '0')
    ground = ground.replace('DCIS/LCIS_2', '1')
    ground = ground.replace('DCIS/LCIS_4', '2')
    ground = ground.replace('DCIS/LCIS_5', '3')
    ground = ground.replace('Healthy_1', '4')
    ground = ground.replace('Healthy_2', '5')
    ground = ground.replace('IDC_1', '6')
    ground = ground.replace('IDC_2', '7')
    ground = ground.replace('IDC_3', '8')
    ground = ground.replace('IDC_4', '9')
    ground = ground.replace('IDC_5', '10')
    ground = ground.replace('IDC_6', '11')
    ground = ground.replace('IDC_7', '12')
    ground = ground.replace('IDC_8', '13')
    ground = ground.replace('Tumor_edge_1', '14')
    ground = ground.replace('Tumor_edge_2', '15')
    ground = ground.replace('Tumor_edge_3', '16')
    ground = ground.replace('Tumor_edge_4', '17')
    ground = ground.replace('Tumor_edge_5', '18')
    ground = ground.replace('Tumor_edge_6', '19')
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground.values.astype(int)
    # print("ground", adata.obs['ground'])
    return adata


def graph_build(adata, adata_X, dataset):
    if dataset == 'DLPFC':
        n = 12
        adj, edge_index = load_adj(adata, n)
        adj2 = load_adj2(adata, n)
        smooth_fea = csr_matrix(adata.obsm['X_pca']).toarray()  # 平滑特征为稀疏处理后的特征矩阵
        smooth_fea = adj2.dot(smooth_fea)
        smooth_fea = torch.FloatTensor(smooth_fea)

    elif dataset == 'MBO':
        n = 10
        adj, edge_index = load_adj(adata, n)
        adj2 = load_adj2(adata, n)
        smooth_fea = csr_matrix(adata.obsm['X_pca']).toarray()  # 平滑特征为稀疏处理后的特征矩阵
        smooth_fea = adj2.dot(smooth_fea)
        smooth_fea = torch.FloatTensor(smooth_fea)
    elif dataset =='MOB_V2':
        n = 7
        adj, edge_index = load_adj(adata, n)
        adj2 = load_adj2(adata, n)
        smooth_fea = csr_matrix(adata.obsm['X_pca']).toarray()  # 平滑特征为稀疏处理后的特征矩阵
        smooth_fea = adj2.dot(smooth_fea)
        smooth_fea = torch.FloatTensor(smooth_fea)

    elif dataset == 'hip':
        n = 50
        adj, edge_index = load_adj(adata, n)
        adj2 = load_adj2(adata, n)
        smooth_fea = csr_matrix(adata.obsm['X_pca']).toarray()  # 平滑特征为稀疏处理后的特征矩阵
        smooth_fea = adj2.dot(smooth_fea)
        smooth_fea = torch.FloatTensor(smooth_fea)

    elif dataset == 'Adult_Mouse_Brain_Section_1':
        n = 5
        adj, edge_index = load_adj(adata, n)
        adj2 = load_adj2(adata, n)
        smooth_fea = csr_matrix(adata.obsm['X_pca']).toarray()  # 平滑特征为稀疏处理后的特征矩阵
        smooth_fea = adj2.dot(smooth_fea)
        smooth_fea = torch.FloatTensor(smooth_fea)

    elif dataset == 'ISH':
        n = 7
        adj, edge_index= load_adj(adata, n)
        adj2 = load_adj2(adata, n)
        smooth_fea = csr_matrix(adata.obsm['X_pca']).toarray()  # 平滑特征为稀疏处理后的特征矩阵
        smooth_fea = adj2.dot(smooth_fea)
        smooth_fea = torch.FloatTensor(smooth_fea)

    else:
        n = 10
        adj, edge_index = load_adj(adata, n)
        adj2 = load_adj2(adata, n)
        smooth_fea = csr_matrix(adata.obsm['X_pca']).toarray()  # 平滑特征为稀疏处理后的特征矩阵
        smooth_fea = adj2.dot(smooth_fea)
        smooth_fea = torch.FloatTensor(smooth_fea)

    return adata, adj, edge_index, smooth_fea


def load_adj(adata, n):
    adj = generate_adj(adata, include_self=False, n=n)
    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj', adj)
    # edge_index = adj_to_edge_index(adj)
    adj_norm, edge_index = preprocess_adj(adj)
    return adj_norm, edge_index


def adj_to_edge_index(adj):
    dense_adj = adj.toarray()
    edge_index = torch.nonzero(torch.tensor(dense_adj), as_tuple=False).t()
    return edge_index


def load_adj2(adata, n):
    adj = generate_adj2(adata, include_self=True)
    adj = sp.coo_matrix(adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # print('adj_norm', adj)
    return adj


def generate_adj(adata, include_self=False, n=6):
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    adj = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj[i, n_neighbors] = 1
    if not include_self:
        x, y = np.diag_indices_from(adj)
        adj[x, y] = 0
    adj = adj + adj.T
    adj = adj > 0
    adj = adj.astype(np.int64)
    return adj


def preprocess_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    # edge_index = adj_to_edge_index(adj)
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    edge_index = adj_to_edge_index(adj_normalized)

    return sparse_mx_to_torch_sparse_tensor(adj_normalized), edge_index


def generate_adj2(adata, include_self=True):
    dist = metrics.pairwise_distances(adata.obsm['spatial'])
    dist = dist / np.max(dist)
    adj = dist.copy()
    if not include_self:
        np.fill_diagonal(adj, 0)
    # print('adj', adj)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated


def refine_label(adata, radius=50, key='label'):  # 修正函数相当于强制修正，
    # 功能，使得每个spot半径小于50的范围内，其他spot 的大部分是哪一类就把这个spot 强制归为这一类。
    n_neigh = radius  # 定义半径
    new_type = []  # spot新的类型
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')  # 用欧氏距离

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def Initialization_D(Z, y_pred, n_clusters, d):  # 对包含子空间基的矩阵进行初始化
    Z_seperate= seperate(Z, y_pred, n_clusters)  # 调用低维嵌入划分函数，将z划分到不同类中
    Z_full = None
    U = np.zeros([Z.shape[1], n_clusters * d])  # 初始化
    print("Initialize D")
    for i in range(n_clusters):
        Z_seperate[i] = np.array(Z_seperate[i])
        u, ss, v = np.linalg.svd(Z_seperate[i].transpose())
        U[:, i * d:(i + 1) * d] = u[:, 0:d]
    D = U
    print("Shape of D: ", D.transpose().shape)
    print("Initialization of D Finished")
    return D


def seperate(Z, y_pred, n_clusters):
    n, d = Z.shape[0], Z.shape[1]
    Z_seperate = defaultdict(list)
    Z_new = np.zeros([n, d])
    for i in range(n_clusters):
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                Z_seperate[i].append(Z[j])
                Z_new[j][:] = Z[j]
    return Z_seperate


def refined_subspace_affinity(s):#函数对亲和矩阵进行归一化并返回加权后的结果
    weight = s**2 / s.sum(0)#将子空间亲和度矩阵作为输入
    return (weight.T / weight.sum(1)).T


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def fix_seed(seed):
    import random
    import torch
    from torch.backends import cudnn
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'



















