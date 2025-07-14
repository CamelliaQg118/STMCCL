from functools import partial
from STMCCL.layers import *
from utils import permutation
from torch_geometric.nn import global_mean_pool


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        # self.gc1 = GraphConvSparse(input_dim, hidden_dim)
        # self.gc2 = GraphConvSparse(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class STMCCL_module(nn.Module):
    def __init__(
            self,
            input_dim,
            nclass,
            latent_dim=128,
            output_dim=64,
            train_dim=128,
            p_drop=0.2,
            dorp_code=0.2,
            dropout=0.2,
            mask_rate=0.8,
            remask_rate=0.8,
            drop_edge_rate=0.1,
            alpha=0.1,
            num_layers=2,
            eta=1,
            d=64,
            use_bias=False,
            decode_type='GCN',
            edgecode_type='GIN',
            fea_type='GCN',
            remask_method='random',
            use_bn='true',
            device='cuda:0'
    ):
        super(STMCCL_module, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.latent_hidden1 = latent_dim
        self.latent_hidden2 = output_dim
        self.edin_dim = input_dim
        self.edlatent_dim = latent_dim
        self.edout_dim = output_dim
        self.dein_dim = output_dim
        self.deout_dim = latent_dim
        self.embedding_dim = output_dim
        self.cluster_dim = output_dim
        self.in_channels = output_dim
        self.hidden_channels = latent_dim
        self.latent_p =output_dim
        self.train_dim = train_dim
        self.ende_dim = output_dim
        self.mask_dim = output_dim
        self.input_latent = output_dim
        self.edsc_cluster_n = nclass
        self.emb_dim = output_dim*3
        self.embout_dim = output_dim*3

        self.nclass = nclass
        self.dropout = dropout
        self.p_drop = p_drop
        self.dorp_code = dorp_code
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.device = device
        self.use_bn = use_bn
        self.remask_method = remask_method
        self.decode_type = decode_type
        self.edgecode_type = edgecode_type
        self.fea_type = fea_type
        self.g_type = fea_type
        self.alpha = alpha
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.drop_edge_rate = drop_edge_rate
        self.eta = eta
        self.d = d
        self.d_dim = output_dim*3

        self.gcn = GCN(self.input_dim, self.latent_dim, self.output_dim, self.dropout)
        self.encoder = Encodeer_Model(self.input_dim, self.latent_hidden1, self.output_dim, self.p_drop, self.device)
        self.encode_edge = self.Code(self.edgecode_type, self.edin_dim, self.edlatent_dim, self.edout_dim, self.dorp_code)
        self.encode_fea = self.Code(self.fea_type, self.input_latent, self.latent_dim, self.output_dim, self.dorp_code)
        self.decoder = self.Code(self.decode_type, self.dein_dim, self.deout_dim, self.input_dim, self.dorp_code)
        self.encode_generate = self.Code(self.g_type, self.input_dim, self.latent_dim, self.output_dim, self.dorp_code)

        self.projector = nn.Sequential(nn.Linear(self.latent_p, self.train_dim),
                                       nn.PReLU(), nn.Linear(self.train_dim, self.latent_p))
        self.projector_generate = nn.Sequential(nn.Linear(self.latent_p, self.train_dim),
                                                nn.PReLU(), nn.Linear(self.train_dim, self.latent_p))
        self.projector_generate.load_state_dict(self.projector.state_dict())

        self.predictor = nn.Sequential(nn.PReLU(), nn.Linear(self.latent_p, self.latent_p))

        self.loss_type1 = self.setup_loss_fn(loss_fn='sce')
        self.loss_type2 = self.setup_loss_fn(loss_fn='mse')

        self.cluster_layer = Parameter(torch.Tensor(self.nclass, output_dim*3))
        self.D = Parameter(torch.Tensor(self.d_dim, self.edsc_cluster_n))
        # print("D", self.D.size())
        # print("d", self.d)
        torch.nn.init.xavier_normal_(self.cluster_layer)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.input_dim)).to(self.device)
        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.mask_dim)).to(self.device)
        self.reset_parameters_for_token()

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)

    def forward(self, adata, X, adj, edge_index):
        adj, X1, (mask_nodes1, keep_nodes1) = self.encoding_mask_noise(adj, X, self.mask_rate)
        Zf1 = self.encoder(X1)

        adj, X2 = self.fea_permutation(adj, X, adata)

        #permutation
        # adj, x_per = self.permutation(adj, x, adata)
        use_edge_index1, masked_edges1 = dropout_edge(edge_index, self.drop_edge_rate)
        use_edge_index2, masked_edges2 = dropout_edge(edge_index, self.drop_edge_rate)
        
        #representation learning
        Gf1, all_hidden_1 = self.encode_edge(X1, use_edge_index1, return_hidden=True)
        Gf2, all_hidden_2 = self.encode_edge(X2, use_edge_index2, return_hidden=True)
        Hmask = Gf1
        Hcor = Gf2

        #fusing embedding
        H = self.encode_fea(Zf1, adj)
        emb1 = torch.cat([H, (Hmask + Hcor) / 2, Zf1], dim=1).to(self.device)
        linear = nn.Linear(self.emb_dim, self.embout_dim).to(self.device)
        emb = linear(emb1).to(self.device)

        #representation predicting
        with torch.no_grad():
            X_target = self.encode_generate(X, adj)
            x_target = self.projector_generate(X_target[keep_nodes1])
        X_pred = self.projector(H[keep_nodes1])
        x_pred = self.predictor(X_pred)
        loss_latent = sce_loss(x_pred, x_target, 1)

        # # ---- feature reconstruction ----
        H1 = Hmask.clone()
        Hg_rec, _, _ = self.random_remask(adj, H1, self.remask_rate)
        rec1 = self.decoder(Hg_rec, adj)
        x_init1 = X[mask_nodes1]
        x_rec1 = rec1[mask_nodes1]
        loss_rec = self.loss_type1(x_init1, x_rec1) 


        q = 1.0 / ((1.0 + torch.sum((emb.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.alpha))
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)

        # Calculate subspace affinity
        s = None
        for i in range(self.edsc_cluster_n):
            si = torch.sum(torch.pow(torch.mm(emb, self.D[:, i * self.d:(i + 1) * self.d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s + self.eta * self.d) / ((self.eta + 1) * self.d)
        s = (s.t() / torch.sum(s, 1)).t()

        return Hmask, Hcor, emb, q, s, loss_rec, loss_latent

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=3)
        # elif loss_fn == 'cos':
        #     criterion = partial(cos_loss)
        else:
            raise NotImplementedError
        return criterion

    def fea_permutation(self, adj, x, adata):
        x_a = permutation(x)
        use_adj = adj.clone()
        # adata.obsm['x'] = x
        # adata.obsm['x_a'] = x_a
        return use_adj, x_a

    def Code(self, m_type, in_dim, num_hidden, out_dim, dropout) -> nn.Module:
        if m_type == "GCN":
            mod = GCN(in_dim, num_hidden, out_dim, dropout)
        elif m_type == "GIN":
            mod = GIN(in_dim, num_hidden, out_dim, dropout)
        elif m_type == "mlp":
            mod = nn.Sequential(nn.Linear(in_dim, num_hidden * 2), nn.PReLU(), nn.Dropout(0.2), nn.Linear(num_hidden * 2, out_dim))
        elif m_type == "linear":
            mod = nn.Linear(in_dim, out_dim)
        else:
            raise NotImplementedError
        return mod

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()
        return use_adj, out_x, (mask_nodes, keep_nodes)

    def random_remask(self, adj, rep, remask_rate=0.5):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]
        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token
        return rep, remask_nodes, rekeep_nodes


def dropout_edge(edge_index, p=0.5, force_undirected=False):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


class Encodeer_Model(nn.Module):
    def __init__(self, input_dim, intermediate_dim, kan_dim, p_drop, device):
        super(Encodeer_Model, self).__init__()
        self.device = device
        self.full_block = full_block(input_dim, intermediate_dim,  p_drop).to(self.device)
        self.KAN = KANLinear(intermediate_dim, kan_dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.full_block(x)
        feat = self.KAN(x)
        return feat


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )











