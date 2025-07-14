import torch.nn.modules.loss
import torch.nn.functional as F
from kmeans_gpu import kmeans_1
from STMCCL.STMCCL_module import STMCCL_module
from tqdm import tqdm
from utils import *
import STMCCL
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
cudnn.deterministic = True
cudnn.benchmark = True


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


class D_constraint1(torch.nn.Module):
    def __init__(self):
        super(D_constraint1, self).__init__()

    def forward(self, d):
        I = torch.eye(d.shape[1]).cuda()
        loss_d1_constraint = torch.norm(torch.mm(d.t(), d) * I - I)
        return 1e-3 * loss_d1_constraint


class D_constraint2(torch.nn.Module):

    def __init__(self):
        super(D_constraint2, self).__init__()

    def forward(self, d, dim, n_clusters):
        S = torch.ones(d.shape[1], d.shape[1]).cuda()
        zero = torch.zeros(dim, dim)
        # print("s", S.size())
        # print("zero", zero.size())

        for i in range(n_clusters):
            # print(f"dim: {dim}, n_clusters: {n_clusters}, d.shape: {d.shape}")
            S[i * dim:(i + 1) * dim, i * dim:(i + 1) * dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d.t(), d) * S)
        return 1e-3 * loss_d2_constraint


class stmccl:
    def __init__(
            self,
            X,
            adata,
            adj,
            edge_index,
            smooth_fea,
            n_clusters,
            dataset,
            rec_w=11,
            latent_w=1,
            cosl_w=1,
            d_w=1,
            kl_w=5,
            dec_tol=0.00,
            threshold=0.5,
            epochs=800,
            dec_interval=3,
            lr=0.0001,
            decay=0.0001,
            device='cuda:0',
    ):
        self.random_seed = 42
        STMCCL.fix_seed(self.random_seed)

        self.n_clusters = n_clusters
        # self.pos_w = pos_w
        # self.neg_w = neg_w
        self.cosl_w = cosl_w
        self.rec_w = rec_w
        # self.cos_w = cos_w
        self.latent_w = latent_w
        self.d_w = d_w
        self.kl_w =kl_w
        self.device = device
        self.dec_tol = dec_tol
        self.threshold = threshold 

        self.adata = adata.copy()
        self.dataset = dataset
        self.cell_num = len(X)
        self.epochs = epochs
        self.dec_interval = dec_interval
        self.learning_rate = lr
        self.weight_decay = decay
        self.adata = adata.copy()
        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]
        self.adj = adj.to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.smooth_fea = torch.FloatTensor(smooth_fea).to(self.device)

        self.model = STMCCL_module(self.input_dim, self.n_clusters).to(self.device)

    def train(self, dec_tol=0.00):
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        # self.model.train()
        # hg, hg_neg, emb, tmp_q, s = self.model_eval()
        hg, hg_neg, emb, q, s = self.model_eval()
        predict_labels, dis, initial = kmeans_1(self.X, self.n_clusters, distance="euclidean", device=self.device)
        expected_output = torch.eye(self.smooth_fea.shape[0]).to(self.device)

        kmeans = KMeans(n_clusters=self.model.edsc_cluster_n, n_init=self.model.edsc_cluster_n * 2, random_state=42)
        y_pred_last = np.copy(kmeans.fit_predict(emb))
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        # Initialize D
        D = Initialization_D(emb, y_pred_last, self.model.edsc_cluster_n, self.model.d)
        D = torch.tensor(D).to(torch.float32) 
        self.model.D.data = D.to(self.device)

        self.model.train()
        list_rec = []
        list_latent = []
        list_cos = []
        list_d = []
        list_kl = []
        list_cosl = []
        list_pos = []
        list_neg = []
        epoch_max = 0
        ari_max = 0
        idx_max = []
        emb_max = []

        if self.dataset in ['Human_Breast_Cancer', 'DLPFC']:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                self.optimizer.zero_grad()
                if epoch % self.dec_interval == 0:
                    hm, hcor, emb, tmp_q, tmp_s = self.model_eval()
                    # tmp_total = np.maximum(tmp_s, tmp_q)
                    s_tilde = refined_subspace_affinity(tmp_s)
                    tmp_p = target_distribution(torch.Tensor(tmp_q))
                    y_pred = tmp_p.cpu().numpy().argmax(1)

                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                if epoch > 50:
                    high_confidence = torch.min(dis, dim=1).values  
                    threshold = torch.sort(high_confidence).values[int(len(high_confidence) * self.threshold)]
                    high_confidence_idx = np.argwhere(high_confidence < threshold)[0] 

                    # pos samples
                    index = torch.tensor(range(self.smooth_fea.shape[0]), device=self.device)[high_confidence_idx]
                    y_sam = torch.tensor(predict_labels, device=self.device)[high_confidence_idx]  # sample high confidence prediction labels

                    index = index[torch.argsort(y_sam)] 
                    class_num = {}  
                    # print("class_nu1", class_num)
                    for idx, label in enumerate(torch.sort(y_sam).values):
                        label = label.item()
                        # print(f"Iteration {idx}: label = {label}")
                        if label in class_num:
                            class_num[label] += 1
                        else:
                            class_num[label] = 1

                    key = sorted(class_num.keys())
                    if len(class_num) < 2:
                        continue
                    pos_contrastive = 0
                    centers_1 = torch.tensor([], device=self.device)
                    centers_2 = torch.tensor([], device=self.device)
                    for i in range(len(key[:-1])):
                        class_num[key[i + 1]] = class_num[key[i]] + class_num[
                            key[i + 1]]
                        now = index[class_num[key[i]]:class_num[key[i + 1]]]
                        pos_embed_1 = hg[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)),
                                                          replace=False)]
                        pos_embed_2 = hg_neg[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]

                        pos_embed_1 = torch.tensor(pos_embed_1, device=self.device)
                        pos_embed_2 = torch.tensor(pos_embed_2, device=self.device)
                        pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()  

                        hg = torch.tensor(hg, device=self.device)
                        hg_neg = torch.tensor(hg_neg, device=self.device)
                        centers_1 = torch.cat([centers_1, torch.mean(hg[now], dim=0).unsqueeze(0)], dim=0)
                        centers_2 = torch.cat([centers_2, torch.mean(hg_neg[now], dim=0).unsqueeze(0)], dim=0)

                    pos_contrastive = -(pos_contrastive / self.n_clusters) 
                    if pos_contrastive == 0: 
                        continue
                    if len(class_num) < 2:
                        loss_col = pos_contrastive  
                    else:
                        centers_1 = F.normalize(centers_1, dim=1, p=2)
                        centers_2 = F.normalize(centers_2, dim=1, p=2)  
                        S = centers_1 @ centers_2.T  
                        S_diag = torch.diag_embed(torch.diag(S))  
                        S = S - S_diag 
                        neg_contrastive = F.mse_loss(S, torch.zeros_like(S))  
                        loss_cosl = pos_contrastive+neg_contrastive  

                    list_pos.append(pos_contrastive.detach().cpu().numpy())
                    list_neg.append(neg_contrastive.detach().cpu().numpy())
                    # print('loss_pos = {:.5f}'.format(pos_contrastive), ' loss_neg = {:.5f}'.format(neg_contrastive))

                else:  
                    S = hg @ hg_neg.T  
                    S = torch.tensor(S, device=self.device)
                    loss_cosl = F.mse_loss(S, expected_output)

                torch.set_grad_enabled(True)
                _, _, _, q, s, loss_rec, loss_latent = self.model(self.adata, self.X, self.adj, self.edge_index)
                d_cons1 = D_constraint1()  # 实例化约束1
                d_cons2 = D_constraint2()  # 实例化约束2
                loss_d1 = d_cons1(self.model.D)
                loss_d2 = d_cons2(self.model.D, self.model.d, self.model.edsc_cluster_n)
                loss_d = loss_d1 + loss_d2

                loss_kl_q = F.kl_div(q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                loss_kl_s = F.kl_div(s.log(), torch.tensor(s_tilde).to(self.device)).to(self.device)
                loss_kl = loss_kl_s + loss_kl_q

                loss_tatal = self.rec_w * loss_rec + self.latent_w * loss_latent + self.cosl_w * loss_cosl + self.kl_w * loss_kl + \
                             self.d_w * loss_d

                loss_tatal.backward()
                self.optimizer.step()

                list_rec.append(loss_rec.detach().cpu().numpy())
                list_latent.append(loss_latent.detach().cpu().numpy())
                list_cos.append(loss_cos.detach().cpu().numpy())
                list_d.append(loss_d.detach().cpu().numpy())
                list_kl.append(loss_kl.detach().cpu().numpy())
                list_cosl.append(loss_cosl.detach().cpu().numpy())
                # print('loss_rec = {:.5f}'.format(loss_rec), #'loss_cos= {:.5f}'.format(loss_cos),
                #       'loss_latent = {:.5f}'.format(loss_latent), 'loss_d= {:.5f}'.format(loss_d),
                #       'loss_kl = {:.5f}'.format(loss_kl), 'loss_cosl = {:.5f}'.format(loss_cosl),
                #       ' loss_total = {:.5f}'.format(loss_tatal))

                kmeans = KMeans(n_clusters=self.n_clusters).fit(emb)
                idx = kmeans.labels_
                self.adata.obsm['STMCCL'] = emb
                labels = self.adata.obs['ground']
                labels = pd.to_numeric(labels, errors='coerce')
                labels = pd.Series(labels).fillna(0).to_numpy()
                idx = pd.Series(idx).fillna(0).to_numpy()
                ari_res = metrics.adjusted_rand_score(labels, idx)
                if ari_res > ari_max:
                    ari_max = ari_res
                    epoch_max = epoch
                    idx_max = idx
                    emb_max = emb
            print("epoch_max", epoch_max)
            print("ARI=======", ari_max)
            nmi_res = metrics.normalized_mutual_info_score(labels, idx_max)
            print("NMI=======", nmi_res)
            self.adata.obs['STMCCL'] = idx_max.astype(str)
            self.adata.obsm['emb'] = emb_max
            return self.adata.obsm['emb'], self.adata.obs['STMCCL']
        else:
            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                self.optimizer.zero_grad()
                if epoch % self.dec_interval == 0:
                    # hg, hg_neg, emb, tmp_q, tmp_s = self.model_eval()
                    hm, hcor, emb, tmp_q, tmp_s = self.model_eval()
                    # tmp_total = np.maximum(tmp_s, tmp_q)
                    s_tilde = refined_subspace_affinity(tmp_s)
                    tmp_p = target_distribution(torch.Tensor(tmp_q))
                    y_pred = tmp_p.cpu().numpy().argmax(1)

                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    self.model.train()
                    if epoch > 0 and delta_label < self.dec_tol:
                        print('delta_label {:.4}'.format(delta_label), '< tol', self.dec_tol)
                        print('Reached tolerance threshold. Stopping training.')
                        break

                if epoch > 50:
                    high_confidence = torch.min(dis, dim=1).values 
                    threshold = torch.sort(high_confidence).values[int(len(high_confidence) * self.threshold)]
                    high_confidence_idx = np.argwhere(high_confidence < threshold)[0]

                    # pos samples
                    index = torch.tensor(range(self.smooth_fea.shape[0]), device=self.device)[high_confidence_idx]
                    y_sam = torch.tensor(predict_labels, device=self.device)[high_confidence_idx]  

                    index = index[torch.argsort(y_sam)] 
                    class_num = {}  
                    # print("class_nu1", class_num)
                    for idx, label in enumerate(torch.sort(y_sam).values):
                        label = label.item()
                        # print(f"Iteration {idx}: label = {label}")
                        if label in class_num:
                            class_num[label] += 1
                        else:
                            class_num[label] = 1

                    key = sorted(class_num.keys())
                    if len(class_num) < 2:
                        continue
                    pos_contrastive = 0
                    centers_1 = torch.tensor([], device=self.device)
                    centers_2 = torch.tensor([], device=self.device)
                    for i in range(len(key[:-1])):
                        class_num[key[i + 1]] = class_num[key[i]] + class_num[
                            key[i + 1]]
                        now = index[class_num[key[i]]:class_num[key[i + 1]]]
                        pos_embed_1 = hg[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)),
                                                          replace=False)]
                        pos_embed_2 = hg_neg[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]

                        pos_embed_1 = torch.tensor(pos_embed_1, device=self.device)
                        pos_embed_2 = torch.tensor(pos_embed_2, device=self.device)
                        pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum() 

                        hg = torch.tensor(hg, device=self.device)
                        hg_neg = torch.tensor(hg_neg, device=self.device)
                        centers_1 = torch.cat([centers_1, torch.mean(hg[now], dim=0).unsqueeze(0)], dim=0)
                        centers_2 = torch.cat([centers_2, torch.mean(hg_neg[now], dim=0).unsqueeze(0)], dim=0)

                    pos_contrastive = -(pos_contrastive / self.n_clusters) 
                    if pos_contrastive == 0: 
                        continue
                    if len(class_num) < 2:
                        loss_col = pos_contrastive  
                    else:
                        centers_1 = F.normalize(centers_1, dim=1, p=2)
                        centers_2 = F.normalize(centers_2, dim=1, p=2) 
                        S = centers_1 @ centers_2.T  
                        S_diag = torch.diag_embed(torch.diag(S))  
                        S = S - S_diag 
                        neg_contrastive = F.mse_loss(S, torch.zeros_like(S)) 
                        loss_cosl = pos_contrastive + neg_contrastive 

                    # list_pos.append(pos_contrastive.detach().cpu().numpy())
                    # list_neg.append(neg_contrastive.detach().cpu().numpy())
                    # print('loss_pos = {:.5f}'.format(pos_contrastive), ' loss_neg = {:.5f}'.format(neg_contrastive))

                else:  
                    S = hg @ hg_neg.T  
                    S = torch.tensor(S, device=self.device)
                    loss_cosl = F.mse_loss(S, expected_output)

                torch.set_grad_enabled(True)
                _, _, _, q, s, loss_rec, loss_latent = self.model(self.adata, self.X, self.adj, self.edge_index)

                d_cons1 = D_constraint1()  
                d_cons2 = D_constraint2()  
                loss_d1 = d_cons1(self.model.D)
                loss_d2 = d_cons2(self.model.D, self.model.d, self.model.edsc_cluster_n)
                loss_d = loss_d1 + loss_d2

                loss_kl_q = F.kl_div(q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                loss_kl_s = F.kl_div(s.log(), torch.tensor(s_tilde).to(self.device)).to(self.device)
                loss_kl = loss_kl_s + loss_kl_q

                loss_tatal = self.rec_w * loss_rec + self.latent_w * loss_latent + self.kl_w * loss_kl + \
                             self.d_w * loss_d + self.cosl_w * loss_cosl


                loss_tatal.backward()
                self.optimizer.step()
                # list_rec.append(loss_rec.detach().cpu().numpy())
                # list_latent.append(loss_latent.detach().cpu().numpy())
                # # list_cos.append(loss_cos.detach().cpu().numpy())
                # list_d.append(loss_d.detach().cpu().numpy())
                # list_kl.append(loss_kl.detach().cpu().numpy())
                # list_cosl.append(loss_cosl.detach().cpu().numpy())
                # print('loss_rec = {:.5f}'.format(loss_rec),  # 'loss_cos= {:.5f}'.format(loss_cos),
                #       'loss_latent = {:.5f}'.format(loss_latent), 'loss_d= {:.5f}'.format(loss_d),
                #       'loss_kl = {:.5f}'.format(loss_kl), 'loss_cosl = {:.5f}'.format(loss_cosl),
                #       ' loss_total = {:.5f}'.format(loss_tatal))

            return emb

    def model_eval(self):
        self.model.eval()
        Hm, Hcor, emb, q, s, loss_rec, loss_latent = self.model(self.adata, self.X, self.adj, self.edge_index)
        emb = emb.data.cpu().numpy()
        q = q.data.cpu().numpy()
        s = s.data.cpu().numpy()
        hg = Hm.data.cpu().numpy()
        hg_neg = Hcor.data.cpu().numpy()
        return hg, hg_neg, emb, q, s
