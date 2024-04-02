import torch.nn as nn
from torch.nn import Linear, PReLU, Sigmoid
from torch_geometric.nn import GCNConv, Sequential
import torch
import numpy as np
import torch.nn.functional as F


class OCDIB(nn.Module):
    def __init__(self, input_dim, hidden_channels_list, output_dim, K, device):
        super(OCDIB, self).__init__()
        
        self.K = K
        self.device = device
        self.output_dim = output_dim
        channels_list = [input_dim] + hidden_channels_list
        hgcn_list = []
        for i in range(len(hidden_channels_list)):
            hgcn_list.append((GCNConv(channels_list[i], channels_list[i+1]), 'x, edge_index -> x'))
            hgcn_list.append(PReLU())
        self.hgcn_seq = Sequential('x, edge_index', hgcn_list)
        
        # no reduction
        self.hk_mu_mod = nn.ModuleList()
        self.hk_logvar_mod = nn.ModuleList()
        self.hk_mod = nn.ModuleList()
        self.hk_to_r_mod = nn.ModuleList()
        for k in range(self.K):
            hk_logvar_list = []
            hk_logvar_list.append((GCNConv(hidden_channels_list[-1],  output_dim), 'x, edge_index -> x'))
            hk_logvar_list.append(PReLU())
            hk_logvar = Sequential('x, edge_index', hk_logvar_list)
            self.hk_logvar_mod.append(hk_logvar)

            hk_mu_list = []
            hk_mu_list.append((GCNConv(hidden_channels_list[-1],  output_dim), 'x, edge_index -> x'))
            hk_mu_list.append(PReLU())
            hk_mu = Sequential('x, edge_index', hk_mu_list)
            self.hk_mu_mod.append(hk_mu)
            
            hk_to_r = Linear(output_dim, 1)
            self.hk_to_r_mod.append(hk_to_r)    
    

    def forward(self, x, edge_index):
        hgcn = self.hgcn_seq(x, edge_index)
        N, _ = hgcn.shape
        D = self.output_dim
        mu = torch.zeros((N, self.K*D),device=self.device) # (N, K*D)
        logvar = torch.zeros_like(mu,device=self.device)  # (N, K*D)
        hk = torch.zeros((N, self.K*D),device=self.device)  # (N, K*D)

        # no reduction
        for k in range(self.K):
            mu[:, k*D:(k+1)*D] = self.hk_mu_mod[k](hgcn, edge_index)
            logvar[:, k*D:(k+1)*D] = Sigmoid()(self.hk_logvar_mod[k](hgcn, edge_index))
            hk[:, k*D:(k+1)*D] = self.reparameterize(mu[:, k*D:(k+1)*D], logvar[:, k*D:(k+1)*D])

        r_list = [self.hk_to_r_mod[k](hk[:, k*D:(k+1)*D]) for k in range(self.K)]
        r = torch.cat(r_list, dim=1)
        r = Sigmoid()(r)

        return hgcn, logvar, mu, hk, r

    def reparameterize(self, mu, logvar,):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        hk = eps * std + mu
        return hk
    
    def _transform_edges(self, edge_index):
        res = torch.LongTensor(np.array([i.cpu().detach().numpy() for i in edge_index])).to(self.device)
        return res

    
    def _semi_mod_loss(self, G, r):
        affiliation_matrix = r.clone()
        affiliation_matrix = F.normalize(affiliation_matrix, p=1)
        adj_matrix = G.adjacency_matrix().to(self.device)

        d_list = G.out_degrees().unsqueeze(dim=0)
        deg_matrix = d_list.t() @ d_list
        deg_matrix_nomalized = (deg_matrix / (G.num_edges())).to(self.device)

        # sim_matrix = affiliation_matrix.dot(affiliation_matrix.transpose())
        sim_matrix = torch.mm(affiliation_matrix, affiliation_matrix.t())

        res_matrix = sim_matrix.multiply( - deg_matrix_nomalized + adj_matrix)
        res = (res_matrix.sum() - res_matrix.diagonal().sum()) / (G.num_edges())

        return res


    def loss_mod(self, G, r_1, r_2):
        G = G.clone()
        l1 = self._semi_mod_loss(G, r_1)
        l2 = self._semi_mod_loss(G, r_2)
        loss = (l1 + l2) * 0.5
        return loss

    def loss_mod(self, G, r):
        loss = self._semi_mod_loss(G, r)
        return -loss



