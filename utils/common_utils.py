import os
import json
import torch
import numpy as np
import dgl
import torch.nn as nn
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



def print_result(final_score_dict):
    # print(*(final_score_dict.keys()), sep='\t', end='\n')
    # for i in final_score_dict.values():
    #     print(f'{i:.5f}', end='\t')
    # print('\n')
    print('[eval]: ', end='')
    for k, v in final_score_dict.items():
        print(f'{v:.5f}', end='\t')
    print(' ')


def dump_result(result_path_prefix, final_score_dict, is_overlapping, affiliation_matrix=None):
    dir_name = "OV_" if is_overlapping else "NO_"
    dir_name += "{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(*(final_score_dict.values()))
    result_path = os.path.join(result_path_prefix, dir_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with open(os.path.join(result_path, 'result.json'), 'w') as f:
        json.dump(final_score_dict, f)
    if not affiliation_matrix is None:
        np.save(os.path.join(result_path, 'affiliation.npy'), affiliation_matrix)
    

def random_aug(graph, edge_drop_rate):
    m = graph.number_of_edges()
    drop_rates = torch.FloatTensor(np.ones(m) * edge_drop_rate)
    masks = torch.bernoulli(1 - drop_rates)
    mask_ids = masks.nonzero().squeeze(1)
    
    ng = dgl.graph([])
    ng.add_nodes(graph.num_nodes())
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[mask_ids]
    ndst = dst[mask_ids]
    ng.add_edges(nsrc, ndst)
    return ng

def label_to_comm(n_com, id_list, label_list):
    com_list = [[]for i in range(n_com)]
    for i in range(n_com):
        l = [id_list[idx] for idx,l_list in enumerate(label_list) if i in l_list]
        com_list[i] = l
    return com_list

def comm_to_label(G, community):
    K = 0
    label_list = [[] for n in G.nodes()]
    for c in community:
        for n in c:
            label_list[n].append(K)
        K += 1
    return label_list


def aff_to_ovlp_label(affiliation_matrix, threshold=0.5):
    l = [[] for _ in range(len(affiliation_matrix))]
    for i, r in enumerate(affiliation_matrix):
        # max_in_row = r.max()
        l[i] = list(np.where(r>threshold)[0])
        if len(l[i]) == 0:
            l[i] = [np.argmax(r)]
    return l

def aff_to_novlp_label(affiliation_matrix):
    return [[int(r.argmax().detach().cpu())] for r in affiliation_matrix]


def filter_by_types(G, label_dict):
    marginal_ids = set()
    src_list, dst_list = G.edges()
    for u,v in zip(src_list.numpy(),dst_list.numpy()):
        if len(label_dict[u]) != len(label_dict[v]):
            marginal_ids.add(u)
            marginal_ids.add(v)
    marginal_ids = list(marginal_ids)
    ovlp_ids = []
    n_ovlp_ids = []
    for u,v in label_dict.items():
        # if u not in marginal_ids:
        if len(v) > 1:
            ovlp_ids.append(u)
        else:
            n_ovlp_ids.append(u)
    return marginal_ids, ovlp_ids, n_ovlp_ids



class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        self.net = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                nn.ReLU(),
                                nn.Linear(hidden_size//2, 2*y_dim))
        self.y_dim = y_dim
        # # p_mu outputs mean of q(Y|X)
        # #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))

        # self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
        #                                nn.ReLU(),
        #                                nn.Linear(hidden_size//2, y_dim))
        # # p_logvar outputs log of variance of q(Y|X)
        # self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
        #                                nn.ReLU(),
        #                                nn.Linear(hidden_size//2, y_dim),
        #                                nn.Tanh())

    def get_mu_logvar(self, x_samples):
        ret = self.net(x_samples)
        mu = ret[:, :self.y_dim]
        logvar = nn.Tanh()(ret[:, self.y_dim:])
        # mu = self.p_mu(x_samples)
        # logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

    

def weight_incrementer(step, gamma_0, gamma_final, current_gamma):
    if step > 1:
        current_gamma = gamma_final - gamma_0*(gamma_final - current_gamma)
    return current_gamma


class LBSign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)
    

def visualize(hidden_feat, label_list):
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    X = hidden_feat.detach().cpu().numpy()
    X_tsne = tsne.fit_transform(X)
    color_node = []
    color_edge = []
    for v in label_list:
        if isinstance(v, list):
            if(len(v) == 1):
                color_node.append(v[0])
                color_edge.append(v[0])
            else:
                color_node.append(v[0])
                color_edge.append(v[1])
        else:
                color_node.append(v)
                color_edge.append(v)

    cmap = plt.cm.Spectral
    x_min, x_max = np.min(color_edge), np.max(color_edge)
    edge_norm = (color_edge - x_min) / (x_max - x_min)
    color_edge_mapped = []
    for e in edge_norm:
        color_edge_mapped.append(cmap(e))
    plt.figure(figsize=(7, 4))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color_node, cmap=cmap, linewidths=0.5, edgecolors=color_edge_mapped, s=15)
    plt.xticks([])
    plt.yticks([])


