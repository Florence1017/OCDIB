import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
import utils
import scipy.sparse as sp
import math
import scipy
import itertools
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, roc_auc_score, average_precision_score
from scipy.optimize import linear_sum_assignment
import networkx as nx
from common_utils import label_to_comm
import torch

logBase=2

class CommunityEval(object):
    def __init__(self, train_id, test_id, test_y, pred_y, K, G):
        self.train_id = train_id
        self.test_id = test_id
        self.test_y = test_y
        # self.train_y = train_y
        self.pred_y = pred_y
        self.id_list = [int(n) for n in G.nodes()]
        self.test_pred_y = [pred_y[i] for i in test_id]

        if isinstance(self.test_pred_y, list):
            pass
        else:
            self.test_pred_y = list(self.test_pred_y)
        # if isinstance(train_pred_y, list):
        #     self.train_pred_y = train_pred_y
        # else:
        #     self.train_pred_y = list(train_pred_y)

        self.n_com = K
        self.n_node = G.number_of_nodes()
        self.G = G
        self.partition = self._get_partition()


    def _get_partition(self):
        # id_list = list(self.train_id) + list(self.test_id)
        # label_list = self.train_pred_y + self.test_pred_y
        # return self._label_to_comm(id_list, label_list)
        id_list = self.id_list
        label_list = self.pred_y
        return label_to_comm(self.n_com, id_list, label_list)


    def _is_partition_overlapped(self):
        # inters = set(self.partition[0]) # initialize
        # for s in self.partition:
        #     inters.intersection_update(set(s))
        # if len(inters) == 0:
        # label_list = self.train_pred_y + self.test_pred_y
        label_list = self.pred_y
        len_list = [len(i) for i in label_list]
        if max(len_list) == 1:
            return False
        return True
    

    def _label_to_affiliation_matrix(self, is_ground_truth=False):
        row = []
        col = []
        data = []
        if is_ground_truth == False:
            # id_list = self.train_id + self.test_id
            # label_list = self.train_pred_y + self.test_pred_y
            id_list = self.id_list
            label_list = self.pred_y
        else:
            id_list = self.test_id
            label_list = self.test_y
        for idx, com_list in zip(id_list, label_list):
            row.extend([idx] * len(com_list))
            col.extend(com_list)
            # data.extend([1 / len(com_list)] * len(com_list)) # normalized
            data.extend([1] * len(com_list))
        matrix = sp.csr_matrix((data, (row, col)), shape=(self.G.num_nodes(), self.n_com), dtype=np.float64)
        return torch.Tensor(matrix.toarray())


    def eval_community(self, is_overlapping=True, affiliation_matrix=None):
        if is_overlapping:
            f1_score = self.compute_ovlp_f1_score()
            onmi_score = self.compute_onmi_score()
            jaccord_score = self.compute_jaccord_score()
            omega_score = self.compute_omega_score()
            if affiliation_matrix is None:
                affiliation_matrix = self._label_to_affiliation_matrix()
            modularity_score = self.compute_ovlp_modularity_score(affiliation_matrix)
            auc_score = self.compute_auc_score(affiliation_matrix)
            map_score = self.compute_map_score(affiliation_matrix)
            return {'f1': f1_score, 'onmi':onmi_score, 'jaccord':jaccord_score, 'omega': omega_score,'auc':auc_score, 'mAP':map_score, 'omodularity':modularity_score}
        else:
            nmi_score = self.compute_nmi_score()
            ari_score = self.compute_ari_score()
            acc_score = self.compute_accuracy_score()
            purity_score = self.compute_purity_score()
            modularity_score = self.compute_modularity_score()
            return {'nmi':nmi_score, 'ari':ari_score, 'acc':acc_score, 'purity':purity_score, 'modularity':modularity_score}

    def _get_matrix(self):
        row = []
        col = []
        data = []
        for idx, com_list in enumerate(self.test_y):
            row.extend(com_list)
            col.extend([self.test_id[idx]] * len(com_list))
            data.extend([1] * len(com_list))
        self.ground_truth_m = sp.csr_matrix((data, (row, col)), shape=(self.n_com, self.n_node), dtype=np.uint32)

        row = []
        col = []
        data = []
        for idx, com_list in enumerate(self.test_pred_y):
            row.extend(com_list)
            col.extend([self.test_id[idx]] * len(com_list))
            data.extend([1] * len(com_list))
        self.pred_m = sp.csr_matrix((data, (row, col)), shape=(self.n_com, self.n_node), dtype=np.uint32)


    def compute_ovlp_f1_score(self):
        # refer to CommunityGAN: https://github.com/SamJia/CommunityGAN/blob/master/src/CommunityGAN/community_detection.py
        self._get_matrix()
        n = (self.ground_truth_m.dot(self.pred_m.T)).toarray().astype(float)  # cg * cd
        p = n / np.array(self.pred_m.sum(axis=1)).clip(min=1).reshape(-1)
        r = n / np.array(self.ground_truth_m.sum(axis=1)).clip(min=1).reshape(-1, 1)
        f1 = 2 * p * r / (p + r).clip(min=1e-10)

        f1_s1 = f1.max(axis=1).mean()
        f1_s2 = f1.max(axis=0).mean()
        f1_s = (f1_s1 + f1_s2) / 2
        return f1_s
    

    def _partial_entropy_a_proba(self, proba):
        # compute H=-plog(p)
        if proba==0:
            return 0
        return -proba * math.log(proba, logBase)


    def _cover_entropy(self, cover, allNodes): #cover is a list of set, no com ID
        allEntr = []
        for com in cover:
            fractionIn = len(com)/len(allNodes)
            allEntr.append(scipy.stats.entropy([fractionIn,1-fractionIn],base=logBase))
        return sum(allEntr)
    

    def _com_pair_conditional_entropy(self, cl, clKnown, allNodes): #cl1,cl2, snapshot_communities (set of nodes)
        #H(Xi|Yj ) =H(Xi, Yj ) − H(Yj )
        # h(a,n) + h(b,n) + h(c,n) + h(d,n)
        # −h(b + d, n)−h(a + c, n)
        #a: count agreeing on not belonging
        #b: count disagreeing : not in 1 but in 2
        #c: count disagreeing : not in 2 but in 1
        #d: count agreeing on belonging
        nbNodes = len(allNodes)
        a =len((allNodes - cl) - clKnown)/nbNodes
        b = len(clKnown-cl)/nbNodes
        c = len(cl-clKnown)/nbNodes
        d = len(cl & clKnown)/nbNodes
        if self._partial_entropy_a_proba(a)+self._partial_entropy_a_proba(d)>self._partial_entropy_a_proba(b)+self._partial_entropy_a_proba(c):
            entropyKnown=scipy.stats.entropy([len(clKnown)/nbNodes,1-len(clKnown)/nbNodes],base=logBase)
            conditionalEntropy = scipy.stats.entropy([a,b,c,d],base=logBase) - entropyKnown
        else:
            conditionalEntropy = scipy.stats.entropy([len(cl)/nbNodes,1-len(cl)/nbNodes],base=logBase)
        return conditionalEntropy #*nbNodes


    def _cover_conditional_entropy(self, cover, coverRef, allNodes, normalized=False): #cover and coverRef and list of set
        X=cover
        Y=coverRef

        allMatches = []
        for com in cover:
            matches = [(com2, self._com_pair_conditional_entropy(com, com2, allNodes)) for com2 in coverRef]
            bestMatch = min(matches,key=lambda c: c[1]) # in overlapping-community settings, we choose the match with the minimum H:H(Xk|Y) = min(l∈1,2,...,|C|) H(Xk|Yl)
            HXY_part=bestMatch[1]
            if normalized:
                HX = self._partial_entropy_a_proba(len(com) / len(allNodes)) + self._partial_entropy_a_proba((len(allNodes) - len(com)) / len(allNodes))# H(X)=H(X=1)+H(X=0)
                if HX==0:
                    HXY_part=1
                else:
                    HXY_part = HXY_part/HX
            allMatches.append(HXY_part)
        to_return = sum(allMatches)
        if normalized:
            to_return = to_return/len(cover)
        return to_return


    def compute_onmi_score(self, allNodes=None):
        # the variant of LFK(2009)
        # refer to the implementation of library cdlib:https://github.com/Yquetzal/onmi
        comm = label_to_comm(self.n_com, self.test_id, self.test_pred_y)
        commRef = label_to_comm(self.n_com, self.test_id, self.test_y)
        cover = set([frozenset(i) for i in comm])
        coverRef = set([frozenset(i) for i in commRef])
        if (len(cover)==0 and len(coverRef)!=0) or (len(cover)!=0 and len(coverRef)==0):
            return 0
        if cover==coverRef:
            return 1

        if allNodes==None:
            allNodes={n for c in coverRef for n in c}
            allNodes|={n for c in cover for n in c}

        HXY = self._cover_conditional_entropy(cover, coverRef, allNodes, normalized=True)
        HYX = self._cover_conditional_entropy(coverRef, cover, allNodes, normalized=True)

        HX = self._cover_entropy(cover, allNodes)
        HY = self._cover_entropy(coverRef, allNodes)

        ONMI = 1 - 0.5 * (HXY+ HYX)
        if ONMI<0 or ONMI>1 or math.isnan(ONMI):
            print("ONMI: %s  from %s %s %s %s "%(ONMI,HXY,HYX,HX,HY))
            raise Exception("incorrect ONMI")
        return ONMI
    

    def _jaccord_similarity(self, a, b):
        if len(a) == 0 and len(b) == 0:
            # print('Empty community, please check!')
            return 0
        return len(set(a).intersection(set(b))) / len(set(a).union(set(b)))


    def compute_jaccord_score(self):
        comm = label_to_comm(self.n_com, self.test_id, self.test_pred_y)
        commRef = label_to_comm(self.n_com, self.test_id, self.test_y)
        pred_y_set = set([frozenset(i) for i in comm])
        test_y_set = set([frozenset(i) for i in commRef])
        s1 = 0
        for i in pred_y_set:
            s = max([self._jaccord_similarity(i,j) for j in test_y_set])
            s1 += s
        s1 = s1 / len(pred_y_set)
        s2 = 0
        for i in test_y_set:
            s = max([self._jaccord_similarity(i,j) for j in pred_y_set])
            s2 += s
        s2 = s2 / len(test_y_set)
        return (s1+s2)/2
    

    def _table(self, dict1, dict2):
        objects = dict1.keys()
        pairs = itertools.combinations(objects, 2) # every pair of objects
        tabledict = {}
        for p in pairs:
            sol1w1 = dict1[p[0]]
            sol1w2 = dict1[p[1]]
            # number of clusters in which pair is together (solution 1)
            tog1 = len(set(sol1w1) & set(sol1w2))
            sol2w1 = dict2[p[0]]
            sol2w2 = dict2[p[1]]
            # number of clusters in which pair is together (solution 2)
            tog2 = len(set(sol2w1) & set(sol2w2))
            #if sol1w1 == ['0'] or sol1w2 == ['0'] or sol2w1 == ['0'] or sol2w2 == ['0']:
            #    continue
            #else:
            tabledict[p] = (tog1, tog2)
        return tabledict
        

    def _margins(self, table, solNumber):
        # calculate marginals from the table
        # solNumber is the number of the solution (0 or 1)
        # returns dictionary of marginals for that solution
        tv = table.values()
        marginals = {}
        d1vals = [e[solNumber] for e in tv]
        for k in list(set(d1vals)):
            marginals[k] = d1vals.count(k)
        return marginals


    def _omega(self, tab):
        # uses cross-tabulation to calculate omega index
        # returns omega index
        sol1 = self._margins(tab, 0)
        sol2 = self._margins(tab, 1)
        maxj = min([max(sol1.keys()), max(sol2.keys())])
        agree = 0
        for pair in tab:
            entry = tab[pair]
            if entry[0] == entry[1]:
                agree += 1
        observed = agree * len(tab)
        count = -1
        expected = 0
        while count < maxj:
            count += 1
            if count in sol1.keys() and count in sol2.keys():
                c1 = sol1[count]
                c2 = sol2[count]
                expected += (c1 * c2)
        num = observed - expected
        den = (len(tab)**2) - expected
        return float(num) / float(den)


    def compute_omega_score(self):
        # refer to the implementation in: https://github.com/gmfraser/omega-index/blob/master/omega.py
        test_y_dict = {id:self.test_y[i] for i,id in enumerate(self.test_id)}
        pred_y_dict = {id:self.test_pred_y[i] for i,id in enumerate(self.test_id)}
        tab = self._table(test_y_dict, pred_y_dict)
        return self._omega(tab)


    # def compute_ovlp_modularity_score(self, affiliation_matrix):
    #     sim_matrix = affiliation_matrix.dot(affiliation_matrix.T)
    #     #边的个数
    #     G_nx = self.G.to_networkx()
    #     edges=G_nx.edges()
    #     m=len(edges)
    #     #每个节点的度
    #     du=G_nx.degree()
    #     ret=0.0
    #     for c in self.partition:
    #         for x in c:
    #             for y in c:
    #                 #边都是前小后大的
    #                 #不能交换x，y，因为都是循环变量
    #                 if x<=y:
    #                     if (x,y) in edges:
    #                         aij=1.0
    #                     else:
    #                         aij=0.0
    #                 else:
    #                     if (y,x) in edges:
    #                         aij=1.0
    #                     else:
    #                         aij=0
    #                 tmp=aij-du[x]*du[y]*1.0/(2*m)
    #                 ret=ret+tmp*sim_matrix[x,y]
    #     ret=ret*1.0/(2*m)
    #     return ret

    def compute_ovlp_modularity_score(self, affiliation_matrix):
        affiliation_matrix = torch.nn.functional.normalize(affiliation_matrix, p=1)
        adj_matrix = self.G.adjacency_matrix()

        d_list = self.G.out_degrees().unsqueeze(dim=0)
        deg_matrix = d_list.t() @ d_list
        deg_matrix_nomalized = deg_matrix / (self.G.num_edges())

        sim_matrix = torch.mm(affiliation_matrix, affiliation_matrix.t())

        res_matrix = sim_matrix.multiply( - deg_matrix_nomalized + adj_matrix)
        res = (res_matrix.sum() - res_matrix.diagonal().sum()) / (self.G.num_edges())
        return float(res)



    def compute_nmi_score(self):
        test_y_list = sum(self.test_y, [])
        pred_y_list = sum(self.test_pred_y, [])
        return normalized_mutual_info_score(test_y_list, pred_y_list)
    
    def compute_ari_score(self):
        test_y_list = sum(self.test_y, [])
        pred_y_list = sum(self.test_pred_y, [])
        return adjusted_rand_score(test_y_list, pred_y_list)
    

    def _clustering_accuracy(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = np.int64(y_true)
        y_pred = np.int64(y_pred)
        # y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w) # find the best match using Kuhn-Munkres or Hungarian Algorithm
        return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

    def compute_accuracy_score(self):
        # refer to the implementation in https://blog.csdn.net/qq_42887760/article/details/105720735
        test_y_list = sum(self.test_y, [])
        pred_y_list = sum(self.test_pred_y, [])
        return self._clustering_accuracy(test_y_list, pred_y_list)
    

    # def _label_to_comm(self, id_list, label_list):
    #     com_list = [[]for i in range(self.n_com)]
    #     for i in range(self.n_com):
    #         l = [id_list[idx] for idx,l_list in enumerate(label_list) if i in l_list]
    #         com_list[i] = l
    #     return com_list


    def compute_purity_score(self):
        # pred_com = self._label_to_comm(self.test_id, self.test_pred_y)
        # test_com = self._label_to_comm(self.test_id, self.test_y)
        pred_com =label_to_comm(self.n_com, self.test_id, self.test_pred_y)
        test_com = label_to_comm(self.n_com, self.test_id, self.test_y)
        score = 0
        for p in pred_com:
            l = [len(set(p).intersection(set(t))) for t in test_com]
            score += max(l)
        return score / len(self.test_y)
    

    def compute_modularity_score(self):
        G_nx = self.G.to_networkx()
        q = nx.community.modularity(G_nx, self.partition)
        return q

    def compute_auc_score(self, affiliation_matrix):
        test_y_oh = self._label_to_affiliation_matrix(is_ground_truth=True)
        return roc_auc_score(test_y_oh[self.test_id], affiliation_matrix[self.test_id])

    def compute_map_score(self, affiliation_matrix):
        test_y_oh = self._label_to_affiliation_matrix(is_ground_truth=True)
        return average_precision_score(test_y_oh[self.test_id], affiliation_matrix[self.test_id])





