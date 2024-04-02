import csv
import dgl
import os
import json
import random
from sklearn.model_selection import train_test_split


def str_list_to_int(str_list):
    return list(map(int, str_list))


class LFRDataLoader(object):
    def __init__(self, data_config):
        self.data_config = data_config
        dataset = self.data_config['dataset']
        network = self.data_config['network']
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data')
        self.path_prefix = os.path.join(base_path, dataset, network) # the prefix of the path storing data
        self.G, self.label_dict, self.K = self.get_raw_data()
        if '-' in network:
            self.is_overlapping = True
        else:
            self.is_overlapping = False


    def get_raw_data(self):
        data_path = os.path.join(self.path_prefix, 'network.dat')
        community_path = os.path.join(self.path_prefix, 'community.dat')
        
        # network loading
        data = csv.reader(open(data_path, 'r'), delimiter='\t')
        src_list, dst_list = [], []
        for d in data:
            src_list.append(int(d[0])-1) # node's id starting from 0
            dst_list.append(int(d[1])-1)
        G = dgl.graph((src_list, dst_list))

        # community loading
        community = csv.reader(open(community_path, 'r'), delimiter='\t')
        label_dict = {}
        for c in community:
            label_list = str_list_to_int(c[1].strip().split(sep=" "))
            label_list = [l-1 for l in label_list]  # community's id starting from 0
            label_dict[int(c[0])-1] = label_list
        label_list_tmp = sum(label_dict.values(), []) # flatten
        K = max(label_list_tmp) + 1
        return G, label_dict, K
    
    def get_train_test_data(self, test_rate=0.2, seed=0):
        path = os.path.join(self.path_prefix, 'train_test', str(test_rate))
        if not os.path.exists(path):
            os.makedirs(path)
            ids = self.G.nodes().numpy()
            labels = [self.label_dict[id] for id in ids] # sort by id
            train_id, test_id, train_y, test_y = train_test_split(ids, labels, 
                                                                    test_size=test_rate,
                                                                    random_state=seed,
                                                                    )
            json.dump({str(id): y for id,y in zip(train_id, train_y)}, 
                        open(os.path.join(path, 'train.json'), 'w'))
            json.dump({str(id): y for id,y in zip(test_id, test_y)}, 
                        open(os.path.join(path, 'test.json'), 'w'))
            
        else:
            train_dict = json.load(open(os.path.join(path, 'train.json'), 'r'))
            test_dict = json.load(open(os.path.join(path, 'test.json'), 'r'))
            train_id, train_y = list(train_dict.keys()), list(train_dict.values())
            test_id, test_y = list(test_dict.keys()), list(test_dict.values())
            train_id = str_list_to_int(train_id)
            test_id = str_list_to_int(test_id)
            train_y = [str_list_to_int(l) for l in train_y]
            test_y = [str_list_to_int(l) for l in test_y]
        
        return train_id, test_id, train_y, test_y


class DBLPDataLoader(object):
    def __init__(self, data_config):
        self.data_config = data_config
        dataset = self.data_config['dataset']
        self.is_sample_new = self.data_config['is_sample_new']
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data')
        self.path_prefix = os.path.join(base_path, dataset) # the prefix of the path storing data
        self.G, self.label_dict, self.K = self.get_raw_data()
        self.is_overlapping = True


    def get_raw_data(self):
        if self.is_sample_new == False:
            data_path = os.path.join(self.path_prefix, 'dblp-5com-graph.txt')
            community_path = os.path.join(self.path_prefix, 'dblp-5com-com.txt')
        else:
            data_path = os.path.join(self.path_prefix, 'dblp-5com-graph_new.txt')
            community_path = os.path.join(self.path_prefix, 'dblp-5com-com_new.txt')
        
        # network loading
        data = csv.reader(open(data_path, 'r'), delimiter='\t')
        src_list, dst_list = [], []
        for d in data:
            src_list.append(int(d[0])) # node's id starting from 0
            dst_list.append(int(d[1]))
        G = dgl.graph((src_list, dst_list))
        G = dgl.add_self_loop(G)

        # community loading
        community = csv.reader(open(community_path, 'r'), delimiter='\t')
        label_dict = {int(n):[] for n in G.nodes()}
        communities = []
        K = 0
        for c in community:
            com_mem = str_list_to_int(c)
            communities.append(com_mem)
            for n in com_mem:
                label_dict[n].append(K)
            K += 1
        return G, label_dict, K
    
    def get_train_test_data(self, test_rate=0.2, seed=0):
        if self.is_sample_new == False:
            path = os.path.join(self.path_prefix, 'train_test', str(test_rate))
        else:
            path = os.path.join(self.path_prefix, 'train_test_new', str(test_rate))
        if not os.path.exists(path):
            os.makedirs(path)
            ids = self.G.nodes().numpy()
            labels = [self.label_dict[id] for id in ids] # sort by id
            train_id, test_id, train_y, test_y = train_test_split(ids, labels, 
                                                                    test_size=test_rate,
                                                                    random_state=seed,
                                                                    )
            json.dump({str(id): y for id,y in zip(train_id, train_y)}, 
                        open(os.path.join(path, 'train.json'), 'w'))
            json.dump({str(id): y for id,y in zip(test_id, test_y)}, 
                        open(os.path.join(path, 'test.json'), 'w'))
            
        else:
            train_dict = json.load(open(os.path.join(path, 'train.json'), 'r'))
            test_dict = json.load(open(os.path.join(path, 'test.json'), 'r'))
            train_id, train_y = list(train_dict.keys()), list(train_dict.values())
            test_id, test_y = list(test_dict.keys()), list(test_dict.values())
            train_id = str_list_to_int(train_id)
            test_id = str_list_to_int(test_id)
            train_y = [str_list_to_int(l) for l in train_y]
            test_y = [str_list_to_int(l) for l in test_y]
        
        return train_id, test_id, train_y, test_y


class AmazonDataLoader(object):
    def __init__(self, data_config):
        self.data_config = data_config
        dataset = self.data_config['dataset']
        self.is_sample_new = self.data_config['is_sample_new']
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data')
        self.path_prefix = os.path.join(base_path, dataset) # the prefix of the path storing data
        self.G, self.label_dict, self.K = self.get_raw_data()
        self.is_overlapping = True


    def get_raw_data(self):
        if self.is_sample_new:
            data_path = os.path.join(self.path_prefix, 'Amazon-5com-graph_new.txt')
            community_path = os.path.join(self.path_prefix, 'Amazon-5com-com_new.txt')
        else:
            data_path = os.path.join(self.path_prefix, 'Amazon-5com-graph.txt')
            community_path = os.path.join(self.path_prefix, 'Amazon-5com-com.txt')
        
        # network loading
        data = csv.reader(open(data_path, 'r'), delimiter='\t')
        src_list, dst_list = [], []
        for d in data:
            src_list.append(int(d[0])) # node's id starting from 0
            dst_list.append(int(d[1]))
        G = dgl.graph((src_list, dst_list))
        G = dgl.add_self_loop(G)

        # community loading
        community = csv.reader(open(community_path, 'r'), delimiter='\t')
        label_dict = {int(n):[] for n in G.nodes()}
        communities = []
        K = 0
        for c in community:
            com_mem = str_list_to_int(c)
            communities.append(com_mem)
            for n in com_mem:
                label_dict[n].append(K)
            K += 1
        return G, label_dict, K
    
    def get_train_test_data(self, test_rate=0.2, seed=0):
        if self.is_sample_new:
            path = os.path.join(self.path_prefix, 'train_test_new_new', str(test_rate))
        else:
            path = os.path.join(self.path_prefix, 'train_test', str(test_rate))
        if not os.path.exists(path):
            os.makedirs(path)
            # ids = self.G.nodes().numpy()
            ids = [k for k,v in self.label_dict.items() if len(v) > 0] # 将没有标签的节点排除在训练/测试集以外
            labels = [self.label_dict[id] for id in ids] # sort by id
            train_id, test_id, train_y, test_y = train_test_split(ids, labels, 
                                                                    test_size=test_rate,
                                                                    random_state=seed,
                                                                    )
            json.dump({str(id): y for id,y in zip(train_id, train_y)}, 
                        open(os.path.join(path, 'train.json'), 'w'))
            json.dump({str(id): y for id,y in zip(test_id, test_y)}, 
                        open(os.path.join(path, 'test.json'), 'w'))
            
        else:
            train_dict = json.load(open(os.path.join(path, 'train.json'), 'r'))
            test_dict = json.load(open(os.path.join(path, 'test.json'), 'r'))
            train_id, train_y = list(train_dict.keys()), list(train_dict.values())
            test_id, test_y = list(test_dict.keys()), list(test_dict.values())
            train_id = str_list_to_int(train_id)
            test_id = str_list_to_int(test_id)
            train_y = [str_list_to_int(l) for l in train_y]
            test_y = [str_list_to_int(l) for l in test_y]
        
        return train_id, test_id, train_y, test_y


class YoutubeDataLoader(object):
    def __init__(self, data_config):
        self.data_config = data_config
        dataset = self.data_config['dataset']
        self.is_sample_new = self.data_config['is_sample_new']
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'data')
        self.path_prefix = os.path.join(base_path, dataset) # the prefix of the path storing data
        self.G, self.label_dict, self.K = self.get_raw_data()
        self.is_overlapping = True


    def get_raw_data(self):
        if self.is_sample_new:
            data_path = os.path.join(self.path_prefix, 'Youtube-5com-graph_new.txt')
            community_path = os.path.join(self.path_prefix, 'Youtube-5com-com_new.txt')
        else:
            data_path = os.path.join(self.path_prefix, 'Youtube-5com-graph.txt')
            community_path = os.path.join(self.path_prefix, 'Youtube-5com-com.txt')
        
        # network loading
        data = csv.reader(open(data_path, 'r'), delimiter='\t')
        src_list, dst_list = [], []
        for d in data:
            src_list.append(int(d[0])) # node's id starting from 0
            dst_list.append(int(d[1]))
        G = dgl.graph((src_list, dst_list))
        G = dgl.add_self_loop(G)

        # community loading
        community = csv.reader(open(community_path, 'r'), delimiter='\t')
        label_dict = {int(n):[] for n in G.nodes()}
        communities = []
        K = 0
        for c in community:
            com_mem = str_list_to_int(c)
            communities.append(com_mem)
            for n in com_mem:
                label_dict[n].append(K)
            K += 1
        return G, label_dict, K
    
    def get_train_test_data(self, test_rate=0.2, seed=0):
        if self.is_sample_new:
            path = os.path.join(self.path_prefix, 'train_test_new', str(test_rate))
        else:
            path = os.path.join(self.path_prefix, 'train_test', str(test_rate))
        if not os.path.exists(path):
            os.makedirs(path)
            # ids = self.G.nodes().numpy()
            ids = [k for k,v in self.label_dict.items() if len(v) > 0] # 将没有标签的节点排除在训练/测试集以外
            labels = [self.label_dict[id] for id in ids] # sort by id
            train_id, test_id, train_y, test_y = train_test_split(ids, labels, 
                                                                    test_size=test_rate,
                                                                    random_state=seed,
                                                                    )
            json.dump({str(id): y for id,y in zip(train_id, train_y)}, 
                        open(os.path.join(path, 'train.json'), 'w'))
            json.dump({str(id): y for id,y in zip(test_id, test_y)}, 
                        open(os.path.join(path, 'test.json'), 'w'))
            
        else:
            train_dict = json.load(open(os.path.join(path, 'train.json'), 'r'))
            test_dict = json.load(open(os.path.join(path, 'test.json'), 'r'))
            train_id, train_y = list(train_dict.keys()), list(train_dict.values())
            test_id, test_y = list(test_dict.keys()), list(test_dict.values())
            train_id = str_list_to_int(train_id)
            test_id = str_list_to_int(test_id)
            train_y = [str_list_to_int(l) for l in train_y]
            test_y = [str_list_to_int(l) for l in test_y]
        
        return train_id, test_id, train_y, test_y



def load_data(data_config):
    dataset = data_config['dataset']
    if dataset == 'LFR':
        return LFRDataLoader(data_config)
    elif dataset == 'DBLP':
        return DBLPDataLoader(data_config)
    elif dataset == 'Amazon':
        return AmazonDataLoader(data_config)
    elif dataset == 'Youtube':
        return YoutubeDataLoader(data_config)
    else:
        raise NotImplementedError('Unsupported dataset {}'.format(dataset))


def sort_by_len(l):
    return len(l)


def subgraph_sampling(G, communities, new_K = 5, choose_top_k=False):
    if choose_top_k == False:
        len_coms = [len(c) for c in communities]
        K = len(communities)
        seed_com = random.choices(range(0, K), len_coms)[0] # randomly choose the first community as seed
        selected_coms = [seed_com]
        for _ in range(new_K-1):
            ovlp_rate = {}
            cur_com = communities[selected_coms[-1]]
            for idx, c in enumerate(communities):
                if set(c).issubset(set(cur_com)) or set(cur_com).issubset(set(c)):
                    continue
                if idx not in selected_coms:
                    ovlp_rate[idx] = len(set(cur_com).intersection(set(c))) / len(set(cur_com).union(set(c)))
            selected_coms.append(max(ovlp_rate, key=ovlp_rate.get))
        new_communities = [communities[c] for c in selected_coms]
    else:
        communities_cp = communities.copy()
        communities_cp.sort(key = sort_by_len)# sort by the size of communities
        new_communities = communities_cp[-new_K:]
    new_nodes = list(set(sum(new_communities, [])))
    sub_G = dgl.node_subgraph(G, new_nodes)
    mapping = {int(idx):i for i, idx in enumerate(sub_G.ndata[dgl.NID])}
    new_communities_nid = []
    for c in new_communities:
        l = []
        for n in c:
            l.append(mapping[n])
        new_communities_nid.append(l)
    return sub_G, new_communities_nid

if __name__ == "__main__":
    data_config = {'dataset':'LFR','network':'5000-500-2'}
    dataloader = load_data(data_config=data_config)
    G = dataloader.G
