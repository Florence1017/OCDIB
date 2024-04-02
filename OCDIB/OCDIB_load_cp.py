import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from utils.data_preprocess import load_data
from utils.early_stop import EarlyStopping
import os
from sklearn.cluster import KMeans
from utils.comm_evaluate import CommunityEval
import numpy as np
from utils.common_utils import random_aug
from OCDIB.OCDIB_model_draft import OCDIB
from utils.common_utils import label_to_comm, aff_to_ovlp_label, print_result, weight_incrementer, filter_by_types
from torch.optim.lr_scheduler import ReduceLROnPlateau
import OCDIB_config as config
import math
import torch.nn.functional as F
from torch.nn import BCELoss
import scipy.sparse as sp

def main():
    # data_config = {'dataset':'LFR','network':'5000-500-2'}
    DataLoader = load_data(data_config=config.data_config)
    G = DataLoader.G
    train_id, test_id, train_y, test_y = DataLoader.get_train_test_data(test_rate=config.train_config['test_size'])
    train_id = train_id[: int(config.train_config['label_rate']*(len(train_id)+len(test_id)))]
    train_y = train_y[: int(config.train_config['label_rate']*(len(train_id)+len(test_id)))]
    print(len(train_id), len(test_id))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model = OCDIB(input_dim=G.num_nodes(), 
            hidden_channels_list=config.model_config['hidden_channels_list'], 
            output_dim=config.model_config['output_dim'], 
            K=DataLoader.K, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train_config['lr'])
    cur_path = os.path.dirname(__file__)
    ck_path = os.path.join(cur_path, 'checkpoint', config.data_config['dataset']+config.data_config['network'])
    stopper = EarlyStopping(checkpoint_path=ck_path, patience=config.train_config['patience'], is_ours=True)


    eps = G.num_edges() / G.num_nodes() / (G.num_nodes()-1)
    threshold = np.sqrt(-(np.log2(1-eps)))
    print('threshold:', threshold)
    
    model = stopper.load_checkpoint(model, filepath="early_stop_2023-05-18_11-12")  

    repeat_times = config.evaluate_config['repeat_times']
    edge_index = torch.LongTensor(np.array([i.cpu().detach().numpy() for i in G.edges()])).to(device)
    feature = torch.eye(G.num_nodes()).to(device)
    
    # Evaluate


    model.eval()
    feature = torch.eye(G.num_nodes()).to(device)
    for j in range(repeat_times):
        hgcn, logvar, mu, hk, r = model(feature, edge_index)
        pred_y = aff_to_ovlp_label(r.detach().cpu().numpy(), threshold=threshold)
        Evaluator = CommunityEval(train_id, test_id, test_y, pred_y, DataLoader.K, G)
        score_dict = Evaluator.eval_community(is_overlapping=DataLoader.is_overlapping, affiliation_matrix=r.detach().cpu())
        if j == 0:
            final_score_dict = score_dict
        else:
            # update the scores in the final dict
            final_score_dict = {k: final_score_dict[k]+v for k,v in score_dict.items()}
    final_score_dict = {k: v/repeat_times for k,v in final_score_dict.items()}
    print_result(final_score_dict)



if __name__ == '__main__':
    main()
        