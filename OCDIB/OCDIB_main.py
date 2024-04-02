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
    print(config.data_config['network'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    row = []
    col = []
    data = []
    for idx, com_list in DataLoader.label_dict.items():
        row.extend([idx] * len(com_list))
        col.extend(com_list)
        data.extend([1] * len(com_list))
    matrix = sp.csr_matrix((data, (row, col)), shape=(G.number_of_nodes(), DataLoader.K), dtype=np.float64)
    aff_ground_truth = torch.Tensor(matrix.toarray()).to(device)

    model = OCDIB(input_dim=G.num_nodes(), 
            hidden_channels_list=config.model_config['hidden_channels_list'], 
            output_dim=config.model_config['output_dim'], 
            K=DataLoader.K, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train_config['lr'])
    cur_path = os.path.dirname(__file__)
    ck_path = os.path.join(cur_path, 'checkpoint', config.data_config['dataset']+config.data_config['network'])
    stopper = EarlyStopping(checkpoint_path=ck_path, patience=config.train_config['patience'], is_ours=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    eps = G.num_edges() / G.num_nodes() / (G.num_nodes()-1)
    threshold = np.sqrt(-(np.log2(1-eps)))
    print('threshold:', threshold)

    repeat_times = config.evaluate_config['repeat_times']
    ib_weight = 0.0001
    feature = torch.eye(G.num_nodes()).to(device)
    for epoch in range(1, 5000+1):
        model.train()
        optimizer.zero_grad()

        G1 = random_aug(G, edge_drop_rate=config.train_config['edge_drop_rate'])

        edge_index_1 = G1.edges()
        edge_index_1 = torch.LongTensor(np.array([i.cpu().detach().numpy() for i in edge_index_1])).to(device)


        hgcn_1, logvar_1, mu_1, hk_1, r_1 = model(x=feature, edge_index=edge_index_1)

        mod_loss = model.loss_mod(G, r_1)
        cl1 = BCELoss()(r_1[train_id], aff_ground_truth[train_id])
        kl1 = -0.5*(1+2*logvar_1[train_id]-mu_1[train_id].pow(2)-logvar_1[train_id].exp()).sum(1).mean().div(math.log(2))
        ib_loss = cl1 + config.train_config['kl_weight']*kl1
        loss = ib_loss + 0.01*mod_loss 
        loss.backward()

        print('[Train]: Epoch: {:03d} | loss:{:.4f} | ib_weight: | ib_loss:{:.2f} | mod_loss:| contras_loss: | contras_loss1: | kl1:'
            .format(epoch, loss, ib_loss))

        # Early stopping
        early_stop = stopper.step(loss=loss.detach().cpu().numpy(), model=model)

            
        # Evaluate
        if epoch % config.evaluate_config['eval_interval'] == 0:
            model.eval() 
            edge_index = torch.LongTensor(np.array([i.cpu().detach().numpy() for i in G.edges()])).to(device)
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

        # Optimize
        optimizer.step()
        scheduler.step(loss)

        if early_stop:
            break
    

    model = stopper.load_checkpoint(model)
    # Evaluate


    model.eval()
    feature = torch.eye(G.num_nodes()).to(device)
    edge_index = torch.LongTensor(np.array([i.cpu().detach().numpy() for i in G.edges()])).to(device)
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
        