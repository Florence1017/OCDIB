import os


config_path = os.path.dirname(__file__)

data_config = {
    'dataset':'Youtube',
    'network':'',
    'is_sample_new':True
}

model_config = {
    'hidden_channels_list':[128],
    'output_dim':32,
}


train_config ={
    'lr':0.1,
    'checkpoint_path':os.path.join(config_path, 'checkpoint', data_config['dataset']+data_config['network']),
    'patience':50,
    'test_size':0.2,
    'label_rate':0.8,
    'edge_drop_rate':0.1,
    'mod_weight':1e3,
    'ib_weight':1,
    'kl_weight':1e-7, 
    'contras_weight':1,
    'tempreture':5,
}

evaluate_config = {
    'result_path': os.path.join(config_path, 'result', data_config['dataset']+data_config['network']),
    'repeat_times':5,
    'eval_interval':200,
}
