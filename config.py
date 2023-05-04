import torch


class Config:
    # Random seed
    random_seed = 0
    
    # Optimizer settings
    lr = 1e-3
    lr_decay = 0.5
    lr_decay_epochs = 1000
    
    # Batch sizes
    obs_batch_size = 5000
    pde_batch_size = 5000
    
    # Training data paths
    train_data_path = 'data/train_data.npy'
    test_data_path = 'data/data.npy'
    additional_anchors_path = ''
    
    # Network settings
    seq_net = [3] + [50]*5 + [4]
    activation = torch.tanh
    
    # Domain settings
    x_min = 0.0
    x_max = 0.2
    y_min = 0.0
    y_max = 0.26
    t_min = 0
    t_max = 10
    pde_points_num = 35000
    
    # Store settings
    history_path = 'logs/history.npy'
    tensorbard_path = 'tb_logs'
    model_path = 'model.pth'
    
    # Training settings
    epochs = 100000
    eval_epochs = 1000
