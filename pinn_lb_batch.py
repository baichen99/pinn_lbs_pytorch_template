import torch
from net import Net
import numpy as np
from losses import ns_pde_loss, mse_loss
from metrics import cal_l2_relative_error
from tqdm import tqdm

from utils.dataset import get_dataloader
from utils.sample import sample_pde_points
from utils.visualize import print_loss_table, print_epoch_err

class Config:
    random_seed = 0
    
    lr = 1e-3
    lr_decay = 0.95
    lr_decay_epochs = 1000
    epochs = 100000
    eval_epochs = 1000
    display_epochs = 1000
    
    optimizer_sigma = torch.optim.Adam
    
    obs_batch_size = 5000
    pde_batch_size = 5000
    train_data_path = 'data/train_data.npy'
    test_data_path = 'data/data.npy'
    additional_acnhors_path = ''
    
    # net
    seq_net = [3] + [50]*5 + [4]
    activation = torch.tanh
    
    # domain
    x_min = 0.0
    x_max = 0.2
    y_min = 0.0
    y_max = 0.26
    t_min = 0
    t_max = 10
    
    pde_points_num = 35000
    
    # store
    history_path = 'logs/history.npy'
    model_path = 'model.pth'
    

def train():
    # set radom seed for reproduce
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    # cal how much memory will be use

    print('Using device:', device)
    
    net = Net(seq_net=Config.seq_net, activation=Config.activation).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.lr_decay_epochs, gamma=Config.lr_decay)
    # observation data for train is a np array of (x, y, t, u, v, p)
    obs_dataloader = get_dataloader(Config.train_data_path, Config.obs_batch_size, shuffle=True, device=device)
    # test_dataloader = get_dataloader(Config.test_data_path, Config.obs_batch_size, shuffle=False, device=device)
    test_data = torch.from_numpy(np.load(Config.test_data_path)).float().to(device)

    # pde points loader
    pde_points = sample_pde_points(Config.pde_points_num, Config.x_min, Config.x_max, Config.y_min, Config.y_max, Config.t_min, Config.t_max, distribution='normal')
    pde_points = pde_points.to(device)
    pde_dataloader = torch.utils.data.DataLoader(pde_points, batch_size=Config.pde_batch_size, shuffle=True)

    # loss has 5 parts, 3 for pde, 2 for observation data
    sigma1 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma2 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma3 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma4 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma5 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    optimizer_sigma = Config.optimizer_sigma([sigma1, sigma2, sigma3, sigma4, sigma5], Config.lr)
    
    MSE_PDE_1 = []
    MSE_PDE_2 = []
    MSE_PDE_3 = []
    MSE_OBS_1 = []
    MSE_OBS_2 = []
    Sigma1 = []
    Sigma2 = []
    Sigma3 = []
    Sigma4 = []
    Sigma5 = []

    for epoch in tqdm(range(Config.epochs), desc='Epoch'):
        optimizer.zero_grad()
        optimizer_sigma.zero_grad()
        # 计算观测数据损失
        total_obs_loss = []
        total_pde_loss = []
        for batch in obs_dataloader:
            
            U_predict = net(batch[:, 0:3])
            mse_obs_1 = mse_loss(U_predict[:, 0:1], batch[:, 3:4])
            mse_obs_2 = mse_loss(U_predict[:, 1:2], batch[:, 4:5])
            loss = (mse_obs_1 / sigma4 ** 2 / 2+  \
                    mse_obs_2 / sigma5 ** 2 / 2)
            total_obs_loss.append(loss)
            
        for batch in pde_dataloader:
            pde_x = batch[:, 0:1].requires_grad_()
            pde_y = batch[:, 1:2].requires_grad_()
            pde_t = batch[:, 2:3].requires_grad_()
            pde_predict = net(torch.cat([pde_x, pde_y, pde_t], dim=1))
            mse_pde_1, mse_pde_2, mse_pde_3 = ns_pde_loss(
                pde_x, pde_y, pde_t, 
                pde_predict[:, 0:1], pde_predict[:, 1:2], pde_predict[:, 2:3], pde_predict[:, 3:4])
            loss = (mse_pde_1 / sigma1 ** 2 / 2 +  \
                    mse_pde_2 / sigma2 ** 2 / 2+  \
                    mse_pde_3 / sigma3 ** 2 / 2)
            total_pde_loss.append(loss)
        
        # sum loss
        total_loss = torch.stack(total_obs_loss).sum() + torch.stack(total_pde_loss).sum()
        total_loss.backward()
        
        optimizer.step()
        optimizer_sigma.step()
        scheduler.step()
        
        # print 5 loss item, use sci notation
        if epoch % Config.display_epochs == 0:
            # cal l2 err on train data
            train_l2_err = []
            for batch in obs_dataloader:
                with torch.no_grad():
                    U_predict = net(batch[:, 0:3])
                u_err = cal_l2_relative_error(U_predict[:, 0:1].cpu(), batch[:, 3:4].cpu())
                v_err = cal_l2_relative_error(U_predict[:, 1:2].cpu(), batch[:, 4:5].cpu())
                train_l2_err.append([u_err, v_err])
            train_l2_err = np.array(train_l2_err).mean(axis=0)
            print_epoch_err(epoch, [train_l2_err[0], train_l2_err[1]])
            print_loss_table(epoch, total_loss, mse_pde_1, mse_pde_2, mse_pde_3, mse_obs_1, mse_obs_2)
        # append to list
        MSE_PDE_1.append(mse_pde_1.item())
        MSE_PDE_2.append(mse_pde_2.item())
        MSE_PDE_3.append(mse_pde_3.item())
        MSE_OBS_1.append(mse_obs_1.item())
        MSE_OBS_2.append(mse_obs_2.item())
        Sigma1.append(sigma1.item())
        Sigma2.append(sigma2.item())
        Sigma3.append(sigma3.item())
        Sigma4.append(sigma4.item())
        Sigma5.append(sigma5.item())
        
        # eval
        if epoch % Config.eval_epochs == 0:
            net.eval()
            # cal metrics on test dataset
            test_pred = net(test_data[:, 0:3])
            u_pred = test_pred[:, 0:1].detach().cpu().numpy()
            v_pred = test_pred[:, 1:2].detach().cpu().numpy()
            u_true = test_data[:, 3:4].detach().cpu().numpy()
            v_true = test_data[:, 4:5].detach().cpu().numpy()
            l2_error_u = cal_l2_relative_error(u_pred, u_true)
            l2_error_v = cal_l2_relative_error(v_pred, v_true)
            print_epoch_err(epoch, [l2_error_u, l2_error_v], train=False)

    
    
    # save model
    torch.save(net.state_dict(), Config.model_path)
    # save loss
    np.save(Config.history_paths, np.array([MSE_PDE_1, MSE_PDE_2, MSE_PDE_3, MSE_OBS_1, MSE_OBS_2, Sigma1, Sigma2, Sigma3, Sigma4, Sigma5]))
    
if __name__ == '__main__':
    import time
    time_start = time.time()  # 开始计时
    train()
    time_end = time.time()  # 结束计时
    print('time cost', time_end - time_start, 's')  # 输出运行时间