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
    lr = 1e-3
    epochs = 1000
    eval_epochs = 100
    display_epochs = 100
    optimizer_sigma = torch.optim.Adam
    obs_batch_size = 1000
    pde_batch_size = 1000
    train_data_path = 'data/train_data.npy'
    test_data_path = 'data/test_data.npy'
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
    
    pde_points_num = 10000
    
    # store
    history_path = 'logs/history.npy'
    model_path = 'model.pth'
    

def train():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)
    
    net = Net(seq_net=Config.seq_net, activation=Config.activation).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr)
    # observation data for train is a np array of (x, y, t, u, v, p)
    obs_dataloader = get_dataloader(Config.train_data_path, Config.obs_batch_size, shuffle=True, device=device)
    test_dataloader = obs_dataloader

    # pde points loader
    pde_points = sample_pde_points(Config.pde_points_num, Config.x_min, Config.x_max, Config.y_min, Config.y_max, Config.t_min, Config.t_max)
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
            loss = (mse_obs_1 / sigma4 ** 2 +  \
                    mse_obs_2 / sigma5 ** 2)
            total_obs_loss.append(loss)
            
        for batch in pde_dataloader:
            pde_x = batch[:, 0:1].requires_grad_()
            pde_y = batch[:, 1:2].requires_grad_()
            pde_t = batch[:, 2:3].requires_grad_()
            pde_predict = net(torch.cat([pde_x, pde_y, pde_t], dim=1))
            mse_pde_1, mse_pde_2, mse_pde_3 = ns_pde_loss(
                pde_x, pde_y, pde_t, 
                pde_predict[:, 0:1], pde_predict[:, 1:2], pde_predict[:, 2:3], pde_predict[:, 3:4])
            loss = (mse_pde_1 / sigma1 ** 2 +  \
                    mse_pde_2 / sigma2 ** 2 +  \
                    mse_pde_3 / sigma3 ** 2)
            total_pde_loss.append(loss)
            
        total_obs_loss = torch.stack(total_obs_loss).mean()
        total_pde_loss = torch.stack(total_pde_loss).mean()
        total_loss = total_obs_loss + total_pde_loss
        total_loss.backward()
        
        optimizer.step()
        optimizer_sigma.step()
        
        # print 5 loss item, use sci notation
        if epoch % Config.display_epochs == 0:
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
            total_err = []
            for batch in test_dataloader:
                with torch.no_grad():
                    U_predict = net(batch[:, 0:3])
                # cal_l2_error
                u_err = cal_l2_relative_error(U_predict[:, 0:1], batch[:, 3:4])
                v_err = cal_l2_relative_error(U_predict[:, 1:2], batch[:, 4:5])
                
                total_err.append([u_err, v_err])
            total_err = np.array(total_err).mean(axis=0)
            # print(total_err.shape)
            print_epoch_err(epoch, total_err)
    
    
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