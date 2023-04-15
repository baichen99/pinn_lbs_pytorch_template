import torch
from net import Net
import numpy as np
from losses import ns_pde_loss, mse_loss
from metrics import cal_l2_relative_error
from tqdm import tqdm
from torchviz import make_dot



class Config:
    lr = 1e-3
    epochs = 1000
    optimizer_sigma = torch.optim.Adam
    
    # domain
    x_min = 0.0
    x_max = 0.2
    y_min = 0.0
    y_max = 0.26
    t_min = 0
    t_max = 10
    
    pde_points_num = 10000
    
def train():
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    net = Net(seq_net=[3] + [50]*5 + [4], activation=torch.tanh).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr)
    
    # loss has 5 parts, 3 for pde, 2 for observation data
    sigma1 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma2 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma3 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma4 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma5 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    optimizer_sigma = Config.optimizer_sigma([sigma1, sigma2, sigma3, sigma4, sigma5], Config.lr)
    
    # random sample pde points in domain, 
    pde_points = torch.rand(Config.pde_points_num, 3, device=device, dtype=torch.float32)
    pde_points[:, 0] = pde_points[:, 0] * (Config.x_max - Config.x_min) + Config.x_min
    pde_points[:, 1] = pde_points[:, 1] * (Config.y_max - Config.y_min) + Config.y_min
    pde_points[:, 2] = pde_points[:, 2] * (Config.t_max - Config.t_min) + Config.t_min
    pde_points = pde_points.to(device).requires_grad_()

    # load observation data
    # observation data for train is a np array of (x, y, t, u, v, p)
    obs_data = np.load('data/train_data.npy')
    # load test data as float32
    obs_data = torch.from_numpy(obs_data).float().to(device)
    # test data
    # test_data = np.load('test_data.npy')
    test_data = obs_data

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

    for epoch in tqdm(range(Config.epochs)):
        optimizer.zero_grad()
        optimizer_sigma.zero_grad()
        pde_x = pde_points[:, 0:1].requires_grad_()
        pde_y = pde_points[:, 1:2].requires_grad_()
        pde_t = pde_points[:, 2:3].requires_grad_()
        pde_predict = net(torch.cat([pde_x, pde_y, pde_t], dim=1))
        
        mse_pde_1, mse_pde_2, mse_pde_3 = ns_pde_loss(
            pde_x, pde_y, pde_t, 
            pde_predict[:, 0:1], pde_predict[:, 1:2], pde_predict[:, 2:3], pde_predict[:, 3:4])
        
        U_predict = net(obs_data[:, 0:3])
        mse_obs_1 = mse_loss(U_predict[:, 0:1], obs_data[:, 3:4])
        mse_obs_2 = mse_loss(U_predict[:, 1:2], obs_data[:, 4:5])
        
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

        
        loss = (
            mse_pde_1 / sigma1 ** 2 +  \
            mse_pde_2 / sigma2 ** 2 +  \
            mse_pde_3 / sigma3 ** 2 +  \
            (mse_obs_1 / sigma4 ** 2 +  \
            mse_obs_2 / sigma5 ** 2)
        )
        
        loss.backward()
        optimizer.step()
        optimizer_sigma.step()
        
        if epoch % 100 == 0:
            net.eval()
            # cal metrics on test dataset
            test_pred = net(test_data[:, 0:3])
            u_pred = test_pred[:, 0:1].detach().numpy()
            v_pred = test_pred[:, 1:2].detach().numpy()
            u_true = test_data[:, 3:4].detach().numpy()
            v_true = test_data[:, 4:5].detach().numpy()
            l2_error_u = cal_l2_relative_error(u_pred, u_true)
            l2_error_v = cal_l2_relative_error(v_pred, v_true)
            print('epoch: {}, loss: {:.4g}, mse_pde_1: {:.4g}, mse_pde_2: {:.4g}, mse_pde_3: {:.4g}, mse_obs_1: {:.4g}, mse_obs_2: {:.4g}, l2_error_u: {:.4g}, l2_error_v: {:.4g}'.format(
                epoch, loss.item(), mse_pde_1.item(), mse_pde_2.item(), mse_pde_3.item(), mse_obs_1.item(), mse_obs_2.item(), l2_error_u, l2_error_v
            ))
    # save history to ./logs
    np.save('logs/MSE_PDE_1.npy', np.array(MSE_PDE_1))
    np.save('logs/MSE_PDE_2.npy', np.array(MSE_PDE_2))
    np.save('logs/MSE_PDE_3.npy', np.array(MSE_PDE_3))
    np.save('logs/MSE_OBS_1.npy', np.array(MSE_OBS_1))
    np.save('logs/MSE_OBS_2.npy', np.array(MSE_OBS_2))
    np.save('logs/Sigma1.npy', np.array(Sigma1))
    np.save('logs/Sigma2.npy', np.array(Sigma2))
    np.save('logs/Sigma3.npy', np.array(Sigma3))
    np.save('logs/Sigma4.npy', np.array(Sigma4))
    np.save('logs/Sigma5.npy', np.array(Sigma5))


    
if __name__ == '__main__':
    import time
    time_start = time.time()  # 开始计时
    train()
    time_end = time.time()  # 结束计时
    print('time cost', time_end - time_start, 's')  # 输出运行时间