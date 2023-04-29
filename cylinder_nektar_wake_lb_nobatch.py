# https://github.com/lululxvi/deepxde/blob/master/examples/pinn_inverse/Navier_Stokes_inverse.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys

from torch.utils.tensorboard import SummaryWriter
from net import Net
from losses import cylinder_ns_loss, mse_loss
from utils.sample import sample_pde_points
from metrics import cal_l2_relative_error

Lx_min, Lx_max = 1.0, 8.0
Ly_min, Ly_max = -2.0, 2.0
t_min, t_max = 0.0, 7.0

pde_points_num = 700

seq_net = [3] + [50] * 6 + [3]



# Load training data
def load_training_data(num):
    data = loadmat("data/cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num, replace=False)
    x_train = data_domain[idx, 0:1]
    y_train = data_domain[idx, 1:2]
    t_train = data_domain[idx, 2:3]
    u_train = data_domain[idx, 3:4]
    v_train = data_domain[idx, 4:5]
    p_train = data_domain[idx, 5:6]
    return [x_train, y_train, t_train, u_train, v_train, p_train]




def train():
    tb = SummaryWriter('./tb_logs/cylinder_nektar_wake_lb')
    pde_points = sample_pde_points(pde_points_num, Lx_min, Lx_max, Ly_min, Ly_max, t_min, t_max)

    [ob_x, ob_y, ob_t, ob_u, ob_v, ob_p] = load_training_data(num=7000)
    ob_xyt = np.hstack((ob_x, ob_y, ob_t))

    # pde_points plus ob_xyt
    pde_points = np.vstack((pde_points, ob_xyt))
    
    # dataloader 
    pde_points = torch.from_numpy(pde_points).float()
    obs_tensor = torch.from_numpy(np.hstack((ob_x, ob_y, ob_t, ob_u, ob_v, ob_p))).float()
    # Define loss scaling factors
    sigma1 = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True))
    sigma2 = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32, requires_grad=True))

    net = Net(seq_net, activation=torch.tanh)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    optimizer_sigma = torch.optim.Adam([sigma1, sigma2], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    scheduler_sigma = torch.optim.lr_scheduler.StepLR(optimizer_sigma, step_size=10000, gamma=0.1)
    for epoch in range(20000):
        optimizer.zero_grad()
        optimizer_sigma.zero_grad()
        # cal pde loss
        pde_x = pde_points[:, 0:1].requires_grad_()
        pde_y = pde_points[:, 1:2].requires_grad_()
        pde_t = pde_points[:, 2:3].requires_grad_()
        pde_predict = net(torch.cat((pde_x, pde_y, pde_t), dim=1))
        pde_u = pde_predict[:, 0:1]
        pde_v = pde_predict[:, 1:2]
        pde_p = pde_predict[:, 2:3]
        mse_pde_1, mse_pde_2, mse_pde_3 = cylinder_ns_loss(pde_x, pde_y, pde_t, pde_u, pde_v, pde_p)
        
        # cal obs loss
        obs_predict = net(obs_tensor[:, 0:3])
        mse_obs_1 = torch.mean((obs_predict[:, 0:1] - obs_tensor[:, 3:4])**2)
        mse_obs_2 = torch.mean((obs_predict[:, 1:2] - obs_tensor[:, 4:5])**2)

        # total loss
        w_1 = 1 / (sigma1.pow(2) * 2)
        w_2 = 1 / (sigma2.pow(2) * 2)
        # weighted_sum
        total_loss = (
                w_1 * (mse_pde_1 +  mse_pde_2 + mse_pde_3) + torch.log(sigma1) + \
                w_2 * (mse_obs_1 + mse_obs_2) + torch.log(sigma2)
            )
        total_loss.backward()
        
        optimizer.step()
        optimizer_sigma.step()
        scheduler.step()
        scheduler_sigma.step()
        
        tb.add_scalar('loss/mse_pde_1', mse_pde_1, epoch)
        tb.add_scalar('loss/mse_pde_2', mse_pde_2, epoch)
        tb.add_scalar('loss/mse_pde_3', mse_pde_3, epoch)
        tb.add_scalar('loss/mse_obs_1', mse_obs_1, epoch)
        tb.add_scalar('loss/mse_obs_2', mse_obs_2, epoch)
        tb.add_scalar('loss/total_loss', total_loss, epoch)
        tb.add_scalar('sigma/sigma1', sigma1, epoch)
        tb.add_scalar('sigma/sigma2', sigma2, epoch)
        tb.add_scalar('weight/w1', w_1, epoch)
        tb.add_scalar('weight/w2', w_2, epoch)


    
        
        if epoch % 100 == 0:
            print('epoch: {}, total_loss: {}'.format(epoch, total_loss.item()))
            # cal error
            [x_star, y_star, t_star, u_star, v_star, p_star] = load_training_data(num=140000)
            
            x_star = torch.from_numpy(x_star).float()
            y_star = torch.from_numpy(y_star).float()
            t_star = torch.from_numpy(t_star).float()
            with torch.no_grad():
                pred = net(torch.cat((x_star, y_star, t_star), 1))
            u_pred = pred[:, 0:1].detach().numpy()
            v_pred = pred[:, 1:2].detach().numpy()
            p_pred = pred[:, 2:3].detach().numpy()
            
            l2_error_u = cal_l2_relative_error(u_pred, u_star)
            l2_error_v = cal_l2_relative_error(v_pred, v_star)
            l2_error_p = cal_l2_relative_error(p_pred, p_star)
            print('l2_error_u: {}, l2_error_v: {}, l2_error_p: {}'.format(l2_error_u, l2_error_v, l2_error_p))
            
            tb.add_scalar('err/l2_error_u', l2_error_u, epoch)
            tb.add_scalar('err/l2_error_v', l2_error_v, epoch)
            tb.add_scalar('err/l2_error_p', l2_error_p, epoch)
            
if __name__ == '__main__':
    train()
            