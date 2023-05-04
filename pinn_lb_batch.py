import torch
from net import Net
import numpy as np
from losses import pns_pde_loss, mse_loss
from metrics import cal_l2_relative_error
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import get_dataloader
from utils.sample import sample_pde_points
from utils.visualize import print_loss_table, print_epoch_err

from config import Config
    

def train():
    tb = SummaryWriter(Config.tensorbard_path)
    # set radom seed for reproduce
    torch.manual_seed(Config.random_seed)
    np.random.seed(Config.random_seed)

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    print('Using device:', device)

    # Define network, optimizer, scheduler
    net = Net(seq_net=Config.seq_net, activation=Config.activation).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=Config.lr_decay_epochs, gamma=Config.lr_decay)

    # Define observation and test data loader
    obs_dataloader = get_dataloader(Config.train_data_path, batch_size=Config.obs_batch_size, shuffle=True, device=device)
    test_data = torch.from_numpy(np.load(Config.test_data_path)).float().to(device)

    # Define PDE points and dataloader
    pde_points = sample_pde_points(
        Config.pde_points_num,
        Config.x_min, Config.x_max,
        Config.y_min, Config.y_max,
        Config.t_min, Config.t_max,
        distribution='normal').to(device)
    pde_dataloader = torch.utils.data.DataLoader(pde_points, batch_size=Config.pde_batch_size, shuffle=True)

    # Define loss scaling factors
    sigma1 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma2 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma3 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)
    sigma4 = torch.tensor(1, dtype=torch.float32, device=device, requires_grad=True)

    # Define optimizer for loss scaling factors
    optimizer_sigma = Config.optimizer_sigma([sigma1, sigma2], Config.lr)

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
            loss = mse_obs_1 + mse_obs_2
            total_obs_loss.append(loss)
            
        for batch in pde_dataloader:
            pde_x = batch[:, 0:1].requires_grad_()
            pde_y = batch[:, 1:2].requires_grad_()
            pde_t = batch[:, 2:3].requires_grad_()
            pde_predict = net(torch.cat([pde_x, pde_y, pde_t], dim=1))
            mse_pde_1, mse_pde_2, mse_pde_3 = pns_pde_loss(
                pde_x, pde_y, pde_t, 
                pde_predict[:, 0:1], pde_predict[:, 1:2], pde_predict[:, 2:3], pde_predict[:, 3:4])
            loss = mse_pde_1 + mse_pde_2 + mse_pde_3
            total_pde_loss.append(loss)
        w_1 = 1 / (2 * sigma1.pow(2))
        w_2 = 1 / (2 * sigma2.pow(2))
        # sum loss
        total_loss = (w_1 * torch.stack(total_obs_loss).mean() + w_2 * torch.stack(total_pde_loss).mean() + \
            torch.log(sigma1) + torch.log(sigma2)) * 1e3
        total_loss.backward()
        
        optimizer.step()
        optimizer_sigma.step()
        scheduler.step()
              
        
        # tensorboard
        tb.add_scalar('loss/total_loss', torch.stack(total_obs_loss).sum() + torch.stack(total_pde_loss).sum(), epoch)
        tb.add_scalar('loss/mse_pde_1', mse_pde_1, epoch)
        tb.add_scalar('loss/mse_pde_2', mse_pde_2, epoch)
        tb.add_scalar('loss/mse_pde_3', mse_pde_3, epoch)
        tb.add_scalar('loss/mse_obs_1', mse_obs_1, epoch)
        tb.add_scalar('loss/mse_obs_2', mse_obs_2, epoch)
        tb.add_scalar('sigma/sigma1', sigma1, epoch)
        tb.add_scalar('sigma/sigma2', sigma2, epoch)
        tb.add_scalar('sigma/sigma3', sigma3, epoch)
        tb.add_scalar('sigma/sigma4', sigma4, epoch)
        tb.add_scalar('sigma/sigma5', sigma5, epoch)
        
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
            err_u1.append(l2_error_u)
            err_u2.append(l2_error_v)
            # tensorboard
            tb.add_scalar('err/l2_error_u', l2_error_u, epoch)
            tb.add_scalar('err/l2_error_v', l2_error_v, epoch)
    
    # save model
    torch.save(net.state_dict(), Config.model_path)
    
if __name__ == '__main__':
    import time
    time_start = time.time()  # 开始计时
    train()
    time_end = time.time()  # 结束计时
    print('time cost', time_end - time_start, 's')  # 输出运行时间