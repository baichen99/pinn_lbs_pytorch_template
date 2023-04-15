import torch


def sample_pde_points(num, x_min, x_max, y_min, y_max, t_min, t_max, distribution='uniform'):
    if distribution == 'uniform':
        # sample using uniform distribution
        # 均匀分布
        x = torch.rand(num, 1) * (x_max - x_min) + x_min
        y = torch.rand(num, 1) * (y_max - y_min) + y_min
        t = torch.rand(num, 1) * (t_max - t_min) + t_min
        pde_points = torch.cat([x, y, t], dim=1)

    elif distribution == 'normal':
        # sample using normal distribution
        # 正态分布
        x = torch.randn(num, 1) * (x_max - x_min) / 2 + (x_max + x_min) / 2
        y = torch.randn(num, 1) * (y_max - y_min) / 2 + (y_max + y_min) / 2
        t = torch.randn(num, 1) * (t_max - t_min) / 2 + (t_max + t_min) / 2
        pde_points = torch.cat([x, y, t], dim=1)
    elif distribution == 'grid':
        # sample using grid distribution
        # 网格分布
        x = torch.linspace(x_min, x_max, int(num ** 0.5))
        y = torch.linspace(y_min, y_max, int(num ** 0.5))
        t = torch.linspace(t_min, t_max, int(num ** 0.5))
        x, y, t = torch.meshgrid(x, y, t)
        pde_points = torch.stack([x, y, t], dim=3).reshape(-1, 3)
    else:
        raise ValueError('Distribution {} not supported'.format(distribution))
    return pde_points
