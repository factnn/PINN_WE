"""
实验数据加载工具
支持多种数据格式
"""
import numpy as np
import torch

def load_data_from_txt(filename, format='standard'):
    """
    从文本文件加载实验数据
    
    Args:
        filename: 数据文件路径
        format: 数据格式
            - 'standard': t, x, rho, u, p (每行5列)
            - 'csv': CSV格式，列名：t, x, rho, u, p
            - 'custom': 自定义格式，需要指定列索引
    
    Returns:
        x_data: (N, 2) array, [t, x]坐标
        rho_data, u_data, p_data: (N,) arrays, 观测值
    """
    if format == 'standard':
        # 标准格式：t x rho u p
        data = np.loadtxt(filename)
        x_data = data[:, [0, 1]]  # t, x
        rho_data = data[:, 2]
        u_data = data[:, 3]
        p_data = data[:, 4]
        
    elif format == 'csv':
        # CSV格式
        import pandas as pd
        df = pd.read_csv(filename)
        x_data = df[['t', 'x']].values
        rho_data = df['rho'].values
        u_data = df['u'].values
        p_data = df['p'].values
        
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return x_data, rho_data, u_data, p_data

def load_data_from_numpy(filename):
    """
    从numpy文件加载数据
    
    Args:
        filename: .npz或.npy文件
    
    Returns:
        x_data, rho_data, u_data, p_data
    """
    if filename.endswith('.npz'):
        data = np.load(filename)
        x_data = data['x_data']  # 或 data['coordinates']
        rho_data = data['rho']
        u_data = data['u']
        p_data = data['p']
    else:
        # .npy文件，假设是字典格式
        data = np.load(filename, allow_pickle=True).item()
        x_data = data['x_data']
        rho_data = data['rho']
        u_data = data['u']
        p_data = data['p']
    
    return x_data, rho_data, u_data, p_data

def load_data_partial(filename, variables=['p']):
    """
    加载部分物理量的数据
    
    Args:
        filename: 数据文件
        variables: 可用的物理量 ['rho', 'u', 'p']的子集
    
    Returns:
        x_data: 坐标
        data_dict: 包含可用物理量的字典
    """
    # 根据实际文件格式实现
    # 这里给一个示例
    data = np.loadtxt(filename)
    x_data = data[:, [0, 1]]
    
    data_dict = {}
    if 'rho' in variables:
        data_dict['rho'] = data[:, 2]
    if 'u' in variables:
        data_dict['u'] = data[:, 3]
    if 'p' in variables:
        data_dict['p'] = data[:, 4]
    
    return x_data, data_dict

def filter_data_by_time(x_data, rho_data, u_data, p_data, t_min=None, t_max=None):
    """
    按时间范围过滤数据
    
    Args:
        t_min, t_max: 时间范围
    """
    if t_min is None:
        t_min = x_data[:, 0].min()
    if t_max is None:
        t_max = x_data[:, 0].max()
    
    mask = (x_data[:, 0] >= t_min) & (x_data[:, 0] <= t_max)
    
    return x_data[mask], rho_data[mask], u_data[mask], p_data[mask]

def filter_data_by_space(x_data, rho_data, u_data, p_data, x_min=None, x_max=None):
    """
    按空间范围过滤数据
    """
    if x_min is None:
        x_min = x_data[:, 1].min()
    if x_max is None:
        x_max = x_data[:, 1].max()
    
    mask = (x_data[:, 1] >= x_min) & (x_data[:, 1] <= x_max)
    
    return x_data[mask], rho_data[mask], u_data[mask], p_data[mask]

def add_noise(data, noise_level=0.01):
    """
    添加噪声（用于测试或模拟测量误差）
    
    Args:
        noise_level: 噪声水平（相对标准差）
    """
    noise = np.random.normal(0, noise_level * np.abs(data))
    return data + noise

def normalize_data(x_data, rho_data, u_data, p_data, 
                   t_ref=None, x_ref=None, rho_ref=None, u_ref=None, p_ref=None):
    """
    归一化数据（可选，如果数据量级差异大）
    """
    if t_ref is None:
        t_ref = x_data[:, 0].max()
    if x_ref is None:
        x_ref = x_data[:, 1].max()
    if rho_ref is None:
        rho_ref = np.abs(rho_data).max()
    if u_ref is None:
        u_ref = np.abs(u_data).max()
    if p_ref is None:
        p_ref = np.abs(p_data).max()
    
    x_data_norm = x_data.copy()
    x_data_norm[:, 0] /= t_ref
    x_data_norm[:, 1] /= x_ref
    
    rho_data_norm = rho_data / rho_ref
    u_data_norm = u_data / u_ref
    p_data_norm = p_data / p_ref
    
    return (x_data_norm, rho_data_norm, u_data_norm, p_data_norm,
            {'t_ref': t_ref, 'x_ref': x_ref, 'rho_ref': rho_ref, 
             'u_ref': u_ref, 'p_ref': p_ref})

def convert_to_tensor(x_data, rho_data, u_data, p_data, dtype=torch.float64, device='cuda'):
    """
    转换为PyTorch tensor
    """
    x_tensor = torch.tensor(x_data, dtype=dtype, requires_grad=False).to(device)
    rho_tensor = torch.tensor(rho_data, dtype=dtype).to(device)
    u_tensor = torch.tensor(u_data, dtype=dtype).to(device)
    p_tensor = torch.tensor(p_data, dtype=dtype).to(device)
    
    return x_tensor, rho_tensor, u_tensor, p_tensor

# ========== 使用示例 ==========
if __name__ == '__main__':
    # 示例1：从文本文件加载
    # x_data, rho_data, u_data, p_data = load_data_from_txt('experimental_data.txt')
    
    # 示例2：过滤数据
    # x_data, rho_data, u_data, p_data = filter_data_by_time(
    #     x_data, rho_data, u_data, p_data, t_min=0.1, t_max=0.2)
    
    # 示例3：转换为tensor
    # x_tensor, rho_tensor, u_tensor, p_tensor = convert_to_tensor(
    #     x_data, rho_data, u_data, p_data)
    
    print("数据加载工具已就绪")
    print("使用方法：")
    print("  from data_loader import load_data_from_txt, convert_to_tensor")
    print("  x_data, rho_data, u_data, p_data = load_data_from_txt('your_data.txt')")
    print("  x_tensor, rho_tensor, u_tensor, p_tensor = convert_to_tensor(x_data, rho_data, u_data, p_data)")

