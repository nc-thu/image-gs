"""
高效的手动反向传播实现
使用tile-based方式，与CUDA前向传播保持一致的效率
"""

from __future__ import annotations
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple


def _cov2d_to_conic_vjp(conic: Tensor, v_conic: Tensor) -> Tensor:
    """计算从conic到cov2d的梯度"""
    a, b, c = conic.unbind(dim=-1)
    va, vb, vc = v_conic.unbind(dim=-1)
    xg00 = a * va + b * vb
    xg01 = a * vb + b * vc
    xg10 = b * va + c * vb
    xg11 = b * vb + c * vc
    v_sigma00 = -(xg00 * a + xg01 * b)
    v_sigma01 = -(xg00 * b + xg01 * c)
    v_sigma10 = -(xg10 * a + xg11 * b)
    v_sigma11 = -(xg10 * b + xg11 * c)
    return torch.stack(
        [v_sigma00, v_sigma01 + v_sigma10, v_sigma11], dim=-1
    )


def _rotmat(theta: Tensor) -> Tensor:
    """计算旋转矩阵"""
    cos_r = torch.cos(theta)
    sin_r = torch.sin(theta)
    row0 = torch.stack([cos_r, -sin_r], dim=-1)
    row1 = torch.stack([sin_r, cos_r], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _rotmat_grad(theta: Tensor) -> Tensor:
    """计算旋转矩阵的梯度"""
    cos_r = torch.cos(theta)
    sin_r = torch.sin(theta)
    row0 = torch.stack([-sin_r, -cos_r], dim=-1)
    row1 = torch.stack([cos_r, -sin_r], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def rasterize_backward_tile_based(
    xys: Tensor,
    conics: Tensor,
    colors: Tensor,
    radii: Tensor,
    grad_output: Tensor,
    tile_size: int = 16,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Tile-based高效反向传播
    
    Args:
        xys: [N, 2] 高斯中心位置
        conics: [N, 3] 二次型参数 (a, b, c)
        colors: [N, C] 颜色/特征
        radii: [N] 半径
        grad_output: [C, H, W] 输出梯度
        tile_size: tile大小，默认16
    
    Returns:
        v_xy: [N, 2] 位置梯度
        v_conic: [N, 3] conic梯度
        v_color: [N, C] 颜色梯度
    """
    device = xys.device
    dtype = xys.dtype
    num_gaussians = xys.shape[0]
    channels, img_h, img_w = grad_output.shape
    
    # 初始化梯度
    v_xy = torch.zeros_like(xys)
    v_conic = torch.zeros_like(conics)
    v_color = torch.zeros_like(colors)
    
    # 计算tile数量
    tile_h = (img_h + tile_size - 1) // tile_size
    tile_w = (img_w + tile_size - 1) // tile_size
    
    # 为每个tile分配高斯
    print(f"  [Manual Backward] Processing {tile_h}x{tile_w} tiles...")
    
    # 批处理高斯以节省内存
    batch_size = min(1000, num_gaussians)
    
    for tile_y in range(tile_h):
        for tile_x in range(tile_w):
            # tile边界
            y_start = tile_y * tile_size
            y_end = min(y_start + tile_size, img_h)
            x_start = tile_x * tile_size
            x_end = min(x_start + tile_size, img_w)
            
            # 提取tile的梯度
            tile_grad = grad_output[:, y_start:y_end, x_start:x_end]  # [C, tile_h, tile_w]
            
            # 批处理高斯
            for batch_start in range(0, num_gaussians, batch_size):
                batch_end = min(batch_start + batch_size, num_gaussians)
                
                # 筛选与该tile相交的高斯
                batch_xys = xys[batch_start:batch_end]
                batch_radii = radii[batch_start:batch_end]
                
                # 判断高斯是否与tile相交
                intersect_mask = (
                    (batch_xys[:, 0] + batch_radii >= x_start) &
                    (batch_xys[:, 0] - batch_radii < x_end) &
                    (batch_xys[:, 1] + batch_radii >= y_start) &
                    (batch_xys[:, 1] - batch_radii < y_end) &
                    (batch_radii > 0)
                )
                
                if not torch.any(intersect_mask):
                    continue
                
                # 获取相交的高斯索引
                intersect_indices = torch.where(intersect_mask)[0] + batch_start
                n_intersect = len(intersect_indices)
                
                if n_intersect == 0:
                    continue
                
                # 提取相交高斯的数据
                g_xy = xys[intersect_indices]  # [n, 2]
                g_conic = conics[intersect_indices]  # [n, 3]
                g_color = colors[intersect_indices]  # [n, C]
                
                # 构建像素网格
                tile_h_actual = y_end - y_start
                tile_w_actual = x_end - x_start
                y_coords = torch.arange(y_start, y_end, device=device, dtype=dtype)
                x_coords = torch.arange(x_start, x_end, device=device, dtype=dtype)
                grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
                
                # [n, 1, 1] - [tile_h, tile_w] = [n, tile_h, tile_w]
                dx = g_xy[:, 0:1, None] - grid_x[None, :, :]
                dy = g_xy[:, 1:2, None] - grid_y[None, :, :]
                
                # 计算高斯权重
                a = g_conic[:, 0:1, None]
                b = g_conic[:, 1:2, None]
                c = g_conic[:, 2:3, None]
                sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
                
                # 权重和有效性检查
                valid = (sigma >= 0) & torch.isfinite(sigma)
                weight = torch.where(valid, torch.exp(-sigma), torch.zeros_like(sigma))
                
                # ===== 反向传播计算 =====
                
                # 1. 颜色梯度: v_color += weight * grad_output
                # weight: [n, tile_h, tile_w], tile_grad: [C, tile_h, tile_w]
                # 需要: [n, C]
                weighted_grad = weight.unsqueeze(1) * tile_grad.unsqueeze(0)  # [n, C, tile_h, tile_w]
                v_color_batch = weighted_grad.sum(dim=(2, 3))  # [n, C]
                v_color.index_add_(0, intersect_indices, v_color_batch)
                
                # 2. sigma梯度: v_sigma = -weight * (color · grad_output)
                color_dot_grad = (g_color.unsqueeze(-1).unsqueeze(-1) * tile_grad.unsqueeze(0)).sum(dim=1)
                # color_dot_grad: [n, tile_h, tile_w]
                v_sigma = -weight * color_dot_grad
                
                # 3. conic梯度
                dx2 = dx * dx
                dy2 = dy * dy
                dxdy = dx * dy
                
                v_conic_a = 0.5 * (v_sigma * dx2).sum(dim=(1, 2))  # [n]
                v_conic_b = (v_sigma * dxdy).sum(dim=(1, 2))
                v_conic_c = 0.5 * (v_sigma * dy2).sum(dim=(1, 2))
                v_conic_batch = torch.stack([v_conic_a, v_conic_b, v_conic_c], dim=-1)
                v_conic.index_add_(0, intersect_indices, v_conic_batch)
                
                # 4. 位置梯度
                vx = (v_sigma * (a.squeeze(-1) * dx + b.squeeze(-1) * dy)).sum(dim=(1, 2))
                vy = (v_sigma * (b.squeeze(-1) * dx + c.squeeze(-1) * dy)).sum(dim=(1, 2))
                v_xy_batch = torch.stack([vx, vy], dim=-1)
                v_xy.index_add_(0, intersect_indices, v_xy_batch)
    
    return v_xy, v_conic, v_color


def project_backward_scale_rot(
    means2d: Tensor,
    scales2d: Tensor,
    rotation: Tensor,
    grad_xy: Tensor,
    grad_conic: Tensor,
    img_height: int,
    img_width: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    投影的反向传播：从xy和conic梯度计算到means2d、scales2d、rotation的梯度
    """
    theta = rotation.view(-1)
    v_cov2d = _cov2d_to_conic_vjp(conic=grad_conic, v_conic=grad_conic)

    v_mean = torch.zeros_like(means2d)
    v_mean[:, 0] = grad_xy[:, 0] * img_width
    v_mean[:, 1] = grad_xy[:, 1] * img_height

    R = _rotmat(theta).to(dtype=scales2d.dtype)
    Rg = _rotmat_grad(theta).to(dtype=scales2d.dtype)

    S = torch.zeros(theta.shape[0], 2, 2, device=scales2d.device, dtype=scales2d.dtype)
    S[:, 0, 0] = scales2d[:, 0]
    S[:, 1, 1] = scales2d[:, 1]

    M = torch.matmul(R, S)
    Mt = M.transpose(-1, -2)

    scale_x_g = torch.zeros_like(S)
    scale_y_g = torch.zeros_like(S)
    scale_x_g[:, 0, 0] = 2.0 * scales2d[:, 0]
    scale_y_g[:, 1, 1] = 2.0 * scales2d[:, 1]

    sigma_x_g = torch.matmul(torch.matmul(R, scale_x_g), R.transpose(-1, -2))
    sigma_y_g = torch.matmul(torch.matmul(R, scale_y_g), R.transpose(-1, -2))

    theta_g = torch.matmul(torch.matmul(Rg, S), Mt) + torch.matmul(M, torch.matmul(S, Rg.transpose(-1, -2)))

    G11 = v_cov2d[:, 0]
    G12 = v_cov2d[:, 1]
    G22 = v_cov2d[:, 2]

    v_scale = torch.zeros_like(scales2d)
    v_scale[:, 0] = (
        G11 * sigma_x_g[:, 0, 0]
        + 2.0 * G12 * sigma_x_g[:, 0, 1]
        + G22 * sigma_x_g[:, 1, 1]
    )
    v_scale[:, 1] = (
        G11 * sigma_y_g[:, 0, 0]
        + 2.0 * G12 * sigma_y_g[:, 0, 1]
        + G22 * sigma_y_g[:, 1, 1]
    )

    v_rot = (
        G11 * theta_g[:, 0, 0]
        + 2.0 * G12 * theta_g[:, 0, 1]
        + G22 * theta_g[:, 1, 1]
    ).unsqueeze(-1)

    return v_mean, v_scale, v_rot


def ssim_loss_backward(pred: Tensor, target: Tensor, loss_value: Tensor) -> Tensor:
    """
    SSIM损失的手动梯度计算（使用数值微分近似）
    
    Args:
        pred: 预测图像 [C, H, W]
        target: 目标图像 [C, H, W]
        loss_value: SSIM损失值
    
    Returns:
        grad: 对pred的梯度 [C, H, W]
    """
    # 使用autograd计算SSIM的梯度（这部分可以保留，因为SSIM很复杂）
    pred_grad = pred.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        from fused_ssim import fused_ssim
        ssim_val = 1 - fused_ssim(pred_grad.unsqueeze(0), target.unsqueeze(0))
        ssim_val.backward()
    
    return pred_grad.grad
