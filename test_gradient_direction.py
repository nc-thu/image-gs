#!/usr/bin/env python3
"""
验证梯度符号的正确性
"""

import torch
import torch.nn.functional as F
import numpy as np

def simple_gaussians_forward(xy, colors, sigma_val=1.0):
    """
    简化的高斯前向传播，用于梯度验证
    """
    H, W = 8, 8  # 小图像
    C = 3  # RGB
    
    # 创建像素网格
    y_coords = torch.arange(H, dtype=torch.float32)
    x_coords = torch.arange(W, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    output = torch.zeros(C, H, W)
    
    for i, (center, color) in enumerate(zip(xy, colors)):
        dx = grid_x - center[0]
        dy = grid_y - center[1]
        sigma = sigma_val * (dx*dx + dy*dy)
        weight = torch.exp(-sigma)
        
        for c in range(C):
            output[c] += color[c] * weight
    
    return output

def test_gradient_direction():
    """测试梯度方向是否正确"""
    print("🔬 测试梯度方向...")
    
    # 简单设置：2个高斯，8x8图像
    xy = torch.tensor([[2.0, 2.0], [4.0, 4.0]], requires_grad=True)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], requires_grad=True)
    target = torch.ones(3, 8, 8) * 0.5  # 目标图像，正确尺寸
    
    # 前向传播
    output = simple_gaussians_forward(xy, colors)
    
    # L1损失
    loss = F.l1_loss(output, target)
    print(f"初始损失: {loss.item():.6f}")
    
    # 反向传播
    loss.backward()
    
    print(f"xy梯度: {xy.grad}")
    print(f"colors梯度: {colors.grad}")
    
    # 验证梯度方向：手动微调参数，看损失是否下降
    with torch.no_grad():
        lr = 0.01
        xy_new = xy - lr * xy.grad
        colors_new = colors - lr * colors.grad
        
        output_new = simple_gaussians_forward(xy_new, colors_new)
        loss_new = F.l1_loss(output_new, target)
        
        print(f"梯度下降后损失: {loss_new.item():.6f}")
        print(f"损失变化: {loss_new.item() - loss.item():.6f} (应该 < 0)")
        
        return loss_new.item() < loss.item()

def test_inverse_scale_gradient():
    """测试inverse scale的梯度处理"""
    print("\n🔬 测试inverse scale梯度...")
    
    # 测试 y = 1/x 的梯度
    scale = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    
    # 前向: inverse_scale = 1/scale
    inverse_scale = 1.0 / scale
    
    # 假设损失对inverse_scale的梯度
    v_inverse_scale = torch.tensor([0.1, -0.2, 0.3])
    
    # 反向传播
    inverse_scale.backward(v_inverse_scale)
    
    print(f"scale梯度 (PyTorch): {scale.grad}")
    
    # 手动计算: d/d(scale) = d/d(1/scale) * d(1/scale)/d(scale) = v_inverse * (-1/scale^2)
    manual_grad = v_inverse_scale * (-1.0 / (scale.detach() ** 2))
    print(f"scale梯度 (手动): {manual_grad}")
    
    # 验证一致性
    diff = torch.abs(scale.grad - manual_grad).max()
    print(f"梯度差异: {diff.item():.8f}")
    
    return diff < 1e-6

if __name__ == "__main__":
    print("🧪 梯度符号和方向验证\n")
    
    try:
        result1 = test_gradient_direction()
        result2 = test_inverse_scale_gradient()
        
        if result1 and result2:
            print("\n✅ 梯度符号验证通过！")
        else:
            print("\n❌ 梯度符号有问题！")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()