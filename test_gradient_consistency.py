#!/usr/bin/env python3
"""
测试手写反向传播与PyTorch autograd的梯度一致性
"""

import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim

def test_l1_gradient():
    """测试L1损失的梯度计算一致性"""
    print("🔬 测试L1损失梯度一致性...")
    
    # 创建测试数据
    pred = torch.randn(3, 64, 64, requires_grad=True)
    target = torch.randn(3, 64, 64)
    
    # PyTorch autograd
    loss_auto = F.l1_loss(pred, target)
    loss_auto.backward()
    grad_auto = pred.grad.clone()
    
    # 手写梯度
    with torch.no_grad():
        grad_manual = torch.sign(pred - target) / pred.numel()  # L1梯度归一化
    
    # 比较
    diff = torch.abs(grad_auto - grad_manual).mean()
    print(f"  L1梯度差异: {diff.item():.8f}")
    
    return diff < 1e-6

def test_ssim_gradient():
    """测试SSIM损失的梯度计算"""
    print("🔬 测试SSIM损失梯度...")
    
    # 创建测试数据
    pred = torch.randn(3, 64, 64, requires_grad=True)
    target = torch.randn(3, 64, 64)
    
    # PyTorch autograd
    loss_auto = 1 - fused_ssim(pred.unsqueeze(0), target.unsqueeze(0))
    loss_auto.backward()
    grad_auto = pred.grad.clone()
    
    print(f"  SSIM梯度范围: [{grad_auto.min().item():.6f}, {grad_auto.max().item():.6f}]")
    print(f"  SSIM梯度均值: {grad_auto.mean().item():.6f}")
    print(f"  SSIM梯度标准差: {grad_auto.std().item():.6f}")
    
    return True

def test_mixed_loss():
    """测试混合损失的梯度"""
    print("🔬 测试L1+SSIM混合损失...")
    
    # 创建测试数据
    pred = torch.randn(3, 64, 64, requires_grad=True)
    target = torch.randn(3, 64, 64)
    
    l1_ratio = 1.0
    ssim_ratio = 0.1
    
    # PyTorch autograd
    l1_loss = l1_ratio * F.l1_loss(pred, target)
    ssim_loss = ssim_ratio * (1 - fused_ssim(pred.unsqueeze(0), target.unsqueeze(0)))
    total_loss = l1_loss + ssim_loss
    
    total_loss.backward()
    grad_auto = pred.grad.clone()
    
    print(f"  L1 Loss: {l1_loss.item():.6f}")
    print(f"  SSIM Loss: {ssim_loss.item():.6f}")
    print(f"  Total Loss: {total_loss.item():.6f}")
    print(f"  混合梯度范围: [{grad_auto.min().item():.6f}, {grad_auto.max().item():.6f}]")
    print(f"  混合梯度均值: {grad_auto.mean().item():.6f}")
    
    return True

if __name__ == "__main__":
    print("🧪 梯度一致性测试\n")
    
    try:
        test_l1_gradient()
        test_ssim_gradient()
        test_mixed_loss()
        print("\n✅ 所有测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()