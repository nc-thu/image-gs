#!/usr/bin/env python3
"""
测试手写反向传播的小型脚本
"""

import torch
import torch.nn as nn
from manual_backward_efficient import rasterize_backward_tile_based, project_backward_scale_rot

def test_manual_backward():
    print("🔬 测试手写反向传播模块...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    N = 100  # 高斯数量
    H, W, C = 64, 64, 3  # 图像尺寸
    
    xys = torch.randn(N, 2, device=device)
    conics = torch.abs(torch.randn(N, 3, device=device)) + 0.1
    colors = torch.randn(N, C, device=device)
    radii = torch.abs(torch.randn(N, device=device)) + 1.0
    grad_output = torch.randn(C, H, W, device=device)
    
    print(f"输入数据形状:")
    print(f"  xys: {xys.shape}")
    print(f"  conics: {conics.shape}")
    print(f"  colors: {colors.shape}")
    print(f"  radii: {radii.shape}")
    print(f"  grad_output: {grad_output.shape}")
    
    try:
        # 测试光栅化反向传播
        v_xy, v_conic, v_color = rasterize_backward_tile_based(
            xys, conics, colors, radii, grad_output, tile_size=16
        )
        
        print(f"光栅化反向传播输出形状:")
        print(f"  v_xy: {v_xy.shape}")
        print(f"  v_conic: {v_conic.shape}")
        print(f"  v_color: {v_color.shape}")
        
        # 测试投影反向传播
        means2d = torch.randn(N, 2, device=device)
        scales2d = torch.abs(torch.randn(N, 2, device=device)) + 0.1
        rotation = torch.randn(N, 1, device=device)
        
        v_mean, v_scale, v_rot = project_backward_scale_rot(
            means2d, scales2d, rotation, v_xy, v_conic, H, W
        )
        
        print(f"投影反向传播输出形状:")
        print(f"  v_mean: {v_mean.shape}")
        print(f"  v_scale: {v_scale.shape}")
        print(f"  v_rot: {v_rot.shape}")
        
        print("✅ 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_manual_backward()