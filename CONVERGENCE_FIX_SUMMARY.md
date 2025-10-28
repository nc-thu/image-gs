# 手写反向传播收敛优化修复

## 🚨 关键问题诊断

通过对比训练日志发现关键问题：

### 收敛速度对比
- **手写反向传播**: Step 200 → PSNR: 18.33, SSIM: 0.555 (收敛缓慢)
- **原始PyTorch**: Step 200 → PSNR: 25.66, SSIM: 0.8363 (收敛快速)

### 根本原因分析
1. **损失函数配置错误**: 
   - 我们只用了 SSIM loss
   - 原始 `model_ori.py` 使用 **L1 (1.0) + SSIM (0.1)** 混合损失

2. **梯度信号太弱**: 
   - 纯SSIM梯度信号相对较弱
   - L1提供强梯度信号，SSIM提供细节优化

## ✅ 修复方案

### 1. 恢复原始损失函数配置
```python
# cfgs/default.yaml 的实际配置
l1_loss_ratio: 1.0      # 主要损失
l2_loss_ratio: 0.0      # 关闭
ssim_loss_ratio: 0.1    # 辅助损失
```

### 2. 修复梯度计算逻辑
```python
# 手写反向传播中的梯度计算顺序调整为：
# L1梯度（主要） + L2梯度 + SSIM梯度（辅助）
grad_images = torch.zeros_like(images)

if self.l1_loss is not None:
    grad_l1 = torch.sign(images - self.gt_images)
    grad_images += self.l1_loss_ratio * grad_l1

if self.ssim_loss is not None:
    grad_ssim = ssim_loss_backward(images, self.gt_images, self.ssim_loss)
    grad_images += self.ssim_loss_ratio * grad_ssim
```

### 3. 优化梯度处理策略
- **移除过度的梯度归一化**: 之前按像素数归一化导致梯度信号被过度削弱
- **移除过度的梯度裁剪**: 之前clamp(±5.0)限制了优化器的自然步长调节
- **保留基本的数值稳定性**: 仅处理NaN/Inf，保持梯度的自然量级

## 🎯 预期效果

修复后应该观察到：

1. **收敛速度显著提升**: 
   - Step 100: PSNR > 22, SSIM > 0.7
   - Step 200: PSNR > 25, SSIM > 0.8

2. **损失值合理**: 
   - Total Loss = L1_loss + 0.1 * SSIM_loss
   - L1占主导，SSIM辅助细节

3. **训练稳定性**: 
   - 梯度信号强而稳定
   - 优化器可以自然调节步长

## 🔧 技术细节

### L1梯度计算
```python
# L1损失: |pred - target|
# L1梯度: sign(pred - target)
grad_l1 = torch.sign(images - self.gt_images)
```

### SSIM梯度计算
```python
# 通过PyTorch autograd计算复杂的SSIM梯度
grad_ssim = ssim_loss_backward(images, self.gt_images, self.ssim_loss)
```

### 梯度组合
```python
# 按权重组合不同损失的梯度
total_grad = l1_ratio * grad_l1 + ssim_ratio * grad_ssim
```

## 📊 验证建议

运行修复后的代码，观察：
1. 前10步的损失下降趋势
2. Step 100/200的PSNR/SSIM指标
3. 总训练时间对比（应该接近原始PyTorch版本）

如果仍有问题，可能需要进一步调整：
- 检查学习率设置
- 验证手写tile-based反向传播的正确性
- 对比单步梯度的数值范围