# 层次化双路径点云Transformer (HDPT)

本仓库实现了层次化双路径点云Transformer (HDPT)，这是一个针对点云分类任务的改进版Point Transformer模型。

## 架构概览

### Baseline与HDPT对比

#### 1. 网络架构

**Baseline:**
- 简单的顺序架构
- 单一路径特征提取
- 固定尺度的特征处理
- 有限的特征交互
- 基础的transformer层结构

**HDPT (我们的改进):**
- 层次化编码器-解码器架构 
- 双路径特征传播
- 多尺度特征提取
- 通过跳跃连接实现丰富的特征交互
- 增强型transformer层与改进的位置编码

#### 2. 特征提取策略

**Baseline:**
```python
# 顺序特征提取
x = self.transformer1(xyz, x)  # 第1层
x = self.transformer2(xyz, x)  # 第2层
x = self.transformer3(xyz, x)  # 第3层
x = self.transformer4(xyz, x)  # 第4层
```

**HDPT:**
```python
# 层次化特征提取
# 编码器路径
f1 = self.transformer1(xyz, x)          # 第1层: N个点
f2 = self.down1(f1, xyz)                # 第2层: N/2个点
f3 = self.transformer2(xyz2, f2)        # 第3层: N/4个点
f4 = self.down2(f3, xyz3)               # 第4层: N/8个点

# 解码器路径
up_f3 = self.up1(f3, f4, xyz3, xyz4)    # 特征融合
up_f2 = self.up2(f2, up_f3, xyz2, xyz3) # 多尺度学习
up_f1 = self.up3(f1, up_f2, xyz, xyz2)  # 丰富特征交互
```

#### 3. 数据处理

**Baseline:**
- 基础的点采样
- 简单的标准化
- 无数据增强

**HDPT:**
- 高级点采样策略
- 全面的数据增强:
  - 随机旋转
  - 随机缩放
  - 随机点丢弃
  - 随机平移
- 增强的标准化技术

#### 4. 训练策略

**Baseline:**
- 固定学习率
- 简单的Adam优化器
- 基础损失函数

**HDPT:**
- 余弦退火学习率调度
- AdamW优化器与权重衰减
- 高级训练技巧:
  - 梯度裁剪
  - 学习率预热
  - 模型检查点保存
  - 最佳模型保存

#### 5. 主要创新点

1. **层次化特征学习**
   - 在不同尺度上逐步抽象特征
   - 同时捕获局部和全局几何信息
   - 更好地处理不同密度的点云

2. **多尺度特征融合**
   ```python
   # 多尺度特征融合
   global_f4 = torch.max(f4, dim=1)[0]  # 全局特征
   global_f3 = torch.max(f3, dim=1)[0]  # 中层特征
   global_f2 = torch.max(f2, dim=1)[0]  # 局部特征
   global_f1 = torch.max(up_f1, dim=1)[0]  # 细粒度特征
   
   # 特征拼接
   global_feature = torch.cat([global_f4, global_f3, global_f2, global_f1], dim=-1)
   ```

3. **改进的Transformer层**
   - 增强的位置编码
   - 更好的特征交互
   - 残差连接与归一化

4. **双路径架构**
   - 编码器路径：逐步降采样，提取高层特征
   - 解码器路径：逐步上采样，恢复细节信息
   - 跳跃连接：保留多尺度特征信息

#### 6. 性能优势

1. **特征提取能力**
   - 基线模型只能获取单一尺度的特征
   - HDPT可以获取多个尺度的几何特征，特征表达更丰富

2. **鲁棒性**
   - 基线模型对点云密度和噪声较敏感
   - HDPT通过多尺度处理和数据增强提高了鲁棒性

3. **计算效率**
   - 基线模型处理所有点的计算复杂度较高
   - HDPT通过层次化处理降低了计算复杂度

4. **泛化能力**
   - 基线模型容易过拟合
   - HDPT通过多种正则化技术和数据增强提高了泛化能力

## 使用说明

### 环境要求
```
torch>=1.7.0
numpy>=1.19.2
sklearn>=0.23.2
```

### 训练模型
```bash
python train.py --config configs/hdpt.yaml
```

### 测试模型
```bash
python test.py --model-path checkpoints/best_model.pth
```

## 引用
如果您使用了本工作，请引用我们的论文：
```
待发表
```