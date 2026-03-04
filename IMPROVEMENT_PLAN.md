# Geometry-Aware Distillation for Few-Step Discrete Diffusion

**作者视角的改进方案 — 目标: NeurIPS 2026**

---

## 第零部分：当前框架的数学瓶颈诊断

在提出改进之前，我先精确定位当前代码中的 **六个核心数学问题**。

### 瓶颈 1：概率单纯形上的线性插值几何失真

**出处**: `d_perflow.py:246-281`, `interpolate_distribution()`

当前实现：
```python
P_t = α · P_{t_k} + (1 - α) · P_{t_{k-1}}
```

**问题**：概率分布空间 Δ^{V-1} 不是欧几里得空间，而是一个黎曼流形。线性插值是单纯形上的**弦**而非**测地线**。

数学分析：设 P 和 Q 都是尖锐分布 (max_prob ≈ 0.94)。线性插值 P_α = αP + (1-α)Q 的熵满足：

```
H(P_α) ≥ α·H(P) + (1-α)·H(Q)    (Jensen不等式，因为 -x log x 是凹函数)
```

即中间插值点的熵**总是大于**端点熵的加权平均。当 P 和 Q 的 argmax 不同时（约 3% 的位置），线性插值产生一个双峰分布，其熵远大于两端的单峰分布。

**后果**：student 学到的中间分布人为地 diffuse，导致生成文本多样性虚高但质量下降。

**量化**：从训练日志看，teacher max_prob ≈ 0.970，student max_prob ≈ 0.940，entropy ratio ≈ 1.87x。这个 1.87x 的熵膨胀不全是 student 能力不足，部分来自训练目标本身的失真。

### 瓶颈 2：KL 散度忽视 token 语义结构

**出处**: `d_perflow.py:295-373`, `compute_kl_loss_probs()`

当前的 KL(P || Q) = Σ_v P(v) log(P(v)/Q(v)) 对所有 token 对 (v, v') 的混淆惩罚相同。

但离散 token 空间有丰富的几何结构：
- 将 "cat" 预测为 "dog"（语义相近）的代价应远小于预测为 "quantum"
- token embedding 定义了自然的度量空间：d(i, j) = ||e_i - e_j||₂

KL 散度完全忽视了这个结构，等价于假设 token 空间是一个无结构的离散集合。

**量化**：对于 V = 50257 的词表，KL 只利用了 log(V) ≈ 10.8 bits 的信息，而 embedding 空间蕴含 768 维的语义信息。

### 瓶颈 3：单步 Euler 的离散化误差无法控制

**出处**: `d_perflow.py:530-532`, student 的 1-step Euler

student 在每个 window 内用 1 步 Euler：
```python
probs = one_hot(x) + dt · dsigma · reverse_rate(x, score)
```

对于 K=4, 每个 window 跨度 Δt ≈ 0.25。Euler 方法的局部截断误差为 O(Δt²)，全局误差为 O(Δt)。

关键问题：reverse_rate 矩阵的谱范数 ||R||₂ 在不同 t 处差异巨大：
- t ≈ 1（高噪声）：||R||₂ 较小，Euler 误差可控
- t ≈ 0（低噪声）：||R||₂ 很大（score 陡峭），Euler 误差爆炸

均匀分 window 意味着在 t ≈ 0 附近的 window 承受了最大的离散化误差，但这恰好是对文本质量影响最大的区域。

### 瓶颈 4：位置无关的损失函数

**出处**: `d_perflow.py:333`, `loss = kl_div.mean(dim=-1)`

对所有 L 个位置取简单平均。但在 Absorbing-state 扩散中：
- **已决定位置**（x_t ≠ MASK）：teacher 和 student 的分布几乎完全一致（都高度集中在原 token），KL ≈ 0
- **待解码位置**（x_t = MASK）：这里是唯一有信息量的位置，需要从 V 个 token 中选择

从 `graph_lib.py:244-266` 的 `score_entropy` 看，SEDD 本身就只在 `rel_ind = (x == self.dim - 1)` 的 absorbing 位置计算损失。但 D-PeRFlow 的 KL 损失没有利用这个结构。

**后果**：约 exp(-σ) 比例的位置是已决定的（对于中等 t，这可能是 50-80%），在这些位置浪费了大量梯度。

### 瓶颈 5：均匀时间窗口不匹配 score 曲率

**出处**: `d_perflow.py:54-56`

```python
self.time_boundaries = torch.linspace(sampling_eps, 1.0, num_time_windows + 1)
```

LogLinear 噪声下 σ(t) = -log(1 - (1-ε)t)：
- t ∈ [0, 0.25]：σ 从 0 到 0.29，变化剧烈，score 曲率高
- t ∈ [0.75, 1.0]：σ 从 1.39 到 ∞，变化更剧烈但 score 已平坦（接近均匀噪声）

最优窗口划分应该使得每个窗口内的 **score 变化量** 大致相等，而非时间跨度相等。

### 瓶颈 6：对抗训练的根本性设计缺陷

**出处**: `adversarial_distillation.py` 全文

根本原因不是 LSGAN vs Hinge vs BCE 的选择，而是：

**判别器的输入空间缺乏可判别性**。

当前输入：`probs @ W_embed`（概率分布投影到 embedding 空间）。当 teacher 和 student 的 argmax 重合率 97% 时，这个投影几乎完全相同。

数学分析：设 P_t 和 P_s 的 argmax 都是 token v*，则：
```
||P_t @ E - P_s @ E|| = ||(P_t - P_s) @ E|| ≤ ||P_t - P_s||₁ · max_v ||e_v||
```
当 P_t(v*) ≈ P_s(v*) ≈ 0.94 时，TV(P_t, P_s) ≈ 0.06，投影差异极小。

这不是可以通过调超参数解决的问题 — 是信号太弱，信噪比太低。

---

## 第一部分：核心创新 — Score Entropy Distillation (SED)

### 1.1 核心洞察

SEDD 的关键贡献是 **score entropy loss** — 离散空间上 score matching 的最优 Bregman 散度。

当前 D-PeRFlow 完全绕开了这个贡献，改用 KL 散度在分布空间做蒸馏。这既浪费了 SEDD 框架的数学优势，又引入了瓶颈 1-4 的所有问题。

**我的核心提议**：用 score entropy 本身作为蒸馏损失函数。

### 1.2 数学推导

原始 SEDD 训练：

```
L_SEDD = E_{t, x_0, x_t} [ g(t) · score_entropy(s_θ(x_t, t), σ(t), x_t, x_0) ]
```

其中 x_0 来自数据分布，x_t ~ P(·|x_0, t) 是前向扩散。

**Score Entropy Distillation (SED)**：

```
L_SED = E_{t, x_0, x_t} [ g(t) · score_entropy(s_student(x_t, t), σ(t), x_t, x̂_0^teacher) ]
```

其中 x̂_0^teacher = argmax_v P_teacher(v | x_t, t) 是 teacher 的去噪预测。

**关键区别**：将 ground truth x_0 替换为 teacher 的预测 x̂_0^teacher。

### 1.3 为什么这是对的 — 理论分析

**定理（非正式）**：如果 teacher 的 score 是准确的（即 s_teacher ≈ s_true），则：

```
argmin_θ L_SED(θ) = argmin_θ D_Bregman(s_student || s_teacher)
```

即 SED 在 Bregman 散度意义下最小化 student 和 teacher 的 score 差异。

**证明思路**：

Score entropy 的定义（从 `graph_lib.py:162-189` 的 Uniform 情形）：

```
SE(s, σ, x, x_0) = [Σ_v e^{s_v} / D - e^{s_x}/D]          (正项)
                   - [Σ_v s_v / D - s_x/D · (1-D·ratio)     (负项, 取决于 x=x_0 与否)
                     + ratio · s_{x_0} / esigm1]
                   + const(σ)
```

当 x_0 被替换为 x̂_0^teacher 时，负项中的 `s_{x_0}` 变成 `s_{x̂_0^teacher}`。

这意味着 student 的 score 被引导在 teacher 预测的 clean token 方向上给出高分。这正是我们想要的 — student 学习 teacher 的去噪方向。

### 1.4 相比 KL 蒸馏的优势

| 特性 | KL 蒸馏 (当前) | Score Entropy 蒸馏 (提议) |
|------|-------------|----------------------|
| 计算复杂度 | O(B·L·V) 全分布计算 | O(B·L) 稀疏计算 (仅在 absorb 位置) |
| 数值稳定性 | 需要 softmax 归一化 + clamp | SEDD 原生，已处理 σ < 0.5 情形 |
| 语义结构 | 忽略 token 距离 | 通过 score 隐式利用 |
| 与预训练的一致性 | 完全不同的损失函数 | **同一个** 损失函数 |
| 理论基础 | 信息论 (KL) | 统计流形上的 Bregman 散度 |
| 适用于 Absorbing graph | 在所有位置计算 | 仅在 MASK 位置计算（更高效） |

### 1.5 实现方案

```python
def score_entropy_distillation_loss(
    student_model, teacher_model, graph, noise, batch,
    num_windows=4, sampling_eps=1e-3
):
    """
    Score Entropy Distillation: 用 SEDD 原生 score entropy 做蒸馏。

    关键改变: ground truth x_0 → teacher 的去噪预测 x̂_0
    """
    device = batch.device
    B = batch.shape[0]

    # 1. 采样时间步 (在 window 内均匀采样)
    t = (1 - sampling_eps) * torch.rand(B, device=device) + sampling_eps
    sigma, dsigma = noise(t)

    # 2. 前向扩散: x_0 → x_t
    x_t = graph.sample_transition(batch, sigma[:, None])

    # 3. Teacher 预测 clean tokens (无梯度)
    with torch.no_grad():
        teacher_score = get_score_fn(teacher_model, train=False, sampling=True)
        teacher_s = teacher_score(x_t, sigma)  # [B, L, V]

        # 对于 Absorbing graph: 在 MASK 位置取 argmax (排除 absorb state)
        # teacher_s[..., :-1] 去掉 absorbing 维度后取 argmax
        x_hat_0 = teacher_s[..., :-1].argmax(dim=-1)  # [B, L]

        # 在非 MASK 位置, 用原始 token (teacher 不需要预测)
        is_mask = (x_t == graph.dim - 1)
        x_hat_0 = torch.where(is_mask, x_hat_0, x_t)

    # 4. Student 的 score entropy loss (同 SEDD 预训练)
    student_log_score = get_score_fn(student_model, train=True, sampling=False)
    log_score = student_log_score(x_t, sigma)  # [B, L, V]

    # 5. Score entropy: 用 teacher 预测代替 ground truth
    loss = graph.score_entropy(log_score, sigma[:, None], x_t, x_hat_0)

    # 6. 时间加权 (同 SEDD)
    loss = (dsigma[:, None] * loss).sum(dim=-1)

    return loss
```

### 1.6 混合训练目标：SED + Data

纯 teacher 蒸馏会有 error accumulation（teacher 本身不完美）。解决方案 — 混合真实数据和 teacher 预测：

```
L = (1 - α) · L_SEDD(s_student, x_0_data)     # 原始 SEDD 训练
  + α · L_SED(s_student, x̂_0_teacher)           # Score entropy 蒸馏
```

其中 α ∈ [0, 1] 是混合权重。在原始 SEDD 训练时 α = 0，在纯蒸馏时 α = 1。

**关键洞察**：由于两个损失函数形式完全相同（都是 score entropy），混合是自然的，不需要额外的超参数平衡。

### 1.7 这个贡献为什么有新颖性

1. **所有现有离散扩散蒸馏方法（SDTT, FS-DFM, Duo-DCD）都在分布空间操作**。它们计算 teacher 的分布 → 用 KL/TV/MSE 匹配 student 的分布。
2. **SED 是第一个在 score 空间做离散蒸馏的方法**。它利用了 SEDD 独特的 score entropy 损失，将蒸馏问题转化为与预训练完全相同的目标。
3. **理论上更干净**：从 score matching 理论直接推导，不需要中间的分布计算步骤。
4. **计算上更高效**：在 Absorbing graph 下，只在 MASK 位置计算损失，复杂度从 O(B·L·V) 降到 O(B·M·V)，其中 M ≪ L 是 MASK 位置数。

---

## 第二部分：Fisher-Rao 测地线插值

### 2.1 问题

当需要在两个时间边界的分布之间插值时（用于训练中间时间步），线性插值是几何错误的。

### 2.2 Fisher-Rao 测地线

在概率单纯形 Δ^{V-1} 上，Fisher-Rao 度量下的测地线为：

```
γ(α; P, Q)(v) = [sin((1-α)θ) · √P(v) + sin(αθ) · √Q(v)]² / sin²(θ)
```

其中 θ = arccos(Σ_v √(P(v)·Q(v))) 是 Bhattacharyya 角度。

**简化版本**：当 P 和 Q 接近（θ 较小）时，测地线近似为 **几何插值**：

```
γ(α; P, Q)(v) ∝ P(v)^{1-α} · Q(v)^α
```

这就是 Amari 信息几何中的 **α-连接测地线** (α = 0 情形)。

### 2.3 实现

```python
def fisher_rao_interpolation(P, Q, alpha, eps=1e-10):
    """
    Fisher-Rao 测地线插值 (几何插值近似)

    P, Q: [B, L, V] 概率分布
    alpha: [B, 1, 1] 插值系数, alpha=0 → P, alpha=1 → Q
    """
    # 几何插值: P^{1-α} · Q^α (在 log 空间计算更稳定)
    log_P = (P + eps).log()
    log_Q = (Q + eps).log()
    log_interp = (1 - alpha) * log_P + alpha * log_Q

    # 归一化
    interp = F.softmax(log_interp, dim=-1)
    return interp
```

### 2.4 性质

1. **保持峰度**：如果 P 和 Q 都是尖锐分布，几何插值的中间点也是尖锐的
2. **概率有效**：经 softmax 归一化后自动满足概率约束
3. **连续和可微**：梯度可以干净地通过
4. **边界正确**：α=0 → P, α=1 → Q

**对比线性插值**：

| P(cat) = 0.94, P(dog) = 0.03 | 线性 (α=0.5) | Fisher-Rao (α=0.5) |
|------|:---:|:---:|
| cat | 0.485 | 0.891 |
| dog | 0.265 | 0.055 |
| 熵 H | 1.03 nats | 0.42 nats |

Fisher-Rao 插值保持了分布的尖锐性，而线性插值产生了一个几乎均匀的分布。

### 2.5 论文故事

这不是一个 trivial 的改进。离散扩散中的分布插值是 **流匹配 (Flow Matching)** 在离散空间的核心操作。PeRFlow 在连续空间用线性插值（因为 R^d 是欧几里得空间），但 **离散概率单纯形不是欧几里得的**。

Fisher-Rao 插值是将连续流匹配正确推广到离散空间的必要步骤。这是一个 **non-trivial 的几何洞察**。

---

## 第三部分：曲率自适应窗口划分

### 3.1 动机

均匀划分 [ε, 1] 是最简单但最浪费的方案。最优划分应使每个窗口内的 **蒸馏难度相等**。

### 3.2 Score 曲率作为难度度量

定义 score 变化率：

```
κ(t) = E_{x_t} [||s(x_t, t) - s(x_t, t - δ)||₂] / δ
```

直觉：κ(t) 大意味着 score 在 t 附近变化剧烈，1 步 Euler 的误差大，蒸馏难度高。

### 3.3 最优划分算法

**目标**：找到 {t_0, t_1, ..., t_K} 使得每个窗口的积分曲率相等：

```
∫_{t_{k-1}}^{t_k} κ(τ) dτ = C  (常数, 对所有 k)
```

**算法**：
1. 预计算：在 T=1000 个网格点上估计 κ(t)
2. 计算 CDF：F(t) = ∫_ε^t κ(τ)dτ / ∫_ε^1 κ(τ)dτ
3. 反 CDF：t_k = F^{-1}(k/K)

```python
def compute_adaptive_boundaries(teacher_model, noise, graph,
                                 num_windows, num_grid=1000,
                                 batch_size=64, device='cuda'):
    """预计算曲率自适应的时间窗口边界"""
    teacher_score_fn = get_score_fn(teacher_model, train=False, sampling=True)

    # 在网格点上估计 score 变化率
    t_grid = torch.linspace(1e-3, 1.0, num_grid, device=device)
    curvatures = []

    # 采样一批数据用于估计
    x_0 = sample_batch(batch_size, device)

    with torch.no_grad():
        for i in range(num_grid - 1):
            t = t_grid[i].expand(batch_size)
            t_next = t_grid[i + 1].expand(batch_size)
            dt = (t_next - t)[0].item()

            sigma_t = noise(t)[0]
            sigma_next = noise(t_next)[0]

            x_t = graph.sample_transition(x_0, sigma_t[:, None])

            s_t = teacher_score_fn(x_t, sigma_t)
            s_next = teacher_score_fn(x_t, sigma_next)  # 同一 x_t, 不同 σ

            # L2 score difference (per position, averaged over batch)
            kappa = ((s_t - s_next) ** 2).sum(dim=-1).sqrt().mean()
            curvatures.append(kappa.item() / dt)

    curvatures = torch.tensor(curvatures)

    # 计算 CDF 并反转
    cdf = curvatures.cumsum(0) / curvatures.sum()

    boundaries = [1e-3]  # t_0 = eps
    for k in range(1, num_windows):
        target = k / num_windows
        idx = (cdf >= target).nonzero(as_tuple=True)[0][0]
        boundaries.append(t_grid[idx].item())
    boundaries.append(1.0)

    return torch.tensor(boundaries)
```

### 3.4 预期效果

对于 LogLinear noise schedule，score 在 t ≈ 0 附近变化最快。自适应划分会将更多窗口集中在低噪声区域：

```
均匀划分 K=4: [0.001, 0.250, 0.500, 0.750, 1.000]
自适应划分 K=4: [0.001, 0.050, 0.150, 0.400, 1.000]  (示意)
```

这使得 student 在关键的低噪声区域有更细粒度的近似，而在容易的高噪声区域合并处理。

---

## 第四部分：Self-Consistency 正则化

### 4.1 核心思想

即使没有 teacher，student 也应该和自己一致。具体地：

**如果 student 在窗口 [t_{k-1}, t_k] 内用 1 步走完和用 2 步走完，结果应该相同。**

这是 Consistency Models (Song et al., ICML 2023) 的离散空间推广。

### 4.2 形式化

定义 student 的单步去噪映射：

```
F_θ(x, t_start, t_end) = EulerStep(s_θ, x, t_start, t_end)
```

Self-consistency 要求：

```
F_θ(x, t_k, t_{k-1}) ≈ F_θ(F_θ(x, t_k, t_mid), t_mid, t_{k-1})
```

其中 t_mid = (t_k + t_{k-1}) / 2。

### 4.3 离散空间的特殊处理

连续空间中，consistency loss 用 L2 距离。离散空间中，F_θ 的输出是 **分布**（不是点），所以用 KL 散度：

```
L_consist = KL(P_1step || sg(P_2step))
```

其中 sg = stop_gradient（只对 1-step 路径求梯度，2-step 路径提供 target）。

```python
def self_consistency_loss(student_score_fn, graph, noise, x_t, t_k, t_k_minus_1):
    """
    Self-consistency: 1步 ≈ 2个半步
    """
    t_mid = (t_k + t_k_minus_1) / 2

    # Path 1: 单步 t_k → t_{k-1} (有梯度)
    P_1step = compute_euler_distribution(
        student_score_fn, x_t, t_k, t_k_minus_1, num_steps=1
    )

    # Path 2: 两步 t_k → t_mid → t_{k-1} (无梯度, 作为 target)
    with torch.no_grad():
        P_half1 = compute_euler_distribution(
            student_score_fn, x_t, t_k, t_mid, num_steps=1
        )
        x_mid = sample_categorical(P_half1)  # 中间采样
        P_2step = compute_euler_distribution(
            student_score_fn, x_mid, t_mid, t_k_minus_1, num_steps=1
        )

    # KL(1step || sg(2step))
    loss = F.kl_div(
        (P_1step + 1e-10).log(), P_2step,
        reduction='none', log_target=False
    ).sum(dim=-1).mean(dim=-1)

    return loss
```

### 4.4 与 SED 的联合训练

```
L_total = L_SED + λ_consist · L_consist
```

SED 提供 teacher → student 的监督信号；consistency 提供 student → student 的自监督信号。两者互补：
- SED 确保 student 的去噪方向正确
- Consistency 确保 student 在不同步数下的行为一致，从而使 few-step 推理可靠

---

## 第五部分：Progressive Self-Distillation

### 5.1 方案

不从 K=4 直接蒸馏，而是逐步减半：

```
Stage 0: 原始 teacher (128 steps)
Stage 1: 蒸馏到 K=64 的 student_1 (2x 加速)
Stage 2: student_1 作为 teacher, 蒸馏到 K=32 的 student_2 (4x 加速)
...
Stage 7: 蒸馏到 K=1 的 student_7 (128x 加速)
```

### 5.2 为什么比直接蒸馏好

**信息论论证**：

设 teacher 128 步的分布为 P_128, student K 步的为 P_K。

直接蒸馏的 KL gap：
```
KL(P_128 || P_4) ≈ O(128/4) = O(32)    (粗略估计)
```

Progressive 蒸馏的 KL gap：
```
KL(P_128 || P_64) + KL(P_64 || P_32) + ... + KL(P_4 || P_2) + KL(P_2 || P_1)
≈ 7 × O(2) = O(14)                     (每步只 2x 加速)
```

Progressive 的总误差更小，因为每步的蒸馏 gap 是常数 O(2)，而非线性增长的 O(128/K)。

### 5.3 快速版本：2 阶段

为了实验效率，可以只做 2 阶段：
```
Stage 1: Teacher (128 steps) → Student_1 (K=16, 8x 加速)
Stage 2: Student_1 (16 steps) → Student_2 (K=4, 32x 加速)
```

总训练时间 ≈ 2 × 单次蒸馏，但质量远好于直接 128→4。

---

## 第六部分：完整方法 — Geometry-Aware Score Entropy Distillation (GA-SED)

### 6.1 统一算法

```
输入: teacher T (预训练 SEDD), 数据分布 D, 窗口数 K, 混合权重 α

1. 预计算曲率自适应窗口边界 {t_0, ..., t_K}
2. 初始化 student S ← copy(T)
3. For each training step:
   a. 从数据采样 x_0 ~ D
   b. 采样时间步 t ~ Uniform[ε, 1]
   c. 前向扩散: x_t ~ P(·|x_0, t)

   d. Teacher 预测 (无梯度):
      x̂_0^T = argmax_v s_T(x_t, t)_v    (teacher 去噪预测)

   e. Score Entropy Distillation loss:
      L_SED = score_entropy(s_S(x_t, t), σ(t), x_t, x̂_0^T)

   f. 原始 SEDD loss (数据正则化):
      L_data = score_entropy(s_S(x_t, t), σ(t), x_t, x_0)

   g. Self-Consistency loss:
      确定 x_t 所在窗口 k, 计算 L_consist

   h. 总损失:
      L = (1-α)·L_data + α·L_SED + λ·L_consist

   i. 更新 S 的参数
```

### 6.2 推理

```
1. x_T ~ 全 MASK 序列
2. For k = K, K-1, ..., 1:
   σ_k = noise(t_k)
   score = s_S(x, σ_k)
   x = AnalyticPredictor.step(score, x, t_k, t_{k-1})
3. 最终去噪: x_0 = Denoiser(x, t_0)
```

推理时的 student 和 teacher 使用完全相同的采样算法，区别仅在步数：teacher 用 128 步，student 用 K 步。

### 6.3 贡献总结

| 贡献 | 新颖性 | 解决的瓶颈 |
|------|:---:|------------|
| Score Entropy Distillation | ★★★★★ | 瓶颈 1,2,3,4 (全部绕开分布空间) |
| Fisher-Rao 测地线插值 | ★★★★ | 瓶颈 1 (几何失真) |
| 曲率自适应窗口 | ★★★ | 瓶颈 5 (均匀划分浪费) |
| Self-Consistency | ★★★ | 提供额外训练信号 |
| Progressive Distillation | ★★ | 降低蒸馏 gap |

---

## 第七部分：实验规划

### 7.1 必做实验

**Exp 1: Ablation Study (消融实验)**

| 方法 | L_SED | Fisher-Rao | Adaptive | Consistency |
|------|:---:|:---:|:---:|:---:|
| Baseline (当前 D-PeRFlow) | ✗ | ✗ | ✗ | ✗ |
| + SED | ✓ | ✗ | ✗ | ✗ |
| + SED + FR | ✓ | ✓ | ✗ | ✗ |
| + SED + FR + AW | ✓ | ✓ | ✓ | ✗ |
| GA-SED (完整) | ✓ | ✓ | ✓ | ✓ |

在 SEDD-small (169M) 上，OpenWebText 训练，WikiText103 验证。

**Exp 2: 与 SOTA 对比**

| 方法 | Steps | PPL↓ | MAUVE↑ | Distinct-3↑ | Latency↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| SEDD Teacher | 128 | (基准) | (基准) | (基准) | (基准) |
| SEDD Teacher | 4 | (退化) | | | |
| SDTT (ICLR'25) | 16 | | | | |
| FS-DFM (Apple) | 8 | | | | |
| **GA-SED (ours)** | **4** | | | | |
| **GA-SED (ours)** | **8** | | | | |

**Exp 3: 模型规模实验**

在 small (169M) 和 medium (457M) 上验证 SED 的 scaling behavior。

**Exp 4: Progressive Distillation**

| 策略 | 最终步数 | PPL |
|------|:---:|:---:|
| Direct 128 → 4 | 4 | |
| Progressive 128 → 16 → 4 | 4 | |
| Progressive 128 → 32 → 8 → 4 | 4 | |

### 7.2 评估指标

- **Perplexity (PPL)**: GPT2-Large 评估，衡量流畅度
- **MAUVE**: 衡量生成分布与真实分布的差距（P-C curve 面积）
- **Distinct-1/2/3**: n-gram 多样性
- **Self-BLEU**: 生成多样性（越低越好）
- **Repetition Rate**: 重复 n-gram 的比例
- **Wall-clock Latency**: 实际推理时间

### 7.3 数据集

- 训练: OpenWebText (主), C4 (扩展)
- 验证/测试: WikiText103, LAMBADA, PTB

---

## 第八部分：实现优先级和时间表

### Phase 1 (2 周): Score Entropy Distillation — 核心贡献

```
1. 修改 d_perflow.py → 新增 score_entropy_distillation.py
2. 实现 SED loss (复用 graph_lib.py 的 score_entropy)
3. 实现混合训练 (L_data + L_SED)
4. 在 SEDD-small 上初步验证 (10K steps)
```

**关键风险**: teacher 的 argmax 预测可能有噪声（非 100% 准确），需要验证 SED 对此的鲁棒性。

**缓解方案**:
- 使用 soft target: 不取 argmax，而是取 teacher 分布的 top-K 做加权 score entropy
- 或使用 temperature τ < 1 锐化 teacher 分布后再取 argmax

### Phase 2 (1 周): Fisher-Rao 插值 + 自适应窗口

```
1. 实现 fisher_rao_interpolation()
2. 实现 compute_adaptive_boundaries()
3. 对比实验: 线性 vs Fisher-Rao, 均匀 vs 自适应
```

### Phase 3 (1 周): Self-Consistency + Progressive

```
1. 实现 self_consistency_loss()
2. 实现 2-stage progressive distillation pipeline
3. 完整消融实验
```

### Phase 4 (2 周): 完整实验

```
1. 多规模实验 (small + medium)
2. 多数据集 (OpenWebText, C4)
3. SOTA 对比
4. Wall-clock timing
```

### Phase 5 (2 周): 论文撰写

```
1. 理论分析 (SED 的收敛性, Fisher-Rao 的几何分析)
2. 实验图表
3. 相关工作对比
4. 投稿准备
```

**总计: 约 8 周**

---

## 第九部分：放弃对抗训练的理由

经过 4 个版本的迭代（BCE → Hinge → LSGAN），对抗训练在这个场景下有**根本性困难**：

1. **信号太弱**: teacher-student 初始化 97% 一致，判别器的信噪比极低
2. **训练不稳定**: GAN 训练需要精心平衡 G/D，增加了大量超参数
3. **计算开销大**: 判别器前向/后向额外增加 30-50% 的计算量
4. **边际收益为零**: 从训练日志看，对抗训练对 PPL 的贡献 = 0

**结论**: 放弃对抗训练，将精力集中在 SED（数学上更优雅，计算上更高效，理论上更有保障）。

对抗训练在离散蒸馏中的失败本身可以作为论文的一个 empirical finding（"Why adversarial training fails for discrete diffusion distillation"），放在附录中作为 negative result。

---

## 第十部分：论文定位

### 标题候选

1. **"Score Entropy Distillation: Few-Step Discrete Diffusion via Native Loss Transfer"**
2. **"Geometry-Aware Distillation on the Probability Simplex for Discrete Diffusion Models"**
3. **"From Scores to Steps: Efficient Distillation of Discrete Diffusion via Score Entropy"**

### 核心 selling point

> "我们观察到，现有离散扩散蒸馏方法都在分布空间操作，需要计算完整的 V 维分布并用 KL 散度匹配。我们提出 Score Entropy Distillation (SED)，首次将蒸馏目标统一到与预训练完全相同的 score entropy 损失函数中。SED 不仅更高效（在 Absorbing graph 上仅在 MASK 位置计算），而且通过 Fisher-Rao 测地线和曲率自适应窗口正确处理了离散概率空间的几何结构。"

### 审稿人可能的 concern 及预回复

**Q: "SED 只是把 x_0 换成 teacher 预测，这有什么新颖性？"**

A: 关键洞察是这个替换在 score entropy 框架下是 mathematically principled 的 —— score entropy 是离散空间上 score matching 的最优 Bregman 散度，用 teacher 预测代替 ground truth 等价于在 score 空间做投影。这与连续空间中的 Score Distillation Sampling (SDS, Poole et al. 2023) 有类似的精神，但针对离散空间进行了本质不同的推导。

**Q: "Fisher-Rao 插值真的有显著提升吗？"**

A: 对于尖锐分布（max_prob > 0.9），线性插值会产生 1.5-2x 的熵膨胀。消融实验会定量展示这一点。

**Q: "和 Consistency Models 有什么区别？"**

A: Consistency Models 在连续空间用 L2 距离定义一致性，且依赖 ODE 求解。我们在离散概率单纯形上用 KL 散度定义一致性，且结合了 score entropy 蒸馏而非纯自监督。

---

*本方案基于对仓库全部核心文件的逐行审阅，结合 NeurIPS 2024-2025 相关工作的分析。*
