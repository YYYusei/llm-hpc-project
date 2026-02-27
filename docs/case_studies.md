# Case Studies: LLM-Assisted HPC Performance Analysis

本文档展示 6 个具体案例，分析 LLM 在 HPC 性能分析和 GPU 代码生成中的表现。

---

## Case Study 1: miniMD LJ Force → CUDA (成功案例)

### 背景

miniMD 是分子动力学模拟基准程序，其核心热点是 Lennard-Jones 力计算。

**原始代码特征**:
- 双层嵌套循环（原子 × 邻居）
- 不规则内存访问（邻居列表）
- 浮点密集计算（力计算公式）

### LLM 分析结果

**Stage 1 (GPT-4o)**:
- 识别热点: ✅ ForceLJ::compute 系列函数
- 瓶颈判断: compute-bound
- GPU 适合度: ✅ True

**Stage 2 (GPT-5.2) 修正**:
- 修正瓶颈为: **memory/latency + synchronization bound**
- 理由: 
  > "LJ 内核的 FLOPs 不算少，但每次迭代需要从 x/type/neighs 进行不规则访存（j 索引导致的 gather），且 halfneigh_threaded 还包含对 f[j] 的 3 次 OpenMP atomic"

### CUDA 代码生成

| 模型 | 状态 | 加速比 | 误差 | 花费 |
|------|------|--------|------|------|
| **GPT-4o** | ✅ 首次成功 | **14.34x** | 3.98e-13 | $0.009 |
| GPT-5.2 | ✅ 首次成功 | 12.03x | 7.28e-12 | $0.017 |

**GPT-4o 生成的优化技术**:
```cuda
// 1. double4 向量化加载
double4 pos_i = reinterpret_cast<const double4*>(x)[i];

// 2. __ldg 只读缓存
int type_i = __ldg(&type[i]);

// 3. FMA 融合乘加
fix = fma(delx, force, fix);
```

### 结论

✅ **成功案例**: LLM 成功识别热点并生成高质量 CUDA 代码，达到 14.34x 加速。级联分析修正了瓶颈类型，提供了更准确的性能分析。

---

## Case Study 2: HPCG SPMV → CUDA (成功案例)

### 背景

HPCG (High Performance Conjugate Gradient) 是稀疏线性代数基准程序，SPMV (Sparse Matrix-Vector Multiplication) 是其核心操作。

**原始代码特征**:
- CSR 格式稀疏矩阵
- 不规则内存访问（间接索引）
- 低算术强度（~0.12-0.17 FLOPs/byte）

### LLM 分析结果

**Stage 1 (GPT-4o)**:
- 识别热点: ✅ ComputeSPMV_ref 内层循环
- 瓶颈判断: **memory-bound** ✅
- GPU 适合度: ✅ True

**Stage 2 (GPT-5.2) 确认**:
- 瓶颈判断: ✅ **正确**
- 补充分析:
  > "算术强度约 0.12-0.17 FLOPs/byte，远低于现代 CPU/GPU 的平衡点，典型 memory-bound"

### CUDA 代码生成

| 模型 | 状态 | 加速比 | 误差 | 花费 |
|------|------|--------|------|------|
| GPT-4o | ❌ 需修复 | 7.11x | 7.11e-15 | $0.011 |
| **GPT-5.2** | ✅ 首次成功 | **10.30x** | 7.11e-15 | $0.006 |

**GPT-5.2 生成的优化技术**:
```cuda
// 1. 针对 27-point stencil 的循环展开
if (max_nnz == 27) {
    #pragma unroll
    for (int j = 0; j < 27; ++j) { ... }
}

// 2. __ldg 只读缓存 + FMA
double xv = __ldg(&x[c]);
sum = fma(a, xv, sum);
```

### 结论

✅ **成功案例**: GPT-5.2 在 SPMV 上表现更好（10.30x vs 7.11x），且首次编译成功。说明不同模型在不同类型 kernel 上各有优势。

---

## Case Study 3: miniMD 级联修正 (部分正确案例)

### 背景

这个案例展示级联分析方案如何修正 Stage 1 的判断错误。

### Stage 1 分析 (GPT-4o)
```json
{
  "bottleneck_type": {
    "primary": "compute",
    "reasoning": "The primary bottleneck is compute-bound due to the intensive 
                  arithmetic operations in the force calculations"
  }
}
```

### Stage 2 修正 (GPT-5.2)
```json
{
  "bottleneck_correct": false,
  "corrected_bottleneck": "memory/latency + synchronization bound",
  "reasoning": "LJ 内核单次相互作用的 FLOPs 不算少，但每次迭代需要从 
                x/type/neighs 进行不规则访存（j 索引导致的 gather），
                且 halfneigh_threaded 还包含对 f[j] 的 3 次 OpenMP atomic"
}
```

### 优化建议对比

| 来源 | 建议数量 | 示例建议 |
|------|----------|----------|
| Stage 1 | 3 条 | GPU 移植、并行化 |
| Stage 2 | **5 条** | 消除 atomic、SoA 布局、预取、full neighbor list |

**Stage 2 新增的关键建议**:
1. 消除 f[j] 写冲突：使用 full neighbor list
2. 数据布局：AoS → SoA
3. 移除 atomic 操作：预计 1.3x-3.0x 加速

### 结论

⚠️ **部分正确案例**: Stage 1 的瓶颈判断过于简化。级联方案通过 Stage 2 修正，提供了更准确的分析和更详细的优化建议。

---

## Case Study 4: GPT-4o SPMV 参数错误 (失败案例)

### 背景

GPT-4o 在生成 SPMV kernel 时出现参数顺序错误，导致编译失败。

### 错误代码

**GPT-4o 生成的函数签名**:
```cuda
__global__ void spmv_kernel_gpt_4o(
    const double* __restrict__ values,    // ❌ 错误顺序
    const int* __restrict__ col_ind,
    const double* __restrict__ x,
    double* __restrict__ y,
    const int* __restrict__ nnz_per_row,
    const int nrow,
    const int max_nnz
)
```

**期望的函数签名**:
```cuda
__global__ void spmv_kernel(
    const int nrow,                       // ✅ 正确顺序
    const int max_nnz,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y
)
```

### 编译错误
```
error: argument of type "int" is incompatible with parameter of type "const double *"
```

### 修复过程

**GPT-5.2 成功修复**:
```cuda
__global__ void spmv_kernel_gpt_4o(
    const int nrow,
    const int max_nnz,
    const int* __restrict__ nnz_per_row,
    const int* __restrict__ col_ind,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y
)
```

### 根因分析

1. **Prompt 中参数顺序不明确**: 未在 prompt 中显式指定参数顺序
2. **GPT-4o 的假设**: 按照"数据优先"的直觉排列参数
3. **与 benchmark 模板不匹配**: 模板中 kernel 调用顺序固定

### 结论

❌ **失败案例**: 参数顺序错误是 LLM 代码生成的常见问题。解决方案：
1. 在 prompt 中明确指定函数签名
2. 使用级联修复机制

---

## Case Study 5: Abinit 级联修正 (部分正确案例)

### 背景

Abinit 是密度泛函理论 (DFT) 计算软件，nonlop_ylm 是其核心热点函数。

### Stage 1 分析 (GPT-4o)
```json
{
  "bottleneck_type": {
    "primary": "compute",
    "reasoning": "The code involves heavy computation with nested loops 
                  and complex mathematical operations"
  }
}
```

### Stage 2 修正 (GPT-5.2)
```json
{
  "bottleneck_correct": false,
  "corrected_bottleneck": "memory+allocation/overhead bound",
  "reasoning": "在最内层循环里大量 ABI_MALLOC/ABI_FREE 每个 block 都分配释放，
                典型会造成显著的内存分配开销、TLB/页抖动与缓存污染"
}
```

### 关键洞察

GPT-5.2 发现了 Stage 1 遗漏的问题：

1. **频繁内存分配**: 每个 atom block 都 ABI_MALLOC/ABI_FREE
2. **低复用率**: ph3d 与 ffnl 的跨维访问导致 cache miss
3. **混合瓶颈**: 不是纯 compute-bound，而是 memory + allocation overhead

### 优化建议

| 建议 | 预计加速 | 难度 |
|------|----------|------|
| Workspace 复用，消除频繁分配 | 1.2x-2.0x | medium |
| 批量 GEMM 替代多次 GEMV | 1.5x-4x | hard |
| 循环重排，改善向量化 | 1.2x-2.5x | medium |

### 结论

⚠️ **部分正确案例**: Stage 1 将瓶颈简单归类为 compute-bound，但实际上内存分配开销是主要问题。级联方案提供了更深入的分析。

---

## Case Study 6: HPCG SYMGS - 数据依赖与完整流程 (复杂案例)

### 背景

SYMGS (Symmetric Gauss-Seidel) 是 HPCG 的核心热点函数，占 67.3% 的 CPU 时间。与前两个 kernel 不同，SYMGS 有**强数据依赖**：

- **前向扫描**: 行 i 依赖行 0..i-1 的更新结果
- **后向扫描**: 行 i 依赖行 i+1..n-1 的更新结果

这意味着**不能直接并行化**，是对 LLM 的严峻挑战。

### 测试方法对比

我们测试了三种不同的方法：

| 方法 | 描述 |
|------|------|
| V1: 直接生成 | 只给代码，让 LLM 自己想办法 |
| V2: 策略提示 | 在 prompt 中明确指定多色排序策略 |
| V3: 完整流程 | 级联分析 → 策略确定 → 代码生成 |

### 测试结果

| 方法 | GPT-4o | GPT-5.2 | 最佳加速比 | 误差 | 花费 |
|------|--------|---------|-----------|------|------|
| V1: 直接生成 | 0.02x ❌ | 编译失败 ❌ | **0.02x** | 2.37e-02 | $0.031 |
| V2: 策略提示 | **3.14x** ✅ | 2.79x ✅ | **3.14x** | 3.42e-02 | $0.008 |
| V3: 完整流程 | - | **5.61x** ✅ | **5.61x** | 3.42e-02 | $0.041 |

### V1 失败分析

GPT-4o 直接生成的代码：
- 编译错误（参数类型不匹配）
- 修复后加速比只有 **0.02x**（比 CPU 慢 50 倍！）
- 没有正确处理数据依赖

**根因**: LLM 不知道如何处理 Gauss-Seidel 的行间依赖，简单并行化导致：
1. 大量同步操作，性能极差
2. 结果数值不准确

### V2 策略提示

在 prompt 中明确指定**多色排序 (Multi-coloring)** 策略：
```
## Parallelization Strategy (MUST USE):
Use Multi-coloring approach:
1. Pre-compute colors for each row
2. Rows with the same color have NO dependencies
3. Process one color at a time in parallel
```

**结果**: GPT-4o 达到 3.14x 加速，首次编译成功。

### V3 完整流程
```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: GPT-4o 分析                                       │
│  - 识别数据依赖问题                                          │
│  - 判断瓶颈类型                                              │
│  - 花费: $0.0035                                            │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: GPT-5.2 深度分析                                  │
│  - 验证瓶颈判断                                              │
│  - 提出多色排序策略                                          │
│  - 提供详细实现建议                                          │
│  - 花费: $0.0240                                            │
├─────────────────────────────────────────────────────────────┤
│  Stage 3: GPT-5.2 生成 CUDA                                 │
│  - 基于策略生成代码                                          │
│  - 首次编译成功                                              │
│  - 花费: $0.0132                                            │
└─────────────────────────────────────────────────────────────┘
```

**结果**: 5.61x 加速，比 V2 提升 **79%**。

### 关于误差 (3.42e-02)

多色 Gauss-Seidel 与顺序 Gauss-Seidel 的**数值结果不完全相同**：
- 顺序 GS：严格按 0,1,2,...,n-1 顺序更新
- 多色 GS：同色行并行更新，不同色串行

这是预期行为，不影响算法收敛性，在实际应用中可接受。

### 关键发现

1. **数据依赖是硬伤**: LLM 不能自动发明多色排序等并行化策略
2. **Prompt 工程关键**: 明确策略提示从 0.02x → 3.14x (提升 157 倍)
3. **完整流程最优**: 级联分析后再生成，3.14x → 5.61x (+79%)
4. **花费换性能**: $0.041 获得最佳性能，性价比高

### 与其他 Kernel 对比

| Kernel | 数据依赖 | 最佳方法 | 加速比 |
|--------|----------|----------|--------|
| miniMD LJ Force | ❌ 无 | 直接生成 | 14.34x |
| HPCG SPMV | ❌ 无 | 直接生成 | 10.30x |
| HPCG SYMGS | ✅ **有** | **完整流程** | 5.61x |

### 结论

⚠️ **复杂案例**: SYMGS 展示了 LLM 在处理数据依赖时的局限性和解决方案：

1. **简单并行代码**: 直接让 LLM 生成即可
2. **复杂依赖代码**: 需要完整流程（分析 → 策略 → 生成）
3. **LLM 的局限性**: 不能自动发明并行化策略，但能很好地执行给定策略

---

---

## Case Study 7: 完整流程对比实验 (方法论验证)

### 背景

为验证"完整流程是否优于直接生成"，我们对三个 kernel 都进行了完整流程测试：

- **完整流程**: Stage 1 (GPT-4o 分析) → Stage 2 (GPT-5.2 策略) → Stage 3 (CUDA 生成)
- **直接生成**: 直接给代码让 LLM 转换为 CUDA

### 测试结果

| Kernel | 直接生成 | 完整流程 | 差异 | 完整流程花费 |
|--------|----------|----------|------|-------------|
| miniMD LJ Force | 14.34x | **15.59x** ✅ | **+8.7%** | $0.091 |
| HPCG SPMV | **10.30x** | 6.18x ❌ | **-40.0%** | $0.075 |
| HPCG SYMGS | 0.02x | **5.61x** ✅ | **+28000%** | $0.041 |

### miniMD: 完整流程略好 (+8.7%)

| 指标 | 直接生成 | 完整流程 |
|------|----------|----------|
| 加速比 | 14.34x | **15.59x** |
| 误差 | 3.98e-13 | 1.16e-10 |
| 花费 | $0.009 | $0.091 |
| 首次成功 | ✅ | ❌ (需修复) |

**分析**: 完整流程略有提升，但花费高 10 倍，性价比不高。

### SPMV: 完整流程失败 (-40%)

| 指标 | 直接生成 | 完整流程 |
|------|----------|----------|
| 加速比 | **10.30x** | 6.18x |
| 误差 | 7.11e-15 | **1.90e+01** ⚠️ |
| 花费 | $0.006 | $0.075 |
| 首次成功 | ✅ | ❌ (需修复) |

**误差 1.90e+01 表明结果完全错误！**

#### 失败根因分析

查看完整流程生成的代码，发现 LLM 错误假设了数据格式：

**LLM 假设的格式 (CSR)**:
```cuda
// 错误：使用 prefix sum 计算行起始位置
int start = 0;
for (int r = 0; r < row; ++r) 
    start += __ldg(&nnz_per_row[r]);  // O(n²) 复杂度！
```

**实际的数据格式 (ELL-like)**:
```cuda
// 正确：使用固定步长
int base = row * max_nnz;
for (int j = 0; j < nnz; ++j) {
    col = col_ind[base + j];
    val = values[base + j];
}
```

**问题**:
1. **数据格式假设错误**: 完整流程让 LLM "过度思考"，假设了更复杂的 CSR 格式
2. **O(n²) 复杂度**: 每个线程都要计算 prefix sum
3. **结果错误**: 读取了错误的内存位置

#### 直接生成为什么成功？

直接生成的 prompt 中明确指定了数据格式：
```
col_ind[i * max_nnz + j]: column index
values[i * max_nnz + j]: matrix value
```

LLM 直接按照格式生成代码，没有"过度分析"。

### SYMGS: 完整流程是必须的 (+28000%)

| 指标 | 直接生成 | 完整流程 |
|------|----------|----------|
| 加速比 | 0.02x | **5.61x** |
| 误差 | 2.37e-02 | 3.42e-02 |
| 花费 | $0.031 | $0.041 |

**分析**: SYMGS 有强数据依赖，直接生成完全失败，完整流程是唯一有效的方法。

### 关键发现

#### 1. 完整流程不一定更好

| 情况 | 结果 |
|------|------|
| 简单并行 + 格式明确 (SPMV) | 直接生成更好 |
| 简单并行 + 格式明确 (miniMD) | 差不多，直接生成性价比高 |
| 复杂依赖 (SYMGS) | 完整流程是必须的 |

#### 2. 过度分析可能有害

完整流程让 LLM 进行更深入的分析，但可能导致：
- 错误假设数据格式
- 选择过于复杂的算法
- "想太多"反而出错

#### 3. 数据格式必须明确

| 方法 | 数据格式 | 结果 |
|------|----------|------|
| 直接生成 | 在 prompt 中明确指定 | ✅ 正确 |
| 完整流程 | 让 LLM 自己推断 | ❌ 可能错误 |

### 方法选择指南
```
                    ┌─────────────────────────────┐
                    │     代码有数据依赖吗？       │
                    └─────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
                  是的                    没有
                    │                       │
                    ▼                       ▼
            ┌───────────────┐      ┌───────────────────┐
            │  完整流程      │      │  直接生成          │
            │  (必须)        │      │  (推荐)            │
            │               │      │                   │
            │  SYMGS: 5.61x │      │  miniMD: 14.34x   │
            │               │      │  SPMV: 10.30x     │
            └───────────────┘      └───────────────────┘
```

### 成本效益分析

| 方法 | 平均花费 | 适用场景 |
|------|----------|----------|
| 直接生成 | $0.01-0.02 | 简单并行，格式明确 |
| 完整流程 | $0.04-0.09 | 复杂依赖，需要策略 |

**结论**: 不要盲目使用完整流程，根据代码特性选择合适的方法。

---

## 总结

### 成功率统计（最终）

| 类型 | 数量 | 案例 |
|------|------|------|
| ✅ 完全成功 | 3 | miniMD CUDA, SPMV CUDA, SYMGS 完整流程 |
| ⚠️ 部分正确 | 2 | miniMD 级联修正, Abinit 级联修正 |
| ❌ 失败 | 3 | GPT-4o SPMV 参数错误, SYMGS 直接生成, SPMV 完整流程 |

### 三个 GPU Kernel 最终结果

| Kernel | 类型 | 最佳方法 | 最佳加速比 | 难度 |
|--------|------|----------|-----------|------|
| miniMD LJ Force | 分子动力学 | 直接生成 / 完整流程 | **14.34x / 15.59x** | 中等 |
| HPCG SPMV | 稀疏矩阵 | **直接生成** | **10.30x** | 简单 |
| HPCG SYMGS | 迭代求解 | **完整流程** | **5.61x** | 困难 |

### 关键发现（最终版）

1. **级联分析有效**: 50% (2/4) 的瓶颈判断被修正
2. **模型各有优势**: 
   - GPT-4o: miniMD 更好 (14.34x vs 12.03x)
   - GPT-5.2: SPMV 更好 (10.30x vs 7.11x)
3. **修复机制重要**: GPT-5.2 成功修复 GPT-4o 的编译错误
4. **完整流程价值有限**: 
   - 简单代码：直接生成更可靠、更便宜
   - 复杂依赖：完整流程是必须的
5. **过度分析可能有害**: SPMV 完整流程因错误假设数据格式而失败
6. **数据格式必须明确**: 在 prompt 中明确指定，不要让 LLM 猜测

### 最佳实践（最终版）

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 简单并行，无依赖 | **直接生成** | 更快、更便宜、更可靠 |
| 复杂依赖（如 GS） | **完整流程** | 需要分析才能找到正确策略 |
| 数据格式复杂 | **明确指定格式** | 避免 LLM 错误假设 |
| 首次失败 | **修复机制** | GPT-5.2 修复成功率高 |

### 成本分析（最终版）

| 任务类型 | 平均花费 | 推荐度 |
|----------|----------|--------|
| 单程序级联分析 | $0.07 | ⭐⭐⭐ 推荐 |
| 简单 CUDA 直接生成 | $0.01-0.02 | ⭐⭐⭐⭐⭐ 强烈推荐 |
| 复杂 CUDA 完整流程 | $0.04-0.09 | ⭐⭐⭐ 按需使用 |
| 代码修复 | $0.01 | ⭐⭐⭐⭐ 必要时使用 |
| **项目总计** | **~$0.70** | - |

### 论文贡献总结

1. **提出级联分析方案**: 两阶段 LLM 分析，修正率 50%
2. **验证 LLM CUDA 生成能力**: 三个 kernel 达到 5-15x 加速
3. **发现方法适用边界**: 
   - 简单并行 → 直接生成
   - 复杂依赖 → 完整流程
4. **识别 LLM 局限性**:
   - 不能自动发明并行化策略
   - 可能错误假设数据格式
   - 参数顺序可能出错
5. **提供最佳实践指南**: 根据代码特性选择合适方法