\# GPT-5.2 实验结果



\*\*测试日期\*\*: 2026-02-19

\*\*模型\*\*: GPT-5.2

\*\*评估器版本\*\*: v2 (含详细度评分)



\## 评估权重



| 维度 | 权重 |

|------|------|

| 热点识别 | 25% |

| 瓶颈类型 | 30% |

| GPU 适合度 | 20% |

| 优化建议 | 10% |

| 详细度 | 15% |



---



\## D2: LLM 分析结果



\### miniMD (C++)



| Prompt | 分数 | 花费 |

|--------|------|------|

| zero\_shot | 81.35 | $0.0331 |

| few\_shot | 81.86 | $0.0340 |

| \*\*contextual\*\* | \*\*93.06\*\* ⭐ | $0.0422 |



\*\*总花费\*\*: $0.1093



---



\### HPCG SPMV (C++)



| Prompt | 分数 | 花费 |

|--------|------|------|

| \*\*zero\_shot\*\* | \*\*96.27\*\* ⭐ | - |

| few\_shot | 89.00 | - |

| contextual | 86.25 | - |



\*\*总花费\*\*: $0.0670



---



\### HPCG SYMGS (C++)



| Prompt | 分数 | 花费 |

|--------|------|------|

| \*\*zero\_shot\*\* | \*\*75.21\*\* ⭐ | - |

| few\_shot | 74.92 | - |

| contextual | 66.75 | - |



\*\*总花费\*\*: $0.0735



---



\### Abinit (Fortran)



| Prompt | 分数 | 花费 |

|--------|------|------|

| \*\*zero\_shot\*\* | \*\*81.38\*\* ⭐ | - |

| few\_shot | 76.69 | - |

| contextual | 54.63 | - |



\*\*总花费\*\*: $0.2186



---



\## D2 汇总



| 程序 | 语言 | 最佳 Prompt | 最高分 | 花费 |

|------|------|-------------|--------|------|

| miniMD | C++ | contextual | \*\*93.06\*\* | $0.1093 |

| HPCG SPMV | C++ | zero\_shot | \*\*96.27\*\* | $0.0670 |

| HPCG SYMGS | C++ | zero\_shot | \*\*75.21\*\* | $0.0735 |

| Abinit | Fortran | zero\_shot | \*\*81.38\*\* | $0.2186 |



\*\*D2 总花费\*\*: $0.4684



---



\## D3: GPU 代码生成



\### 生成结果



| 指标 | 结果 |

|------|------|

| 目标函数 | miniMD ForceLJ::compute\_fullneigh |

| 生成模型 | GPT-5.2 |

| 一次成功 | ✅ |

| 编译通过 | ✅ (nvcc -arch=sm\_86) |

| 正确性验证 | ✅ 误差 1.93e-15 (阈值 1e-6) |



\### 性能基准



\*\*硬件\*\*: NVIDIA RTX 3060 Laptop (6GB)



| 原子数 | CPU (ms) | GPU (ms) | 加速比 |

|--------|----------|----------|--------|

| 10,000 | 0.78 | 0.08 | \*\*10.1x\*\* |

| 50,000 | 7.20 | 0.53 | \*\*13.5x\*\* |

| 100,000 | 18.03 | 1.47 | \*\*12.3x\*\* |

| 200,000 | 46.58 | 3.82 | \*\*12.2x\*\* |



\### LLM 生成的 CUDA Kernel

```cuda

\#define PAD 4

\_\_global\_\_ void compute\_fullneigh\_kernel(

&nbsp;   int nlocal, int ntypes,

&nbsp;   const double\* \_\_restrict\_\_ x,

&nbsp;   double\* \_\_restrict\_\_ f,

&nbsp;   const int\* \_\_restrict\_\_ type,

&nbsp;   const int\* \_\_restrict\_\_ neighbors,

&nbsp;   const int\* \_\_restrict\_\_ numneigh,

&nbsp;   int maxneighs,

&nbsp;   const double\* \_\_restrict\_\_ cutforcesq,

&nbsp;   const double\* \_\_restrict\_\_ epsilon,

&nbsp;   const double\* \_\_restrict\_\_ sigma6)

{

&nbsp;   int i = blockIdx.x \* blockDim.x + threadIdx.x;

&nbsp;   if (i >= nlocal) return;

&nbsp;   

&nbsp;   double xtmp = x\[i \* PAD + 0];

&nbsp;   double ytmp = x\[i \* PAD + 1];

&nbsp;   double ztmp = x\[i \* PAD + 2];

&nbsp;   int type\_i = type\[i];

&nbsp;   double fix = 0.0, fiy = 0.0, fiz = 0.0;

&nbsp;   int nneigh = numneigh\[i];

&nbsp;   int base = i \* maxneighs;

&nbsp;   

&nbsp;   for (int k = 0; k < nneigh; k++) {

&nbsp;       int j = neighbors\[base + k];

&nbsp;       double delx = xtmp - x\[j \* PAD + 0];

&nbsp;       double dely = ytmp - x\[j \* PAD + 1];

&nbsp;       double delz = ztmp - x\[j \* PAD + 2];

&nbsp;       double rsq = delx \* delx + dely \* dely + delz \* delz;

&nbsp;       int type\_ij = type\_i \* ntypes + type\[j];

&nbsp;       

&nbsp;       if (rsq < cutforcesq\[type\_ij]) {

&nbsp;           double sr2 = 1.0 / rsq;

&nbsp;           double sr6 = sr2 \* sr2 \* sr2 \* sigma6\[type\_ij];

&nbsp;           double force = 48.0 \* sr6 \* (sr6 - 0.5) \* sr2 \* epsilon\[type\_ij];

&nbsp;           fix += delx \* force;

&nbsp;           fiy += dely \* force;

&nbsp;           fiz += delz \* force;

&nbsp;       }

&nbsp;   }

&nbsp;   

&nbsp;   f\[i \* PAD + 0] = fix;

&nbsp;   f\[i \* PAD + 1] = fiy;

&nbsp;   f\[i \* PAD + 2] = fiz;

}

```



\*\*特点\*\*:

\- 使用 `\_\_restrict\_\_` 优化指针别名

\- 一个线程处理一个原子

\- 正确的边界检查

\- 与手写参考版本性能相当



---



\## 总花费



| 类别 | 花费 |

|------|------|

| D2 分析测试 | $0.4684 |

| D3 代码生成 | ~$0.01 |

| \*\*总计\*\* | \*\*~$0.48\*\* |

