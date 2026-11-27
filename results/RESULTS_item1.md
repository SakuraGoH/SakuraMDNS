# Item 1 — C++ 重写 MCMC 扫描算法：实现、校验与 benchmark

日期：2026-11-26　硬件：Intel Core Ultra（4P+8E+2LPE，18 逻辑核）+ Intel Arc iGPU (Meteor Lake, 112 EU)

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `mdns/samplers/cpp/samplers.cpp` | 5 个 MCMC kernel 的 C++17 实现，OpenMP 按链并行 |
| `mdns/samplers/cpp_ext.py` | ctypes 绑定 + 与上游同签名的 drop-in 包装 |
| `scripts/build_cpp.ps1` | 用 MSVC 14.52 编译 `mdns_samplers.dll`（无需 pybind11/ninja） |
| `scripts/validate_samplers.py` | 对可枚举分布做精确校验 |
| `scripts/check_equilibration.py` | L=16 平衡性交叉检验（以 SW 为基准） |
| `scripts/bench_samplers.py` | 速度 / 自相关时间 / 有效样本率 benchmark（**仓库清理时已删除**，见文末说明；数字存档在 `results/bench_samplers.json`） |

已实现的 kernel：`ising_mh`、`ising_sw`、`potts_mh`、`potts_glauber`、`potts_sw`。
统一 C ABI：`(S, B, L, q, beta, J, h, sweeps, e_series, seed)`，`S` 为 `(B, L*L)` 连续数组
（Ising `int8` 存 0/1，Potts `int32` 存 0..q-1），链内 RNG 为 splitmix64 播种的 xoshiro128+，
每条链独立可复现。

**一个扫描（sweep）= L² 次单点更新**，或 SW 的一次完整 cluster 扫描。

## 2. 正确性校验：8/8 通过

对可精确枚举的体系比对能量分布（TV 距离），并额外检查**单点态边缘分布**（对提议偏置敏感）：

```
[PASS] ising_mh       L=4 beta=0.4407  ham_err=0.00e+00 TV=0.0060 label_dev=0.0149
[PASS] ising_sw       L=4 beta=0.4407  ham_err=0.00e+00 TV=0.0033 label_dev=0.0076
[PASS] ising_sw       L=4 beta=0.28    ham_err=0.00e+00 TV=0.0030 label_dev=0.0023
[PASS] potts_mh       L=3 beta=1.005   ham_err=0.00e+00 TV=0.0024 label_dev=0.0032
[PASS] potts_glauber  L=3 beta=1.005   ham_err=0.00e+00 TV=0.0022 label_dev=0.0055
[PASS] potts_sw       L=3 beta=1.005   ham_err=0.00e+00 TV=0.0035 label_dev=0.0021
[PASS] potts_sw       L=3 beta=1.2     ham_err=0.00e+00 TV=0.0021 label_dev=0.0011
[PASS] potts_glauber  L=3 beta=0.5     ham_err=0.00e+00 TV=0.0059 label_dev=0.0010
```

`ham_err=0` 表示 C++ 上报的能量与参考哈密顿量逐样本完全一致。

### 发现并修复的 bug：Potts MH 提议分布有偏

```cpp
// 错：cur+1 拿到 2/3 概率，破坏 Metropolis 的对称提议假设
nw = rng.below(q); if (nw == cur) nw = (nw + 1) % q;

// 对：在除 cur 外的 q-1 个态上均匀
int32_t nw = rng.below(q - 1); if (nw >= cur) ++nw;
```

后果很严重且**只有 L=16 的交叉检验能抓出来**（L=3 的精确测试 TV 仅从 0.0062 变到 0.0024，几乎看不出来）：

| Potts L=16, β=1.2 | 修复前 | 修复后 | SW 基准 |
|---|---|---|---|
| `potts_mh` 平均能量/格点 | −1.8017 (Δ=+0.096) ❌ | **−1.8979 (Δ=−0.0003)** ✅ | −1.8976 |
| `potts_mh` β=1.005 | −1.4378 (Δ=+0.184) ❌ | **−1.6223 (Δ=−0.0006)** ✅ | −1.6217 |

加长 burn-in 到 40000 扫也无法消除该偏差 —— 这正是"偏差"而非"混合慢"的判据。

## 3. Benchmark（L=16, B=128, J=1）

τ = 能量的积分自相关时间（Sokal 自适应窗口，单位：扫）；
ESS/s = B · sweeps/s / (2τ)，即每秒产生的**有效独立样本**数。

| sampler | β | τ_int (扫) | Python 扫/s | C++ 扫/s (线程) | 加速 | C++ ESS/s | ESS 加速 |
|---|---|---|---|---|---|---|---|
| Ising MH | 0.4407 | 19 | 63.11 | 5671 @4t | **89.9×** | 19 029 | 89.9× |
| Ising SW | 0.4407 | 3 | 8.82 | 3878 @8t | **439.6×** | 75 605 | 439.6× |
| Potts MH | 1.2 | 5 | 30.71 | 3820 @4t | **124.4×** | 53 234 | 124.4× |
| Potts Glauber | 1.2 | 3 | 44.79 | 2413 @4t | **53.9×** | 58 049 | 53.9× |
| Potts SW | 1.2 | 3 | 8.02 | 3924 @4t | **489.0×** | 73 020 | 489.0× |

要点：

- 单点算法（MH/Glauber）：**54–124×**。Python 版虽已用 numpy 对 B 条链向量化，
  但每条链每步仍有几十次 numpy 调用，C++ 把整层开销消掉了。
- 团簇算法（SW）：**440–489×**。Python 版是纯 Python 三重循环 + Union-Find 闭包，
  这正是数据生成的真正瓶颈。
- 按"有效样本/秒"看，**SW 与 Glauber 并列最优**（7.3–7.6 万 ESS/s），而 MH 的
  τ 随 β→β_c 迅速增大（Ising 临界点 τ≈19 扫），这是单点算法的经典临界慢化。

## 4. 硬件相关发现：OpenMP 不是越多线程越好

| 线程数 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Ising MH 扫/s | 1627 | 3378 | **5671** | 4836 |
| Potts SW 扫/s | 1410 | 2671 | **3924** | 3707 |
| Potts Glauber 扫/s | 872 | 1670 | **2413** | 1574 |

4 线程（≈P 核数量）最佳，8 线程已饱和甚至退化，18 逻辑核更差。
原因是这台机器是 4P+8E+2LPE 的异构 CPU，E 核显著拖后腿；且每条链工作集仅 L²·1B = 256 B，
调度与内存带宽比算力更早成为瓶颈。**benchmark 因此对每种算法单独扫描线程数取最优。**

热路径优化（相对首版）：把 `%` 取模换成比较+加减、把 `below(n)` 换成乘法移位，
单线程 MH 从 1491 → 2126 扫/s（**+43%**）。

## 5. 注意事项（不要误读这些数字）

1. **τ 只测了能量**。在 Potts 有序相（β=1.2）真正的慢模式是磁化/畴壁，能量会偏快，
   所以 ESS/s 是偏乐观的上界；下一阶段的 GPU 采样 benchmark 会补上磁化强度的 τ。
2. **Python 基线**用实测速率，τ 复用 C++ 长链的估计（τ 是马尔可夫链的性质、与实现无关，
   而 Python 版跑不出足够长的链）。这一处理在 `bench_samplers.py` 的 docstring 里写明。
3. 表里的加速比是"同一算法、同一分布"的纯实现加速，**不含算法改进**。
   算法层面的提升（如 checkerboard、Wolff）留给 Item 2。

## 6. 复现

```powershell
pwsh scripts/build_cpp.ps1 -Force
python scripts/validate_samplers.py      # 精确校验
python scripts/check_equilibration.py    # L=16 平衡性
python scripts/bench_samplers.py         # 速度 / ESS
```

结果 JSON：`results/validate_samplers.json`、`results/equilibration_l16.json`、`results/bench_samplers.json`

---

## 附：仓库清理说明

`scripts/bench_samplers.py` 已在最终整理中删除：它的对照基线正是被删掉的纯 Python 采样器
（用户要求移除 Python 扫描代码）。本文表格中的所有数字仍可追溯到 `results/bench_samplers.json`，
C++ 侧可用 `python scripts/validate_samplers.py` 复核正确性。
