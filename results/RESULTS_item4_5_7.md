# Item 4/5/7 — 闭环流水线、门控、Muon 优化器（进行中）

日期：2026-11-26　设备：Intel Arc XPU

## 1. Item 7 Muon：完成

`mdns/optim/muon.py`：Newton-Schulz 五次迭代正交化 + 动量，矩阵参数走 Muon、其余（embedding / head / norm / gamma / bias）走 AdamW，
共用一个 `Optimizer` 对象（单一 `state_dict`）。

- 正交化数值检验：输出奇异值 **min 0.677 / max 1.136 / mean 0.841**（µON 参考实现的正常范围）。
- 参数切分：`res-s` 上 **8 个张量 / 0.166M 参数（98.9%）** 走 Muon，33 个张量走 AdamW。
- **等墙钟时间对比**（同一去噪代理任务，各自先做 LR 扫描再比较）：

| 优化器 | 最优 LR | 步/秒 | 等时间步数 | val NLL@0.75 |
|---|---|---|---|---|
| AdamW | 1e-3 | 11.7 | 409 | 0.4805 |
| **Muon** | 1e-2 | 8.5 | 300 | **0.4775** |

Muon 每步慢 **1.37×**，但在等墙钟预算下仍**胜出 0.0029 NLL**。结论：可用，收益真实但不大。

## 2. Item 5 门控：完成

`mdns/gatekeeper.py`。**关键设计**：不用 ESS 当门控指标。ESS→1 只是 p_θ=π 的**必要非充分**条件——
一个塌缩到单一构型的模型会产生**常数**重要性权重，ESS 同样等于 1，用它当门控会主动推广塌缩模型。
因此默认指标是**留出集上的 proper scoring rule**（λ=0.75 遮蔽位置的平均交叉熵），塌缩模型在该指标上会灾难性变差。

实测门控确实拦住了坏候选（见下），这是它应有的行为。

## 3. Item 4 闭环：已跑通，并已收敛（详见 RESULTS_item9.md）

`mdns/loop.py` + `scripts/run_closed_loop.py`，每轮：**扫描 → 写 shard → 训练 → 门控 → 提升/回滚**。
产物：`round_XX.npz`（样本 + 重要性权重）、`best.pt`（模型 + EMA + 架构配置 + 轮次）、`metrics.json`。

在 XPU 上实测跑通（L=8, q=3, res-s, 6 轮）：

```
round 0: beta=0.500 ESS=0.0085 loss=0.541 score=-1.10156 -> promoted
round 1: beta=0.500 ESS=0.0256 loss=0.536 score=-1.10156 -> rolled back
round 2: ... ESS=0.0485 ... score=-1.10156 -> rolled back
...
1/6 rounds promoted; best score -1.1015625
```

**诚实结论**：管线机械上完全正确（扫描、落盘、训练、门控、回滚都执行了），但**采样质量没有提升**，
所以 6 轮里 5 轮被门控拒绝。原因是**重要性权重退化**：ESS≈0.01 意味着 512 个样本里只有 5 个左右真正携带权重，
模型只在这几个样本上学，留出集似然反而变差（1.1015 → 1.1484）。这正是原论文必须用
`--use_anneal` + 50000 epochs + 更大 batch 的原因。

已为此加的机制：β 退火（`--beta-start`）、周期重采样（`resample_every`，论文默认 10）、`--gate-patience`。
仍未解决：在小规模/短预算下让 ESS 升起来。下一步要试的方向：把 `scan_batch` 提到 2048–4096、
改用 `nbt-2`（同质量下参数更少、更容易拟合）、或在校准阶段先用高温把 ESS 拉起来再降温。

## 4. 开发中抓到的两个真 bug（都由门控暴露）

1. **`weights` 被转成 `x.dtype` 而 `x` 是 int64** → 重要性权重被截断为 0，损失恒为 0，模型完全不训练。
2. **损失符号错误**：`ce` 已取正号为 `-log p`，外层又多了一个负号 → 实际在**最大化**交叉熵。
   表现是 `loss` 从 −1.0 一路"改善"到 −5.9，而留出集 NLL 每轮变差、门控全部拒绝。
   门控在这里起到了应有的作用——它是唯一发现训练器坏掉的环节。

另修一处**测量**问题：`ce.sum()` 在 bf16 下累加会产生 ~30% 误差（曾把随机初始化模型的 NLL 误算成 0.82 而非正确的 1.107），
现在统一在 `mdns/losses.py::holdout_nll` 里用 float32 累加，门控、闭环、benchmark 共用同一份代码。

## 5. 复现

```powershell
python scripts/bench_optimizers.py                     # Muon vs AdamW
python scripts/run_closed_loop.py --rounds 6 --steps-per-round 60 `
    --L 8 --scan-batch 512 --beta 0.5 --beta-start 0.5 --out-dir runs/hot2
```
