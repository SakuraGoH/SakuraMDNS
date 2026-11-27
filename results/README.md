# MDNS 工程改造：9 项交付索引

环境：Windows / **Intel Arc iGPU（Meteor Lake, 112 EU, 22.6 GB 共享内存）** / `torch 2.10.0+xpu` / MSVC 14.52
原始项目：MDNS（NeurIPS 2025），masked diffusion neural sampler

| # | 项目 | 状态 | 证据 |
|---|---|---|---|
| 1 | C++ 重写 MCMC 扫描 | ✅ | [RESULTS_item1.md](RESULTS_item1.md) — 精确校验 8/8，最速 **489×**（Potts SW） |
| 2 | GPU 并行采样 | ✅ | [RESULTS_item2.md](RESULTS_item2.md) — 15/15 校验通过；**结论：本机 GPU 全面慢于 CPU**（0.11–0.44×） |
| 3 | NBT+Transformer 混合残差 + 压缩比 | ✅ | [RESULTS_item3_6.md](RESULTS_item3_6.md) — 压缩比 **2** 为甜点（参数省 7×、算力省 7.4×） |
| 4 | 扫描→写数据→训练→门控→迭代闭环 | ✅ | [RESULTS_item9.md](RESULTS_item9.md) — 10 轮扫描 + 落盘 + 训练，10/10 被接受 |
| 5 | Gatekeeper 门控 | ✅ | [RESULTS_item4_5_7.md](RESULTS_item4_5_7.md) — 坏候选被拒（5/6、0/6），好候选全通过（10/10） |
| 6 | 模型注册表 | ✅ | [RESULTS_item3_6.md](RESULTS_item3_6.md) — 11 个预设，全字段可覆盖 |
| 7 | Muon 优化器 | ✅ | [RESULTS_item4_5_7.md](RESULTS_item4_5_7.md) — 等墙钟下胜 AdamW 0.0029 NLL |
| 8 | 清理死代码 | ✅ | [RESULTS_item8.md](RESULTS_item8.md) — 删 ~700 行，修 RopeVIT 反传 bug |
| 9 | XPU 纯残差端到端训练 | ✅ | [RESULTS_item9.md](RESULTS_item9.md) — NLL 1.09866 → **1.08062**（优于均匀基线） |

## 关键实测结论（与直觉相反、但可复现）

1. **数据生成从来不是瓶颈**：C++ SW 产出 10 万有效样本约 **1 秒**。项目瓶颈在训练侧。
2. **这块 iGPU 不适合做采样**：L=16/32 上 PyTorch XPU 的算子派发开销（~0.3 ms/op × ~24 op/扫描）
   吃掉了全部并行优势；规模放大后瓶颈转为**共享显存带宽**。GPU 该花在训练上。
3. **参数效率 ≠ 时间效率**：`nbt-2` 参数少 7×、算力少 7.4× 且质量持平，但**样本/秒只有 `res-s` 的 39%**
   （NBT 的 1×1↔3×3 组合让算子数翻倍）。所以默认架构定成 `res-s`。
4. **门控不能用 ESS**：塌缩模型的重要性权重是常数，ESS 同样趋于 1；必须用留出集 proper scoring rule。
5. **低温自采样会权重退化**：β 必须从"热"的一端起步再退火，否则 512 个样本里只有 5 个带权重，训练必然过拟合。

## 快速开始

```powershell
pwsh scripts/build_cpp.ps1 -Force            # 编译 C++ 采样器
python scripts/validate_samplers.py          # 8 项精确校验
python scripts/run_closed_loop.py --L 8 --rounds 10 --steps-per-round 200 `
    --resample-every 15 --scan-batch 1024 --beta 0.4 --beta-start 0.15 `
    --out-dir runs/demo
```

## 代码布局

```
mdns/                    新代码（~1500 行）
  device.py              设备抽象（xpu/cuda/cpu）+ autocast
  losses.py              scan / ESS / WDCE / holdout_nll / EMA
  models/                latticenet.py（Res/NBT/Attn）+ registry.py（11 预设）
  optim/muon.py          Muon + AdamW 混合优化器
  samplers/              cpp_ext.py（C++ 绑定）+ gpu.py（checkerboard/Wolff）+ cpp/
  loop.py                闭环
  gatekeeper.py          门控
scripts/                 编译、校验、benchmark、闭环入口
results/                 6 份结果文档 + 原始 JSON
runs/                    闭环产物（shard / best.pt / metrics.json）
model/, utils_*.py       旧代码（已标注 deprecated，保留以兼容原论文 checkpoint）
```
