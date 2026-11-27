# Item 6 + Item 3 — 模型注册表、NBT+Transformer 混合残差架构、压缩比研究

日期：2026-11-26　设备：Intel Arc iGPU（XPU）+ Core Ultra　L=16，q=3

## 1. 交付物

| 文件 | 内容 |
|---|---|
| `mdns/models/latticenet.py` | `LatticeNet` 主干、`ResBlock`、`NBTBlock`、`MidBlock`、`AttnBlock` |
| `mdns/models/registry.py` | 11 个命名预设、`build()` / `build_from_config()` / `list_presets()` |
| `scripts/bench_architectures.py` | 成本表（参数/FLOPs/吞吐）+ 固定步数去噪代理质量研究 |
| `results/bench_architectures.json` | 原始数据 |

## 2. 架构：为什么换掉 ViT

原始 MDNS 是"把 L² 个格点当 token 序列的 ViT"，每层全注意力，完全没有利用输入**本身就是周期边界上的二维点阵**这一结构。
`LatticeNet` 保持**完全相同的调用接口**（`model(x) -> (B, D, N)`、`model.logits(x)`、`vocab_size`、`length`），可以直接塞进
`utils_train.rnd` / `loss_wdce`，但主干换成：

```
token embedding (vocab → C)
  → 1x1 stem (embed_dim → C)
  → [ ResBlock | NBTBlock ] × n_blocks          # 全部 circular padding
  → GroupNorm
  → 1x1 conv head → N logits/site
```

**关键设计点**

1. **circular padding**。物理用的是周期边界，零填充会凭空造出一个哈密顿量里没有的边界。
2. **GroupNorm 而非 BatchNorm**：bf16 + batch 256 下更稳。
3. **LayerScale**：每个残差块乘一个初值为 0 的 `gamma`，训练初期整块≈恒等，深层堆叠稳定（沿用原仓库做法）。

**NBTBlock（瓶颈 + Transformer）**

```
x ──► 1x1 压缩  C → C/ratio ──► [3x3 depthwise-separable] × n_inner ──► [AttnBlock] ──► 1x1 扩张 C/ratio → C ──► + x
```
- `ratio` 就是**压缩比**，所有贵的运算都发生在 `C/ratio` 宽度上；
- `n_inner` 是 KataGo 式的"嵌套内层块"；
- `AttnBlock` 在**瓶颈宽度**上做全局注意力（BoTNet 思路），默认**不加位置编码**——
  周期点阵应当平移等变，注意力只做"集合聚合"，位置信息由卷积特征携带。可选 `attn_pos="sincos"`。

## 3. 注册表（Item 6）

```python
from mdns.models.registry import build, list_presets
model = build("nbt-2", L=16, vocab_size=4, device="xpu:0")
model.num_params(); model.config()
```

| 预设 | block | C | 块数 | 压缩比 | 内层 | 注意力 | 参数量 |
|---|---|---|---|---|---|---|---|
| `res-s` **(默认)** | res | 48 | 4 | 1 | – | – | 0.168 M |
| `res-m` | res | 64 | 6 | 1 | – | – | 0.446 M |
| `res-l` | res | 96 | 8 | 1 | – | – | 1.333 M |
| `nbt-1` | nbt | 64 | 4 | 1 | 2 | ✓ | 0.209 M |
| `nbt-2` | nbt | 64 | 4 | 2 | 2 | ✓ | 0.064 M |
| `nbt-4` | nbt | 64 | 4 | 4 | 2 | ✓ | 0.023 M |
| `nbt-8` | nbt | 64 | 4 | 8 | 2 | ✓ | 0.010 M |
| `nbt-4-noattn` | nbt | 64 | 4 | 4 | 2 | ✗ | 0.014 M |
| `nbt-4-deep` | nbt | 64 | 6 | 4 | 3 | ✓ | 0.037 M |
| `nbt-4-sincos` | nbt | 64 | 4 | 4 | 2 | ✓ + sincos | 0.023 M |
| `vit-64x6` | vit | 64 | 6 | 1 | – | ✓(全注意力) | 0.318 M |

所有字段（`channels` / `n_blocks` / `compression_ratio` / `n_inner` / `use_attention` / `n_heads` / `mlp_ratio` / `attn_pos` / `embed_dim` / `dtype`）
都可以在 `build(..., **overrides)` 里覆盖，训练脚本不需要再改代码。
`model.config()` 会把完整配置写进 checkpoint，`build_from_config()` 能原样重建。

## 4. 压缩比研究

**代理任务**：用 C++ Swendsen-Wang 生成真实 Potts 样本（β=1.005 临界点），把每个格点以 λ~U(0,1) 遮蔽，
只在被遮蔽位置算交叉熵。**这不是 MDNS 的目标函数**，而是隔离出"这个主干对 p(x_masked | x_visible) 建模得多好"，
在相同数据、相同优化器、相同随机种子、相同步数下比较。

评估用**固定高遮蔽率**。这一步很关键：β=1.2 时点阵近乎完全有序，λ 小的时候单个可见近邻就足以确定该点，
CE 直接饱和到 0，无法区分架构（第一版就是这样翻车的）。改用 β=1.005 且 λ=0.75/0.9 后才有区分度。

**300 步结果**（`NLL@0.75` 越低越好）：

| 预设 | 参数量 (M) | 前向 GFLOPs | 吞吐 (样本/s) | NLL@0.5 | **NLL@0.75** | NLL@0.9 |
|---|---|---|---|---|---|---|
| `res-s` | 0.168 | 5.44 | **777** | 0.4404 | **0.4814** | 0.5449 |
| `res-m` | 0.446 | 14.50 | 314 | 0.4365 | **0.4814** | 0.5430 |
| `nbt-2` | 0.064 | 1.96 | 304 | 0.4355 | 0.4844 | 0.5527 |
| `nbt-1` | 0.209 | 6.60 | 242 | 0.4365 | 0.4873 | 0.5527 |
| `vit-64x6` | 0.318 | 16.20 | 327 | 0.4467 | 0.4884 | 0.5526 |
| `nbt-4-noattn` | 0.014 | 0.38 | 581 | 0.4395 | 0.4912 | 0.5684 |
| `nbt-4` | 0.023 | 0.65 | 361 | 0.4424 | 0.4922 | 0.5625 |
| `nbt-8` | 0.010 | 0.25 | 458 | 0.4785 | 0.5254 | 0.6211 |

**100 步结果对比**（说明为什么训练预算会改变结论）：

| 预设 | NLL@0.75 @100步 | NLL@0.75 @300步 |
|---|---|---|
| `res-s` | 0.4932 | 0.4814 |
| `vit-64x6` | 0.4954 | 0.4884 |
| `nbt-1` | 0.6074 | 0.4873 |
| `nbt-2` | 0.6211 | 0.4844 |
| `nbt-4` | 0.7344 | 0.4922 |
| `nbt-8` | 0.8086 | 0.5254 |

### 结论

1. **压缩比 2 是甜点**。300 步时 `nbt-2` 用 **0.064 M 参数 / 1.96 GFLOPs** 达到 0.4844，
   与 `res-m`（0.446 M / 14.5 GFLOPs）只差 0.003 —— **参数省 7×、算力省 7.4×**，质量基本持平。
2. **压缩比 4 仍可接受**（0.4922），**压缩比 8 明显退化**（0.5254，比 `res-s` 差 0.044）。
3. **短训练预算下压缩会吃亏**：100 步时 `nbt-4` 是 0.7344（几乎没学会），300 步降到 0.4922。
   压缩网络需要更多步才追平，所以"压缩比"必须和训练预算一起选。
4. **瓶颈处的注意力在本问题（L=16）上没有可测收益**：`nbt-4` 0.4922 vs `nbt-4-noattn` 0.4912，在噪声内。
   原因：4 个 3×3 残差块的感受野约 9 格 ≈ 0.56L，已经覆盖大半张 16×16 点阵。**L=32 时注意力预计会变得重要**
   （感受野只占 0.28L），这一点留待 L=32 复测。
5. **但对本机训练吞吐而言，`res-s` 是最好的默认值**：777 样本/s，比 `nbt-2`（304）快 2.6×、比 `vit`（327）快 2.4×，
   质量并列第一。原因很现实——XPU 上算子派发是瓶颈，`nbt` 的 1x1↔3x3 组合多出好几倍算子数。
   **参数效率 ≠ 时间效率**，这台机器上时间效率更重要。

因此 `DEFAULT_PRESET` 定为 **`res-s`**；如果将来显存/参数成为约束，改用 `nbt-2`。

## 5. 过程中发现的问题

**上游 RopeVIT 在 torch 2.10 上无法反传。** `RoPEAttention.forward` 里

```python
q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis)
```

对 `view_as_complex` 的源张量做了原地写，反向时报
`one of the variables needed for gradient computation has been modified by an inplace operation`。
`registry._patch_rope_attention()` 给出了等价的非原地实现，使 `vit-64x6` 可以作为基线参与对比。
这是 Item 8 要清理的遗留问题之一。

## 6. 注意事项

- 代理任务是**有监督去噪**，不是 MDNS 的 `wdce` 目标；它只用来给主干排序。
- 300 步时 `res-s`/`res-m` 并列 0.4814，很可能是 λ=0.75 下该任务的**条件熵下限**附近（指标开始饱和），
  0.01 量级的差异不应过度解读；0.04 以上（如 `nbt-8`）才是可靠信号。
- 只测了 L=16、单一种子、一个温度。L=32 的复测留给后续。

## 7. 复现

```powershell
python scripts/bench_architectures.py            # 300 步完整研究（约 8 分钟）
python scripts/bench_architectures.py --quick    # 100 步
```
