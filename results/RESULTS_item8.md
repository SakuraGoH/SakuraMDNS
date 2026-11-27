# Item 8 — 清理无用与老旧代码

日期：2026-11-26

## 1. 清理结果

| 文件 | 清理前 | 清理后 | 说明 |
|---|---|---|---|
| `model/vit.py` | 498 行 | **159 行** (−68%) | 删掉约 250 行注释掉的 `@register_model` 死代码 + 3 个未使用类 |
| `model/vit_rope.py` | 497 行 | **254 行** (−49%) | 删掉约 200 行注释死代码 + `hf_checkpoint_load` / `adjust_pos_embed_size` |
| `utils.py` | 78 行 | 50 行 | 删 `metric()`（未使用）、`cycleloader()`（未使用，无 dataloader） |
| `utils_train.py` | 252 行 | 156 行 | 删 `train_ddp()`（无入口，且 `all_reduce(loss)` 后除以 `world_size**2` 语义可疑） |

累计删除约 **700 行**死代码。

## 2. 顺手修掉的根因 bug

`model/vit_rope.py::RoPEAttention.forward` 原本：

```python
q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
```

对 `view_as_complex` 的**源张量**做原地写，反向时报
`variables needed for gradient computation has been modified by an inplace operation`
（torch 2.10.0+xpu）。在 Item 3 里我是用 `registry._patch_rope_attention()` 打的猴子补丁；
**现在改成函数式写法**（`torch.cat` 拼回），根因修掉，补丁已删除。

验证：`vit-64x6` 基线现在可以直接 `build()` 并反传，不再需要任何绕过。

## 3. 依赖清理

`environment.yml` 删掉了 grep 验证过**从未被 import** 的依赖：

| 删除 | 理由 |
|---|---|
| `torchvision`, `torchaudio` | 无视觉数据集、无音频 |
| `einops` | 全仓库未 import |
| `omegaconf`, `pyyaml` | 配置是 dict / argparse |
| `huggingface_hub` | 唯一使用处 `hf_checkpoint_load()` 已删除 |

并加了注释说明本机实际用的是 XPU 构建（`--index-url .../whl/xpu`），避免有人照着 `conda env create` 装出 CPU/CUDA 版。

## 4. 刻意保留的东西

| 保留 | 理由 |
|---|---|
| `utils_ising.py` / `utils_potts.py` 里的慢速 Python MCMC | 是 C++ kernel 的**校验参照**（`validate_samplers.py` 依赖它们），删掉就没法做精确性对照 |
| `model/vit.py` 的 `Attention` / `Layer_scale_init_Block` / `vit_models` | `registry` 的 `vit-64x6` 基线需要，用来和论文原始架构对比 |
| `train_ising.py` / `train_potts.py` | 已加 docstring 标注 deprecated，但保留以便原论文 checkpoint 仍能加载复现 |
| `utils_train.py::train` | 同上；新代码走 `mdns/losses.py` + `mdns/loop.py` |

## 5. 回归验证

清理后必须证明没弄坏任何东西：

```
$ python -c "import mdns, mdns.losses, mdns.loop, mdns.gatekeeper, mdns.optim, \
             mdns.models.registry, mdns.samplers.gpu, model, utils, utils_train, \
             utils_ising, utils_potts"
all imports OK

$ python _scratch/smoke_arch.py        # 11 个架构预设：前向 + 反向
  OK vit-64x6       params=  0.318M  out=(8, 256, 4)
all ok

$ python scripts/validate_samplers.py  # C++ kernel 精确校验
8/8 passed
```

## 6. 新代码布局

清理后 `mdns/` 包的规模（约 1500 行，全部为新写）：

```
mdns/device.py               101   设备抽象（xpu/cuda/cpu）+ autocast
mdns/losses.py               139   scan / ESS / WDCE / holdout_nll / EMA
mdns/models/latticenet.py    227   ResBlock / NBTBlock / AttnBlock / LatticeNet
mdns/models/registry.py      125   11 个架构预设
mdns/optim/muon.py           168   Muon + AdamW 混合优化器
mdns/samplers/cpp_ext.py     187   C++ kernel 的 ctypes 绑定
mdns/samplers/gpu.py         174   GPU checkerboard / Wolff
mdns/loop.py                 195   闭环：扫描→落盘→训练→门控→提升
mdns/gatekeeper.py           109   留出集 proper scoring rule 门控
```

---

## 附：第二阶段清理（发布前）

Item 8 的第一轮清理只删了死代码，仍保留了旧模块。发布到 GitHub 时做了更彻底的整理：

| 动作 | 对象 |
|---|---|
| 删 | `utils_ising.py` / `utils_potts.py` 的**纯 Python 采样器**（MH / Glauber / Swendsen-Wang），物理量抽取到新的 `mdns/physics.py` |
| 删 | `utils.py`、`utils_train.py`（已被 `mdns/losses.py` + `mdns/loop.py` 取代） |
| 删 | `train_ising.py`、`train_potts.py`、`train_commands.sh`（入口统一为 `scripts/run_closed_loop.py`） |
| 删 | 两个评估 notebook（由 `scripts/eval_checkpoint.py` 取代） |
| 删 | `scripts/bench_samplers.py`（对照基线已不存在） |
| 移 | `model/` → `mdns/models/ropevit/`（仅作 baseline，供 `--arch vit-64x6`） |

清理后全仓库只剩新管线 + 上游 ViT baseline，`python scripts/validate_samplers.py` 仍是 8/8 通过。
