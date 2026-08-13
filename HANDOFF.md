# TAX-DPD 复现交接文档

> 交接对象：接手在 H200 上继续本任务的 agent
> 撰写时间：2026-08-03
> 原仓库：<https://github.com/lyuxinghe/TAX-DPD>（ICRA 2026, arXiv:2604.11793）

---

## 0. 一句话现状

环境、数据、缺失代码补齐、GMM 阶段训练、扩散短跑验证**均已完成**；
扩散长跑（论文主结果来源）**尚未开始**，这是下一步。

**但有一个前提必须先读懂**：论文的核心模型文件 `tax3d_v2.py` 在原仓库里
**根本不存在**（作者未发布），当前仓库里那份是**我基于配置文件和仓库内其他
组件重建的**。因此即使全部跑通、数字漂亮，也不能声称"复现了论文"。诚实的
表述是"在 RPDiff 数据上训练了一个 TAX-DPD 架构的近似实现"。详见 §6。

---

## 1. 原仓库的缺失情况（这是整件事的起点）

`src/non_rigid/utils/script_utils.py` 导入两个文件，原仓库都没有：

```python
from non_rigid.models.tax3d import (DiffusionTransformerNetwork, CrossDisplacementModule)
from non_rigid.models.tax3d_v2 import (TAX3Dv2Network, TAX3Dv2MuFrameModule, TAX3Dv2FixedFrameModule)
```

处理方式：

| 文件 | 来源 | 可信度 |
|---|---|---|
| `tax3d.py` (528 行) | 作者本人另一个仓库 `ey-cai/non-rigid` 的 `articulated` 分支，**原始代码** | 高 |
| `tax3d_v2.py` (449 行) | **我重建**。骨干（DiT / diffusion / encoder）全部来自本仓库已有文件，Lightning 模块的训练逻辑照 `tax3d.py` 的模式写 | **低，未经作者验证** |

`tax3d_v2.py` 在 `ey-cai/non-rigid` 的全部 14 个分支里都搜过（含第一作者
Lyuxing He 的 `articulated_lh` 分支），均无。v2 模型代码确实没公开。

**建议**：向 <https://github.com/lyuxinghe/TAX-DPD/issues> 提 issue 请求发布
`tax3d_v2.py`。一旦拿到官方版本，应当替换掉我这份重建，重跑全部实验。

---

## 2. 环境

用 pixi 管理（原仓库的 conda/Dockerfile 方案锁死在 CUDA 11.8 + torch 2.0.1，
**不支持 Blackwell 架构**，PTX ISA 版本不够）。

```bash
pixi install          # conda + pip 依赖
pixi run post-install  # pyg 扩展 + third_party + 本项目（editable）
pixi run python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

关键版本：Python 3.9 / torch 2.7.0+cu128 / torch-geometric 2.6.1 / pytorch3d 0.7.8
/ lightning 2.0.3 + lightning-cloud 0.5.57（版本必须配对，否则 `AppinstancesIdBody`
导入失败）。

### ⚠️ 一处 in-tree 之外的补丁，`pixi install` 后会丢

安装包 `rpad-pyg` 的 `pointnet2.py` 从 `torch_geometric.nn` 导入 `PointConv`，
该符号在 torch_geometric ≥ 2.3 已改名 `PointNetConv`。我直接改了
`.pixi/envs/default/lib/python3.9/site-packages/rpad/pyg/nets/pointnet2.py`。

**在 H200 上重建环境后必须重新打这个补丁**，否则 import 就炸：

在仓库根目录直接运行（幂等，重复执行安全）：

```bash
pixi run python - <<'PYEOF'
import pathlib, rpad.pyg.nets.pointnet2 as m

p = pathlib.Path(m.__file__)
s = p.read_text()
old = "from torch_geometric.nn import PointConv, fps, global_max_pool, knn_interpolate, radius"
new = ("try:\n    " + old + "\nexcept ImportError:\n    "
       "from torch_geometric.nn import PointNetConv as PointConv, "
       "fps, global_max_pool, knn_interpolate, radius")

if "PointNetConv as PointConv" in s:
    print(f"already patched: {p}")
elif old in s:
    p.write_text(s.replace(old, new, 1))
    print(f"patched: {p}")
else:
    raise SystemExit(f"import line not found in {p} -- upstream changed, patch by hand")
PYEOF
```

通过 `rpad.pyg.nets.pointnet2.__file__` 定位文件，不依赖硬编码路径。
若该模块本身 import 失败（正是这个 bug 的症状），改用：

```bash
pixi run python -c "import rpad.pyg, pathlib; print(pathlib.Path(rpad.pyg.__file__).parent / 'nets/pointnet2.py')"
```

拿到路径后手工加 try/except。验证：

```bash
pixi run python -c "from non_rigid.utils.script_utils import create_model; print('✅ import 链通')"
```

同样的改动我也在 in-tree 的 `src/non_rigid/nets/pn2.py` 做了（那份在 git 里，
不会丢）。

### 已知未解决的环境问题

`gym==0.21.0` 装不上（元数据不合规，pip ≥ 24 拒绝）。`pixi.toml` 里已跳过，
`third_party/dedo` 和 `3D-Diffusion-Policy` 是用 `--no-deps` 装的。这不影响
RPDiff 任务；只有 DEDO 仿真需要它。

---

## 3. 数据

RPDiff 官方只提供 3 个任务包，从 Dropbox 下载（URL 在 rpdiff 仓库的
`scripts/dl_train_data.bash` 里）：

| 任务 | 大小 | 预处理缓存 |
|---|---|---|
| `mug_on_rack_multi_large_proc_gen_demos` | 23 GB | ✅ 3600 文件 |
| `can_in_cabinet_stack` | 33 GB | ❌ 未做 |
| `book_on_bookshelf_double_view_rnd_ori` | 17 GB | ❌ 未做 |
| `descriptions/objects`（物体资产） | 999 MB | — |

放在 `/home/yuchi/data/rpdiff/`。`configs/dataset/rpdiff.yaml` 里的两个路径
已从作者的绝对路径改成这个位置——**换机器要改**。

论文表格里的 `mug_rack_easy_single` / `mug_rack_med_single` 等任务，RPDiff
官方数据包**不提供**，需要自己用 RPDiff 的仿真生成。

### 预处理：性能关键，且有一个陷阱

不做预处理时，每取一个样本要在 CPU 上对全分辨率点云（child 29417 点 /
parent 60006 点）跑 FPS 降采样，实测 **275 ms/样本**，而读文件只要 3.1 ms。
GPU 会一直空转等数据（实测 GPU 0% / worker 98%）。

```bash
pixi run python scripts/preprocess_rpdiff.py \
  --data-dir /home/yuchi/data/rpdiff/data/task_demos \
  --task-name <TASK> --task-type <TYPE> \
  --sample-size-action 2048 --sample-size-anchor 4096 \
  --downsample-type fps --split all
```

**陷阱**：缓存点数不能等于训练时请求的点数。`maybe_apply_augmentations`
要求遮挡后点数 `> min_num_points`，缓存点数 == 请求点数时遮挡结果必然更少，
于是每次都被丢弃 —— **遮挡增强完全失效**。实测：

| 缓存尺寸 | action 遮挡命中 | anchor 遮挡命中 | 载入耗时 |
|---|---|---|---|
| 不预处理 | 83/100 | 84/100 | 275 ms |
| 512 / 1024（== 请求值） | **0/100** | **0/100** | 0.4 ms |
| **2048 / 4096（当前采用）** | 79/100 | 83/100 | 19 ms |

作者配置默认 `preprocess: True`，而预处理脚本默认 512/512 —— 两者一撞遮挡就
死。这可能是作者的疏漏，也可能他们本就没用遮挡训练，无法判断。我选了大缓存
方案保住增强，代价是载入多 19 ms（被 worker 预取掩盖，不影响 epoch 时间）。

**验证方法**（把场景级 SE(3) 增强关掉，隔离出遮挡）：

```bash
pixi run python -c "
import torch, os; from pathlib import Path; from hydra import compose, initialize
with initialize(version_base='1.3', config_path='configs'):
    cfg = compose(config_name='train_gmm', overrides=[
        'model.rel_pos=True','model.point_encoder=pn2','model.object_scale=3.0',
        'dataset=rpdiff','dataset.preprocess=True',
        'dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos',
        'dataset.rpdiff_task_type=task_name_mug_on_rack_multi',
        'dataset.sample_size_anchor=1024',
        'dataset.scene_transform_type=identity',
        'dataset.rotation_variance=0.0','dataset.translation_variance=0.0',
        'dataset.train_dataset_size=3200','dataset.val_dataset_size=400',
        'dataset.test_dataset_size=400','training.batch_size=32',
        'resources.num_workers=0','resources.gpus=[0]'])
from non_rigid.datasets.rigid import RPDiffDataset
ds = RPDiffDataset(Path(os.path.expanduser(cfg.dataset.data_dir)), cfg.dataset, 'train')
print('action 唯一签名', len({round(float(ds[0]['pc_action'].sum()),4) for _ in range(30)}), '/30')
print('anchor 唯一签名', len({round(float(ds[0]['pc_anchor'].sum()),4) for _ in range(30)}), '/30')
"
```

期望 30/30。如果是 1/30，说明遮挡失效（缓存尺寸设错了）。

---

## 4. 代码改动清单

### 4.1 新增文件（未跟踪，全部需要保留）

| 文件 | 行数 | 说明 |
|---|---|---|
| `src/non_rigid/models/tax3d.py` | 528 | 作者原码，取自 `ey-cai/non-rigid@articulated` |
| `src/non_rigid/models/tax3d_v2.py` | 449 | **我重建**，见 §6 |
| `src/non_rigid/utils/vis_utils.py` | 49 | `logging_utils.py` 需要 `get_color`，原仓库无此文件。返回 [N,6] 的 XYZRGB 数组供 `wandb.Object3D` 用 |
| `src/non_rigid/datasets/real_world.py` | 15 | 存根。`script_utils.py` 顶层导入 `RealWorldDataModule`，缺了整个模块导入链就断。**未实现**，只在 `dataset.name == "insertion"`（NIST 任务）时才会被真正调用，届时会 raise |
| `pixi.toml` / `pixi.lock` | 90 / — | 环境定义 |

### 4.2 修改文件（9 个，共 +94 / −25 行）

| 文件 | 改动 |
|---|---|
| `configs/dataset/rpdiff.yaml` | 两处绝对路径改本地 |
| `configs/model/df_cross.yaml` | 补 `rotary: False` |
| `configs/model/tax3dv2.yaml` | 补 `rotary: False` |
| `configs/train_gmm.yaml` | 补 `resume_ckpt: null` / `resume_epoch: 0` |
| `scripts/train_gmm.py` | 加续训支持（+48 行） |
| `scripts/eval_rigid.py` | GMM checkpoint 加载兼容新旧两种格式 |
| `src/non_rigid/datasets/rigid.py` | 空点云守卫 + 补 `T_action2goal` |
| `src/non_rigid/models/dit/models.py` | 4 个向后兼容别名 |
| `src/non_rigid/nets/pn2.py` | `PointConv` → `PointNetConv` 兼容 |

完整 diff 用 `git diff` 查看。以下只解释非显然的几处。

**`models.py` 的别名**（第 1384 行附近）

`tax3d.py` 和 `df_base.py` 引用 4 个当前 `models.py` 里不存在的类名。全部映射
到 `DiT_PointCloud_Cross`：

```python
DiT_PointCloud_Unc       = DiT_PointCloud_Cross   # self-attn → cross-attn（近似！）
DiT_PointCloud_Unc_Cross = DiT_PointCloud_Cross   # 纯改名
DiT_PointCloud           = DiT_PointCloud_Cross   # scene-level 变体
Rel3D_DiT_PointCloud_Unc_Cross = DiT_PointCloud_Cross
```

⚠️ 只有第二行是严格等价的改名。其余三个是**近似映射** —— 它们原本可能是
self-attention 或 scene-level 变体，语义并不完全一致。这只影响 `df_cross`
（TAX3D v1 对照组）的行为，**不影响 tax3dv2 主模型**（它走 `TAX3Dv2_*_DiT`，
不经过这些别名）。如果要跑 df_cross 基线，这里需要重新核实。

**`rigid.py` 的 `T_action2goal`**

`get_pred_pcd_rigid_errors` 需要这个键，数据集只有 `T_action2world` /
`T_goal2world`。合成方式：`T_goal2world⁻¹ ∘ T_action2world`。
注：在 RPDiff 里这两个变换恰好相同（都是 `Translate(scene_center)∘T⁻¹`），
所以结果接近单位矩阵；但按定义合成才是对的。

**`train_gmm.py` 的续训**

原脚本只 `torch.save(model.state_dict())`，无任何加载逻辑。改动：
checkpoint 改存 `{model, optimizer, epoch}`；`resume_ckpt` 非空时复用原 run
目录、跳过初始可视化、循环从 `start_epoch` 起。旧格式（裸 state_dict）仍能
加载，会打 WARNING 说明 AdamW 动量从零重建。

用法：

```bash
... resume_ckpt=/path/to/epoch_N.pt resume_epoch=N
```

`scripts/train.py`（扩散）**本来就有续训**，不需要改 —— 它有
`save_weights_only=False` + `save_last=True` 的 ModelCheckpoint，用
`ckpt_path=<path>` 即可从 `last.ckpt` 干净续训。

---

## 5. 已修复的 bug（6 个）

前 3 个是我重建 `tax3d_v2.py` 时写错的，后 3 个是接入真实数据/流程时暴露的。

| # | 症状 | 根因 | 修在哪 |
|---|---|---|---|
| 1 | `ValueError: Unsupported function type False` | 我把 `time_based_weighting` 从字符串 `"even"` 转成了布尔。`_time_based_weights` 只认 `"even"/"linear"/"sigmoid"`，无 falsy 分支 | `tax3d_v2.py`，原样透传字符串 |
| 2 | `TypeError: forward() got an unexpected keyword argument 'noisy_goal'` | 我把 `noisy_goal` 塞进了 `model_kwargs`。该 dict 会被 splat 进 DiT（`model(xr_t, xs_t, t, **model_kwargs)`），只能含 DiT 签名接受的键（`y`/`x0`/`rel_pos`/`finetune_frame`）。mu-frame 的参考系是通过 `p_sample_loop` 里 `pred_ref = noise_r.clone()` 传的 | `tax3d_v2.py`，删掉该键 |
| 3 | ModelCheckpoint 监控的 metric 不存在（会在首次验证时抛 `MisconfigurationException`） | 我照抄了 `df_base.py` 的键名 `val_wta_rmse_*`，但那是遗留文件（`script_utils.py` 从不导入）。真正接进 `train.py` 的 `tax3d.py` 用 `val_rmse_wta_*` | `tax3d_v2.py`，键名对齐 |
| 4 | `KeyError: 'T_action2goal'` | 数据集不产出该键 | `rigid.py`，合成后加入 batch |
| 5 | `AssertionError: T_action2distactor_list must be provided for distractor_min error` | 训练配置默认 `prediction_error_type: distractor_min`，但 RPDiff 数据没有 distractor 位姿 | `tax3d_v2.py`，误差计算包 try/except（KeyError, AssertionError），失败则跳过该指标 |
| 6 | `RuntimeError: max(): input.numel() == 0` | 个别 demo 的点云为空 | `rigid.py`，空点云时跳到下一个 demo |

### 一个仓库自带的、未修的隐性 bug

`maybe_apply_augmentations` 在**球体和平面遮挡同时触发**时，mask 合成的形状
会不匹配（`new_mask` 是原始点数，`temp_mask` 是第一次遮挡后的点数）。
仓库自带配置走不到这条路（action 是 `plane=0.8, ball=0`，anchor 是
`ball=0.8, plane=0`，同一点云只会触发一种），所以不影响复现。若要同时开启
两种遮挡，需要先修这里。

---

## 6. ⚠️ `tax3d_v2.py` 是重建的 —— 必读

### 确定正确的部分

模型骨干**全部来自仓库内已有文件**，不是我写的：

- `TAX3Dv2_FixedFrame_Token_DiT` / `TAX3Dv2_MuFrame_DiT`（`dit/models.py`，作者原码）
- `JointFeatureEncoder` / `DisjointFeatureEncoder`（`encoders.py`，作者原码）
- `create_diffusion_ddrd_separate` / `create_diffusion_mu`（`dit/diffusion/`，作者原码）

我写的只是 Lightning 模块的外壳：优化器、训练/验证循环、WTA 采样、日志。

### 已验证的

- 合成张量冒烟测试：fixed 和 mu 两条路径的 `training_losses` 与
  `p_sample_loop` 调用约定均正确，loss 有限、输出形状对
- 真实数据 50 epoch：train/loss 1.080 → 0.044，`val_rmse_wta_0` = 0.139
- 训练能收敛、验证路径能走通、checkpoint 能保存

### 未验证 / 存疑的

1. **是否与作者实现等价** —— 无法验证，没有参照
2. **mu-frame 的坐标系约定**。我在 `_get_x_start` 里做
   `pc_goal - noisy_goal`（在 GMM 估计的相对系里学），推理时再加回
   `noisy_goal`。但 mu 的 `p_sample_loop` 语义是 `noise_r` 承载**绝对**帧
   估计（`pred_ref = noise_r.clone()`），且末尾不加回任何偏移。这两种约定
   可能不一致 —— 我当前的 `_predict_wta` 里 `noise_r` 用的是随机噪声而非
   GMM 预测，末尾手动加回 `noisy_goal`。**建议：优先用 fixed frame**
   （`model.frame_type=fixed`，也是配置默认值），mu-frame 需要先核实
3. **`num_wta_trials` 在训练/验证间的一致性**、可视化的 viz_idx 采样等细节

### 参数量对照

重建版实例化后 **6.5 M** 参数（DiT xS: depth=5, hidden=128, heads=4 +
PointNet++ 编码器）。论文说骨干是 xS，量级看起来合理，但没有官方数字可比。

---

## 7. 已完成的训练与产物

### GMM 阶段（第一阶段，已完成）

Run ID `kkqzj33eok`，1000 epoch，7 小时 12 分（26 s/epoch，RTX 5060 Ti）。
产物在 `/home/yuchi/data/gmm_outputs/kkqzj33eok/`。

| epoch | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| val_loss | 84 | **77** | 82.3 | 93.2 | 112 | 132 | 158 | 178 | 189 | 200 |

train_loss 一路降到 24.7，val_loss 在 **epoch 200 触底后单调回升到 200**
—— 明确的过拟合。

> **⚠️ 评估时必须用 `epoch_200.pt`，不要用 `epoch_1000.pt`。**
> 脚本不做 best-checkpoint 选择，10 个 checkpoint 全部保留。
> H200 上重跑时建议把 `epochs` 降到 200~300，或加 early stopping。

覆盖率曲线（top-0.99/0.90/0.50 需要多少个高斯分量）在
`logs/top_probs.html`，是比 loss 更能反映 GMM 质量的指标，值得先看一眼。

### 扩散短跑（验证用，已完成）

50 epoch，产物在
`scripts/logs/train_rpdiff_tax3dv2/2026-07-31/11-47-59/checkpoints/`。

| 指标 | 值 |
|---|---|
| train/loss | 1.080 → **0.044** |
| val_rmse_wta_0 | **0.139** |
| val_rmse_0 | 0.302 |

这份 checkpoint 只用于证明管道可用，没有科研价值（epoch 数远不够）。

### 一处待解决的路径不一致

`train_gmm.py` 写到 `{gmm_log_dir}/{run_id}/`，而 `eval_rigid.py` 拼的是
`{gmm_log_dir}/{task_name}/{run_id}/`（多一层 task_name）。我传的
`task_name=mug_multi` 存进了 config.yaml 但没参与路径。跑 eval 时要么加 CLI
覆盖、要么建软链接。**这个我没动**，留给评估阶段一并处理。

---

## 8. 下一步：H200 上的扩散长跑

### RTX 5060 Ti 上的实测基线

| | per-batch | per-epoch | 全程 |
|---|---|---|---|
| 训练步 | 0.248 s | 25 s（100 batch） | 20000 epoch = **5.7 天** |
| 验证 | 25.5 s（10 WTA × 100 步去噪） | 25 batch ≈ 10.6 min | 每 100 epoch 一次 × 200 次 = **1.4 天** |

合计约 7 天。验证开销出乎意料大：400 样本 × 10 WTA × 100 步 = 40 万次前向。

### H200 预期

| | RTX 5060 Ti | H200 SXM5 | 比值 |
|---|---|---|---|
| FP32 | 22 TFLOPS | 67 TFLOPS | 3× |
| 显存带宽 | 900 GB/s | 4.8 TB/s | 5.3× |
| 显存 | 16 GB | 141 GB | 8.8× |

PointNet++ 是带宽密集、DiT 是计算密集，综合 **4–5×**，即 5.7 天 → **约 1.3 天**。

### 加速手段（按收益排序，附风险）

| 手段 | 收益 | 风险 |
|---|---|---|
| 加大 batch（32 → 128/256，H200 显存充裕） | 墙钟时间近线性下降 | **中**。扩散模型的 batch scaling 非严格线性，LR 需同步放大（线性或 √ 规则），收敛行为需要实测确认。建议先 128 跑 200 epoch 对比 loss 曲线 |
| `check_val_every_n_epochs` 100 → 500 | 验证开销 1.4 天 → 0.28 天 | 低。只影响观测粒度 |
| `num_wta_trials` 10 → 3 | 验证再快 3.3× | 低-中。WTA 是估计量，样本少方差大；最终报数时仍应用 10 |
| `torch.compile(self.network)` | 10–30% | 低。PyTorch 2.7 支持良好；首次编译有几分钟开销 |
| `diff_inference_steps` 100 → 20（配 DDIM） | 验证 5× | **高**。直接改变采样质量，会影响所有报告指标。不建议在主实验里用 |

### 建议的启动命令（H200）

```bash
cd ~/projects/TAX-DPD/scripts

nohup env WANDB_MODE=offline pixi run python train.py \
  model=tax3dv2 model.type=point \
  model.frame_type=fixed \
  model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.object_scale=3.0 \
  dataset=rpdiff dataset.preprocess=True \
  dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos \
  dataset.rpdiff_task_type=task_name_mug_on_rack_multi \
  dataset.sample_size_anchor=1024 \
  dataset.train_dataset_size=3200 \
  dataset.val_dataset_size=400 \
  dataset.test_dataset_size=400 \
  training.batch_size=128 \
  training.val_batch_size=64 \
  training.check_val_every_n_epochs=500 \
  training.num_wta_trials=3 \
  wandb.group=rigid wandb.project=tax_dpd_h200 \
  resources.num_workers=16 'resources.gpus=[0]' \
  > ~/diffusion_train.log 2>&1 &
```

⚠️ `batch_size=128` 是 4× 于作者配置，**LR 没有同步调整**。要么把
`training.lr` 从 1e-4 提到 2e-4~4e-4，要么先用 `batch_size=32` 对齐作者设置
跑一遍作为基准。这个取舍需要你自己定，我没有实测数据支撑。

### ⚠️ 进程必须脱离会话

我第一次跑 GMM 时用了会话托管的后台机制，约 4 小时后被回收（退出码 144），
epoch 139 的进度全丢。正确做法是 `setsid`：

```bash
setsid nohup env WANDB_MODE=offline pixi run python train.py ... \
  > ~/diffusion_train.log 2>&1 < /dev/null &
```

验证是否真的脱离（PPID 应为 1，SID 应与当前 shell 不同）：

```bash
ps -o pid,ppid,sid,cmd -p $(pgrep -f "train.py" | head -1)
ps -o sid= -p $$
```

### 监控

tqdm 用 `\r` 而非 `\n`，必须转换才读得到：

```bash
tail -c 2000 ~/diffusion_train.log | tr '\r' '\n' | grep -v "^$" | tail -3
tr '\r' '\n' < ~/diffusion_train.log | grep -o "train/loss=[0-9.]*" | awk 'NR%100==1' | tail -20
tr '\r' '\n' < ~/diffusion_train.log | grep -o "val_rmse_wta_0=[0-9.]*" | uniq | tail
```

**注意**：`pgrep -f "train.py"` 会匹配到检查命令自身而误报"运行中"。我踩过
这个坑，报告了错误状态。请用 PID 直查（`ps -p <PID>`），并交叉验证 GPU 占用
和 checkpoint 时间戳。

---

## 9. 完整待办

| # | 任务 | 预计（H200） | 阻塞 |
|---|---|---|---|
| 1 | 扩散长跑（mug_on_rack） | ~1.3 天 | 无，可立即开始 |
| 2 | 评估（`eval_rigid.py`） | 数小时 | 需先解决 §7 的 GMM 路径不一致 |
| 3 | can_in_cabinet（预处理 + GMM + 扩散） | ~1.5 天 | 无 |
| 4 | book_on_bookshelf（同上） | ~1.5 天 | 无 |
| 5 | Mug-Rack Easy/Med Single | — | **数据不存在**，需用 RPDiff 仿真生成 |
| 6 | NIST 插入 | — | `real_world.py` 是存根，未实现 |
| 7 | DEDO 布料 | — | 数据未下载，且需要 `gym==0.21.0` |
| 8 | 与论文数字对比 | — | 依赖 1–4 |

### 优先级建议

1. **先提 issue 要 `tax3d_v2.py`**。这是唯一能把"近似实现"变成"真复现"的
   途径，且不阻塞其他工作，越早发出越好。
2. 用 fixed frame 跑通 mug_on_rack 全流程（训练 → 评估），拿到第一个可对比
   的数字。
3. 再铺开到另外两个任务。
4. mu-frame 和 df_cross 基线放最后 —— 前者坐标系约定存疑（§6），后者依赖
   §4.2 里那几个近似别名。

---

## 10. 关键文件索引

```
configs/
  dataset/rpdiff.yaml          ← 数据路径（换机器必改）
  model/tax3dv2.yaml           ← 主模型配置
  training/rpdiff_tax3dv2.yaml ← epochs=20000, batch=32, val 每 100 epoch
  train_gmm.yaml               ← GMM 配置 + 续训字段
scripts/
  train.py                     ← 扩散训练（自带续训）
  train_gmm.py                 ← GMM 训练（续训是我加的）
  eval_rigid.py                ← 评估
  preprocess_rpdiff.py         ← 预处理（注意 §3 的尺寸陷阱）
src/non_rigid/
  models/tax3d_v2.py           ← ⚠️ 我重建的核心模型
  models/tax3d.py              ← 作者原码（v1）
  models/dit/models.py         ← DiT 骨干（含 TAX3Dv2_*_DiT）+ 我加的别名
  models/dit/diffusion/        ← ddrd_separate（fixed）/ mu（mu-frame）
  models/encoders.py           ← Joint/Disjoint 特征编码器
  datasets/rigid.py            ← RPDiff dataloader
  utils/script_utils.py        ← create_model / create_datamodule 工厂

/home/yuchi/data/rpdiff/                       ← 数据（73 GB）
/home/yuchi/data/gmm_outputs/kkqzj33eok/       ← GMM 产物（用 epoch_200.pt）
~/gmm_train.log                                ← GMM 训练日志
```

