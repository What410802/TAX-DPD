# TAX-DPD 复现状态记录（截至 2026-08-13）

本文档记录当前 TAX-DPD 复现工作的实际状态、使用的数据、已经做过的工程修改、训练/评估结果、和论文结果的差距、已排查的问题以及后续建议。

结论先行：当前结果不能认为已经复现到论文水平。当前最稳定的 simulator success 结果约为 `28.6%`，明显低于论文 Table I 中最接近任务 `Mug/Multi-MedRack` 的 TAX-DPD full result `95%`。当前代码更准确的描述是：在 RPDiff 官方可下载数据上，训练并评估了一个基于重构 `tax3d_v2.py` 的 TAX-DPD-like / adapter-based 近似实现。

## 1. 当前仓库和相关路径

主仓库：

```text
/home/yuchi/projects/TAX-DPD
```

H200 上使用的同名仓库：

```text
/home/yuchi/projects/TAX-DPD
```

RPDiff 仿真评估仓库：

```text
/home/yuchi/projects/rpdiff
```

主要 checkpoint：

```text
/home/yuchi/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-34-11/checkpoints/last.ckpt
```

主要日志：

```text
/home/yuchi/tax_dpd_eval_rigid.log
/home/yuchi/tax_dpd_rpdiff_success_100.log
/home/yuchi/tax_dpd_rpdiff_success_500.log
/home/yuchi/tax_dpd_rpdiff_diag50.log
/home/yuchi/tax_dpd_rpdiff_init_diag50.log
/home/yuchi/tax_dpd_rpdiff_identityrot30.log
/home/yuchi/tax_dpd_rpdiff_invsvd30.log
```

## 2. 使用的数据

训练和评估使用的是 RPDiff 官方数据，不是 TAX-DPD 论文作者额外发布的数据。

数据来自 RPDiff 官方仓库的 Dropbox 下载链接，链接在 RPDiff 仓库：

```text
scripts/dl_train_data.bash
```

本地/H200 数据目录：

```text
/home/yuchi/data/rpdiff/
```

主要数据包：

```text
/home/yuchi/data/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos
/home/yuchi/data/rpdiff/descriptions/objects
```

规模：

```text
mug_on_rack_multi_large_proc_gen_demos: 约 23GB，3600 个 demo 文件
descriptions/objects: 约 999MB
```

训练时使用的任务配置覆盖：

```text
dataset=rpdiff
dataset.preprocess=True
dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos
dataset.rpdiff_task_type=task_name_mug_on_rack_multi
dataset.sample_size_action=512
dataset.sample_size_anchor=1024
dataset.train_dataset_size=3200
dataset.val_dataset_size=400
dataset.test_dataset_size=400
```

注意：论文 Table I 中的 `Mug/EasyRack`、`Mug/MedRack`、`Mug/Multi-MedRack` 和我们当前用的数据最接近的是 `Mug/Multi-MedRack`。但 RPDiff 官方可下载数据包没有完全按论文表格命名的 `mug_rack_easy_single` / `mug_rack_med_single` 任务包；当前使用的是官方可下载的 `mug_on_rack_multi_large_proc_gen_demos`。

## 3. 重要前提和限制

当前复现有一个核心限制：原始 TAX-DPD 仓库没有提供官方 `src/non_rigid/models/tax3d_v2.py`。因此当前 `tax3d_v2.py` 是根据仓库中现有组件和论文描述重构出来的。

这会影响结论解释：

```text
不能说“严格复现了论文 TAX-DPD”。
只能说“训练并评估了一个重构 TAX3Dv2 的近似实现”。
```

此外，当前 simulator success eval 不是论文官方完整 pipeline。它通过 RPDiff simulator 外接 adapter 实现：

```text
TAX-DPD 点云预测
→ SVD 拟合 SE(3)
→ 将 SE(3) 送入 RPDiff PyBullet execution
→ 统计 Place Success / touching_surf
```

论文 full method 包含更完整的 global/GMM initialization 和 local refinement/inference 逻辑；当前 adapter 没有完整复现这些部分。

## 4. 训练结果

长训练已经在 H200 上完成。

最终 checkpoint：

```text
/home/yuchi/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-34-11/checkpoints/last.ckpt
```

checkpoint 对应的关键配置：

```text
model.name: tax3dv2
model.type: point
model.frame_type: fixed
model.pred_frame: noisy_goal
model.noisy_goal_scale: 1.0
model.object_scale: 3.0
model.zero_shape: true
dataset: rpdiff
dataset.rpdiff_task_name: mug_on_rack_multi_large_proc_gen_demos
dataset.rpdiff_task_type: task_name_mug_on_rack_multi
```

这里存在一个可疑点：`frame_type=fixed` 和 `pred_frame=noisy_goal` 同时出现。当前重构的 fixed-frame module 不直接使用 `batch["noisy_goal"]`，而 dataset 在 `pred_frame=noisy_goal` 时会生成该字段。这个组合可能意味着训练目标、frame 假设和推理 adapter 之间存在不一致，需要后续重点检查。

## 5. RMSE / rigid eval 结果

评估日志：

```text
/home/yuchi/tax_dpd_eval_rigid.log
```

配置特点：

```text
val/test full
num_wta_trials=10
batch_size=8
train_dataset_size=1  # 仅为了跳过耗时 train eval，所以 Train RMSE 为 nan
```

结果：

```text
Val RMSE:       0.2914450788497925
Val RMSE_WTA:   0.17192791342735292
Val T_err:      0.2533973169326782
Val T_err_WTA:  0.1250347302109003
Val R_err:      122.71263549804688 deg
Val R_err_WTA:  55.20724136352539 deg

Test RMSE:      0.2865487539768219
Test RMSE_WTA:  0.16276203230023384
Test T_err:     0.24810121893882753
Test T_err_WTA: 0.11754266023635865
Test R_err:     120.96321044921875 deg
Test R_err_WTA: 51.38385108947754 deg
```

解释：

- WTA 后 RMSE 和 translation error 有下降。
- rotation error 仍然很大，test `R_err_WTA ≈ 51.38°`。
- 这和后续 simulator success 低相符：挂杯任务对姿态很敏感，几十度的姿态误差会显著降低成功率。

## 6. RPDiff simulator success eval

### 6.1 环境和依赖处理

RPDiff 仓库路径：

```text
/home/yuchi/projects/rpdiff
```

H200 上 TAX-DPD pixi 环境安装过以下依赖：

```text
meshcat
urdfpy
healpy
rtree
mesh-to-sdf
easydict
configargparse
yacs
airobot  # git+https://github.com/Improbable-AI/airobot.git@panda-2f140 --no-deps
rpdiff editable install --no-deps
cython
colorlog
opencv-python-headless<4.9
pyassimp
rospkg
networkx==2.8.8
numpy==1.23.5
```

注意事项：

- 不能安装未 pin 的 `opencv-python-headless`，否则会把 NumPy 升到 2.x，导致 pybullet/trimesh 等依赖异常。
- 已恢复到 `numpy==1.23.5`。
- 官方 RPDiff README 里的 `knn_cuda` wheel URL 已失效；因此修改了 RPDiff `evaluate_rpdiff.py`，将依赖 `knn_cuda` 的 RPDiff policy import 变为 lazy import，并在 TAX-DPD external adapter 路径中绕过。

物体资产 symlink：

```text
/home/yuchi/projects/rpdiff/src/rpdiff/descriptions/objects
→ /home/yuchi/data/rpdiff/descriptions/objects
```

Meshcat server：

```text
PID file: /home/yuchi/meshcat_server.pid
log:      /home/yuchi/meshcat_server.log
command:  /home/yuchi/projects/TAX-DPD/.pixi/envs/default/bin/meshcat-server --zmq-url=tcp://127.0.0.1:6000
```

### 6.2 RPDiff adapter 改动

修改文件：

```text
/home/yuchi/projects/rpdiff/src/rpdiff/eval/evaluate_rpdiff.py
```

主要改动：

- lazy import RPDiff policy 相关模块，避免 `knn_cuda` 缺失导致直接失败；
- 新增 `tax_dpd_external` inference method；
- 加载 TAX-DPD checkpoint；
- 对每个 trial：
  - 获取 parent/child 点云；
  - 调用 TAX-DPD `_predict_wta`；
  - 用 SVD 将 child input points 对齐到预测 goal points；
  - 将得到的 SE(3) 送入 RPDiff execution；
  - 记录 `Place Success` 和 `touching_surf`。

新增过的 RPDiff 配置包括：

```text
mug_on_rack_multi_tax_dpd_smoke.yaml
mug_on_rack_multi_tax_dpd_100.yaml
mug_on_rack_multi_tax_dpd_500.yaml
mug_on_rack_multi_tax_dpd_diag50.yaml
mug_on_rack_multi_tax_dpd_init_diag50.yaml
mug_on_rack_multi_tax_dpd_identityrot30.yaml
mug_on_rack_multi_tax_dpd_invsvd30.yaml
```

### 6.3 simulator eval 结果

Smoke eval：

```text
5 / 5 completed
0 / 5 success
success rate = 0.0%
log: /home/yuchi/tax_dpd_rpdiff_success_smoke.log
```

100-trial：

```text
100 / 100 completed
26 / 100 success
success rate = 26.0%
log: /home/yuchi/tax_dpd_rpdiff_success_100.log
```

500-trial：

```text
500 / 500 completed
143 / 500 success
success rate = 28.6%
log: /home/yuchi/tax_dpd_rpdiff_success_500.log
```

诊断实验：

```text
diag50, random SO(3) + random parent point init:
19 / 50 = 38.0%
log: /home/yuchi/tax_dpd_rpdiff_diag50.log

init_diag50, init_orig_ori=True + init_parent_mean_pos=True:
12 / 50 = 24.0%
log: /home/yuchi/tax_dpd_rpdiff_init_diag50.log

identity_rotation30, 只用预测点云中心平移，不使用 SVD 旋转:
8 / 30 = 26.7%
log: /home/yuchi/tax_dpd_rpdiff_identityrot30.log

invsvd30, 反向 SVD sanity check:
8 / 30 = 26.7%
log: /home/yuchi/tax_dpd_rpdiff_invsvd30.log
```

这些结果说明：

- 500-trial 主结果稳定在 `26–29%` 附近。
- 改初始化没有提升。
- 不使用 SVD 旋转没有提升。
- 反向 SVD 也没有提升。
- 因此当前低成功率不是简单由随机初始化、SVD 方向写反、或者单独的旋转使用方式造成的。

## 7. 和论文结果对比

论文 Table I 中最接近当前任务的是 `Mug/Multi-MedRack`。

| 方法 / 结果 | Success Rate |
|---|---:|
| 论文 TAX-DPD full | 95% |
| 论文 TAX-DPD w/o GMM | 74% |
| 论文 RPDiff baseline | 86% |
| 当前 100-trial adapter eval | 26.0% |
| 当前 500-trial adapter eval | 28.6% |
| 当前 diagnostic ablations | 24.0%–38.0% |

差距：

```text
相对论文 TAX-DPD full:
95.0% - 28.6% = 66.4 个百分点

相对论文 TAX-DPD w/o GMM:
74.0% - 28.6% = 45.4 个百分点

相对论文 RPDiff baseline:
86.0% - 28.6% = 57.4 个百分点
```

相对比例：

```text
28.6 / 95.0 ≈ 30.1%
```

结论：

```text
当前 adapter-based 复现稳定在论文 full TAX-DPD 的约 30% 相对水平，
不能认为已复现论文结果。
```

## 8. 已遇到的问题和处理方式

### 8.1 官方 `tax3d_v2.py` 缺失

问题：

```text
原仓库没有发布官方 src/non_rigid/models/tax3d_v2.py。
```

处理：

```text
根据仓库内 DiT backbone、旧 TAX3D 模块、论文描述和训练接口重构了 tax3d_v2.py。
```

影响：

```text
这是当前最大不确定性来源。
即使代码跑通，也不能等价于论文官方实现。
```

### 8.2 RPDiff 官方数据和论文任务命名不完全一致

问题：

```text
RPDiff 官方下载包提供 mug_on_rack_multi_large_proc_gen_demos，
但论文表格中的 Mug/EasyRack、Mug/MedRack、Mug/Multi-MedRack 并不是以完全相同路径提供。
```

处理：

```text
使用官方可下载的 mug_on_rack_multi_large_proc_gen_demos，
并将其作为最接近 Mug/Multi-MedRack 的任务来比较。
```

影响：

```text
可以作为近似对比，但不能视为完全同任务、同 pipeline 对比。
```

### 8.3 RPDiff simulator 依赖问题

问题：

- `knn_cuda` 官方 wheel URL 失效；
- RPDiff 原始 import 会在不走 RPDiff policy 时也导入 knn_cuda 相关模块；
- unpinned OpenCV 会升级 NumPy 到 2.x，破坏 pybullet/trimesh 兼容性；
- old networkx 会触发 `fractions.gcd` 问题。

处理：

- RPDiff policy import 改 lazy import；
- TAX-DPD external adapter 路径绕过 RPDiff policy；
- pin `opencv-python-headless<4.9`；
- 恢复 `numpy==1.23.5`；
- 安装 `networkx==2.8.8`。

### 8.4 simulator success 判据偏弱

RPDiff eval 当前主要记录：

```text
touching_surf
```

在 mug-rack 任务上，`Place Success` 和 `touching_surf=True` 对齐。视觉抽样显示，部分成功样本只是接触/套到 peg 附近，不一定代表精细、稳定、符合论文直觉的挂杯姿态。

因此当前 simulator success 是一个有参考价值的执行指标，但不要过度解释为严格论文成功率。

### 8.5 debug 文件最初缺少中间信息

问题：

初始 RPDiff eval 的 `success_rate_relation.npz` 只保存了：

```text
parent_id
child_id
place_success
success_criteria_dict
mesh_file
args
```

没有保存预测 transform、点云中心、SVD 残差等中间信息。

处理：

后续给 adapter 增加 `tax_dpd_debug.npz`，保存：

```text
tax_action_mean
tax_anchor_mean
tax_scene_center
tax_pred_mean_centered
tax_svd_fit_rmse
tax_relative_trans_pred
relative_trans_guess
relative_trans
start_child_pose_mat
final_child_pose_mat
child_pcd_guess_mean
transformed_child_pred_mean
parent_pcd_mean
parent_pcd_min
parent_pcd_max
delta_pc_final_list
place_parent_idx_pred
place_parent_pose_mat
final_child_pos
final_child_ori_xyzw
tax_transform_variant
```

## 9. 当前 debug 观察

从 `diag50` / `init_diag50` 的 debug 数据看：

- 成功和失败样本的 `tax_svd_fit_rmse` 没有明显分界；
- 成功和失败样本的 predicted rotation angle 没有明显分界；
- final child pose 到 selected parent 的距离也没有稳定分界；
- 视觉抽样中，失败常见表现是杯子落在 rack 外侧、偏离 peg、未接触目标 rack；
- 成功样本通常只是接触到/套到 peg 附近，姿态不一定非常精细。

这支持当前判断：

```text
问题不是一个简单阈值或单一 transform 方向 bug，
更可能是 TAX3Dv2 重构实现、frame convention、训练目标和 simulator adapter 之间存在系统性不一致。
```

## 10. 当前最可能的原因

按优先级排序：

1. `tax3d_v2.py` 重构实现和论文官方实现不一致。
2. `frame_type=fixed`、`pred_frame=noisy_goal`、`zero_shape=true` 的组合在当前重构实现中可能没有正确表达论文方法。
3. 当前 adapter 将点云预测直接 SVD 成 SE(3)，没有论文 full pipeline 的 GMM/global initialization 和 local refinement。
4. 当前 RPDiff simulator task/setup 与论文 Table I 的任务并非完全同名同配置。
5. 模型虽然在 RMSE_WTA 上有一定拟合能力，但 WTA rotation error 仍然很大，导致挂杯执行成功率低。

## 11. 不建议继续做的事

不建议马上继续：

```text
继续增加 trial 数到 1000+
继续盲目加训练 epoch
只调 RPDiff eval 参数
```

原因：

- 500-trial 已经足够说明当前 adapter 结果稳定在 `26–29%`；
- ablation 显示初始化、SVD 方向、是否使用旋转都不是简单主因；
- 当前和论文差距是几十个百分点，优先应修模型/接口一致性，而不是继续堆算力。

## 12. 推荐下一步

### 12.1 做 TAX-DPD dataloader 内部可视化

目标：确认 checkpoint 在 TAX-DPD 自己的训练/测试数据坐标系里是否预测正确。

建议新增脚本：

```text
scripts/debug_tax3dv2_predictions.py
```

功能：

- 加载 checkpoint；
- 加载 RPDiff test dataset；
- 跑 32 或 64 个样本；
- 保存：
  - `pc_action`
  - `pc_anchor`
  - GT `pc`
  - `pred_actions_wta`
  - per-sample RMSE / T_err / R_err
  - object id / parent id，如果 dataset 能提供；
  - `.npz` 和可视化 `.ply` / `.html`。

判断：

```text
如果 pred 在 TAX-DPD dataloader 内部已经明显偏/旋转错：
    问题在 tax3d_v2.py 重构实现、训练目标或模型配置。

如果 pred 在 dataloader 内部基本对：
    问题在 RPDiff simulator adapter / frame conversion。
```

### 12.2 做 oracle simulator sanity check

目标：确认 RPDiff simulator setup 本身是否可靠。

做法：

```text
用 RPDiff demo 里的 GT transform 或 dataset 里的 GT pose 直接送进 RPDiff simulator。
```

判断：

```text
如果 oracle transform 也低：
    当前 RPDiff eval/task setup 有问题。

如果 oracle transform 很高：
    simulator 没问题，问题在 TAX-DPD 模型预测或 adapter。
```

### 12.3 再考虑 GMM/global initialization

论文 full TAX-DPD `95%`，w/o GMM `74%`。当前 `28.6%` 连 w/o GMM 都远低于，因此现在不能只把问题归因于缺 GMM。

更合理的顺序是：

```text
先修基础 TAX3Dv2 prediction / frame convention，
让无 GMM 版本至少接近论文 w/o GMM 水平，
再补 GMM/global initialization 和 local refinement。
```

## 13. 给后续接手人的简短结论

当前已经完成：

- RPDiff 官方 mug_on_rack_multi 数据准备；
- TAX3Dv2 重构模型训练；
- RMSE/rigid eval；
- RPDiff PyBullet simulator adapter eval；
- 100/500 trial simulator success；
- 多个诊断 ablation；
- 初步 debug 中间量保存。

当前最可信数字：

```text
Test RMSE_WTA: 0.1628
Test T_err_WTA: 0.1175
Test R_err_WTA: 51.38°
RPDiff simulator 500-trial success: 143 / 500 = 28.6%
```

当前和论文的关系：

```text
论文 Mug/Multi-MedRack TAX-DPD full: 95%
当前 adapter-based result: 28.6%
差距: 66.4 个百分点
```

当前判断：

```text
没有复现到论文水平。
主要风险在官方 tax3d_v2.py 缺失、重构实现不确定、frame/prediction target 和 simulator adapter 不一致。
```

最推荐下一步：

```text
先写 TAX-DPD dataloader 内部预测可视化/逐样本指标脚本，
确认模型本身在自己的数据坐标系里是否预测正确。
```
