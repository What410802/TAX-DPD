# 复现

> 根据/home/yuchi/projects/TAT-DPD/HANDOFF.md,重新在ssh h200-1复现本仓库.h200-1的使用方式可以看/home/yuchi/Downloads/金山 H200 集群使用指南(H200-1_2).md

<details><summary>2 previous messages</summary>

> 我会先读取交接文档、集群使用指南以及本仓库当前状态，确认“复现”所需的代码版本、环境、数据和预期结果，再按文档在 `h200-1` 上执行并持续核验。远端操作会严格限定在交接文档指定的工作目录内。
>
> 本地命令启动时被环境的 `bwrap` UID 映射拦截了，这不是仓库或 SSH 的报错。我先改用非登录 shell 做只读检查；若仍受限，我会明确给出受阻点和可直接执行的命令。
</details>

当前执行环境无法启动任何本地或 SSH 命令，均在命令运行前失败：

```text
bwrap: setting up uid map: Permission denied
```

我已分别尝试登录 shell、非登录 shell及 `/tmp` 工作目录，结果一致。因此目前无法读取 `HANDOFF.md` 和集群指南，也无法连接 `h200-1` 完成复现。这是 Codex 沙箱的 UID 映射故障，需要修复或重启当前执行环境后继续。

> 继续尝试

<details><summary>2 previous messages</summary>

> 我继续从最小命令开始验证执行通道；一旦 shell 恢复，会立即读取两份文档并接入 `h200-1`。
>
> 最小的 `/bin/true` 仍在进程创建前被同一 UID 映射错误拦截，说明与具体命令、路径或 SSH 配置无关。我再测试一次默认工作目录入口，以排除显式 `workdir` 触发的问题。
</details>

再次尝试了最小命令和默认工作目录，仍在进程启动前失败：

```text
bwrap: setting up uid map: Permission denied
```

当前工具无法执行任何 shell 命令，因此也无法读取文档或运行 `ssh h200-1`。需要重启/重新创建 Codex 执行会话或修复沙箱的 user namespace 配置后才能继续；仓库尚未产生任何修改。

> 继续尝试

<details><summary>33 previous messages</summary>

> 执行权限已经切换为不受限模式。我现在重新读取交接文档、H200 使用指南和仓库状态，然后按其中约定连接 `h200-1` 开始复现。
>
> 沙箱问题已解除。新的实际问题是用户给出的 `/home/yuchi/projects/TAT-DPD/HANDOFF.md` 不存在；当前仓库名是 `TAX-DPD`，我正在定位交接文档，同时单独读取集群指南，判断这里是否只是路径笔误。
>
> 已确认是路径笔误，交接文件实际位于当前仓库 `/home/yuchi/projects/TAX-DPD/HANDOFF.md`。仓库含一批未提交的复现相关改动，我会保留它们原样；下一步依据交接文件确认远端目录和剩余任务，并先做无交互 SSH 连通性检查。
>
> 交接要求的主任务是：把当前含未提交补丁的工作树、RPDiff 数据和 GMM `epoch_200.pt` 放到 H200，然后重建 pixi 环境、做冒烟验证并启动 fixed-frame 扩散长跑。当前 GPU 5 完全空闲；我先检查远端是否已有同名目录/数据/进程和剩余空间，避免覆盖既有成果，再决定增量同步范围。
>
> 远端尚无项目、数据或 GMM 目录，统一存储池余量约 2.7 TB，足够复现。项目本地 8.9 GB 很可能主要是可重建的 `.pixi` 环境，我先拆分容量并只同步源码与必要产物，随后以可续传的 `rsync` 传输约 100 GB 数据；GPU 5 将作为训练卡，当前没有计算进程。
>
> 容量核对后，当前主实验只需 `mug_on_rack` 的 23 GB 数据和约 1 GB 物体描述，无需先搬另外两个未预处理任务。远端非登录 shell 没在 `PATH` 中暴露 pixi，我会在同步期间定位其安装位置；现在开始增量同步源码（排除 `.pixi`）以及 mug 数据和 GMM 产物。
>
> 源码同步正在进行，`rsync` 已保持为可续传模式；由于仓库内还包含约 1 GB 的日志和第三方代码，跳板链路下会需要一些时间。同步结束后我会立即校验关键文件哈希，不会直接假设传输完整。
>
> 源码已完整同步，实际传输约 1.1 GB，平均约 12.7 MB/s。GMM 产物正在传输；按当前链路速率，随后 24 GB 的 mug 数据预计约半小时，我会保持任务并在数据到齐后继续环境安装与启动。
>
> GMM 的全部 checkpoint 与日志已同步完成。物体描述资产正在传输，之后是 23 GB 的 mug demo；这两部分目录结构会保持为配置文件期望的 `/home/yuchi/data/rpdiff/...`。
>
> mug 数据首次同步在远端缺少父目录时立即退出，没有传输或覆盖任何文件。我已定位为纯目录初始化问题，先创建 `data/task_demos` 后继续同一条可续传同步。
>
> mug demo 的 23 GB 主传输已开始。当前链路吞吐约 6–13 MB/s，预计还需 30–60 分钟；传输采用 `--partial`，即使跳板连接波动也能从已有文件继续。
>
> 传输仍稳定运行，没有报错或重连。完成后我会用第二次 `rsync --checksum --dry-run` 校验差异，并对 `epoch_200.pt` 单独比对 SHA-256；这是后续训练可复现性的关键输入。
>
> 主数据同步仍在进行，远端没有出现空间不足或权限错误。由于数据由大量 `.npz` 和预处理文件组成，实际耗时会受文件数影响，不只是总字节数；我会保持传输会话直至明确成功退出。
>
> 传输已持续约十多分钟，仍正常进行。接下来的环境安装会使用远端共享缓存，预计明显快于数据搬运；安装完成后会先跑 import、CUDA、数据增强和一个短训练批次验证，再启动长跑。
>
> 数据同步会话仍存活且无错误输出，说明不是挂起在认证环节，而是在持续搬运。为了避免中断这个长连接，我保持当前会话，完成信号到达后再进行远端校验和安装。
>
> 传输时间已进入预估区间的后半段，仍未出现失败。完成后我会优先检查远端预处理缓存确实有 3600 个文件；若缓存缺失或数量不符，不会贸然启动会导致 CPU 数据瓶颈的长训练。
>
> 主数据仍在传输，状态稳定。由于输出只在进程完成时汇总，我目前不虚报百分比；`rsync` 最终退出码为 0 后才视为传输成功。
>
> 已超过半小时但仍处于合理范围，尤其当前链路经两级跳板且数据含大量小文件。会话没有超时（已配置两小时上限），我继续等到完整退出，不中途切断造成额外目录扫描。
>
> 长传输仍未退出，未观察到错误。当前阶段没有可安全并行的本地操作（同一跳板链路会争抢带宽），因此继续保持单一 rsync 会话，避免降低吞吐或破坏续传状态。
>
> 当前已接近一小时，仍是活动中的 rsync 进程而非超时；数据集在本地约 23 GB，但大量文件导致每个文件都要经过 SSH 往返，吞吐明显低于大文件同步。继续等待至退出码明确返回。
>
> 仍在持续，暂无异常。等传输结束后我会优先执行远端环境探测（`pixi`/`uv`/模块缓存位置），若 pixi 不在 PATH，会按交接文档用官方安装方式建立环境，不会复用不兼容的旧 CUDA 环境。
>
> 主数据已完整同步成功（24,527,869,089 字节，7314 个文件，退出码 0）。现在进入远端复现阶段：校验目录与 checkpoint、定位 pixi，然后执行环境安装和补丁。
>
> 远端数据文件数和 GMM checkpoint SHA-256 与本地完全一致，源码改动也完整保留。远端没有预装 `pixi`，只有 `uv`；我按交接文档的 pixi 方案安装用户级 pixi，随后运行 `pixi install` 和 `post-install`。
>
> 远端正在安装 pixi 并解析锁定环境；该步骤会访问已配置的代理/缓存，预计比数据同步短，但首次 conda 解包仍可能需要数分钟。当前安装进程未报错。
>
> pixi 0.75.0 已安装，默认环境成功创建。现在执行交接文档要求的 `post-install`（PyG 扩展、第三方和 editable 安装），然后重打 `rpad` 的 `PointConv` 兼容补丁并验证模型导入链。
>
> 环境安装完成，但交接文档中的 `rpad` 包在该锁定环境里没有安装（补丁脚本因此无法定位模块），这说明 `post-install` 的依赖集合与原环境有差异。我先确认 `rpad` 是否以源码目录存在、以及当前 `script_utils` 的首个缺失导入，再针对实际错误补齐，不会盲目改环境。
>
> 实际错误已明确：`post-install` 使用 `--no-deps` 安装本项目，导致 `pyproject.toml` 中的 `rpad-pyg` 与 `rpad-visualize-3d` 被跳过；这不是代码问题。我现在按项目声明的 Git 依赖显式安装这两个包，再重新打 `PointConv` 兼容补丁。CUDA 探测在未指定设备时返回 false，我会在训练命令中显式 `CUDA_VISIBLE_DEVICES=5` 验证。
>
> `rpad` 已补丁成功；模型导入链现在推进到下一个交接文档已预见的缺项：`pytorch3d` 未安装。按文档从 GitHub stable 安装，完成后再验证 import。CUDA 仍显示 unknown error，即使 `CUDA_VISIBLE_DEVICES=5`，后续会单独诊断驱动/容器可见性。
>
> PyTorch3D 正在从 stable 分支构建安装；该包包含 C++/CUDA 扩展，编译时间会明显长于普通 Python 包。安装命令仍在运行且未报错。
>
> PyTorch3D 0.7.8 已成功编译安装。现在并行验证完整 import 链与 CUDA 设备访问；若 import 已通而 CUDA 仍失败，我会根据 `/dev/nvidia*` 权限、环境变量和 PyTorch 驱动报错定位，训练不会在 CPU 上误启动。
>
> 完整模型 import 链已经通过（上一条仅在最终测试输出的引号上触发了 `NameError`，所有导入均已完成）。剩余阻塞是 CUDA `cuInit` 返回 unknown error；设备节点权限正常，且 H200 驱动可见。我现在检查这是整机新进程的驱动状态问题还是 pixi 环境库冲突。
>
> 已确认不是 pixi 冲突：系统 `/usr/lib/x86_64-linux-gnu/libcuda.so.1` 直接调用 `cuInit(0)` 也返回 CUDA 999，且 `nvidia-smi` 报告 GPU Recovery Action 为 `Reboot`。这是 `h200-1` 主机驱动状态故障，新 CUDA 进程无法创建；我不会启动一个必然失败的长跑。等待管理员处理前，我继续完成 CPU 侧的数据集与增强验证，把除主机重启外的风险清空。
>
> CPU 侧验证全部通过：3200 个样本可加载，实际训练划分 2880，action/anchor 遮挡签名均为 30/30，预处理增强有效。最后再复查一次 CUDA 999 与 GPU 5 状态；若仍要求 Reboot，本轮唯一剩余动作只能由集群管理员完成。
</details>

已在 `h200-1` 完成复现准备：

- 同步 TAX-DPD 源码及未提交补丁，排除本地 `.pixi`
- 同步 mug 数据约 24 GB，共 7314 个文件
- 同步 GMM 全部产物；`epoch_200.pt` SHA-256 与本地一致
- 安装 pixi 0.75.0、锁定环境、PyG 扩展、`rpad` 依赖和 PyTorch3D 0.7.8
- 应用 `PointConv` 兼容补丁
- 模型完整 import 链通过
- RPDiff 数据加载通过：3200 个样本，训练划分 2880
- 遮挡增强验证通过：action/anchor 均为 `30/30`

目前唯一阻塞是 `h200-1` 驱动故障：

```text
libcuda.so.1 cuInit(0) = 999
GPU Recovery Action: Reboot
```

这发生在系统 CUDA 驱动层，与 pixi 环境无关。GPU 训练尚未启动，避免误跑到 CPU。需要管理员重启或修复 `h200-1` 的 NVIDIA 驱动；恢复后即可在 GPU 5 上进行短跑验证并用 `setsid` 启动扩散长跑。

> 继续尝试

<details><summary>2 previous messages</summary>

> 我继续检查 `h200-1` 的驱动恢复情况；只有 `cuInit` 成功后才会启动训练，避免长跑再次因宿主机状态直接失败。
>
> `h200-1` 本身未重启（已运行近 5 天），GPU 5 仍空闲；CUDA 探测命令依然在 30 秒内无法返回，表明驱动层尚未恢复。我会再用远端进程级超时获取明确状态，并核对是否只影响本用户或整机 CUDA 初始化。
</details>

继续检查后，`h200-1` 仍未恢复：

```text
cuInit = 999
torch.cuda.is_available() = False
GPU 5 Recovery Action = Reboot
```

主机自上次检查以来没有重启，GPU 5 虽空闲但不能被任何新 CUDA 进程使用。因此长跑仍无法启动；远端环境、数据、GMM 产物和 CPU 数据验证均已就绪。需要管理员重启 `h200-1` 后再继续。

> 给我详细步骤我来手动执行

目前环境、代码和数据均已放在 `h200-1`。你需要先让管理员恢复驱动，然后按下面步骤操作。

**1. 联系管理员重启 H200-1**

当前明确状态：

```text
cuInit(0) = 999
GPU Recovery Action = Reboot
```

这是共享服务器，其他 GPU 上有用户任务，不要自行执行 `sudo reboot` 或重启 NVIDIA 驱动。请管理员协调后重启 `h200-1`。

**2. 重启后登录并验证主机**

```bash
ssh h200-1
uptime
nvidia-smi
```

`uptime` 应显示主机刚重启。确认 GPU 5 没有其他任务：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader
```

GPU 5 应接近：

```text
5, 1 MiB, 0 %
```

检查驱动是否仍要求重启：

```bash
nvidia-smi -i 5 -q | grep -A1 "Recovery Action"
```

预期不再显示 `Reboot`。

**3. 验证系统 CUDA**

```bash
python3 - <<'PY'
import ctypes
result = ctypes.CDLL("libcuda.so.1").cuInit(0)
print("cuInit:", result)
assert result == 0
PY
```

必须输出：

```text
cuInit: 0
```

若仍是 `999`，不要启动训练，继续让管理员处理驱动。

**4. 验证 TAX-DPD 环境**

```bash
export PATH="$HOME/.pixi/bin:$PATH"
cd ~/projects/TAX-DPD

CUDA_VISIBLE_DEVICES=5 pixi run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
print("device:", torch.cuda.get_device_name(0))

from non_rigid.utils.script_utils import create_model
print("model import: OK")
PY
```

预期：

```text
torch: 2.7.0+cu128
CUDA available: True
visible GPUs: 1
device: NVIDIA H200
model import: OK
```

这里的设备 `0` 是被 `CUDA_VISIBLE_DEVICES=5` 映射后的物理 GPU 5。

**5. 再确认数据和 GMM checkpoint**

```bash
du -sh ~/data/rpdiff
find \
  ~/data/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos \
  -type f | wc -l

sha256sum \
  ~/data/gmm_outputs/kkqzj33eok/checkpoints/epoch_200.pt
```

当前预期结果：

```text
约 24G
7314
d1316edaef38d0490338f08967d6529f7f2e21f8a6c7b4119031a774a87e64bd
```

**6. 启动较忠实的基准长跑**

这个版本保留作者配置的 `batch_size=32`、验证频率和 10 次 WTA，速度较慢，但比 batch 128 的加速配置更适合作为复现基准。

```bash
cd ~/projects/TAX-DPD/scripts
export PATH="$HOME/.pixi/bin:$PATH"

setsid nohup env \
  CUDA_VISIBLE_DEVICES=5 \
  WANDB_MODE=offline \
  "$HOME/.pixi/bin/pixi" run python train.py \
  model=tax3dv2 \
  model.type=point \
  model.frame_type=fixed \
  model.pred_frame=noisy_goal \
  model.noisy_goal_scale=1.0 \
  model.object_scale=3.0 \
  dataset=rpdiff \
  dataset.preprocess=True \
  dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos \
  dataset.rpdiff_task_type=task_name_mug_on_rack_multi \
  dataset.sample_size_anchor=1024 \
  dataset.train_dataset_size=3200 \
  dataset.val_dataset_size=400 \
  dataset.test_dataset_size=400 \
  training.batch_size=32 \
  training.val_batch_size=32 \
  wandb.group=rigid \
  wandb.project=tax_dpd_h200 \
  resources.num_workers=16 \
  'resources.gpus=[0]' \
  > "$HOME/diffusion_train.log" 2>&1 < /dev/null &

echo $! | tee "$HOME/tax_dpd_train.pid"
```

注意保存最后输出的 PID。

**7. 确认进程真正脱离 SSH**

```bash
PID=$(cat ~/tax_dpd_train.pid)
ps -o pid,ppid,sid,etimes,cmd -p "$PID"
```

预期：

- 进程存在
- `PPID` 为 `1`
- 命令中有 `train.py`

检查 GPU 5：

```bash
nvidia-smi -i 5
```

显存应明显增加，GPU utilization 应在训练批次期间上升。

**8. 检查启动日志**

```bash
tail -c 5000 ~/diffusion_train.log \
  | tr '\r' '\n' \
  | grep -v '^$' \
  | tail -30
```

查看 loss：

```bash
tr '\r' '\n' < ~/diffusion_train.log \
  | grep -o 'train/loss=[0-9.]*' \
  | tail -20
```

查看验证指标：

```bash
tr '\r' '\n' < ~/diffusion_train.log \
  | grep -o 'val_rmse_wta_0=[0-9.]*' \
  | uniq \
  | tail
```

查看 checkpoint：

```bash
find ~/projects/TAX-DPD/scripts/logs \
  -path '*/checkpoints/*' \
  -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' \
  | sort \
  | tail
```

**9. 安全停止训练**

```bash
PID=$(cat ~/tax_dpd_train.pid)
kill -TERM "$PID"
```

等待约 20 秒再确认：

```bash
ps -p "$PID"
```

不要使用宽泛的 `pkill -f train.py`，可能误杀其他训练或检查命令。

**10. 从 checkpoint 续训**

先定位最新 `last.ckpt`：

```bash
find ~/projects/TAX-DPD/scripts/logs \
  -name last.ckpt \
  -printf '%T@ %p\n' \
  | sort -n \
  | tail -1
```

然后在原启动命令末尾增加：

```bash
ckpt_path=/完整路径/last.ckpt
```

仍然使用 `setsid nohup`、同一物理 GPU 和相同训练配置。

最后需要保留科研表述上的限制：当前 `tax3d_v2.py` 是依据仓库组件重建的近似实现，并非作者发布的官方模型代码，因此结果只能称为“在 RPDiff 上训练 TAX-DPD 架构的近似实现”。

> 我重启了h200-1，你继续尝试

<details><summary>5 previous messages</summary>

> 我先确认重启后的 `cuInit`、PyTorch CUDA 和 GPU 5 空闲状态；全部通过后会做一个短启动验证，再用 `setsid` 正式启动 fixed-frame 长跑并记录 PID。
>
> 重启已生效：`cuInit=0`，PyTorch 识别 H200，模型 import 链通过，GPU 5 空闲，也没有残留训练进程。接下来我先读取训练配置的准确字段并做单 epoch/少量样本的 GPU 短跑，确认前向、反向和 checkpoint 路径无误后再启动 20,000 epoch 长跑。
>
> 配置字段已核对。为避免测试路径里的 WandB 收尾缺陷，我会用真实 Trainer 跑 1 epoch、64/16/16 个样本、WTA=1 的独立 smoke run；它会实际执行 H200 前向、反向、验证和 checkpoint 保存，但不会污染正式项目的配置。
>
> GPU smoke run 已完整通过：H200 前向/反向/验证正常，6.5M 参数模型完成 1 epoch，`train/loss=1.36474`，checkpoint 与离线 WandB 收尾均成功。现在按较忠实基准配置启动正式 20,000 epoch 长跑（batch 32、WTA 10、每 100 epoch 验证），物理 GPU 5，并用 `setsid` 脱离 SSH 会话。
>
> 正式长跑已稳定启动并成功脱离会话：PID `48046`、PPID `1`、SID `48046`；训练子进程 PID `48253`。GPU 5 使用约 6.9 GB、利用率约 77%，每 epoch 当前约 8.8 秒，loss 已从初始水平下降到约 0.3–0.5。再核对正式 run 目录和 `last.ckpt` 是否持续更新后交付监控信息。
</details>

正式训练已在 `h200-1` 的物理 GPU 5 上启动并稳定运行。

关键信息：

```text
启动器 PID: 48046
训练进程 PID: 48253
PPID: 1
SID: 48046
日志: /home/yuchi/diffusion_train.log
Run 目录:
~/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-21-33
```

当前状态：

- CUDA 与 H200 验证通过
- 1 epoch GPU smoke test 完整通过
- 正式配置为 20,000 epoch、batch 32、WTA 10
- GPU 显存约 7.5 GB，训练功耗约 264 W
- 当前约 10 秒/epoch
- epoch 4 时 loss 已降到约 `0.05–0.16`
- 进程已通过 `setsid` 脱离 SSH，不会因断开终端停止
- 首个正式 checkpoint 预计在第 100 epoch 验证后出现

监控：

```bash
ssh h200-1

PID=$(cat ~/tax_dpd_train.pid)
ps -o pid,ppid,sid,etimes,stat,cmd -p "$PID"

tail -c 3000 ~/diffusion_train.log \
  | tr '\r' '\n' \
  | grep -v '^$' \
  | tail -20

nvidia-smi -i 5
```

查看 checkpoint：

```bash
find \
  ~/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-21-33/checkpoints \
  -type f -ls
```

训练目前无需手动干预。

> 大概需要跑多久？

<details><summary>6 previous messages</summary>

> 我用当前日志中的实际 epoch 速度和首轮验证耗时来估算，比仅按 H200 理论算力更可靠。
>
> 刚检查发现训练并未继续到预期的首轮验证，而是在 epoch 50 开始时退出了；因此当前不能给出有效完成时间。我先提取完整 traceback 确认是数据样本、worker 还是 CUDA 错误，并恢复训练后再给 ETA。
>
> 崩溃原因已定位：每 5000 global steps 触发一次额外训练指标，仍使用 `distractor_min`，但 RPDiff 没有 distractor 位姿。交接文档声称已用 try/except 修复，但当前实现只覆盖了另一条路径或覆盖范围不完整。我会补齐这处保护并添加针对第 5000 step 分支的测试，然后重新启动；由于第 100 epoch 前没有 checkpoint，这次需要从头跑，损失约 8 分钟。
>
> 我会把训练额外指标与验证指标采用相同的可选错误处理：缺少 distractor 元数据时只跳过刚体误差日志，不影响 diffusion loss、WTA 指标或优化步骤。修改仅限 `tax3d_v2.py` 这一处，并同步到远端。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./src/non_rigid/models/tax3d_v2.py` (+17 -13)
>
> </details>
>
> 针对该分支的回归 smoke test 已通过：把额外日志周期设为每一步，连续触发缺失 distractor 的路径，1 epoch 正常完成。现在保留失败日志并从头重启正式长跑；之后 ETA 以实测约 8.8 秒/epoch 加首轮验证耗时估算。
</details>

按实测速度，预计约 **2.5–3 天**。

估算依据：

- 稳定训练约 `8.8–10 秒/epoch`
- 20,000 epoch 纯训练约 `49–56 小时`
- 每 100 epoch 验证一次，共约 200 次
- 验证包含 10 WTA × 100 步去噪，预计额外约 `8–15 小时`

刚才训练在 epoch 50 暴露了额外指标分支的 bug。我已修复并通过强制触发该分支的回归测试，正式训练已从头重启：

```text
PID: 74600
GPU 5: 约 6.9 GB，利用率约 70%
状态: 正常运行
```

首轮验证在 epoch 100 完成后，就能根据 H200 的真实验证耗时给出更准确的 ETA。目前保守预计在 **2026-08-06 左右**完成。

> 现在是什么状态？大概需要跑多久？

<details><summary>2 previous messages</summary>

> 我现在读取训练 PID、最新 epoch、首轮验证耗时、checkpoint 时间戳和 GPU 状态，再根据已经发生的实际进度重新计算剩余时间。
>
> 当前进程正常，刚完成 epoch 100 前的训练并正在做第一次完整验证：GPU 5 利用率 90%、显存约 84 GB、功耗约 497 W。第一次验证的真实耗时决定总 ETA，我再等它完成并读取首个 checkpoint，预计几分钟内能给出更准确数字。
</details>

当前运行正常，已完成首轮完整验证并生成可续训 checkpoint。

状态：

- 当前约 epoch 100 / 20,000，完成约 0.5%
- PID：`74600`，已脱离 SSH 会话
- GPU 5：利用率约 90%，显存约 84 GB
- 首轮验证：
  - `val_rmse_wta_0 = 0.142`
  - `val_rmse_0 = 0.304`
- `last.ckpt` 已保存，异常中断后可以续训

基于首个 100-epoch 周期实测：

- 100 epoch 训练约 15 分钟
- 一次完整验证约 7–8 分钟
- 每 100 epoch 总计约 22–23 分钟
- 剩余时间约 **74–76 小时，即约 3.1 天**

若速度保持稳定，预计在 **8 月 6 日傍晚至夜间**完成。

> 告诉我自己检查进度的指令

登录服务器：

```bash
ssh h200-1
```

检查训练进程是否存活：

```bash
PID=$(cat ~/tax_dpd_train.pid)
ps -o pid,ppid,sid,etimes,stat,cmd -p "$PID"
```

有进程信息表示仍在运行；无输出表示已退出。

查看最新训练进度：

```bash
tail -c 5000 ~/diffusion_train.log \
  | tr '\r' '\n' \
  | grep -E 'Epoch [0-9]+:' \
  | tail -1
```

查看最近的 loss：

```bash
tail -c 20000 ~/diffusion_train.log \
  | tr '\r' '\n' \
  | grep -o 'train/loss=[0-9.]*' \
  | tail -20
```

查看最新验证指标：

```bash
tr '\r' '\n' < ~/diffusion_train.log \
  | grep -oE 'val_rmse_wta_0=[0-9.]+|val_rmse_0=[0-9.]+' \
  | tail -10
```

检查 GPU 5：

```bash
nvidia-smi -i 5
```

简洁输出：

```bash
nvidia-smi -i 5 \
  --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader
```

查看最新 checkpoint：

```bash
find \
  ~/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-34-11/checkpoints \
  -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %f\n' \
  | sort \
  | tail
```

检查是否报错：

```bash
tr '\r' '\n' < ~/diffusion_train.log \
  | grep -E 'Error executing|Traceback|CUDA out of memory|Killed' \
  | tail -20
```

持续刷新综合状态，每 10 秒一次：

```bash
watch -n 10 '
PID=$(cat ~/tax_dpd_train.pid)
ps -o pid,etimes,stat -p "$PID"
nvidia-smi -i 5 \
  --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader
tail -c 5000 ~/diffusion_train.log \
  | tr "\r" "\n" \
  | grep -E "Epoch [0-9]+:" \
  | tail -1
'
```

按 `Ctrl+C` 退出监控，不会停止训练。

> 现在是什么状态？大概需要跑多久？

<details><summary>5 previous messages</summary>

> Context compaction
>
> Source: automatic
> Status: completed
>
> 我会直接连 `h200-1` 查当前训练进程、日志最新 epoch、错误和 GPU 状态，然后按实测速度更新剩余时间。
>
> 上一次输出被进度条日志刷屏截断了，但关键信息已经看到：训练进程还活着，`PID=74600`，已脱离 SSH 作为 `PPID=1` 运行。现在我再用更窄的命令只取摘要，避免被 tqdm 日志淹没。
>
> 现在训练正在第 500 多个 epoch，checkpoint 已经到 `epoch=499-step=50000`。GPU 5 上有训练进程占用约 84 GB 显存；利用率瞬时只有 36% 是因为刚好可能在验证/数据阶段，进程本身正常。
>
> 刚才 Perl 命令写法会把数组引用打印出来，我修正一下只取最后一个匹配行。
</details>

当前状态：训练正常运行中。

- 训练进程：`PID=74600`，`PPID=1`，说明已经脱离 SSH 会话，不会因为你断开连接而停。
- 当前进度：最新日志到 `Epoch 561`，上一完整 epoch 是 `Epoch 560`。
- 最近 checkpoint：已保存到 `epoch=499-step=50000.ckpt`，同时有 `last.ckpt`。
- 验证指标目前是：
  - `val_rmse_wta_0=0.124`
  - `val_rmse_0=0.295`
- GPU：物理 GPU 5 上进程还在，占用约 `84 GB / 144 GB` 显存。
- 目前没看到新的崩溃错误。

按现在实测速度估算，完整 `20000 epoch` 总耗时大约 `75 小时` 左右。现在已经跑了约 `2.1 小时`、完成约 `2.8%`，剩余大约 `73 小时`，也就是约 `3 天多一点`。

预计完成时间大概在 `2026-08-06 18:00-20:00 CST` 附近。这个估算会受验证阶段、GPU 负载、IO 波动影响，误差按几个小时看比较合理。

> 如果我用多张卡可以让训练快点完成吗

<details><summary>3 previous messages</summary>

> 我先看一下这个仓库的训练配置和 Lightning 启动方式，确认它是不是天然支持多卡，以及会不会因为 batch size/验证逻辑导致加速不明显。
>
> 训练脚本确实交给 Lightning 的 `devices=cfg.resources.gpus`，所以命令层面可以指定多张卡；但我还要看模型里是否有按固定 batch size 生成张量的写法，这会影响 DDP 下每卡 batch 的正确性。
>
> 代码里 `training_step` 用的是实际 batch 的 `x_start.shape[0]`，不是硬编码 `self.batch_size`，这点对多卡是好的。主要风险不在能不能启动，而在“多卡后等效 batch size 变了，结果不再严格等价”。
</details>

可以用多张卡加速，但不是“直接无风险地快 N 倍”。

这个仓库的训练脚本用 Lightning：

```python
Trainer(accelerator="gpu", devices=cfg.resources.gpus)
```

所以命令层面支持多卡，例如在 H200 上如果想用物理 GPU 5,6,7,8，可以用：

```bash
CUDA_VISIBLE_DEVICES=5,6,7,8 pixi run python train.py ... resources.gpus=[0,1,2,3]
```

但当前这次已经在单卡跑了，不能在不中断的情况下动态加 GPU。要多卡，只能停掉当前进程，然后从 `last.ckpt` 续训。

关键取舍：

1. 如果保持 `training.batch_size=32`，用 4 张卡时 Lightning DDP 通常是每卡 batch 32，等效全局 batch 变成 128。这样会明显更快，可能接近 2.5-3.5 倍，但训练动力学变了，不是严格同一个实验。

2. 如果想保持全局 batch 仍是 32，4 张卡要设 `training.batch_size=8`。这样结果更接近原设置，但加速会小很多，甚至可能因为通信开销和小 batch 低效率，不划算。

3. 当前最耗时的部分不只是训练，还有每 100 epoch 一次的验证。验证阶段也可能受益于多卡，但 WTA 验证比较重，实际加速要看 Lightning 分布式验证和日志聚合表现，不能保证线性。

我的建议：如果目标是尽快得到一个可用复现实验结果，可以切到 4 卡续训，保持 `batch_size=32`，接受“全局 batch 改变”的差异，并在记录里写清楚。如果目标是尽量保持当前实验可比性，就让这次单卡继续跑完，预计还剩约 3 天。

如果要切 4 卡，操作大致是：

```bash
ssh h200-1
kill $(cat ~/tax_dpd_train.pid)
```

然后用 `last.ckpt` 续训，命令里需要加类似：

```bash
checkpoint.local_ckpt=/home/yuchi/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-34-11/checkpoints/last.ckpt
resources.gpus=[0,1,2,3]
```

并通过：

```bash
CUDA_VISIBLE_DEVICES=5,6,7,8
```

把逻辑卡 `0,1,2,3` 映射到物理卡 `5,6,7,8`。这一步我建议先做一个 5-10 分钟多卡 smoke resume，确认 DDP、checkpoint 和验证都能过，再放长跑。

> 看一下/home/yuchi/Desktop/discoverse_robotics/papers/TAT-DPD.pdf是用的什么实验设置

<details><summary>3 previous messages</summary>

> 我会先从 PDF 抽取文本，重点查实验设置、训练细节、数据集、网络/扩散配置和评估指标；如果文本抽取不完整，再看相关页截图。
>
> 文本抽取可用。论文主体和 appendix 都有实验设置，主体给任务/指标，附录给超参和点数；我继续把这些段落按页附近抽出来，避免漏掉真实世界设置。
>
> 论文没有把 RPDiff 每个任务的训练样本数放在主体表格里，但明确给了任务集合、100 trial 成功率评估、FPS 下采样点数和统一训练超参。我再查一下 “demo/data” 是否有隐藏的样本规模说明。
</details>

我看了 `/home/yuchi/Desktop/discoverse_robotics/papers/TAT-DPD.pdf`。这篇 PDF 标题是 **“Disentangled Point Diffusion for Precise Object Placement”**，方法名在文中是 **TAX-DPD**，不是 TAT-DPD。

论文里的主要实验设置如下。

**仿真实验 RPDiff**

使用完整 RPDiff placement task suite，PyBullet 环境：

- `Mug/EasyRack`：一个 rack，一个 peg，挂 mug
- `Mug/MedRack`：一个 rack，两个 pegs，挂 mug
- `Mug/Multi-MedRack`：多个 racks，每个两个 pegs，挂 mug
- `Book/Shelf`：把书插入部分填充的书架
- `Can/Cabinet`：把罐子放到一摞罐子上或开放 shelf 上

评估方式：

- 每个 placement task 做 `100 trials`
- 每个 trial 使用 held-out test suite 中未见过的 object/scene mesh，随机 pose
- 成功率通过最终 simulator state 判断
- 使用 RPDiff 的 insertion controller 执行预测 placement
- 对比 baseline：
  - `TAX3D`
  - `RPDiff`
  - `RPDiff w/o classifier-based reranking`
  - 多个 TAX-DPD ablation

论文主表结果里，TAX-DPD 平均成功率是 `0.97`，RPDiff 是 `0.88`，TAX3D 是 `0.48`。

**训练超参**

附录 Table V 给的是：

```text
Batch size:                 16
Learning rate:              1e-4
Learning rate warmup steps: 100
Weight decay:               1e-5
Epochs:                     20,000
Number of DiT blocks:       5
Number of heads per block:  4
Hidden size per block:      128
Diffusion steps:            100
```

**点云下采样**

附录 Table VI：

```text
RPDiff-Mug/EasyRack:
  object PO: 512 points
  scene  PS: 512 points

RPDiff-Mug/MedRack:
  object PO: 512 points
  scene  PS: 512 points

RPDiff-Mug/Multi-MedRack:
  object PO: 512 points
  scene  PS: 1024 points

RPDiff-Book/Shelf:
  object PO: 512 points
  scene  PS: 1024 points

RPDiff-Can/Cabinet:
  object PO: 256 points
  scene  PS: 1024 points
```

你现在跑的 `mug_on_rack_multi_large_proc_gen_demos` 基本对应论文里的 `RPDiff-Mug/Multi-MedRack`，所以论文设置应是：

```text
object/action points: 512
scene/anchor points: 1024
batch size: 16
epochs: 20000
diffusion steps: 100
lr: 1e-4
weight decay: 1e-5
warmup: 100
```

**和我们当前 H200 长跑的差异**

当前运行命令里是：

```text
training.batch_size=32
training.val_batch_size=16
dataset.sample_size_anchor=1024
epochs=20000
num_wta_trials=10
frame_type=fixed
```

也就是说：`scene/anchor=1024` 和论文的 Mug/Multi-MedRack 一致；但训练 batch size 当前是 `32`，论文表里是 `16`。如果你想更贴论文设置，后续应该用 `training.batch_size=16`。当前 run 更像是为了 H200 单卡吞吐做了 batch 放大。

> 现在是什么状态?

<details><summary>2 previous messages</summary>

> 我直接连 `h200-1` 查进程、最新日志、checkpoint 和 GPU，占用/错误一起看。
>
> 训练还在跑，已经到 `18073/20000 epoch`，约 90%。我注意到 GPU 5 上除了训练子进程还有另一个进程占了约 26GB 显存，我查一下 PID 是什么，判断是否影响速度。
</details>

当前状态：还在正常跑，已经接近结束。

- 时间：`2026-08-06 14:25 CST`
- 主启动进程：`PID=74600`，仍然存活，`PPID=1`
- 实际训练子进程：`PID=74812`
- 当前进度：`Epoch 18073 / 20000`，约 `90.4%`
- 上一个完整 epoch：`Epoch 18072`
- 最近 checkpoint：`epoch=17999-step=1800000.ckpt` 和 `last.ckpt`，保存时间 `2026-08-06 14:09`
- 最近日志没有看到 `Traceback`、OOM、Killed 等错误

当前指标：

```text
val_rmse_wta_0 = 0.170
val_rmse_0     = 0.294
val_rmse_wta_1 = 0.166
val_rmse_1     = 0.292
```

预计剩余时间：大约 `6.5-7.5 小时`，如果速度稳定，预计 `2026-08-06 21:00-22:00 CST` 完成。

注意：GPU 5 上现在除了我们的训练进程外，还有另一个用户 `GGB` 的进程占了约 `26GB` 显存。我们的训练仍占约 `84GB`，没有崩，但这可能会让瞬时利用率和速度有波动。

> 现在是什么状态?

<details><summary>1 previous message</summary>

> 我会重新查 `h200-1` 上的 PID、训练日志末尾、checkpoint 和 GPU。现在日期已经到 8 月 7 日，重点确认它是完成、还在跑，还是中途退出。
</details>

训练已经完成。

关键信息：

- `tax_dpd_train.pid` 里还是 `74600`，但 `ps` 查不到该进程，说明训练进程已经退出，PID 文件是旧的。
- 日志最后到 `Epoch 19999 / 20000`，正常完成。
- W&B 显示 `Waiting for W&B process to finish... (success)`，没有看到错误。
- 最终 checkpoint 已保存：

```text
/home/yuchi/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-34-11/checkpoints/epoch=19999-step=2000000.ckpt
/home/yuchi/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-34-11/checkpoints/last.ckpt
```

最终 summary 指标：

```text
train/loss        0.08695
train/rmse        0.18793
train_wta/rmse    0.08492

val_rmse_0        0.29153
val_rmse_1        0.29212
val_rmse_wta_0    0.17193
val_rmse_wta_1    0.17037

val_cos_sim_0     0.40120
val_cos_sim_1     0.35124
val_cos_sim_wta_0 0.72577
val_cos_sim_wta_1 0.69267
```

注意：目前 GPU 5 还有其他进程在占显存，但不是我们的训练进程。下一步应该做 `eval_rigid.py` 评估，用最终 checkpoint 或按 best checkpoint 评估。

> 继续评估

<details><summary>57 previous messages</summary>

> 我会接着在 `h200-1` 上跑评估。先确认 `HANDOFF.md` 里的评估命令、`eval_rigid.py` 的 checkpoint/GMM 参数名，以及最终 checkpoint 路径，避免把 GMM 路径或 ckpt 传错。
>
> 交接文档很长，我已经确认里面提到 `eval_rigid.py` 的 GMM checkpoint 兼容和 GMM 路径不一致问题。接下来直接看 `eval_rigid.py` 和配置，确定实际 CLI 参数。
>
> 本地没有 `configs/eval_rigid.yaml`，`eval_rigid.py` 实际用的是 `configs/eval.yaml` 或同名缺失会失败；我继续找 Hydra 配置和现有评估命令痕迹。
>
> 评估脚本硬编码 `config_name="eval_rigid"`，但仓库当前没有这个配置文件；这会直接导致 Hydra 启动失败。先检查远端是否也缺，再决定用已有 `eval_cli.yaml` 补一个最小 `eval_rigid.yaml` 还是改脚本。
>
> 还有一个不一致：`eval_rigid.py` 里访问的是 `cfg.gmm.gmm / cfg.gmm.run_id / cfg.gmm.task_name`，而 `eval_cli.yaml` 只有顶层 `gmm`。所以评估需要补一个和脚本匹配的配置文件，不能只靠 CLI 覆盖。
>
> 我发现 `eval_rigid.py` 里还残留了一个 `breakpoint()`，正式评估会卡进调试器；另外缺 `eval_rigid.yaml` 和 `inference/rpdiff_tax3dv2.yaml`。我会做最小补丁：补配置、去掉断点，不改评估逻辑。
>
> `eval_rigid.py` 还有一个更大的问题：它无条件去 W&B API 拉训练配置。我们现在是离线本地 checkpoint，必须跳过 W&B 配置加载，改为用 CLI 显式传入训练时的 dataset/model 参数。否则还没到加载 checkpoint 就会失败。
>
> 远端非交互 shell 里 `pixi` 不在 PATH，我改用绝对路径检查 checkpoint 内容。
>
> checkpoint 里没有保存完整 Hydra 配置，所以评估配置只能从 CLI 传入。接下来我补 `eval_rigid.yaml`、`rpdiff_tax3dv2` inference 配置，并让 `eval_rigid.py` 在本地 checkpoint 时跳过 W&B 配置读取。
>
> 我现在开始改本地文件；改完会同步到 H200，再先跑一个小规模 smoke eval，确认不会进断点、不会找错 GMM/ckpt。
>
> <details><summary>Created 2 files, edited a file</summary>
>
> - Wrote `./configs/eval_rigid.yaml` (+32 -0)
> - Wrote `./configs/inference/rpdiff_tax3dv2.yaml` (+9 -0)
> - Wrote `./scripts/eval_rigid.py` (+9 -9)
>
> </details>
>
> 本地补丁已完成。现在同步这 3 个文件到远端，并为 GMM 建一个兼容 `eval_rigid.py` 路径规则的软链接：脚本要找 `gmm_outputs/<task_name>/<run_id>`，实际训练产物在 `gmm_outputs/<run_id>`。
>
> 刚才 `rsync` 目标少了目录层级，把三个文件也放到了远端仓库根目录；这是我刚创建的误放副本。我会删除这三个明确路径的副本，并把文件同步到正确目录。
>
> 补丁已同步到正确路径。接下来先看哪张 GPU 空，再用 1/1/1 样本做 smoke eval；这个只验证评估管道，不代表最终指标。
>
> 所有 GPU 都有占用，GPU 5 目前余量最大，smoke eval 用 `CUDA_VISIBLE_DEVICES=5` 并把 batch/WTA/数据量降到 1，避免抢太多资源。
>
> smoke eval 已经跑到加载 GMM 和数据集，失败点是 `tax3d_v2` 的模块初始化在 eval 模式也要求 `cfg.training`。我会在 `eval_rigid.yaml` 里加入训练配置默认项，仅用于模型初始化；dataloader 仍走 `cfg.inference`。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./configs/eval_rigid.yaml` (+1 -0)
>
> </details>
>
> 我把 `training` 默认配置补进 eval 配置了；现在同步并重跑小样本 smoke。
>
> 第二次 smoke 已经成功加载本地 checkpoint，失败在 DataLoader：仓库的 dataloader 固定 `persistent_workers=True`，所以 `num_workers=0` 不允许。我改用 `num_workers=1` 重跑。
>
> 评估入口现在进入 `run_eval` 了，下一处失败是脚本沿用了旧模型接口 `update_batch_frames/predict`，而我们重建的 `TAX3Dv2FixedFrameModule` 只有训练/验证用的 `_predict_wta`。我需要把 eval 适配到当前模块接口，复用 `_predict_wta` 输出。
>
> 对 fixed-frame 重建模块来说，GMM 实际没有被 `_predict_wta` 使用；训练/验证指标也是直接对 `batch["pc"]` 做 WTA。为了让 `eval_rigid.py` 能跑，我会加一条 TAX3Dv2 分支：调用 `_predict_wta`，再用 `T_goal2world` 把预测点云转到 world frame 后计算原脚本的 RMSE/位姿误差。
>
> 我还看到 `pc_world` 也是旧接口 `update_batch_frames` 补出来的字段，原始 dataset 没有。TAX3Dv2 分支里我会显式用 `T_goal2world` 生成 `pc_world` 和 `pred_world`。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./scripts/eval_rigid.py` (+18 -3)
>
> </details>
>
> 我刚补的分支里发现一个维度问题：TAX3Dv2 分支不应该提前把 ground truth 展开，否则后面的通用逻辑会重复 expand。我马上修正为“GT 保持 B，预测保持 B×WTA”。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./scripts/eval_rigid.py` (+4 -3)
>
> </details>
>
> TAX3Dv2 eval 分支已经修好并保持通用误差计算不变。现在同步脚本，再跑 1 样本 smoke。
>
> smoke eval 已通过，能加载最终 checkpoint 并完成 train/val/test 各 1 个样本。正式评估我会用 `num_wta_trials=10`、`val_batch_size=8` 启动，理由是训练期间 WTA=10 且当前 GPU 5 不是独占；用默认 WTA=20 + batch16 很可能因显存被其他进程占用而 OOM。
>
> 由于当前 fixed-frame 重建模型的 `_predict_wta` 不使用 GMM frame，我不会在正式 eval 里加载 GMM，避免额外占显存和制造“用了 GMM”的误导。这个评估是最终 diffusion checkpoint 的点云/刚体误差评估，不是 RPDiff 仿真成功率。
>
> 评估已启动，PID `3891109`，日志在 `~/tax_dpd_eval_rigid.log`。我等它初始化一会儿，再确认是否真正脱离 SSH、有没有立刻报错/OOM。
>
> 进程已脱离 SSH：`PPID=1`、`SID=3891109`。GPU 5 上评估子进程刚初始化，占用还很小；我再等一会儿看是否进入 tqdm 进度。
>
> 正式评估已经进入 train split：`2/400` batches，约 `6.8-7.4s/batch`。按这个速度，train+val+test 总共约 `490` batches，预计 `55-65 分钟`；显存占用约 `138GB/144GB`，接近上限但目前没 OOM。
>
> 进度到 `7/400`，GPU 利用率 100%，显存 `140.6GB/143.8GB`。这比预期更贴近上限；我先不打断，继续观察几分钟。如果出现 OOM，我会用 `batch_size=4` 自动重启正式评估。
>
> 当前正式 eval 大头卡在 train split，而且 train 指标不是最终复现实验的关键；照这个速度完整 train+val+test 要 2.5 小时以上。为了尽快拿到有效 held-out 指标，我会重启评估：`train_dataset_size=1` 只做占位，`val/test=400` 保持完整。
>
> 已重启为 val/test 完整评估。旧日志保存在 `~/tax_dpd_eval_rigid.full_train_aborted.log`，新 PID 是 `620890`，新日志仍是 `~/tax_dpd_eval_rigid.log`。我确认它脱离 SSH 并进入进度。
>
> 新评估已进入 50 个 batch 的 split，说明 1 个 train batch 已经越过，正在跑完整 val 或 test。当前约 `26s/batch`，val+test 预计 `40-45 分钟`。显存仍接近上限但稳定。
>
> 进度 `4/50`，速度稳定在 `22-23s/batch`。如果后面 test 同速，总剩余大约 `35-40 分钟`。
>
> 到 `8/50`，单 batch 约 `20s`，显存没有继续上涨。当前评估看起来稳定。
>
> `12/50`，按当前速度这个 split 还剩约 13-14 分钟；之后还会跑另一个 50-batch split。
>
> `15/50`，仍稳定。显存约 `139.2GB`，没有 OOM。
>
> `19/50`，这个 split 还剩约 10-11 分钟；整体剩余约 30 分钟左右。
>
> `22/50`，无错误。速度维持在每 batch 20-21 秒。
>
> 第一个完整 split 已过半：`27/50`。预计这个 split 还剩约 8 分钟，然后开始最后一个 split。
>
> `31/50`，这个 split 还剩约 6-7 分钟。没有错误输出。
>
> `35/50`，继续稳定。预计第一个完整 split 还有约 5 分钟。
>
> `39/50`，第一个完整 split 接近尾声；还没有输出最终 RMSE，说明还没切到下一个 split。
>
> `43/50`，约 2-3 分钟后完成当前 split。
>
> `47/50`，当前 split 马上结束。下一轮 50 batch 应该是最后的 test split。
>
> 第一个完整 split 用时 `17:10`，现在进入最后一个 50-batch split，`1/50`。预计还剩约 17-20 分钟。
>
> 最后一个 split 到 `5/50`，速度略快一些。剩余约 15 分钟。
>
> 最后 split 到 `9/50`，当前还剩约 13-14 分钟。
>
> 最后 split 到 `13/50`，还剩大约 12 分钟。
>
> 最后 split 到 `17/50`，还剩约 11 分钟。显存略升到 `139.3GB`，仍稳定。
>
> 最后 split 已到 `21/50`，而且显存降到 `138.9GB`，更稳了。这个评估已经跑完前两个 split，预计最终结果还要约 10 分钟。
>
> 最后 split 到 `25/50`，还剩半程，约 8 分钟。
>
> 最后 split 到 `32/50`，剩余约 6 分钟。
>
> 最后 split 到 `39/50`，还剩约 3-4 分钟。
>
> 评估已经完成，进程退出，val/test 指标已打印。`Train RMSE` 是 `nan`，这是因为我把 train split 缩成 1 个占位样本来节省时间；正式可用的是完整 val/test。
</details>

评估已完成。结果在 H200：

```text
/home/yuchi/tax_dpd_eval_rigid.log
```

这次跑的是最终 diffusion checkpoint：

```text
/home/yuchi/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-34-11/checkpoints/last.ckpt
```

评估设置：`val/test` 完整跑，`num_wta_trials=10`，`batch_size=8`。`train` 只跑了 1 个占位样本来节省时间，所以 `Train RMSE = nan` 不看。

关键结果：

```text
Val RMSE:      0.29145
Val RMSE_WTA:  0.17193
Val T_err:     0.25340
Val T_err_WTA: 0.12503
Val R_err:     122.71 deg
Val R_err_WTA: 55.21 deg

Test RMSE:      0.28655
Test RMSE_WTA:  0.16276
Test T_err:     0.24810
Test T_err_WTA: 0.11754
Test R_err:     120.96 deg
Test R_err_WTA: 51.38 deg
```

我还做了几个必要补丁才能让评估跑起来：

- 新增 `configs/eval_rigid.yaml`
- 新增 `configs/inference/rpdiff_tax3dv2.yaml`
- 修改 `scripts/eval_rigid.py`
  - 本地 checkpoint 时跳过 W&B API 配置加载
  - 移除残留 `breakpoint()`
  - 为重建的 `tax3dv2` 模型适配 `_predict_wta`
  - 显式生成 world-frame 点云用于 RMSE/刚体误差

注意：这次 fixed-frame 重建模型的 eval 分支没有实际使用 GMM；它评估的是点云 diffusion checkpoint 的预测误差，不是 RPDiff 仿真成功率。

> 所以你的复现结果可以跟论文对上吗

不能严格对上论文结果。

目前我们拿到的是：在 H200 上把这个仓库的 **近似 TAX-DPD fixed-frame 实现** 跑通了训练和点云/刚体误差评估。但它还不能作为“论文 TAX-DPD 结果复现成功”的证据，主要原因有三个：

1. 核心代码不是官方完整实现
   论文主模型 `tax3d_v2.py` 原仓库没有发布。现在这份是根据仓库里已有 DiT、diffusion、encoder 组件重建的 Lightning 外壳，能训练、能收敛，但不能保证和作者实现一致。

2. 我们评估的不是论文 Table I 的同一指标
   论文 RPDiff 表格报的是 PyBullet 中执行 placement 后的 `success rate`，每个任务 100 trials。
   我们现在跑出来的是 point-cloud RMSE、WTA RMSE、translation/rotation error，不是仿真成功率。

3. fixed-frame eval 目前没有真正接入 GMM
   训练时用了 `pred_frame=noisy_goal` 的数据设定，但我最后的 `eval_rigid.py` 适配分支为了让重建模型跑通，评估的是 diffusion point prediction；fixed-frame `_predict_wta` 没有使用 GMM 输出。因此它不是论文的完整“两阶段 GMM + local refinement + simulator execution”流程。

当前结果更适合这样表述：

```text
We trained and evaluated a reconstructed TAX-DPD-like fixed-frame point diffusion model on the RPDiff Mug/Multi-MedRack dataset. The pipeline trains to completion and obtains test RMSE_WTA ≈ 0.163 under our point-cloud evaluation.
```

不应该表述为：

```text
We reproduced TAX-DPD's reported RPDiff success rate.
```

如果要真正和论文对齐，下一步需要做两件事：

- 拿到或重建更可信的官方 `tax3d_v2.py` 推理逻辑，尤其是 GMM frame 如何进入 local diffuser。
- 接 RPDiff simulator/controller，按论文的 `100 trials success rate` 跑 `Mug/Multi-MedRack`，然后才能和 Table I 里的 `0.95` success rate比较。

> 目前的复现结果跟论文里相差多少呢

<details><summary>1 previous message</summary>

> 我会先把论文里能直接对比的数字再查一遍，尤其是是否有 RPDiff 的 RMSE/coverage/precision 表；如果论文只报 success rate，就只能说明“指标不一致，不能算差多少”。
</details>

目前不能给出“相差 X%”这种严格数字，因为我们和论文报的不是同一个指标。

论文里 `Mug/Multi-MedRack` 的主结果是：

```text
TAX-DPD full: 0.95 success rate
TAX-DPD w/o GMM: 0.74 success rate
TAX-DPD w/ SE(3) diffusion: 0.89 success rate
RPDiff: 0.86 success rate
TAX3D: 0.32 success rate
```

我们现在得到的是点云误差评估：

```text
Val RMSE_WTA:  0.17193
Test RMSE_WTA: 0.16276
Test T_err_WTA: 0.11754
Test R_err_WTA: 51.38 deg
```

所以严格说：

```text
论文：仿真执行成功率 success rate
我们：点云 RMSE / 刚体配准误差
```

这两个不能直接相减，也不能说“比论文低多少”。

如果按复现程度判断，当前离论文主结果还差比较大：

1. 论文主表要的是 PyBullet/RPDiff controller 执行后的 `100 trials success rate`，我们还没跑这个。
2. 我们的 eval 目前没有真正用上 GMM，所以更接近“local diffusion 点云误差评估”，不是论文完整 TAX-DPD。
3. `Test R_err_WTA ≈ 51°` 偏大，这说明当前预测点云经 SVD 估计出来的刚体姿态并不稳定；即使点云 RMSE_WTA 看起来能收敛，也不能乐观地推断能达到论文 `0.95` success rate。

最诚实的结论是：

```text
目前结果还不能对上论文。论文 Mug/Multi-MedRack 是 95% success rate；
我们当前没有对应 success rate，因此数值差距未知，但从评估链路和旋转误差看，不能认为已经接近论文主结果。
```

下一步要真正量化差距，需要跑 RPDiff simulator success evaluation。跑完后才能直接比较：

```text
gap = 0.95 - our_success_rate
```

> 继续跑

<details><summary>43 previous messages</summary>

> 我理解你是要继续往“和论文可比”的方向跑，也就是尝试 RPDiff simulator/controller 的 success rate，而不是再跑同一个 RMSE eval。我先检查仓库里有没有现成的 RPDiff success-eval 入口；如果没有，就确认缺口在哪里。
>
> 搜索结果显示本仓库的 `eval_rigid.py` 只写了 RMSE/刚体误差，代码注释也说 coverage/precision/success 需要 RPDiff env 另算。我继续确认第三方目录里是否包含 RPDiff 仿真代码；如果没有，就只能先跑可执行的下一项（例如完整 train split RMSE 或尝试搭 RPDiff 依赖）。
>
> 本地第三方目录没有 RPDiff 仿真环境，只有 DEDO 和 3D-Diffusion-Policy；所以没法直接从当前仓库调用论文的 PyBullet success-rate eval。我会先在远端确认状态和可用空间/已有 RPDiff 代码，然后如果没有现成环境，就启动一个更完整的 RMSE 评估补齐 train split，同时把 success-rate eval 的缺口列清楚。
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `RPDiff GitHub relational pose diffusion multi-modal rearrangement`
>
> </details>
>
> 我查到 RPDiff 官方 repo 是 `anthonysimeonov/rpdiff`。我会先只克隆/查看它的评估脚本和依赖，不改当前训练产物；如果它能接收外部 pose/prediction，再把我们的预测接进去。
>
> 我已经把 RPDiff repo 克隆到 `/home/yuchi/projects/rpdiff`。接下来重点看它的 `evaluate_rpdiff.py` 是否能替换 policy/model 输出；如果它强绑定 RPDiff 自己的模型权重，就需要写一个 adapter。
>
> RPDiff 的 eval 脚本是可以改造的：它在中间得到 `parent_pcd`、`child_pcd`，然后调用 RPDiff 自己的 `infer_relation_policy` 得到相对变换。我们的接入点就是替换这一段：用 TAX-DPD 预测 goal point cloud，再 SVD 得到 child 的最终 pose。接下来先做依赖/import smoke，确认 RPDiff eval 环境能不能启动。
>
> 本地刚克隆的 RPDiff 还没在 H200 上；H200 当前 TAX-DPD 环境里有 `pybullet`，但缺 `meshcat`、`airobot`、`rpdiff` 包本身。下一步我把 RPDiff repo clone 到 H200，并尝试按它的 requirements 做最小安装。
>
> RPDiff 需要自己的 assets 和若干老依赖；我不会改 TAX-DPD 的锁文件，但会在 H200 的现有 pixi 环境里做最小 pip 安装和 RPDiff editable 安装。如果依赖冲突，我会停下来汇报，不强行降级 torch/numpy。
>
> 依赖安装到一半可用，但 `pixi run` 在 RPDiff 目录下找不到 `pixi.toml`，所以 editable install 没执行。这个不是依赖失败，我改用 TAX-DPD 环境里的绝对 Python 路径继续安装。
>
> RPDiff editable install 缺 `Cython`。我会只补这个构建依赖，再继续安装。
>
> `rpdiff` 包本身已经装上了；现在剩两个运行时问题：`airobot` 缺 `colorlog`，以及 `urdfpy/mesh_to_sdf` 依赖的老 `networkx` 在 Python 3.9 下还引用 `fractions.gcd`。我先补 `colorlog`，然后测试 RPDiff eval 脚本实际 import 是否卡在 urdfpy。
>
> RPDiff eval import 下一步卡在 `cv2`，这是 airobot 的依赖。我继续补最小运行依赖 `opencv-python-headless`，再测 import。
>
> `opencv-python-headless` 把 TAX-DPD 环境里的 NumPy 拉到了 `2.0.2`，导致 `pybullet/trimesh/shapely` 这类二进制包 import 失败。我要先把 NumPy 恢复到锁文件兼容版本，避免影响已有 TAX-DPD 环境；然后再考虑单独环境跑 RPDiff。
>
> 锁文件里 NumPy 是 `1.23.5`。我会在 H200 把 TAX-DPD pixi 环境恢复到 `numpy==1.23.5`，并测试之前的 TAX-DPD import 链，避免环境被 RPDiff 尝试污染。
>
> 现在 import 卡在 `urdfpy` 拉的老 `networkx==2.2`，它在 Python 3.9 下不可用。这个可以用较新的 `networkx` 兼容绕过；我安装 `networkx==2.8.8` 后再测。
>
> RPDiff eval import 继续卡在 `knn_cuda`。官方给的是一个老 wheel，可能和 torch 2.7/CUDA 12 不兼容，但我先试安装；如果不兼容，再考虑给 eval 脚本做 lazy import，避开 RPDiff 自己的 policy。
>
> 官方 README 里的 `knn_cuda` wheel URL 已经 404。因为我们替换 RPDiff policy 后并不需要它，我接下来走 adapter 路线：复制/改造 eval 脚本，移除 RPDiff policy 模型导入，直接用 TAX-DPD 输出 pose。
>
> 我会给 RPDiff eval 脚本加一个 `tax_dpd_external` inference_method：不加载 RPDiff 自己的模型，不需要 `knn_cuda`，直接加载我们训练好的 TAX-DPD checkpoint。先在本地补丁，再同步 H200。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/eval/evaluate_rpdiff.py` (+98 -1)
>
> </details>
>
> 已加基础 helper。现在继续补 main 里的分支：`tax_dpd_external` 时跳过 RPDiff policy dict，并在每个 trial 调用 TAX-DPD helper 生成相对变换。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/eval/evaluate_rpdiff.py` (+33 -22)
>
> </details>
>
> 主分支补好了。现在添加一个 RPDiff eval 配置：5 个 trial 的 smoke success-rate eval，使用 TAX-DPD checkpoint，先验证整个 PyBullet 执行链能跑。
>
> <details><summary>Created a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/config/full_eval_cfgs/mug_on_rack_multi/mug_on_rack_multi_tax_dpd_smoke.yaml` (+46 -0)
>
> </details>
>
> 还有一个顶层 import 会拉 `knn_cuda`：RPDiff 的 transformer policy 类。TAX-DPD adapter 不需要这些类，我把这组 import 也改成 lazy/可缺失。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/eval/evaluate_rpdiff.py` (+11 -5)
>
> </details>
>
> RPDiff eval import 现在能过了。我启动 5-trial 的 TAX-DPD adapter smoke success eval，使用物理 GPU 1 映射为 `cuda:0`；先看它能否完整跑通 PyBullet trial。
>
> 进程刚启动，状态是 `D`，通常是在加载/IO 或 PyBullet 初始化。日志目前只有 pybullet build 行；我等一会儿确认是否继续推进或卡住。
>
> 评估进程没有进入 GPU，也没有继续输出，最可能是在等 meshcat ZMQ server。RPDiff eval 脚本即使不看可视化也会初始化 meshcat。我启动一个后台 `meshcat-server` 后再观察。
>
> 刚才 `pgrep` 被检查命令自己匹配了，meshcat-server 没真正起来。我直接启动一次并验证 PID。
>
> 现在 RPDiff eval 已经越过 meshcat/PyBullet 初始化，加载了 test objects，并进入 trial 前准备。继续观察是否能完成第一个 trial。
>
> Context compaction
>
> Source: automatic
> Status: completed
>
> 我先接着查 H200 上 smoke eval 的当前进度；如果已经结束，就直接读成功率/失败原因，再决定是否启动更大规模评估。
>
> 当前 smoke eval 还在跑：PID 593318，已跑约 4 分钟，完成了 2/5 个 trial，前两个 Place Success 都是 0。GPU1 正在用，约 69GB 显存、89% 利用率。按现在速度约 50–60 秒/iteration，剩余 3 个 trial 大概还要 3 分钟。我继续等它跑完。
>
> 5-trial smoke eval 已经跑完；5 个 iteration 都是 Place Success 0。下一步我会确认没有隐藏 traceback，并查看评估输出目录。如果技术上正常结束，我会启动一个更大的 100-trial 评估；按当前速度预计约 80–100 分钟。
>
> 确认：smoke 进程已退出，没有 traceback；5/5 全部失败，当前 adapter 的模拟成功率样本结果是 0/5。现在我准备启动 100-trial 长评估，用独立日志和 PID，避免覆盖 smoke 日志。
>
> 我发现远端 shell 默认没有 `python` 命令，这不影响评估，因为实际运行用的是 TAX-DPD pixi 环境里的 Python。配置文件我用已有 smoke 配置复制成 100-trial 配置，只改实验名和 iteration 数。
>
> 本地和 H200 看到的是同一路径，我会用补丁方式新增 100-trial 配置，然后在 H200 后台启动。
>
> <details><summary>Created a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/config/full_eval_cfgs/mug_on_rack_multi/mug_on_rack_multi_tax_dpd_100.yaml` (+46 -0)
>
> </details>
>
> 100-trial 配置已新增。现在启动后台任务，并把 PID/日志路径固定下来。
>
> 100-trial 第一次启动失败，原因很明确：我在本机路径新增了配置，但 H200 上 `/home/yuchi/projects/rpdiff/...` 没看到这个新文件。smoke 配置在 H200 上存在，所以我现在把新增配置同步到 H200，再重新启动。
>
> 配置已同步到 H200。重新启动 100-trial 评估。
>
> 100-trial 已重新启动成功：PID 609727，日志 `/home/yuchi/tax_dpd_rpdiff_success_100.log`。它刚加载 OpenGL，GPU1 正在满载。再等一小段确认第一个 iteration 能完成。
</details>

已继续跑起来。

当前状态：

- 100-trial RPDiff simulator success eval 已在 H200 后台运行
- PID: `609727`
- 日志: `/home/yuchi/tax_dpd_rpdiff_success_100.log`
- 配置: `mug_on_rack_multi_tax_dpd_100.yaml`
- 当前已完成 `2/100`
- 当前结果：`0/2` success
- 进程正常，无 traceback / OOM / killed
- 速度：约 `50–60 秒/iteration`
- 预计总时长：约 `80–100 分钟`
- 预计剩余：约 `75–95 分钟`

你可以自己检查进度：

```bash
ssh h200-1

PID=$(cat ~/tax_dpd_rpdiff_success_100.pid)
ps -o pid,ppid,sid,etimes,stat,cmd -p "$PID"

grep -E "Iteration: .*Place Success|Traceback|Error|CUDA out of memory|Killed" \
  ~/tax_dpd_rpdiff_success_100.log | tail -50

nvidia-smi -i 1
```

统计当前成功数/总数：

```bash
grep -E "Iteration: .*Place Success" ~/tax_dpd_rpdiff_success_100.log | wc -l

grep -E "Iteration: .*Place Success: 1.000" ~/tax_dpd_rpdiff_success_100.log | wc -l
```

注意：刚才 5-trial smoke eval 已完整跑完，结果是 `0/5`。现在这个 100-trial 是更正式一点的模拟成功率评估，但仍然是基于我们写的 TAX-DPD → RPDiff adapter，不是论文官方完整 pipeline。

> 现在是什么状态

<details><summary>1 previous message</summary>

> 我直接查 H200 上 100-trial 评估的 PID、日志和当前成功率。
</details>

100-trial 评估已经跑完。

结果：

- 进程：已退出
- 日志：`/home/yuchi/tax_dpd_rpdiff_success_100.log`
- 完成：`100/100`
- 最后一行：

```text
Iteration: 99, Place Success: 0.260
```

也就是当前这个 RPDiff simulator adapter 评估的成功率是：

```text
26 / 100 = 26.0%
```

补充说明：日志里的 `Place Success` 是累计成功率，不是单个 trial 的 0/1。所以不能用 `grep "Place Success: 1.000"` 统计成功数；最终 `0.260` 才是关键结果。

当前结论：这个 adapter 版模拟成功率结果是 `26%`。它比之前 RMSE 指标更接近论文里的 evaluation 形式，但仍然不是论文官方完整 pipeline，因为 TAX-DPD 官方 `tax3d_v2.py / GMM / local frame` 部分缺失，我们现在用的是重构模型 + RPDiff 外接 adapter。

> 下一步要做什么  继续

<details><summary>4 previous messages</summary>

> 下一步我会做两件事：先把 100-trial 结果解析清楚（确认单次成功/失败和输出文件），然后继续跑更稳的评估。优先选择 500-trial 同配置，因为 100 次的方差还比较大；如果中途发现日志里有异常或结果文件缺失，会先修解析/保存问题。
>
> 100-trial 输出目录存在，日志最终累计成功率和 `touching_surf=True` 数一致：`26/100`。没有错误。现在我继续启动 500-trial 同配置评估，目标是降低随机方差；预计按当前速度约 6–8 小时。
>
> <details><summary>Created a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/config/full_eval_cfgs/mug_on_rack_multi/mug_on_rack_multi_tax_dpd_500.yaml` (+46 -0)
>
> </details>
>
> 500-trial 已启动：PID `851549`，日志 `/home/yuchi/tax_dpd_rpdiff_success_500.log`，seed=1。现在等第一个 iteration 完成，确认不是只启动成功。
</details>

已继续执行，当前进入更稳的 500-trial 评估。

当前状态：

- 任务：RPDiff simulator success eval，500 trials
- PID：`851549`
- 日志：`/home/yuchi/tax_dpd_rpdiff_success_500.log`
- seed：`1`
- GPU：H200 GPU1，约 `76.9GB / 143.8GB`，利用率 `100%`
- 当前已确认完成：`1/500`
- 当前累计成功率：`0.000`
- 没有 traceback / OOM / killed

预计时间：

- 当前速度约 `45–60 秒 / trial`
- 500 trials 总时长约 `6.5–8.5 小时`
- 已经开始跑，不需要你保持 ssh 会话

你可以检查进度：

```bash
ssh h200-1

PID=$(cat ~/tax_dpd_rpdiff_success_500.pid)
ps -o pid,ppid,sid,etimes,stat,cmd -p "$PID"

grep -E "Iteration: .*Place Success|Traceback|Error|CUDA out of memory|Killed" \
  ~/tax_dpd_rpdiff_success_500.log | tail -50

nvidia-smi -i 1
```

当前已完成的 100-trial 结果仍是：

```text
26 / 100 = 26.0%
```

500-trial 的目的不是“提高分数”，而是确认这个 26% 是否稳定。若 500-trial 最终仍在 25% 左右，就可以把它作为当前 adapter 复现的主要 simulator success 结果。

> 根据现有的结果，跟论文中的结果做一个暂时的对比呢

<details><summary>3 previous messages</summary>

> 我先从本地论文 PDF 里提取实验表格/指标，避免凭记忆对比。然后把我们现在已有的 RMSE 和 simulator success 跟论文数字放在一起说明“能比”和“不能严格比”的部分。
>
> 本地有 `pdftotext`，我会提取 PDF 文本并定位论文中的实验表格。
>
> 论文表格已定位：RPDiff 仿真实验主表是 Table I，RPDiff suite 每个任务 100 trials。我们的 100-trial adapter 结果最接近其中 `Mug/Multi-MedRack`，但不是严格同 pipeline。再查一下 500-trial 当前进度后给你临时对比。
</details>

临时对比结论：目前复现结果和论文主结果差距很大，不能认为已经对上。

最接近的论文任务是 Table I 里的 `Mug/Multi-MedRack`，因为我们现在跑的是 `mug_on_rack_multi` + `syn_rack_med` + 多 rack 场景。

| 项目 | Success Rate |
|---|---:|
| 论文 TAX-DPD full，Mug/Multi-MedRack | `0.95` |
| 论文 TAX-DPD w/o GMM，Mug/Multi-MedRack | `0.74` |
| 论文 RPDiff baseline，Mug/Multi-MedRack | `0.86` |
| 我们当前 100-trial adapter 结果 | `0.26` |

差距：

- 对论文 full TAX-DPD：低 `0.69`，也就是少 `69 个百分点`
- 对论文 w/o GMM：低 `0.48`
- 对论文 RPDiff baseline：低 `0.60`

当前 500-trial 还在跑，临时状态：

```text
Iteration 15 / 500
Place Success: 0.188
```

也就是目前早期约 `3/16` 成功，样本太少，不稳定；最终还是要等 500 跑完。但它当前趋势和 100-trial 的 `26%` 是同一量级，没有显示出接近论文 `95%` 的迹象。

关键原因不是单纯“训练没跑够”，而是 pipeline 不完整：

1. 仓库缺失官方 `tax3d_v2.py`，我们现在是重构实现。
2. 论文 full 方法用了 GMM/global initialization + local diffuser；我们当前 adapter 没有完整复现这套。
3. 我们是把 TAX-DPD 输出外接进 RPDiff simulator，用 SVD 拟合 transform，这更像工程近似评估，不是论文官方 evaluation code。
4. 之前 RMSE eval 也显示姿态误差很大：test `R_err ≈ 121°`，WTA 后也还有 `≈51°`，这会直接导致挂杯成功率低。

所以现在可以写成：

```text
当前 adapter-based reproduction 在 Mug/Multi-MedRack-like 任务上得到 26% success，
相比论文 TAX-DPD 的 95% 低 69 个百分点；相比论文 w/o GMM 的 74% 也低 48 个百分点。
该结果说明当前复现尚未达到论文水平，主要限制在官方 TAX3Dv2/GMM/local pipeline 缺失和 adapter 近似。
```

下一步优先级：等 500-trial 完成，确认 `26%` 是否稳定；然后应集中补 GMM/global initialization 和 local refinement，而不是继续盲目加训练轮数。

> 下一步要做什么

下一步建议按优先级这样做：

1. 等 500-trial 跑完

当前它已经在跑，先拿稳定成功率。100-trial 是 `26%`，500-trial 能判断这是不是偶然波动。

检查：

```bash
ssh h200-1
grep -E "Iteration: .*Place Success" ~/tax_dpd_rpdiff_success_500.log | tail -20
```

2. 不要马上继续训练

目前问题主要不是 epoch 不够，而是 pipeline 不完整。现有 RMSE 里旋转误差很大，sim success 也低，继续堆训练时间大概率收益有限。

3. 优先补论文关键模块

最该补的是：

- GMM / global placement initialization
- local refinement / local diffuser
- 更接近论文的 inference sampling 和 rerank/selection
- 检查 object/scene frame 是否和论文一致

目前 adapter 是：

```text
TAX-DPD point prediction → SVD 拟合 SE(3) → RPDiff simulator
```

这和论文 full TAX-DPD 不等价。

4. 做一个 sanity check

在 simulator 里保存若干失败 case 的预测位姿图像/点云，看失败模式：

- 是方向完全错？
- 是位置偏 rack？
- 是挂在错误 peg？
- 是 mug 穿模/没接触？
- 是 frame 定义反了？

这一步能判断该修 rotation/frame，还是该补 GMM。

5. 如果要继续自动跑，我建议下一步是：

```text
先等 500-trial 结束 → 汇总最终 success → 抽样查看失败 case → 定位主要失败原因 → 决定修 GMM/local/frame
```

我不建议现在再开更长的 1000-trial；如果 500-trial 仍在 20–30%，结论已经足够明确。

> 现在是什么状态

<details><summary>1 previous message</summary>

> 我查一下 500-trial 的进程、已完成 iteration、当前累计成功率和错误日志。
</details>

500-trial 已经跑完。

结果：

```text
500 / 500 completed
Final Place Success: 0.286
成功数约: 143 / 500
```

没有发现：

```text
Traceback
CUDA OOM
Killed
Exception
```

当前进程已经退出，但 GPU1 仍显示有占用/利用率，可能是别的任务或刚退出后的残留，需要单独看进程。

和 100-trial 对比：

```text
100-trial: 26.0%
500-trial: 28.6%
```

结论更稳定了：当前 adapter-based 复现大约是 `26–29%` success，确实远低于论文 `Mug/Multi-MedRack` 的 `95%`。

下一步建议不是继续跑更多 trial，而是开始定位失败原因：

1. 抽样看成功/失败案例图像。
2. 检查预测 transform 是否有系统性旋转错误。
3. 补/还原论文的 GMM global initialization。
4. 再补 local refinement / local diffuser。

> 继续下一步

<details><summary>32 previous messages</summary>

> 我开始做失败原因定位：先解析 500-trial 的逐 trial 成功/失败，找出典型成功和失败样本，然后检查保存的 npz/图像里有什么字段，判断能不能直接分析预测位姿误差/失败模式。
>
> 500-trial 数据完整：143 成功、357 失败，输出包含每个 trial 的 `success_rate_relation.npz`、`results.txt` 和渲染图。现在我检查 npz 字段，确认是否能量化位姿/接触失败类型。
>
> 远端系统 `python3` 没有 numpy；我改用 TAX-DPD pixi 环境里的 Python 来读 npz。
>
> npz 里只保存了最终 success 和对象 ID，没有保存预测 transform/中间点云；所以定量失败模式需要改代码加记录。先用已有渲染图做视觉抽样，确认失败是不是大多“没挂上/错位/姿态错”。
>
> 我把几个成功/失败 trial 的渲染图拉到本地，做一个拼图方便直接看。
>
> 第一次拉图没成功：scp 的 brace 在远端没有展开，本地也没有 PIL。改用远端循环明确复制；拼图用系统 ImageMagick，如果没有就逐张查看。
>
> 命令里用了清理 `/tmp` 目录，被安全规则拒绝。我不用删除，改建一个新的临时目录。
>
> 图片已拉到本地临时目录；系统没有生成 montage，我直接打开几个关键视角看。
>
> 临时目录名需要重新定位一下。
>
> 目录是 `/tmp/tax_dpd_vis.8KKdIz`。
>
> trial 0 是失败样本。从两个视角看，场景里基本只看到 rack，杯子不明显挂在 peg 上；这符合 `touching_surf=False`。我再看成功样本和 teleport 图。
>
> 成功样本显示杯子确实接触/套在 peg 附近，但姿态看起来并不精细；成功判据主要是 `touching_surf`，不是严格“好看地挂住”。我继续看后期成功样本和 teleport 图，判断是否只是接触判据宽松。
>
> teleport 的失败样本能看到杯子在 rack 外侧/边上，位置没有进入挂杯区域。继续看成功/后期样本。
>
> 视觉结果初步判断：失败常见是杯子落在 rack 外侧/离 peg 偏掉；成功只是接触到/套到 peg，精度不高。下一步我改 RPDiff adapter，让它保存 TAX-DPD 预测 transform、输入点云中心、SVD 拟合残差，跑一个小规模 diagnostic eval，用数据判断是 translation 还是 rotation 主导失败。
>
> 代码里当前保存点在每个 trial 结束处，能直接加诊断字段；`relative_trans_pred` 在推理处生成，后面会变成最终 child pose。我要加一个可选 diagnostic 保存：`tax_dpd_debug.npz`，包含预测 relative transform、initial/final child pose、父/子点云中心、SVD RMSE。然后跑 50-trial diagnostic，不改原 500 结果。
>
> 我会最小改动：`_tax_dpd_infer` 返回 `(T_world, debug)`；只有 `tax_dpd_external` 分支用 debug，保存到 npz。这样不影响普通 RPDiff 分支。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/eval/evaluate_rpdiff.py` (+39 -2)
>
> </details>
>
> 刚才补丁有一个变量顺序问题：`start_parent_pose_mat` 在保存前还没定义。我马上修掉，避免 diagnostic 运行崩。
>
> 确实有变量顺序问题；我把 `place_parent_pose_mat` 移到 parent pose 读完之后保存。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/eval/evaluate_rpdiff.py` (+2 -1)
>
> </details>
>
> 修完。现在同步修改到 H200，并新增 50-trial diagnostic 配置。
>
> <details><summary>Created a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/config/full_eval_cfgs/mug_on_rack_multi/mug_on_rack_multi_tax_dpd_diag50.yaml` (+46 -0)
>
> </details>
>
> 代码已同步并通过语法检查。现在启动 50-trial diagnostic 评估，预计 35–50 分钟；先等 1–2 个 trial 确认 debug 文件正常生成。
>
> diag50 已启动：PID `2083566`，日志 `/home/yuchi/tax_dpd_rpdiff_diag50.log`。我继续等第一个 trial 完成并检查 `tax_dpd_debug.npz`。
>
> debug 文件已经正常生成，说明诊断版运行没问题。先读前两个失败 trial 的 debug 数值，看看有没有明显 translation/rotation 异常。
>
> 前两个失败样本的信号很明显：SVD 拟合残差很小（约 1–1.4cm），说明“点云预测 → rigid transform 拟合”本身不是主问题；预测给出的 relative rotation 很大（111°、141°），最终姿态/接近方向容易错。继续等 diag50 多跑一些样本，再做 success vs fail 统计。
>
> 这里还有一个可操作的假设：当前 adapter 每个 trial 都用 RPDiff 的随机 SO(3) 初始姿态和随机 parent 点作为 guess，日志里一直有 warning。论文 full 方法有 global/GMM 初始化，当前没有。为了验证“初始化是否主因”，我会同时准备一个小 ablation：固定原始 upright orientation + parent mean 初始化，跑 50 trials，对比 diag50。
>
> 先看 H200 哪些 GPU 空闲；如果有空卡，就并行跑初始化 ablation，不占用 diag50 的 GPU1。
>
> 没有合适空卡：GPU6 虽然利用率 0，但显存已占 129GB；我们的单进程大约需要 68–77GB，不能安全并行开。初始化 ablation 我先准备配置，等 diag50 完成后再接着跑。
>
> <details><summary>Created a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/config/full_eval_cfgs/mug_on_rack_multi/mug_on_rack_multi_tax_dpd_init_diag50.yaml` (+47 -0)
>
> </details>
>
> 我会放一个远端后台 watcher：等当前 `diag50` 进程结束后，自动用同一张 GPU1 启动 `init_orig_ori + parent_mean` 的 50-trial ablation。这样不用我一直占着会话，也避免两任务抢显存。
>
> watcher 已挂好：`diag50` 结束后会自动启动 `init_diag50`。我再给一次当前进度和后续检查指令。
</details>

已继续推进，当前状态如下：

1. 已给 RPDiff adapter 加了诊断保存

新增每个 trial 的：

```text
tax_dpd_debug.npz
```

里面包含：

- TAX-DPD 预测的 relative transform
- 初始 guess transform
- 最终 world transform
- child / parent 点云中心
- SVD 拟合 RMSE
- final child pose
- parent 距离列表

2. 当前正在跑 `diag50`

```text
PID: 2083566
日志: /home/yuchi/tax_dpd_rpdiff_diag50.log
配置: mug_on_rack_multi_tax_dpd_diag50.yaml
seed: 2
```

已确认：

```text
trial_0 和 trial_1 的 tax_dpd_debug.npz 已正常生成
```

3. 初步诊断结果

前两个失败样本：

```text
SVD fit RMSE: 1.35cm / 1.09cm
relative rotation: 111° / 141°
```

这说明目前主要问题不像是 “SVD 拟合坏了”，而更像是：

```text
TAX-DPD adapter 输出的目标姿态/初始化校正太大，导致最终杯子姿态和接近方向不稳定。
```

视觉上也看到失败样本常见情况是杯子在 rack 外侧/偏离 peg，没有真正挂上。

4. 已准备并挂好自动下一步

因为现在没有安全空 GPU，我没有并行开第二个实验。我挂了一个 watcher：

```text
watcher PID: 2092514
watcher 日志: /home/yuchi/tax_dpd_init_after_diag50_watcher.log
```

它会在 `diag50` 结束后自动启动一个初始化 ablation：

```text
配置: mug_on_rack_multi_tax_dpd_init_diag50.yaml
改动:
  init_orig_ori: True
  init_parent_mean_pos: True
```

这个实验用来验证“随机 SO(3) + 随机 parent point 初始化”是不是主要拖累。

你可以检查：

```bash
ssh h200-1

# 当前 diagnostic
grep -E "Iteration: .*Place Success|Traceback|Error|CUDA out of memory|Killed" \
  ~/tax_dpd_rpdiff_diag50.log | tail -30

# watcher
cat ~/tax_dpd_init_after_diag50_watcher.log

# 如果 init ablation 已启动
grep -E "Iteration: .*Place Success|Traceback|Error|CUDA out of memory|Killed" \
  ~/tax_dpd_rpdiff_init_diag50.log | tail -30
```

下一步等两个 50-trial 都完成后，我会对比：

```text
random init diag50 vs fixed/upright init_diag50
```

如果 `init_diag50` 明显提高，优先补 GMM/global initialization；如果没提高，优先查 frame/rotation convention 和 local refinement。

> 现在是什么状态

<details><summary>2 previous messages</summary>

> 我查 H200 上 diagnostic 和 init ablation 的进程、日志、最终成功率以及是否有错误。
>
> 两个 50-trial 都已经结束。我再确认 GPU1 上是谁在占用，避免误判为我们的评估还在跑。
</details>

当前状态：两个后续诊断实验都已经跑完，我们自己的评估进程都已退出。

结果汇总：

```text
500-trial 主结果:
143 / 500 = 28.6%

diag50，随机 SO(3) + 随机 parent point 初始化:
19 / 50 = 38.0%

init_diag50，原始姿态 + parent mean 初始化:
12 / 50 = 24.0%
```

诊断文件也完整生成：

```text
diag50 tax_dpd_debug.npz: 50 / 50
init_diag50 tax_dpd_debug.npz: 50 / 50
```

没有发现：

```text
Traceback
OOM
Killed
Exception
```

关键结论：

- 改成 `init_orig_ori=True` 和 `init_parent_mean_pos=True` 没有提升，反而从 `38%` 掉到 `24%`。
- 所以当前低成功率不是简单由“随机初始化”造成的。
- 更可能的问题在：
  - frame / rotation convention 不一致；
  - TAX-DPD 输出的姿态方向不稳定；
  - 缺少论文 full pipeline 的 GMM/global placement 和 local refinement；
  - 当前 adapter 用 SVD 从点预测拟合 SE(3)，但没有论文里的完整 inference 逻辑。

GPU1 现在还有占用，但不是我们的评估进程，是其他用户进程。我们的 PID `2083566 / 2172810 / 2092514` 都已经不在了。

下一步我建议做定量 debug 分析：比较成功 vs 失败样本里的 rotation angle、translation distance、SVD RMSE、final pose 到 parent 的距离，找出最主要失败因子。

> 继续下一步

<details><summary>17 previous messages</summary>

> 我继续做定量 debug：把 `diag50` 和 `init_diag50` 的 `tax_dpd_debug.npz` 汇总成成功/失败对比，重点看 rotation、translation、SVD RMSE、final pose 到 parent 的距离是否有明显分界。
>
> 统计结果出来了：成功/失败在 SVD RMSE、rotation angle、final-to-parent distance 上没有清晰分界。也就是说现在不是某个简单阈值问题。下一步我查 adapter 的 transform 方向和 TAX-DPD 数据字段，重点确认我们有没有把“预测点云所在坐标系/方向”用反。
>
> 我发现关键线索：当前仓库里已经有重构的 `tax3d_v2.py`，其中 `noisy_goal` 分支明确要求输入 `batch["noisy_goal"]`，否则用 0。我们的 adapter 之前没有传 `noisy_goal`，等价于把 GMM/global goal frame 置零，这和论文 full pipeline 差很多。下一步我会改 adapter，把 RPDiff 初始 guess 的 child centroid 作为 noisy_goal 传进去，跑一个 50-trial ablation 验证。
>
> 注意一个细节：当前 adapter 加载的是 `frame_type=fixed`，但 checkpoint/训练可能对应 `noisy_goal` 语义。我要先确认实际实例类型和 config，再决定 patch 用法。
>
> 实际 checkpoint 是 `frame_type=fixed`，所以 `noisy_goal` 不会被模型类读取；但训练 config 里 `pred_frame=noisy_goal`，这通常由 dataset 预处理决定。现在要查 RPDiffDataset 在 `pred_frame=noisy_goal` 时 `pc/pc_action/pc_anchor/T_goal2world` 到底怎么变。
>
> dataset 逻辑确认了：模型训练/评估时 `pc_action/pc_anchor/pc` 都是在 scene-centered 坐标系里；我们 adapter 这部分基本匹配。`noisy_goal` 对 fixed-frame 模型不生效。现在更可疑的是尺度/归一化和 checkpoint 的 `frame_type=fixed + zero_shape` 行为。我继续查 model config 是否用了 `pcd_scale/object_scale`，adapter 是否漏了同样缩放。
>
> 为了验证“transform 方向/组合是否用反”，我准备加一个可配置的 adapter ablation：默认保持现在的 `action_to_pred`，新增 `pred_to_action_inverse` / `use_pred_as_world` 这种小规模测试模式。先改代码和配置；是否立刻跑取决于 GPU1 是否能安全容纳。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/eval/evaluate_rpdiff.py` (+15 -2)
>
> </details>
>
> 补了三个 transform variant，其中 `identity_rotation` 是关键 sanity check：只用预测点云中心做平移，不采用模型/SVD 给出的旋转。如果这个反而更好，说明当前旋转预测/使用方式是主问题。现在加 30-trial 配置。
>
> <details><summary>Created a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/config/full_eval_cfgs/mug_on_rack_multi/mug_on_rack_multi_tax_dpd_identityrot30.yaml` (+47 -0)
>
> </details>
>
> 再加一个反向 SVD 的 30-trial 配置，用来确认 SVD 方向是否影响很大。
>
> <details><summary>Created a file</summary>
>
> - Wrote `~/projects/rpdiff/src/rpdiff/config/full_eval_cfgs/mug_on_rack_multi/mug_on_rack_multi_tax_dpd_invsvd30.yaml` (+47 -0)
>
> </details>
>
> 我同步代码/配置到 H200，然后先做语法检查。之后如果 GPU1 可容纳，我会按顺序跑 `identity_rotation30`，再跑 `invsvd30`。
>
> H200 当前所有 GPU 都在高利用率，只有 GPU1 显存余量理论上够跑一个 30-trial 小实验。我会先启动 `identity_rotation30`；如果 OOM 或被系统杀，会从日志里确认，不会影响已有结果。
>
> `identity_rotation30` 已启动：PID `2379116`，日志 `/home/yuchi/tax_dpd_rpdiff_identityrot30.log`。我等第一个 trial 完成确认没有 OOM。
>
> identity_rotation30 正常在跑，已完成 2/30，暂时 0%。我设置一个 watcher：它结束后自动启动 `invsvd30`，这样两个 ablation 连续跑完。
>
> watcher 已就绪：`identityrot30` 完成后会自动启动 `invsvd30`。我再取一次当前进度。
</details>

已继续下一步，当前正在跑两个 transform 方向/旋转使用方式的 ablation。

已经完成的分析结论：

- `diag50`：`19/50 = 38%`
- `init_diag50`：`12/50 = 24%`
- 固定原始姿态 + parent mean 初始化没有提升，所以“随机初始化”不是主因。
- 成功/失败样本的 SVD RMSE、rotation angle、parent distance 没有明显分界。
- 现在重点转向 adapter 的 transform/rotation 使用方式。

我已改代码加入 `transform_variant`，并启动：

```text
identity_rotation30
```

含义：只用 TAX-DPD 预测点云中心给出的平移，不使用 SVD 拟合出的旋转。这个实验用于判断“旋转预测/旋转使用方式”是不是主要问题。

当前状态：

```text
PID: 2379116
日志: /home/yuchi/tax_dpd_rpdiff_identityrot30.log
配置: mug_on_rack_multi_tax_dpd_identityrot30.yaml
进度: 2 / 30
当前成功率: 0.000
```

同时我已挂 watcher，`identity_rotation30` 结束后自动启动：

```text
invsvd30
```

含义：反向 SVD / inverse transform sanity check，用于确认 transform 方向有没有写反。

watcher：

```text
PID: 2384637
日志: /home/yuchi/tax_dpd_invsvd_after_identityrot_watcher.log
```

你可以检查：

```bash
ssh h200-1

# identity_rotation30
grep -E "Iteration: .*Place Success|Traceback|Error|CUDA out of memory|Killed" \
  ~/tax_dpd_rpdiff_identityrot30.log | tail -30

# watcher / invsvd 是否已启动
cat ~/tax_dpd_invsvd_after_identityrot_watcher.log

# invsvd30，如果已经启动
grep -E "Iteration: .*Place Success|Traceback|Error|CUDA out of memory|Killed" \
  ~/tax_dpd_rpdiff_invsvd30.log | tail -30
```

判断标准：

- 如果 `identity_rotation30` 明显高于 28–38%，说明旋转预测/旋转使用是主问题。
- 如果 `invsvd30` 明显高于当前结果，说明 transform 方向可能写反。
- 如果两个都不提升，下一步就该集中查 frame convention / 模型重构实现，而不是继续调 RPDiff eval 参数。

> 现在是什么状态

<details><summary>1 previous message</summary>

> 我查一下 `identity_rotation30`、`invsvd30` 和 watcher 的状态与最终结果。
</details>

两个 ablation 都已经跑完，我们自己的相关进程都退出了。

结果：

```text
identity_rotation30:
8 / 30 = 26.7%

invsvd30:
8 / 30 = 26.7%
```

对比已有结果：

```text
500-trial 主结果:             28.6%
diag50 random init:            38.0%
init_diag50 orig/parent_mean:  24.0%
identity_rotation30:           26.7%
invsvd30:                      26.7%
```

结论：

- 不使用 SVD 旋转没有提升：`identity_rotation30` 仍然只有 `26.7%`
- 反向 SVD 也没有提升：`invsvd30` 仍然只有 `26.7%`
- 所以问题大概率不是简单的：
  - SVD transform 方向写反；
  - 旋转直接导致全部失败；
  - 随机初始化导致失败。

更可能的问题现在集中在：

```text
TAX3Dv2 重构实现 / frame convention / training target 与 simulator adapter 不一致
```

也就是说，当前 adapter 只是能跑通并得到 26–29% success，但没有复现论文 full pipeline 的核心行为。

下一步最有价值的是检查模型本身是否学到了正确的目标点云：

1. 从 TAX-DPD 原始 dataloader 取 batch；
2. 用 checkpoint 预测；
3. 直接保存 `pc_action / pc_anchor / gt pc / pred pc` 可视化；
4. 对比预测点云是不是在训练数据坐标系里已经偏/旋转错；
5. 如果在 dataloader 内部已经错，就修 `tax3d_v2.py`；
6. 如果 dataloader 内部是对的，但 simulator adapter 错，再修 RPDiff adapter/frame 转换。

> 现在的结果跟论文的结果对比

当前结果和论文差距很大，不能说复现对上了。

最接近的论文任务是 `Mug/Multi-MedRack`。

| 方法 / 结果 | Success Rate |
|---|---:|
| 论文 TAX-DPD full | `95%` |
| 论文 TAX-DPD w/o GMM | `74%` |
| 论文 RPDiff baseline | `86%` |
| 我们当前 500-trial 主结果 | `28.6%` |
| 我们 100-trial 结果 | `26.0%` |
| 当前几个 diagnostic ablation | `24%–38%` |

差距：

```text
对论文 TAX-DPD full:
95.0% - 28.6% = 66.4 个百分点

对论文 w/o GMM:
74.0% - 28.6% = 45.4 个百分点

对论文 RPDiff baseline:
86.0% - 28.6% = 57.4 个百分点
```

所以当前结论是：

```text
当前 adapter-based 复现稳定在约 26–29% success，
明显低于论文 Mug/Multi-MedRack 的 95%，
也低于论文中去掉 GMM 的 74% ablation。
```

原因判断：

- 当前不是论文官方完整 pipeline。
- 仓库缺失官方 `tax3d_v2.py`，现在用的是重构实现。
- 当前 RPDiff simulator eval 是外接 adapter：`TAX-DPD 点云预测 → SVD 拟合 SE(3) → RPDiff execution`。
- 论文 full TAX-DPD 包含更完整的 global/GMM initialization + local refinement/inference 逻辑。
- 我们做的 ablation 说明：简单改初始化、去掉旋转、反向 SVD 都不能把结果拉近论文。

一句话：

```text
目前复现结果约为论文 full TAX-DPD 的 30% 相对水平：
28.6 / 95 ≈ 30.1%。
```

如果要继续往论文靠，下一步应该查 TAX3Dv2 重构实现和训练/推理 frame convention，而不是继续增加 trial 数。

> 具体怎么做 你有思路吗

有思路。现在不能再盲跑实验，应该按“先定位模型内错不错，再定位 adapter 错不错”的顺序做。

具体路线如下。

1. 先做 TAX-DPD dataloader 内部可视化

目的：确认 checkpoint 在它自己的训练数据坐标系里是不是能预测对。

做法：

- 从 `RPDiffDataset` 取 test batch；
- 输入 checkpoint；
- 保存四组点云：
  - `pc_action`
  - `pc_anchor`
  - `pc`，也就是 GT goal point cloud
  - `pred_actions_wta`
- 看 pred 和 GT 是否重合。

判断：

```text
如果 pred 在 TAX-DPD dataloader 内部已经偏很多：
    问题在 tax3d_v2.py 重构/训练/模型实现。

如果 pred 和 GT 在 dataloader 内部基本对：
    问题在 RPDiff simulator adapter/frame 转换。
```

这是最高优先级。

2. 对 dataloader 内部计算 SE(3) 指标

现在已有 RMSE 结果很差，但要更细：

```text
Test RMSE_WTA: 0.1628
Test T_err_WTA: 0.1175
Test R_err_WTA: 51.38°
```

下一步要分样本保存：

- 每个样本的 `rmse_wta`
- `T_err`
- `R_err`
- object id
- parent id
- 预测 transform
- GT transform

然后看：

```text
是所有样本都差？
还是某些 mug/rack 组合特别差？
是 translation 主导？
还是 rotation 主导？
```

3. 检查 `tax3d_v2.py` 重构实现

重点查这些地方：

- `TAX3Dv2FixedFrameModule._get_x_start`
- `_predict_wta`
- `sample_r + sample_s` 的组合是否和训练 loss 一致
- `zero_shape=True` 时 diffusion 的 `sample_s` 是否被正确使用
- `rotation_noise_scale=45.0` 是否和论文/作者实现一致
- `TAX3Dv2_FixedFrame_Token_DiT` 输入 `y/x0` 是否和作者原版一致
- `frame_type=fixed` + `pred_frame=noisy_goal` 这个组合是否合理

现在 checkpoint config 是：

```text
frame_type: fixed
pred_frame: noisy_goal
zero_shape: true
object_scale: 3.0
```

这套组合很可疑。`pred_frame=noisy_goal` 在 dataset 里生成了 `noisy_goal`，但 fixed-frame module 没用它。可能训练目标/模型假设已经和论文不一致。

4. 补一个“oracle adapter”验证 simulator

用 TAX-DPD dataset 或 RPDiff demo 里的 GT transform 直接送进 RPDiff simulator，看 simulator success 能不能接近高分。

判断：

```text
如果 oracle transform 在 simulator 里也低：
    RPDiff eval/task setup 本身有问题。

如果 oracle transform 很高：
    simulator 没问题，问题在模型预测/adapter。
```

这是确认环境是否靠谱的关键 sanity check。

5. 再考虑 GMM/global initialization

论文 full 结果 `95%`，w/o GMM 是 `74%`。我们当前 `28.6%` 比 w/o GMM 还低很多，所以现在不能只说“缺 GMM”。

更合理判断是：

```text
当前模型/adapter 基础预测就不对。
先把基础点云预测修到接近论文 w/o GMM 水平，再补 GMM。
```

6. 具体下一步我会做什么

我建议马上写一个脚本：

```text
scripts/debug_tax3dv2_predictions.py
```

功能：

- 加载 checkpoint；
- 加载 RPDiff test dataset；
- 跑 32 或 64 个样本；
- 输出：
  - per-sample RMSE/T_err/R_err CSV
  - 点云 `.npz`
  - 可视化 `.html` 或 `.ply`
- 保存到：

```text
/home/yuchi/projects/TAX-DPD/debug_outputs/tax3dv2_pred_debug/
```

然后看结果：

```text
Case A: pred 点云在 dataset 内就错
    修 TAX3Dv2 重构实现。

Case B: pred 点云在 dataset 内对
    修 RPDiff adapter。
```

这是现在最短路径。

> 你复现用的什么数据  哪里来的

<details><summary>1 previous message</summary>

> 我查一下本地交接文档和配置，给你一个准确的数据来源/路径说明。
</details>

复现用的是 RPDiff 官方数据，不是论文作者额外发布的 TAX-DPD 数据。

具体是这个任务包：

```text
mug_on_rack_multi_large_proc_gen_demos
```

来源：

```text
RPDiff 官方仓库的 Dropbox 下载链接
```

交接文档里写的位置是 RPDiff 仓库的：

```text
scripts/dl_train_data.bash
```

本地/H200 上的数据路径是：

```text
/home/yuchi/data/rpdiff/
```

主要内容：

```text
/home/yuchi/data/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos
/home/yuchi/data/rpdiff/descriptions/objects
```

规模：

```text
mug_on_rack_multi_large_proc_gen_demos: 约 23GB，3600 个 demo 文件
descriptions/objects: 约 999MB
```

训练时用的配置覆盖是：

```text
dataset=rpdiff
dataset.preprocess=True
dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos
dataset.rpdiff_task_type=task_name_mug_on_rack_multi
dataset.train_dataset_size=3200
dataset.val_dataset_size=400
dataset.test_dataset_size=400
```

训练出的 checkpoint 是：

```text
/home/yuchi/projects/TAX-DPD/scripts/logs/train_rpdiff_tax3dv2/2026-08-03/15-34-11/checkpoints/last.ckpt
```

RPDiff simulator eval 用的物体资产来自：

```text
/home/yuchi/data/rpdiff/descriptions/objects
```

并在 H200 上 symlink 到：

```text
/home/yuchi/projects/rpdiff/src/rpdiff/descriptions/objects
```

需要注意：论文表格里的 `Mug/EasyRack`、`Mug/MedRack`、`Mug/Multi-MedRack` 等任务和我们用的数据最接近的是 `Mug/Multi-MedRack`，但 RPDiff 官方包里没有完全按论文表格命名的 `mug_rack_easy_single / mug_rack_med_single` 数据包。我们用的是 RPDiff 官方可下载的 `mug_on_rack_multi_large_proc_gen_demos`。

> 确认一下ssh h200-1上的仓库内容跟本地的是同步的。然后把所有对应的修改提交commit，让当前仓库指向git@github.com:Yuchi-Zhang-00/TAX-DPD.git，然后推送修改。

> 把当前为止的复现过程出现的问题和结论及原因记录成一个详细的文档。我要让其他人了解当前复现的状态。

<details><summary>5 previous messages</summary>

> 我会在仓库里新增一份复现状态文档，内容覆盖：数据来源、环境/代码改动、训练与评估结果、和论文差距、已排查的问题、当前结论、下一步建议。先看一下现有文档结构和 git 状态，避免覆盖已有记录。
>
> 我会新增一份独立文档，不改 `HANDOFF.md`，避免把“交接文档”和“当前实验状态报告”混在一起。文件放在 `docs/REPRODUCTION_STATUS_2026-08-13.md`。
>
> 我会把文档写成“事实记录 + 可复查路径 + 当前结论”，并明确标注哪些是严格结果、哪些是 adapter-based 近似结果。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./docs/REPRODUCTION_STATUS_2026-08-13.md` (+633 -0)
>
> </details>
>
> 文档已新增。我会快速检查渲染和内容关键段落，并确认 git 状态只新增了这份文档。
</details>

已新增详细状态文档：

[docs/REPRODUCTION_STATUS_2026-08-13.md](./docs/REPRODUCTION_STATUS_2026-08-13.md)

里面记录了：

- 当前复现结论
- 数据来源和路径
- checkpoint / 日志路径
- 训练配置
- RMSE 评估结果
- RPDiff simulator adapter 设置
- 100 / 500 trial success 结果
- diagnostic ablation 结果
- 和论文 Table I 的差距
- 已遇到的问题及处理方式
- 当前最可能原因
- 不建议继续做的事
- 推荐下一步排查路线

核心结论也写进去了：

```text
当前 500-trial simulator success = 143 / 500 = 28.6%
论文 Mug/Multi-MedRack TAX-DPD full = 95%
差距 = 66.4 个百分点
```

文档目前是新增文件，尚未 commit。

> 确认一下ssh h200-1上的仓库内容跟本地的是同步的。然后把所有对应的修改提交commit，让当前仓库指向git@github.com:Yuchi-Zhang-00/TAX-DPD.git，然后推送修改。

<details><summary>11 previous messages</summary>

> 我先确认本地和 H200-1 的仓库状态、分支、HEAD、remote 是否一致；确认后再切 remote、提交并推送。
>
> 我先做精确对比：列出本地和 H200-1 各自的未提交改动文件名差集，确认是否还差几个文件没同步。
>
> 我已经确认：H200-1 和本地并不完全同步，差异主要是本地多了 `scripts/train.py` 和 `docs/REPRODUCTION_STATUS_2026-08-13.md` 两个未在 H200-1 出现的改动/文件。先把这两项同步过去，再做提交和改 remote。
>
> 现在两边文件集一致了。接下来我只做三件事：切 remote 到你的 GitHub 仓库、提交当前改动、推送。提交前我再看一下变更摘要，确保没混进意外文件。
>
> 我已经确认两边文件内容同步了。现在只处理 git：把 origin 改到你的 GitHub 仓库，提交当前改动，然后 push。
>
> 我现在提交。提交内容会包含文档、配置和之前已经做过的代码改动；不会丢弃任何现有修改。
>
> 提交成功。下一步我把本地与 H200-1 的 commit/remote 再核对一下，然后 push 到你的 GitHub 仓库。
>
> 先把本地 commit 推到你的 GitHub 仓库，再让 H200-1 的 `origin` 指向同一个远端并 fetch，至少把 HEAD/remote 对齐；工作区如果还有未提交内容，我会如实说明。
>
> 本地已经推上去了。现在我同步 H200-1 的 remote 到同一个 GitHub 仓库，并拉取最新 main；这样至少仓库指向和提交历史一致。
>
> H200-1 拉取失败的原因很明确：它那台机器当前没有可用的 GitHub SSH key。我先确认一下它是否已经把 remote 改过去了，以及 HEAD 现在停在哪个提交。
>
> H200-1 的 remote 已切到你的 GitHub 仓库，但它本地工作区还停在旧 commit `c4a2f79`，而且有一堆未提交改动。因为 SSH key 缺失，我不能直接从那台机器拉取更新。接下来我会把当前本地提交号给你，并说明 H200-1 上还需要你自己执行的同步命令。
</details>

已完成一半：

- 本地仓库已切到你的 remote：
  - `git@github.com:Yuchi-Zhang-00/TAX-DPD.git`
- 本地已提交并推送：
  - commit: `070002f3959cc7ba463425801004c646ed06c037`
- 这个 commit 已经推到 GitHub 的 `main`

但 H200-1 上还没真正同步到这个最新 commit，原因是：

```text
h200-1 上当前没有可用的 GitHub SSH key
```

所以我在 H200-1 上执行 `git fetch origin` 时失败了：

```text
Permission denied (publickey)
```

现在 H200-1 的状态是：

- `origin` 已改成你的仓库：
  - `git@github.com:Yuchi-Zhang-00/TAX-DPD.git`
- 但本地 `HEAD` 仍是旧的：
  - `c4a2f796911bd3cf9b43abef848aaa7651c3043a`
- 工作区还有未提交改动

如果你要让 H200-1 真正和本地一致，最直接的下一步是在 H200-1 上配置 GitHub SSH key，然后执行：

```bash
cd /home/yuchi/projects/TAX-DPD
git fetch origin
git reset --hard origin/main
```

但这一步会改动 H200-1 上当前工作区，所以我没替你做。
如果你愿意，我可以下一条直接给你一份 H200-1 上可手动执行的最短同步命令清单。

> 可以把当前的结果提一个issue到github上吗。我之前已经提过了一个Hi,
>
> I am trying to implement this work. Looks like there are two missing scripts "tax3d, tax3d_v2" in src/non_rigid/models, while in src/non_rigid/utils/script_utils.py. It writes
>
> from non_rigid.models.tax3d import (
>     DiffusionTransformerNetwork,
>     CrossDisplacementModule,
> )
> from non_rigid.models.tax3d_v2 import (
>     TAX3Dv2Network,
>     TAX3Dv2MuFrameModule,
>     TAX3Dv2FixedFrameModule
> )。具体可以看https://github.com/lyuxinghe/TAX-DPD/issues/1

> 可以把当前的结果提一个issue到github上吗。我之前已经提过了一个Hi,
>
> I am trying to implement this work. Looks like there are two missing scripts "tax3d, tax3d_v2" in src/non_rigid/models, while in src/non_rigid/utils/script_utils.py. It writes
>
> from non_rigid.models.tax3d import (
>     DiffusionTransformerNetwork,
>     CrossDisplacementModule,
> )
> from non_rigid.models.tax3d_v2 import (
>     TAX3Dv2Network,
>     TAX3Dv2MuFrameModule,
>     TAX3Dv2FixedFrameModule
> )。具体可以看https://github.com/lyuxinghe/TAX-DPD/issues/1。提issue之前让我先看一下内容

<details><summary>3 previous messages</summary>

> 我先看一下你之前的 issue 内容和当前仓库文档里的结果，随后给你一版英文 issue 草稿；只草拟，不会提交。
>
> <details><summary>Searched the web</summary>
>
> Searched the web for `https://github.com/lyuxinghe/TAX-DPD/issues/1`
>
> </details>
>
> 我看到了 issue #1：目前只是报告缺少 `tax3d.py` / `tax3d_v2.py`，没有后续复现实验结果。下面我给你一版建议作为 comment 发到同一个 issue，而不是新开 issue；这样上下文更连续。
</details>

建议发到已有 issue #1 下面作为 follow-up comment。内容如下，你先看：

```markdown
Hi,

I wanted to follow up with some reproduction details after trying to reconstruct the missing `tax3d.py` / `tax3d_v2.py` modules locally.

Since the official `src/non_rigid/models/tax3d.py` and `src/non_rigid/models/tax3d_v2.py` are missing from the repository, I implemented a best-effort reconstruction based on the available DiT backbone, config files, and the import interfaces in `script_utils.py`. I was able to run training and evaluation end-to-end on the RPDiff mug-on-rack data, but the results are still far from the paper numbers.

## Data and setup

I used the official RPDiff downloadable data:

```text
mug_on_rack_multi_large_proc_gen_demos
```

with:

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

The closest paper task appears to be `Mug/Multi-MedRack` in Table I.

## Rigid prediction evaluation

After training a reconstructed TAX3Dv2 fixed-frame model, I got the following evaluation metrics:

```text
Val RMSE:       0.2914
Val RMSE_WTA:   0.1719
Val T_err:      0.2534
Val T_err_WTA:  0.1250
Val R_err:      122.71 deg
Val R_err_WTA:  55.21 deg

Test RMSE:      0.2865
Test RMSE_WTA:  0.1628
Test T_err:     0.2481
Test T_err_WTA: 0.1175
Test R_err:     120.96 deg
Test R_err_WTA: 51.38 deg
```

The WTA rotation error remains quite high.

## RPDiff simulator evaluation

I also connected the reconstructed TAX-DPD prediction to the RPDiff PyBullet simulator through an external adapter:

```text
TAX-DPD predicted goal point cloud
→ SVD fit to SE(3)
→ RPDiff PyBullet execution
→ Place Success / touching_surf
```

The simulator success results were:

```text
100 trials: 26 / 100 = 26.0%
500 trials: 143 / 500 = 28.6%
```

I also tried several small diagnostic ablations:

```text
random SO(3) + random parent point init, 50 trials:
19 / 50 = 38.0%

init_orig_ori=True + init_parent_mean_pos=True, 50 trials:
12 / 50 = 24.0%

identity rotation only, 30 trials:
8 / 30 = 26.7%

inverse-SVD sanity check, 30 trials:
8 / 30 = 26.7%
```

These ablations did not close the gap.

For comparison, Table I reports approximately:

```text
TAX-DPD full, Mug/Multi-MedRack: 95%
TAX-DPD w/o GMM, Mug/Multi-MedRack: 74%
RPDiff baseline, Mug/Multi-MedRack: 86%
```

So the current reconstructed implementation is much lower than both the full method and the w/o-GMM ablation.

## Current suspicion

The main issue seems unlikely to be only the RPDiff simulator wrapper or a simple SVD direction bug. The likely causes are:

1. The reconstructed `tax3d_v2.py` does not match the official implementation.
2. The expected frame convention / prediction target is unclear.
3. The combination in the trained config is somewhat confusing:

```text
frame_type: fixed
pred_frame: noisy_goal
zero_shape: true
object_scale: 3.0
```

In my reconstruction, the fixed-frame module does not directly use `batch["noisy_goal"]`, while the dataset does create this field when `pred_frame=noisy_goal`. This may indicate a mismatch between the intended training target and the reconstructed model logic.

4. The full paper pipeline likely also depends on GMM/global initialization and local refinement details that are not recoverable from the current released code.

Would it be possible to release the official implementations of:

```text
src/non_rigid/models/tax3d.py
src/non_rigid/models/tax3d_v2.py
```

and, if available, the exact evaluation/inference config used for the RPDiff Table I results?

That would make it much easier to determine whether the discrepancy is due to my reconstruction, missing GMM/local inference pieces, or a dataset/evaluation mismatch.

Thanks!
```

我建议不要新开 issue，直接评论到你已有的 #1。原因是这个 comment 本质上是对“缺失脚本导致无法复现”的补充证据：我们已经尝试重构并跑出结果，但和论文差距很大，说明需要官方实现。
