# N12 单样本诊断可视化

`placegen-visualize-overfit` 只读取已经生成的单样本 overfit report 和该 report 绑定的训练 NPZ；它不加载 checkpoint、不运行模型，也不读取 grouped final/test 数据。输出是包含 Plotly JavaScript 的单文件 HTML。

```bash
pixi run placegen-visualize-overfit \
  --report /path/to/step1000-rescaled-v1.json \
  --output-html /path/to/step1000-rescaled-v1.html \
  --summary /path/to/step1000-rescaled-v1.visualization.json
```

若报告记录的是服务器绝对路径，而 NPZ 已复制到本机，可增加 `--training-npz /relocated/rack-plate-000000.npz`。文件名、sample ID 和 SHA-256 必须仍与 report 一致。`--show` 可选；在服务器上通常只生成 HTML，再用浏览器打开。

颜色语义固定为：灰色 rack parent、蓝色 source child、绿色 target child、红色 predicted child。四类点均为世界系米制坐标。工具从 start child 和 source pose 反解物体系点，再分别乘 report 中的 target/predicted `world_from_object`；加载时还会检查 target 重建是否与训练 NPZ 的 final child 对齐。

训练 NPZ 是 RPDiff 兼容的 object-array 格式，因此只应可视化可信的本项目 N12 产物。工具会先验证 report schema、sample ID 和整个 NPZ 的 SHA-256，再打开固定六字段的训练文件，并拒绝覆盖已有 HTML/旁车 JSON。

## 旋转噪声消融报告

`placegen_overfit_one.py --disable-rotation-noise` 是仅限诊断 harness 的单变量实验开关。它只把当前进程中已实例化 diffusion 对象的 `rotation_noise_scale` 置为 `0.0`，不会改写 `configs/model/tax3dv2.yaml` 或训练配置。生成的 report 会额外带有三个可选字段：

在与普通单样本 overfit 相同的参数后追加该开关即可，例如：

```bash
pixi run placegen-overfit-one \
  --data-root /path/to/data \
  --input-npz /path/to/observation.npz \
  --manifest /path/to/manifest.json \
  --report /path/to/step1000-no-rotation-noise.json \
  --steps 1000 \
  --disable-rotation-noise
```

- `diagnostic_flags`：记录 `disable_rotation_noise` 与 `rescale_learned_sigmas` 两个 CLI 布尔值；
- `diagnostic_actual_state`：记录网络/扩散方差与损失类型，以及实际采用的旋转噪声尺度；
- `rotation_noise_override`：记录开关状态、修改前后的尺度和 `changed` 判定。

可视化器对这些字段采用严格的可选 schema：字段出现时必须完整匹配预期键集合、布尔值/字符串/非负有限尺度类型，并交叉核对 flag、override 和实际 diffusion 状态。旧版没有这些字段的 report 仍可读取；字段缺失、增加未知键或三者互相矛盾都会拒绝加载。

固定方差（`learn_sigma=false`）目前不属于本实验节点：重建的 DDRD-separate 固定方差采样路径尚未实现，因此没有暴露一个会在采样阶段才失败的 CLI 假设。

## B300 匹配重复与 TAX-Pose fallback

2026-08-28 在 B300-2/GPU 2 用同一 `rack-plate-000000`、`seed=7`、
`sample_seed=1701` 和 1000 optimizer steps 重跑了两次 native `45°` 与两次 harness-only
rotation-off `0°`：

| run | rotation | ordered RMSE (m) | world translation (mm) | world rotation (deg) | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| native-v1 | 45° | 1.189097 | 1153.094 | 18.829 | severe red |
| native-v2 | 45° | 1.153506 | 1145.168 | 8.011 | severe red |
| rotation-off-v1 | 0° | 0.712311 | 619.529 | 16.696 | severe red |
| rotation-off-v2 | 0° | 0.867246 | 793.240 | 13.688 | severe red |

四次均 `finite=true`、参数确实更新，但都没有记住一个样本。rotation-off 在两次匹配运行中仍有
方向性改善，却没有消除米级错误；因此它是贡献因素，不是可提交到生产配置的充分解释。对应 report
SHA 为 `0892b72a...e066c`、`27cc1180...30212`、`dbdc9c6e...7ae2b`、
`d548909f...d849`；新 native/rotation-off-v2 的自包含 Plotly HTML 也已用本页工具生成并镜像到
H200。完整 SHA 与路径记录在 PlaceGen 的迁移指南 §13.6。

这次持续失败后的上游审计确认，官方 `lyuxinghe/TAX-DPD@c4a2f796` 并不是论文链路的完整发布：

- README 仍把安装、训练、评测和 checkpoint 标为未发布；
- `script_utils.py` 导入的 `tax3d.py`、`tax3d_v2.py`、`datasets/real_world.py` 等文件不在官方树；
- 论文要求的完整刚体链为 Dense GMM → 局部解耦点扩散 → RANSAC-SVD，而当前 N12 明确是
  `reconstructed-fixed-frame-w/o-GMM`。

因此当前结果只能判断**本项目的重建 wrapper 失败**，不能反驳论文的完整方法，也不能靠增加训练步数
消除实现不确定性。论文或公开代码没有证据证明还存在一个神秘的、未披露的位姿后处理器；可以确认的
缺口是 wrapper、配置/checkpoint、数据生成/尺度统计、GMM 接线、RANSAC 参数和执行器配置。

作为 fallback，已审计 `r-pad/taxpose`。当前 main `0c4298fa` 的一个 PM adapter 调用点与
`CustomTransformer` 的字典返回值不一致；若采用祖先基线，必须先固定到
`corl_2022_release@8bf3b96592310a5f4ce656ff83ca431f560e9e62`，再处理 B300 上
`torch.det` 的 NVRTC 不兼容和硬编码 `.cuda()`。TAX-Pose 是 cross-pose flow + weighted SVD，盘子—
沥水架没有现成 checkpoint，仍需用 development demonstrations 重新训练；它不能直接替换当前
TAX-DPD checkpoint。一个 main/dgcnn、128-dim、直接点损失的 N12 harness 在临时 3×3 determinant
兼容改动下，500/2000 steps 得到 `27.705/26.887 mm` ordered RMSE、约 `187.624 MiB` allocator peak。
这个 harness 不是官方 TAX-Pose loss、也不是非过拟合结果，只证明模型 forward/backward 和适配方向
可行，不能进入 AnyPlace-vs-TAX-DPD 比较表。

服务器上对应的两个 JSON 末尾目前写入了字面量 `}\\n`（反斜杠加 `n`），不是合法 JSON 的普通
换行；`json.tool` 因而会报告 `Extra data`。原始文件保持 no-replace 以便审计；正式 fallback 运行
应重新生成规范 JSON，并把新 SHA 写入 manifest，不要就地覆盖这些诊断 artifact。
