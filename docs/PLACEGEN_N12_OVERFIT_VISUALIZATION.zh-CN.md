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
