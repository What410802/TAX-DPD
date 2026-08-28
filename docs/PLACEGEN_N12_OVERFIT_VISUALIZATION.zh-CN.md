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
