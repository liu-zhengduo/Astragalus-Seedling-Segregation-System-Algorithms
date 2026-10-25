# 苗体抓取检测 V2（效果优先版）

兼容性更新（2026-07-31）：兼容不同 OpenCV 版本中 `HoughLinesP`
返回的 `(N, 1, 4)`、`(N, 4)` 和 `(4,)` 数组形状，修复
`TypeError: cannot unpack non-iterable numpy.int32 object`。

V2 保留了欧氏距离变换和 marker-controlled watershed，并增加“厚度核心、骨架短枝剪除、交叉方向 Hough 拆分、距离加权 PCA、碎段合并和置信度”几层处理。目标从“所有黑色线条”调整为“可供夹爪抓取的粗壮主体”，从而减少细根误检，并把 X/L/Y 形交叉中的不同方向尽量拆开。

## 安装与运行

```powershell
pip install -r requirements.txt
python seedling_grasp_v2.py "D:\images\test.jpg" -o "D:\results\test" --show
```

批量处理：

```powershell
python batch_run_v2.py "D:\images" "D:\results"
```

批量模式不需要 `--show`。每张图片单独生成 8 张过程图和一个 `poses.csv`。

## 输出说明

- 红点：置信度不低于 0.5 的候选；
- 橙点：置信度低于 0.5，需要人工或后续逻辑复核；
- 紫线：姿态角；
- CSV 新增 `confidence`、`thickness_px`、`length_px` 和 `method` 字段。

若机械手只抓高置信度主体，可运行：

```powershell
python seedling_grasp_v2.py input.jpg -o result --min-confidence 0.5
```

## 关键参数

`--core-radius` 默认是 `0.004`，表示主体最小半径约为图片短边的 0.4%。

- 细根误检仍多：提高到 `0.0045` 或 `0.005`；
- 细苗漏检：降低到 `0.0035`；
- 同一批相机、镜头和工作距离固定后，应固定这个参数，不建议每张图单独改变。

## 六张现场图回归

| 图片 | V1 候选 | V2 全部 | V2 置信度≥0.5 |
|---|---:|---:|---:|
| 8772… | 4 | 5 | 5 |
| 6043… | 5 | 7 | 5 |
| 51d3… | 14 | 10 | 3 |
| ceca… | 17 | 16 | 15 |
| cc3d… | 17 | 15 | 12 |
| 9b78… | 19 | 16 | 13 |

这些数字是候选数量，不是准确率。六张图尚未提供逐株人工真值，因此不能据此计算 precision、recall 或 F1。肉眼检查可确认：V2 找回了第一张右侧粗壮主体，能把第六张的大型 X 形交叉拆成两个方向，并显著减少第三、第五、第六张的细根候选；个别弯折主体仍可能出现两个候选。

如果识别效果是最终验收标准，下一轮需要给这些图片各标一个“真实抓取点”（或提供每株实例掩膜）。有了真值后才能自动搜索 `core-radius`、合并距离和置信度阈值，给出可复现的准确率，而不是只凭肉眼继续调参。
